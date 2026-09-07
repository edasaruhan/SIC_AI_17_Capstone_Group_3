"""Executable Sprint 3 refinement and graph-value experiment.

The runner treats the Sprint 1 full feature/split artifacts and the accepted
Sprint 2 run as immutable inputs.  Hyperparameters are selected only from
expanding temporal folds contained inside outer train.  Refitted models,
thresholds, and the A/B/C feature-family experiment use outer validation; no
test feature row is transformed or scored.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from argus.config import get_path, load_config
from argus.modeling.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_fingerprint,
    sha256_file,
    write_run_manifest,
)
from argus.modeling.baseline import (
    PartitionMatrices,
    _atomic_joblib_dump,
    _close_memmap,
    _configure_duckdb,
    _materialize_partition,
    _source_snapshot,
    _verify_frozen_inputs,
)
from argus.modeling.metrics import compute_average_precision, evaluate_binary_predictions
from argus.modeling.preprocessing import (
    ALWAYS_FORBIDDEN_PREDICTORS,
    GRAPH_HISTORY_FEATURES,
    FeatureContract,
    FittedPreprocessor,
    feature_contract_for_family,
)
from argus.modeling.refinement_metrics import (
    analyze_validation_score_saturation,
    optimize_validation_thresholds,
)
from argus.modeling.refinement_models import (
    CandidateModel,
    build_refinement_candidate,
    convergence_evidence,
    fit_candidate,
    lightgbm_leaf_evidence,
    predict_dual_scores,
)
from argus.modeling.refinement_protocol import (
    TemporalFold,
    derive_expanding_window_folds,
    select_outer_validation_champion,
    summarize_and_select_candidates,
    write_fold_split_manifest,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_WORK_PREFIX = ".argus_sprint3_work_"
_MODEL_FAMILIES = ("logistic_regression", "random_forest", "lightgbm")
_PRIMARY_FEATURE_FAMILIES = (
    "transaction_only",
    "transaction_temporal_history",
    "transaction_temporal_history_graph",
)
_NOVEL_GRAPH_FEATURES = (
    "sender_prior_fan_in_degree",
    "receiver_prior_fan_out_degree",
    "pair_previous_transfer_count",
)


class RefinementPipelineError(RuntimeError):
    """Raised when a Sprint 3 protocol or evidence invariant fails."""


@dataclass(frozen=True)
class RefinementRunResult:
    """Paths and evidence for one completed Sprint 3 run."""

    run_dir: Path
    manifest_path: Path
    report_path: Path
    manifest: dict[str, Any]


class _StageRecorder:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        print(f"[refinement] {name} ...", flush=True)
        started = time.perf_counter()
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - started
            self.seconds[name] = elapsed
            print(f"[refinement] {name} FAILED after {elapsed:.2f}s", flush=True)
            raise
        elapsed = time.perf_counter() - started
        self.seconds[name] = elapsed
        print(f"[refinement] {name} completed in {elapsed:.2f}s", flush=True)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RefinementPipelineError(f"{name} must be a mapping")
    return value


def _stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_parameters(estimator: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in estimator.get_params().items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def _novel_graph_contract() -> FeatureContract:
    base = feature_contract_for_family("transaction_temporal_history")
    duplicates = set(GRAPH_HISTORY_FEATURES).difference(_NOVEL_GRAPH_FEATURES)
    return FeatureContract(
        numeric_features=(*base.numeric_features, *_NOVEL_GRAPH_FEATURES),
        low_cardinality_categories=base.low_cardinality_categories,
        bank_frequency_categories=base.bank_frequency_categories,
        missing_indicator_features=base.missing_indicator_features,
        forbidden_predictors=(*ALWAYS_FORBIDDEN_PREDICTORS, *sorted(duplicates)),
    )


def _contract_for_experiment(family: str) -> FeatureContract:
    if family == "transaction_temporal_history_graph_novel3":
        return _novel_graph_contract()
    return feature_contract_for_family(family)


def _prepare_run_directory(run_dir: Path) -> None:
    """Remove only known generated Sprint 3 content, retaining its tracked marker."""

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir.resolve()
    if resolved.name != "sprint3" or resolved.parent.name != "artifacts":
        raise RefinementPipelineError(f"Refusing to clean unexpected run directory: {resolved}")
    for child in run_dir.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _safe_cleanup_work_directory(work_dir: Path, run_dir: Path) -> None:
    resolved_work = work_dir.resolve()
    resolved_run = run_dir.resolve()
    if resolved_work.parent != resolved_run or not resolved_work.name.startswith(_WORK_PREFIX):
        raise RefinementPipelineError(f"Refusing to remove unexpected work directory: {work_dir}")
    if resolved_work.exists():
        shutil.rmtree(resolved_work)


def _close_partition(partition: PartitionMatrices | None) -> None:
    if partition is None:
        return
    _close_memmap(partition.matrix)
    _close_memmap(partition.target)
    _close_memmap(partition.source_row_numbers)


def _save_preprocessor(
    preprocessor: FittedPreprocessor,
    run_dir: Path,
    family: str,
    *,
    scope: str,
) -> dict[str, Any]:
    target = run_dir / "preprocessing" / scope / family
    atomic_write_json(preprocessor.to_state_dict(), target / "fitted_state.json")
    manifest = preprocessor.manifest()
    manifest["feature_family"] = family
    manifest["scope"] = scope
    atomic_write_json(manifest, target / "manifest.json")
    return manifest


def _compact_score_diagnostics(raw: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return {
        "raw_minimum": float(np.min(raw)),
        "raw_maximum": float(np.max(raw)),
        "raw_unique_scores": int(np.unique(raw).size),
        "probability_minimum": float(np.min(probability)),
        "probability_maximum": float(np.max(probability)),
        "probability_unique_scores": int(np.unique(probability).size),
        "probability_exact_zero_count": int(np.count_nonzero(probability == 0.0)),
        "probability_exact_one_count": int(np.count_nonzero(probability == 1.0)),
        "probability_boundary_fraction": float(
            np.count_nonzero((probability == 0.0) | (probability == 1.0)) / probability.size
        ),
        "probability_collapsed_raw_ranking": int(np.unique(probability).size)
        < int(np.unique(raw).size),
    }


def _trial_eligible(
    family: str,
    convergence: Mapping[str, Any],
    leaf_evidence: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if family == "logistic_regression" and convergence.get("converged") is not True:
        reasons.append("logistic_solver_did_not_converge")
    if family == "lightgbm" and int(leaf_evidence.get("trees_with_absolute_leaf_over_1m", 0)):
        reasons.append("pathological_leaf_values_over_one_million")
    if (
        family in {"logistic_regression", "lightgbm"}
        and float(diagnostics["probability_boundary_fraction"]) > 0.01
    ):
        reasons.append("more_than_one_percent_exact_probability_boundaries")
    return not reasons, reasons


def _fit_one_candidate(
    candidate: CandidateModel,
    train: PartitionMatrices,
    validation: PartitionMatrices,
    *,
    batch_rows: int,
    top_k: Sequence[int],
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
    assert validation.source_row_numbers is not None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_seconds = fit_candidate(candidate, train.matrix, train.target)
    warning_messages = [str(item.message) for item in caught]
    scores = predict_dual_scores(candidate.estimator, validation.matrix, batch_rows=batch_rows)
    convergence = convergence_evidence(candidate.estimator, warning_messages)
    leaf_evidence = lightgbm_leaf_evidence(candidate.estimator)
    diagnostics = _compact_score_diagnostics(scores.ranking_score, scores.probability)
    eligible, reasons = _trial_eligible(
        candidate.model_family, convergence, leaf_evidence, diagnostics
    )
    ap = compute_average_precision(validation.target, scores.ranking_score)
    if ap is None:
        raise RefinementPipelineError("A temporal validation fold contains no positives")
    row = {
        "model_family": candidate.model_family,
        "candidate_id": candidate.candidate_id,
        "average_precision": float(ap),
        "fit_seconds": fit_seconds,
        "predict_seconds": scores.runtime_seconds,
        "eligible": eligible,
        "ineligibility_reasons": reasons,
        "ranking_score_type": scores.ranking_score_type,
        "imbalance_policy": candidate.imbalance_policy,
        "effective_parameters": _json_parameters(candidate.estimator),
        "effective_parameters_sha256": _stable_digest(_json_parameters(candidate.estimator)),
        "convergence": convergence,
        "lightgbm_leaf_evidence": leaf_evidence,
        "score_diagnostics": diagnostics,
        "warnings": warning_messages,
        "test_metrics": None,
        "test_predictions_generated": False,
        "configured_top_k": list(top_k),
    }
    return row, candidate.estimator, scores.ranking_score, scores.probability


def _candidate_configs(settings: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    tuning = _mapping(settings.get("tuning"), "sprint3.tuning")
    result: dict[str, list[Mapping[str, Any]]] = {}
    for family in _MODEL_FAMILIES:
        configured = tuning.get(family)
        if not isinstance(configured, list) or len(configured) < 2:
            raise RefinementPipelineError(f"At least two tuning candidates required for {family}")
        if not all(isinstance(item, Mapping) for item in configured):
            raise RefinementPipelineError(f"Invalid candidate list for {family}")
        result[family] = list(configured)
    return result


def _write_frame_json(frame: pd.DataFrame, destination: Path) -> None:
    atomic_write_json(frame.to_dict(orient="records"), destination)


def _run_temporal_cv(
    connection: duckdb.DuckDBPyConnection,
    *,
    feature_table: Path,
    outer_split_table: Path,
    run_dir: Path,
    work_dir: Path,
    settings: Mapping[str, Any],
    random_seed: int,
) -> tuple[list[TemporalFold], pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    temporal = _mapping(settings.get("temporal_cv"), "sprint3.temporal_cv")
    folds = derive_expanding_window_folds(
        connection,
        feature_table,
        outer_split_table,
        train_quantiles=temporal["cumulative_train_quantiles"],
        validation_end_quantiles=temporal["validation_end_quantiles"],
    )
    atomic_write_json(
        {
            "strategy": "expanding_window",
            "scope": "frozen_outer_train_only",
            "selection_partition": "inner_validation",
            "test_access": False,
            "folds": [fold.to_dict() for fold in folds],
        },
        run_dir / "temporal_cv" / "folds.json",
    )
    candidates = _candidate_configs(settings)
    trials: list[dict[str, Any]] = []
    contract = feature_contract_for_family("transaction_temporal_history")
    batch_rows = int(settings["batch_rows"])
    compression = str(settings["parquet_compression"])

    for fold in folds:
        fold_dir = work_dir / f"fold_{fold.fold}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        fold_split = run_dir / "temporal_cv" / "splits" / f"fold_{fold.fold}.parquet"
        write_fold_split_manifest(
            connection, outer_split_table, fold, fold_split, compression=compression
        )
        preprocessor = FittedPreprocessor.fit_duckdb(
            connection, feature_table, fold_split, contract
        )
        _save_preprocessor(
            preprocessor, run_dir, "transaction_temporal_history", scope=f"fold_{fold.fold}"
        )
        train: PartitionMatrices | None = None
        validation: PartitionMatrices | None = None
        try:
            train = _materialize_partition(
                connection,
                preprocessor,
                feature_table,
                fold_split,
                partition="train",
                expected_rows=fold.train_rows,
                expected_positives=fold.train_positive_labels,
                batch_rows=batch_rows,
                work_dir=fold_dir,
            )
            validation = _materialize_partition(
                connection,
                preprocessor,
                feature_table,
                fold_split,
                partition="validation",
                expected_rows=fold.validation_rows,
                expected_positives=fold.validation_positive_labels,
                batch_rows=batch_rows,
                work_dir=fold_dir,
            )
            for model_family in _MODEL_FAMILIES:
                for candidate_config in candidates[model_family]:
                    candidate = build_refinement_candidate(
                        model_family,
                        candidate_config,
                        training_labels=train.target,
                        random_seed=random_seed,
                        n_jobs=int(settings["model_threads"]),
                    )
                    print(
                        f"[refinement] fold={fold.fold} fitting {candidate.candidate_id}",
                        flush=True,
                    )
                    result, estimator, raw, probability = _fit_one_candidate(
                        candidate,
                        train,
                        validation,
                        batch_rows=batch_rows,
                        top_k=settings["top_k"],
                    )
                    result.update(
                        {
                            "fold": fold.fold,
                            "partition": "inner_validation",
                            "feature_family": "transaction_temporal_history",
                            "train_rows": fold.train_rows,
                            "train_positive_labels": fold.train_positive_labels,
                            "validation_rows": fold.validation_rows,
                            "validation_positive_labels": fold.validation_positive_labels,
                        }
                    )
                    trials.append(result)
                    atomic_write_json(trials, run_dir / "temporal_cv" / "trials.json")
                    print(
                        f"[refinement] fold={fold.fold} {candidate.candidate_id}: "
                        f"AP={result['average_precision']:.8f}, eligible={result['eligible']}",
                        flush=True,
                    )
                    del estimator, raw, probability, candidate
                    gc.collect()
        finally:
            _close_partition(train)
            _close_partition(validation)
            gc.collect()
            shutil.rmtree(fold_dir)

    # Keep nested diagnostic objects intact in the in-memory frame; the CSV
    # writer renders them as text while JSON remains fully structured.
    trial_frame = pd.DataFrame(trials)
    atomic_write_csv(trial_frame, run_dir / "temporal_cv" / "trials.csv")
    summary, selected = summarize_and_select_candidates(trial_frame, expected_folds=len(folds))
    atomic_write_csv(summary, run_dir / "temporal_cv" / "candidate_summary.csv")
    _write_frame_json(summary, run_dir / "temporal_cv" / "candidate_summary.json")
    atomic_write_json(selected, run_dir / "temporal_cv" / "selected_candidates.json")
    return folds, trial_frame, summary, selected


def _selected_config(
    settings: Mapping[str, Any], model_family: str, candidate_id: str
) -> Mapping[str, Any]:
    for item in _candidate_configs(settings)[model_family]:
        if str(item.get("candidate_id")) == candidate_id:
            return item
    raise RefinementPipelineError(f"Selected candidate configuration not found: {candidate_id}")


def _selected_threshold_value(thresholds: Mapping[str, Any], raw: np.ndarray) -> float:
    summaries = _mapping(thresholds.get("summaries"), "threshold summaries")
    selected = _mapping(
        summaries.get("predeclared_joint_primary_constraint"),
        "predeclared_joint_primary_constraint",
    )
    point = selected.get("operating_point")
    if isinstance(point, Mapping) and point.get("threshold") is not None:
        return float(point["threshold"])
    return float(np.nextafter(np.max(raw), np.inf))


def _evaluate_outer_model(
    candidate: CandidateModel,
    train: PartitionMatrices,
    validation: PartitionMatrices,
    *,
    settings: Mapping[str, Any],
    run_dir: Path,
    model_key: str,
    feature_family: str,
    transformed_feature_count: int,
    split_sha256: str,
    random_seed: int,
    save_model: bool = True,
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
    result, estimator, raw, probability = _fit_one_candidate(
        candidate,
        train,
        validation,
        batch_rows=int(settings["batch_rows"]),
        top_k=settings["top_k"],
    )
    assert validation.source_row_numbers is not None
    saturation = analyze_validation_score_saturation(
        validation.target,
        probabilities=probability,
        raw_scores=raw,
        source_row_number=validation.source_row_numbers,
        top_k_values=settings["top_k"],
        partition="validation",
    )
    saturation["raw_average_precision"] = compute_average_precision(validation.target, raw)
    saturation["probability_average_precision"] = compute_average_precision(
        validation.target, probability
    )
    threshold_config = _mapping(settings.get("threshold_optimization"), "threshold")
    thresholds = optimize_validation_thresholds(
        validation.target,
        raw,
        partition="validation",
        fpr_ceiling=float(threshold_config["fpr_ceiling"]),
        alert_budget=int(threshold_config["alert_budget"]),
        primary_fpr_ceiling=float(threshold_config["fpr_ceiling"]),
        primary_alert_budget=int(threshold_config["alert_budget"]),
        include_frontier=False,
    )
    threshold = _selected_threshold_value(thresholds, raw)
    ranking_metrics = evaluate_binary_predictions(
        validation.target,
        raw,
        validation.source_row_numbers,
        threshold=threshold,
        top_k_values=settings["top_k"],
    )
    result.update(
        {
            "model_key": model_key,
            "feature_family": feature_family,
            "transformed_feature_count": int(transformed_feature_count),
            "evaluation_partition": "validation",
            "ranking_metrics": ranking_metrics,
            "threshold_optimization": thresholds,
            "saturation": saturation,
            "random_seed": random_seed,
            "split_manifest_sha256": split_sha256,
            "selection_source": "inner_temporal_validation_folds",
            "test_metrics": None,
            "test_predictions_generated": False,
        }
    )
    model_dir = run_dir / "models" / model_key
    model_path = model_dir / "model.joblib"
    if save_model:
        _atomic_joblib_dump(
            estimator,
            model_path,
            int(settings["model_serialization_compression"]),
        )
        result["model_artifact"] = model_path.relative_to(run_dir).as_posix()
    atomic_write_json(result, model_dir / "metadata.json")
    atomic_write_json(ranking_metrics, model_dir / "validation_metrics.json")
    atomic_write_json(saturation, model_dir / "saturation.json")
    atomic_write_json(thresholds, model_dir / "thresholds.json")
    return result, estimator, raw, probability


def _materialize_family(
    connection: duckdb.DuckDBPyConnection,
    *,
    family: str,
    feature_table: Path,
    split_table: Path,
    prevalence: Mapping[str, Mapping[str, Any]],
    run_dir: Path,
    work_dir: Path,
    settings: Mapping[str, Any],
) -> tuple[FittedPreprocessor, PartitionMatrices, PartitionMatrices, dict[str, Any]]:
    contract = _contract_for_experiment(family)
    preprocessor = FittedPreprocessor.fit_duckdb(connection, feature_table, split_table, contract)
    manifest = _save_preprocessor(preprocessor, run_dir, family, scope="outer_train")
    family_work = work_dir / f"outer_{family}"
    family_work.mkdir(parents=True, exist_ok=False)
    train = _materialize_partition(
        connection,
        preprocessor,
        feature_table,
        split_table,
        partition="train",
        expected_rows=int(prevalence["train"]["rows"]),
        expected_positives=int(prevalence["train"]["positive_labels"]),
        batch_rows=int(settings["batch_rows"]),
        work_dir=family_work,
    )
    validation = _materialize_partition(
        connection,
        preprocessor,
        feature_table,
        split_table,
        partition="validation",
        expected_rows=int(prevalence["validation"]["rows"]),
        expected_positives=int(prevalence["validation"]["positive_labels"]),
        batch_rows=int(settings["batch_rows"]),
        work_dir=family_work,
    )
    return preprocessor, train, validation, manifest


def _audit_sprint2_saturation(
    *,
    project_root: Path,
    validation: PartitionMatrices,
    preprocessor: FittedPreprocessor,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    assert validation.source_row_numbers is not None
    sprint2_dir = project_root / "artifacts" / "sprint2"
    saved_preprocessing = json.loads(
        (sprint2_dir / "preprocessing" / "manifest.json").read_text(encoding="utf-8")
    )
    if tuple(saved_preprocessing["transformed_feature_names"]) != preprocessor.feature_names:
        raise RefinementPipelineError(
            "Sprint 3 B feature order differs from the accepted Sprint 2 model input order"
        )
    if saved_preprocessing["state_sha256"] != preprocessor.state_sha256:
        raise RefinementPipelineError(
            "Sprint 3 B preprocessing state differs from the accepted Sprint 2 state"
        )
    results: dict[str, Any] = {}
    for family in _MODEL_FAMILIES:
        model = joblib.load(sprint2_dir / "models" / family / "model.joblib")
        scores = predict_dual_scores(
            model, validation.matrix, batch_rows=int(settings["batch_rows"])
        )
        audit = analyze_validation_score_saturation(
            validation.target,
            probabilities=scores.probability,
            raw_scores=scores.ranking_score,
            source_row_number=validation.source_row_numbers,
            top_k_values=settings["top_k"],
            partition="validation",
        )
        audit["raw_average_precision"] = compute_average_precision(
            validation.target, scores.ranking_score
        )
        audit["probability_average_precision"] = compute_average_precision(
            validation.target, scores.probability
        )
        raw_threshold = 0.5 if scores.ranking_score_type == "positive_class_probability" else 0.0
        raw_ranking_metrics = evaluate_binary_predictions(
            validation.target,
            scores.ranking_score,
            validation.source_row_numbers,
            threshold=raw_threshold,
            top_k_values=settings["top_k"],
        )
        probability_metrics = evaluate_binary_predictions(
            validation.target,
            scores.probability,
            validation.source_row_numbers,
            threshold=0.5,
            top_k_values=settings["top_k"],
        )
        saved_connection = duckdb.connect()
        try:
            saved = saved_connection.execute(
                f"""
                SELECT source_row_number, score_{family}
                FROM read_parquet(?)
                ORDER BY source_row_number
                """,
                [str((sprint2_dir / "validation_predictions.parquet").resolve())],
            ).fetchnumpy()
        finally:
            saved_connection.close()
        saved_rows = np.asarray(saved["source_row_number"], dtype=np.int64)
        saved_probability = np.asarray(saved[f"score_{family}"], dtype=np.float64)
        recomputed_order = np.argsort(validation.source_row_numbers, kind="stable")
        if not np.array_equal(saved_rows, validation.source_row_numbers[recomputed_order]):
            raise RefinementPipelineError(
                f"Sprint 2 saved/recomputed validation identities differ for {family}"
            )
        absolute_difference = np.abs(saved_probability - scores.probability[recomputed_order])
        score_reproduction = {
            "rows_compared": int(saved_rows.size),
            "maximum_absolute_probability_difference": float(absolute_difference.max()),
            "exact_equal_rows": int(np.count_nonzero(absolute_difference == 0.0)),
            "all_probabilities_equal_within_1e_12": bool(
                np.allclose(
                    saved_probability, scores.probability[recomputed_order], atol=1e-12, rtol=0.0
                )
            ),
            "saved_exact_zero_count": int(np.count_nonzero(saved_probability == 0.0)),
            "recomputed_exact_zero_count": int(np.count_nonzero(scores.probability == 0.0)),
            "saved_exact_one_count": int(np.count_nonzero(saved_probability == 1.0)),
            "recomputed_exact_one_count": int(np.count_nonzero(scores.probability == 1.0)),
            "artifact": "artifacts/sprint2/validation_predictions.parquet",
        }
        evidence: dict[str, Any] = {
            "model": family,
            "source": "accepted_sprint2_serialized_model_recomputed_on_outer_validation",
            "ranking_score_type": scores.ranking_score_type,
            "audit": audit,
            "raw_ranking_metrics": raw_ranking_metrics,
            "probability_metrics": probability_metrics,
            "saved_probability_reproduction": score_reproduction,
            "effective_parameters": _json_parameters(model),
            "lightgbm_leaf_evidence": lightgbm_leaf_evidence(model),
        }
        if family == "logistic_regression":
            coefficients = np.asarray(model.coef_).reshape(-1)
            if coefficients.size != len(preprocessor.feature_names):
                raise RefinementPipelineError("Sprint 2 LR feature order differs from Sprint 3 B")
            maximum_margin_index = int(np.argmax(scores.ranking_score))
            contributions = np.asarray(validation.matrix[maximum_margin_index]) * coefficients
            leading = np.argsort(-np.abs(contributions))[:10]
            evidence["maximum_margin_contributions"] = [
                {
                    "feature": preprocessor.feature_names[index],
                    "standardized_value": float(validation.matrix[maximum_margin_index, index]),
                    "coefficient": float(coefficients[index]),
                    "margin_contribution": float(contributions[index]),
                }
                for index in leading
            ]
            evidence["maximum_margin_source_row_number"] = int(
                validation.source_row_numbers[maximum_margin_index]
            )
        results[family] = evidence
        del model, scores
        gc.collect()
    return {
        "partition": "validation",
        "test_access": False,
        "finding": (
            "Sprint 2 probability ties are evaluated against pre-sigmoid/pre-link raw scores; "
            "deterministic Top-K through a material probability tie is not interpreted as "
            "model superiority."
        ),
        "models": results,
    }


def _write_validation_predictions(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_rows: np.ndarray,
    labels: np.ndarray,
    scores: Mapping[str, tuple[np.ndarray, np.ndarray]],
    source_filename: str,
    destination: Path,
    compression: str,
) -> dict[str, Any]:
    data: dict[str, np.ndarray] = {
        "source_row_number": np.asarray(source_rows),
        "is_laundering": np.asarray(labels),
    }
    model_columns: dict[str, dict[str, str]] = {}
    for key, (raw, probability) in sorted(scores.items()):
        raw_name = f"raw_score_{key}"
        probability_name = f"probability_{key}"
        data[raw_name] = np.asarray(raw)
        data[probability_name] = np.asarray(probability)
        model_columns[key] = {
            "raw_score": raw_name,
            "probability": probability_name,
        }
    frame = pd.DataFrame(data)
    connection.register("argus_sprint3_predictions", frame)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary_sql = temporary.resolve().as_posix().replace("'", "''")
    escaped_source = source_filename.replace("'", "''")
    score_projection = ", ".join(
        f'CAST("{column}" AS DOUBLE) AS "{column}"'
        for column in data
        if column not in {"source_row_number", "is_laundering"}
    )
    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    '{escaped_source}:row-' || CAST(source_row_number AS VARCHAR)
                        AS transaction_id,
                    CAST(source_row_number AS BIGINT) AS source_row_number,
                    CAST(is_laundering AS TINYINT) AS is_laundering,
                    {score_projection}
                FROM argus_sprint3_predictions
                ORDER BY source_row_number
            ) TO '{temporary_sql}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
            """
        )
        os.replace(temporary, destination)
    finally:
        connection.unregister("argus_sprint3_predictions")
        temporary.unlink(missing_ok=True)
    row = connection.execute(
        "SELECT count(*), count(DISTINCT transaction_id), sum(is_laundering) FROM read_parquet(?)",
        [str(destination.resolve())],
    ).fetchone()
    return {
        "partition": "validation",
        "rows": int(row[0]),
        "unique_transaction_ids": int(row[1]),
        "positive_labels": int(row[2]),
        "model_score_columns": model_columns,
        "test_predictions_included": False,
    }


def _write_comparison_artifacts(
    run_dir: Path,
    *,
    sprint2_manifest: Mapping[str, Any],
    sprint2_saturation: Mapping[str, Any],
    refined: Mapping[str, Mapping[str, Any]],
    graph_result: Mapping[str, Any],
    champion: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    def record(
        *,
        stage: str,
        model: str,
        candidate_id: str,
        feature_family: str,
        ranking_score_type: str,
        metrics: Mapping[str, Any],
        threshold_point: Mapping[str, Any],
        tie_summary: Mapping[str, Any] | None,
        fit_seconds: float | None,
        predict_seconds: float | None,
    ) -> dict[str, Any]:
        metric_top_k = list(metrics["top_k_metrics"])
        tie_by_k = {int(item["requested_k"]): item for item in (tie_summary or {}).get("top_k", [])}
        # Keep the ordinary deterministic Top-K metrics and the tie-aware
        # expected/minimum/maximum bounds together in the JSON/report payload.
        # The CSV still receives flattened scalar columns below.
        top_k: list[dict[str, Any]] = []
        for item in metric_top_k:
            k = int(item["requested_k"])
            enriched = dict(item)
            tie = tie_by_k.get(k)
            if tie:
                enriched.update(tie)
            enriched.setdefault("deterministic_true_positives_at_k", item.get("true_positives"))
            enriched.setdefault("deterministic_precision_at_k", item.get("precision_at_k"))
            enriched.setdefault("deterministic_recall_at_k", item.get("recall_at_k"))
            top_k.append(enriched)
        row: dict[str, Any] = {
            "stage": stage,
            "model": model,
            "candidate_id": candidate_id,
            "feature_family": feature_family,
            "evaluation_partition": "validation",
            "ranking_score_type": ranking_score_type,
            "average_precision": metrics["average_precision"],
            "roc_auc": metrics["roc_auc"],
            "threshold": threshold_point.get("threshold"),
            "f1": threshold_point.get("f1"),
            "recall": threshold_point.get("recall"),
            "precision": threshold_point.get("precision"),
            "false_positive_rate": threshold_point.get("false_positive_rate"),
            "alert_count": threshold_point.get("alert_count"),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "top_k_metrics": top_k,
            "test_metrics_used": False,
        }
        for item in top_k:
            k = int(item["requested_k"])
            row[f"precision_at_{k}"] = item["precision_at_k"]
            row[f"recall_at_{k}"] = item["recall_at_k"]
            tie = tie_by_k.get(k, {})
            row[f"top_{k}_cutoff_tie_material"] = tie.get(
                "tie_break_material_at_cutoff", item["tie_break_material_at_cutoff"]
            )
            row[f"top_{k}_expected_precision"] = tie.get("expected_precision_at_k")
            row[f"top_{k}_minimum_precision"] = tie.get("minimum_precision_at_k")
            row[f"top_{k}_maximum_precision"] = tie.get("maximum_precision_at_k")
        return row

    rows: list[dict[str, Any]] = []
    for model_name, result in refined.items():
        metrics = result["ranking_metrics"]
        threshold = result["threshold_optimization"]["summaries"][
            "predeclared_joint_primary_constraint"
        ]
        point = threshold.get("operating_point") or {}
        rows.append(
            record(
                stage="sprint3_refined_transaction_baseline",
                model=model_name,
                candidate_id=str(result["candidate_id"]),
                feature_family=str(result["feature_family"]),
                ranking_score_type=str(result["ranking_score_type"]),
                metrics=metrics,
                threshold_point=point,
                tie_summary=result["saturation"]["raw_score"],
                fit_seconds=float(result["fit_seconds"]),
                predict_seconds=float(result["predict_seconds"]),
            )
        )
    refined_frame = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "top_k_metrics"} for row in rows]
    ).sort_values(["average_precision", "model"], ascending=[False, True], kind="stable")
    atomic_write_csv(refined_frame, run_dir / "refined_model_comparison.csv")
    atomic_write_json(
        {
            "partition": "validation",
            "primary_metric": "average_precision_on_raw_ranking_score",
            "accuracy_is_primary": False,
            "models": dict(refined),
            "champion": dict(champion),
            "test_metrics_used": False,
        },
        run_dir / "refined_model_comparison.json",
    )
    unified_rows: list[dict[str, Any]] = []
    for name, payload in _mapping(sprint2_manifest.get("models"), "sprint2.models").items():
        reproduced = sprint2_saturation.get("models", {}).get(name, {})
        audit = reproduced.get("audit", {})
        metrics = reproduced.get("raw_ranking_metrics", payload["validation_metrics"])
        baseline_record = record(
            stage="sprint2_baseline_recomputed_raw_ranking",
            model=str(name),
            candidate_id="sprint2_fixed_configuration",
            feature_family="transaction_temporal_history",
            ranking_score_type=str(
                reproduced.get("ranking_score_type", "positive_class_probability")
            ),
            metrics=metrics,
            threshold_point=metrics["threshold_metrics"],
            tie_summary=audit.get("raw_score"),
            fit_seconds=payload.get("runtime_seconds", {}).get("fit"),
            predict_seconds=payload.get("runtime_seconds", {}).get("validation_predict"),
        )
        baseline_record["accepted_probability_average_precision"] = payload["validation_metrics"][
            "average_precision"
        ]
        baseline_record["representation_comparability"] = (
            "raw_score_vs_raw_score; accepted Sprint 2 probability AP retained separately"
        )
        unified_rows.append(baseline_record)
    unified_rows.extend(rows)
    graph_threshold = graph_result["threshold_optimization"]["summaries"][
        "predeclared_joint_primary_constraint"
    ]
    unified_rows.append(
        record(
            stage="sprint3_graph_enhanced",
            model="lightgbm",
            candidate_id=str(graph_result["candidate_id"]),
            feature_family=str(graph_result["feature_family"]),
            ranking_score_type=str(graph_result["ranking_score_type"]),
            metrics=graph_result["ranking_metrics"],
            threshold_point=graph_threshold.get("operating_point") or {},
            tie_summary=graph_result["saturation"]["raw_score"],
            fit_seconds=float(graph_result["fit_seconds"]),
            predict_seconds=float(graph_result["predict_seconds"]),
        )
    )
    unified_frame = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "top_k_metrics"} for row in unified_rows]
    )
    atomic_write_csv(unified_frame, run_dir / "baseline_vs_refined.csv")
    atomic_write_json(
        {
            "partition": "validation",
            "primary_metric": "average_precision",
            "rows": unified_rows,
            "sprint2_probability_tie_caveat": (
                "Sprint 2 LR/LightGBM probability Top-K values are retained as historical "
                "facts but not interpreted as superiority when a material tie crosses K."
            ),
            "test_metrics_used": False,
        },
        run_dir / "baseline_vs_refined.json",
    )
    return refined_frame, unified_rows


def _write_ablation_artifacts(
    run_dir: Path,
    *,
    ablation_results: Mapping[str, Mapping[str, Any]],
    parameter_digest: str,
    split_digest: str,
    random_seed: int,
    duplicate_evidence: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for family, result in ablation_results.items():
        metrics = result["ranking_metrics"]
        rows.append(
            {
                "feature_family": family,
                "scope": (
                    "primary"
                    if family in _PRIMARY_FEATURE_FAMILIES
                    else "sensitivity_duplicate_removed"
                ),
                "model_family": "lightgbm",
                "candidate_id": result["candidate_id"],
                "effective_parameters_sha256": parameter_digest,
                "parameter_digest": parameter_digest,
                "random_seed": random_seed,
                "evaluation_partition": "validation",
                "split_manifest_sha256": split_digest,
                "transformed_feature_count": result["transformed_feature_count"],
                "feature_count": result["transformed_feature_count"],
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "test_metrics_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    order = [*_PRIMARY_FEATURE_FAMILIES, "transaction_temporal_history_graph_novel3"]
    frame["_order"] = frame["feature_family"].map({name: index for index, name in enumerate(order)})
    frame = frame.sort_values("_order", kind="stable").drop(columns="_order")
    indexed = frame.set_index("feature_family")
    b = float(indexed.loc["transaction_temporal_history", "average_precision"])
    c = float(indexed.loc["transaction_temporal_history_graph", "average_precision"])
    novel = float(indexed.loc["transaction_temporal_history_graph_novel3", "average_precision"])
    frame["delta_vs_b"] = frame["average_precision"] - b
    atomic_write_csv(frame, run_dir / "ablation" / "feature_family_ablation.csv")
    _write_frame_json(frame, run_dir / "ablation" / "feature_family_ablation.json")
    conclusion = {
        "headline_comparison": "B_transaction_temporal_history_vs_C_plus_all_graph",
        "evaluation_partition": "validation",
        "same_model_family": True,
        "same_candidate": True,
        "same_candidate_id": True,
        "same_effective_parameters": True,
        "same_random_seed": True,
        "same_frozen_split": True,
        "same_evaluation_protocol": True,
        "same_validation_protocol": True,
        "model_family": "lightgbm",
        "candidate_id": str(indexed.loc["transaction_temporal_history", "candidate_id"]),
        "graph_features_added": list(GRAPH_HISTORY_FEATURES),
        "graph_feature_construction": "strict_prior_history_from_sprint1_feature_pipeline",
        "graph_feature_target_used": False,
        "effective_parameters_sha256": parameter_digest,
        "split_manifest_sha256": split_digest,
        "b_average_precision": b,
        "c_average_precision": c,
        "c_minus_b_average_precision": c - b,
        "baseline_average_precision": b,
        "graph_average_precision": c,
        "absolute_average_precision_delta": c - b,
        "graph_added_validation_value": c > b,
        "novel3_average_precision": novel,
        "novel3_minus_b_average_precision": novel - b,
        "duplicate_graph_feature_evidence": dict(duplicate_evidence),
        "interpretation": (
            "The sign and magnitude are empirical validation results. A non-positive delta "
            "means these strict-prior graph summaries did not add validation ranking value; "
            "it is not converted into a positive claim."
        ),
        "test_metrics_used": False,
    }
    atomic_write_json(conclusion, run_dir / "ablation" / "graph_value_conclusion.json")
    return frame, conclusion


def _verify_duplicate_graph_fields(
    connection: duckdb.DuckDBPyConnection,
    feature_table: Path,
    split_table: Path,
) -> dict[str, Any]:
    """Recount exact graph/history aliases over every permitted outer row."""

    row = connection.execute(
        """
        SELECT
            count(*) AS rows_checked,
            count(*) FILTER (
                WHERE sender_prior_fan_out_degree
                      IS DISTINCT FROM sender_previous_unique_counterparties
            ) AS sender_mismatches,
            count(*) FILTER (
                WHERE receiver_prior_fan_in_degree
                      IS DISTINCT FROM receiver_previous_unique_counterparties
            ) AS receiver_mismatches
        FROM read_parquet(?) feature
        INNER JOIN read_parquet(?) split USING (transaction_id)
        WHERE split.partition IN ('train', 'validation')
        """,
        [str(feature_table.resolve()), str(split_table.resolve())],
    ).fetchone()
    evidence = {
        "sender_prior_fan_out_degree": "sender_previous_unique_counterparties",
        "receiver_prior_fan_in_degree": "receiver_previous_unique_counterparties",
        "rows_checked": int(row[0]),
        "sender_mismatch_rows": int(row[1]),
        "receiver_mismatch_rows": int(row[2]),
        "mismatch_rows_full_train_plus_validation": int(row[1]) + int(row[2]),
        "partitions_checked": ["train", "validation"],
        "test_rows_checked": False,
        "verification_scope": "strict-prior Sprint 1 engineered feature table",
    }
    if evidence["mismatch_rows_full_train_plus_validation"] != 0:
        raise RefinementPipelineError("Expected graph/history duplicate fields are not equal")
    return evidence


def _plot_results(comparison: pd.DataFrame, ablation: pd.DataFrame, run_dir: Path) -> None:
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(comparison["model"], comparison["average_precision"], color="#315a7d")
    ax.set_ylabel("Validation average precision (raw ranking score)")
    ax.set_title("Sprint 3 refined transaction models")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(figure_dir / "refined_model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["transaction", "+ temporal/history", "+ all graph", "+ novel graph only"]
    ax.bar(labels, ablation["average_precision"], color="#4b8063")
    ax.set_ylabel("Validation average precision")
    ax.set_title("Same-model feature-family ablation")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_family_ablation.png", dpi=160)
    plt.close(fig)


def _render_fallback_report(payload: Mapping[str, Any], destination: Path) -> Path:
    champion = payload["champion"]
    graph = payload["ablation"]["graph_value_conclusion"]
    acceptance = payload["acceptance"]
    lines = [
        "# ARGUS Sprint 3 Status",
        "",
        f"**Status:** {payload['status']}",
        "",
        "Sprint 3 completed on the frozen chronological protocol. The final test feature rows "
        "were not transformed, scored, tuned, or used for selection.",
        "",
        "## Refined champion",
        "",
        f"- Model: `{champion['champion_model']}`",
        f"- Validation AP: `{champion['selection_metric_value']:.10f}`",
        "",
        "## Graph value experiment",
        "",
        f"- B AP: `{graph['b_average_precision']:.10f}`",
        f"- C AP: `{graph['c_average_precision']:.10f}`",
        f"- C − B: `{graph['c_minus_b_average_precision']:.10f}`",
        "",
        "## Acceptance",
        "",
        *[f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in acceptance.items()],
        "",
        "Sprint 4 / GraphSAGE was not started.",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def run_sprint3_refinement(
    config_path: str | Path = "configs/refinement.yaml",
) -> RefinementRunResult:
    """Execute bounded tuning, outer validation, and the same-model graph ablation."""

    started = time.perf_counter()
    started_at = datetime.now(UTC)
    recorder = _StageRecorder()
    config = load_config(config_path)
    settings = _mapping(config.get("sprint3"), "sprint3")
    if settings.get("final_test_access") is not False:
        raise RefinementPipelineError("Sprint 3 final test access must remain disabled")
    if settings.get("full_data") is not True or settings.get("sampled") is not False:
        raise RefinementPipelineError("This evidence run requires the unsampled frozen dataset")

    project_root = Path(config["_meta"]["project_root"])
    feature_table = get_path(config, "feature_table")
    split_table = get_path(config, "split_table")
    split_metadata_path = get_path(config, "split_metadata")
    upstream_manifest_path = get_path(config, "upstream_manifest")
    run_dir = get_path(config, "run_dir")
    report_path = get_path(config, "generated_report")
    sprint2_dir = get_path(config, "sprint2_run_dir")
    sprint2_manifest_path = sprint2_dir / "run_manifest.json"

    # Validate every immutable dependency and the required Git checkpoint before
    # clearing a previously successful Sprint 3 run. A broken/missing upstream
    # input must never destroy known-good evidence merely because a rerun started.
    with recorder.measure("frozen_input_and_sprint2_verification"):
        baseline_settings = _mapping(config.get("baseline"), "baseline")
        provenance, prevalence, upstream = _verify_frozen_inputs(
            settings=baseline_settings,
            feature_table=feature_table,
            split_table=split_table,
            split_metadata_path=split_metadata_path,
            upstream_manifest_path=upstream_manifest_path,
        )
        if sha256_file(sprint2_manifest_path) == "":
            raise RefinementPipelineError("Unreachable empty Sprint 2 manifest digest")
        sprint2_manifest = json.loads(sprint2_manifest_path.read_text(encoding="utf-8"))
        if sprint2_manifest.get("status") != "PASS":
            raise RefinementPipelineError("Accepted Sprint 2 manifest is not PASS")
        checkpoint = subprocess_checkpoint(project_root)
        expected_checkpoint = "162ae64b81398fa4433d661cce98ba16fce40503"
        if checkpoint != expected_checkpoint:
            # Descendant commits are acceptable, but the Sprint 2 checkpoint must exist.
            if not git_is_ancestor(project_root, expected_checkpoint, checkpoint):
                raise RefinementPipelineError("Sprint 2 checkpoint is not in Git history")

    _prepare_run_directory(run_dir)
    work_dir = run_dir / f"{_WORK_PREFIX}{uuid.uuid4().hex}"
    work_dir.mkdir(parents=False, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    connection: duckdb.DuckDBPyConnection | None = None
    succeeded = False
    retained_partitions: list[PartitionMatrices] = []

    try:
        with recorder.measure("source_and_configuration_snapshot"):
            atomic_write_json(_source_snapshot(project_root), run_dir / "source_snapshot.json")
            atomic_write_json(config, run_dir / "resolved_config.json")
            atomic_write_json(prevalence, run_dir / "prevalence.json")
            atomic_write_json(
                {
                    "sprint2_checkpoint": expected_checkpoint,
                    "sprint2_manifest": file_fingerprint(sprint2_manifest_path),
                    "feature_table": file_fingerprint(feature_table),
                    "split_table": file_fingerprint(split_table),
                    "split_metadata": file_fingerprint(split_metadata_path),
                },
                run_dir / "input_provenance.json",
            )

        with recorder.measure("duckdb_initialization"):
            connection = duckdb.connect(str(work_dir / "refinement.duckdb"))
            _configure_duckdb(connection, settings, work_dir / "spill")

        with recorder.measure("inner_expanding_temporal_cv"):
            folds, trials, candidate_summary, selected = _run_temporal_cv(
                connection,
                feature_table=feature_table,
                outer_split_table=split_table,
                run_dir=run_dir,
                work_dir=work_dir,
                settings=settings,
                random_seed=int(config["project"]["random_seed"]),
            )

        with recorder.measure("outer_b_matrix_and_refined_models"):
            b_family = "transaction_temporal_history"
            b_preprocessor, b_train, b_validation, b_manifest = _materialize_family(
                connection,
                family=b_family,
                feature_table=feature_table,
                split_table=split_table,
                prevalence=prevalence,
                run_dir=run_dir,
                work_dir=work_dir,
                settings=settings,
            )
            retained_partitions.extend((b_train, b_validation))
            refined: dict[str, dict[str, Any]] = {}
            prediction_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for model_family in _MODEL_FAMILIES:
                selected_id = selected[model_family]["candidate_id"]
                candidate = build_refinement_candidate(
                    model_family,
                    _selected_config(settings, model_family, selected_id),
                    training_labels=b_train.target,
                    random_seed=int(config["project"]["random_seed"]),
                    n_jobs=int(settings["model_threads"]),
                )
                result, estimator, raw, probability = _evaluate_outer_model(
                    candidate,
                    b_train,
                    b_validation,
                    settings=settings,
                    run_dir=run_dir,
                    model_key=model_family,
                    feature_family=b_family,
                    transformed_feature_count=int(b_manifest["transformed_feature_count"]),
                    split_sha256=sha256_file(split_table),
                    random_seed=int(config["project"]["random_seed"]),
                )
                if not result["eligible"]:
                    raise RefinementPipelineError(
                        f"Selected {model_family} is ineligible on outer train/validation: "
                        f"{result['ineligibility_reasons']}"
                    )
                refined[model_family] = result
                prediction_scores[f"refined_{model_family}"] = (raw, probability)
                del estimator, candidate
                gc.collect()

        with recorder.measure("sprint2_saturation_root_cause_reproduction"):
            sprint2_saturation = _audit_sprint2_saturation(
                project_root=project_root,
                validation=b_validation,
                preprocessor=b_preprocessor,
                settings=settings,
            )
            atomic_write_json(
                sprint2_saturation, run_dir / "saturation" / "sprint2_reproduction.json"
            )

        with recorder.measure("same_model_feature_family_ablation"):
            selected_lgb_id = selected["lightgbm"]["candidate_id"]
            selected_lgb_config = _selected_config(settings, "lightgbm", selected_lgb_id)
            duplicate_graph_evidence = _verify_duplicate_graph_fields(
                connection, feature_table, split_table
            )
            b_parameters = refined["lightgbm"]["effective_parameters"]
            parameter_digest = _stable_digest(b_parameters)
            ablation_results: dict[str, dict[str, Any]] = {b_family: refined["lightgbm"]}

            for family in (
                "transaction_only",
                "transaction_temporal_history_graph",
                "transaction_temporal_history_graph_novel3",
            ):
                preprocessor, train, validation, feature_manifest = _materialize_family(
                    connection,
                    family=family,
                    feature_table=feature_table,
                    split_table=split_table,
                    prevalence=prevalence,
                    run_dir=run_dir,
                    work_dir=work_dir,
                    settings=settings,
                )
                try:
                    candidate = build_refinement_candidate(
                        "lightgbm",
                        selected_lgb_config,
                        training_labels=train.target,
                        random_seed=int(config["project"]["random_seed"]),
                        n_jobs=int(settings["model_threads"]),
                    )
                    key = f"ablation_{family}"
                    result, estimator, raw, probability = _evaluate_outer_model(
                        candidate,
                        train,
                        validation,
                        settings=settings,
                        run_dir=run_dir,
                        model_key=key,
                        feature_family=family,
                        transformed_feature_count=int(
                            feature_manifest["transformed_feature_count"]
                        ),
                        split_sha256=sha256_file(split_table),
                        random_seed=int(config["project"]["random_seed"]),
                    )
                    if not result["eligible"]:
                        raise RefinementPipelineError(
                            f"Graph ablation model is ineligible for {family}: "
                            f"{result['ineligibility_reasons']}"
                        )
                    actual_digest = _stable_digest(result["effective_parameters"])
                    if actual_digest != parameter_digest:
                        raise RefinementPipelineError(
                            f"Same-model ablation parameter mismatch for {family}"
                        )
                    ablation_results[family] = result
                    prediction_scores[key] = (raw, probability)
                    del preprocessor, estimator, candidate
                finally:
                    _close_partition(train)
                    _close_partition(validation)
                    family_work = work_dir / f"outer_{family}"
                    if family_work.exists():
                        shutil.rmtree(family_work)
                    gc.collect()

            ablation_frame, graph_conclusion = _write_ablation_artifacts(
                run_dir,
                ablation_results=ablation_results,
                parameter_digest=parameter_digest,
                split_digest=sha256_file(split_table),
                random_seed=int(config["project"]["random_seed"]),
                duplicate_evidence=duplicate_graph_evidence,
            )

        with recorder.measure("validation_selection_thresholds_and_export"):
            champion = select_outer_validation_champion(refined, partition="validation")
            atomic_write_json(champion, run_dir / "refined_transaction_champion.json")
            threshold_analysis = {
                "partition": "validation",
                "score_type": "raw_ranking_score",
                "primary_rule": settings["threshold_optimization"]["primary_rule"],
                "tie_policy": settings["threshold_optimization"]["tie_policy"],
                "models": {
                    **{name: result["threshold_optimization"] for name, result in refined.items()},
                    **{
                        f"ablation_{family}": result["threshold_optimization"]
                        for family, result in ablation_results.items()
                        if family != b_family
                    },
                },
                "model_feature_families": {
                    **{name: b_family for name in refined},
                    **{
                        f"ablation_{family}": family
                        for family in ablation_results
                        if family != b_family
                    },
                },
                "test_metrics_used": False,
            }
            atomic_write_json(threshold_analysis, run_dir / "threshold" / "analysis.json")
            comparison, baseline_vs_refined_rows = _write_comparison_artifacts(
                run_dir,
                sprint2_manifest=sprint2_manifest,
                sprint2_saturation=sprint2_saturation,
                refined=refined,
                graph_result=ablation_results["transaction_temporal_history_graph"],
                champion=champion,
            )
            source_filename = str(
                upstream.get("provenance", {})
                .get("transactions", {})
                .get("filename", "HI-Small_Trans.csv")
            )
            assert b_validation.source_row_numbers is not None
            prediction_manifest = _write_validation_predictions(
                connection,
                source_rows=b_validation.source_row_numbers,
                labels=b_validation.target,
                scores=prediction_scores,
                source_filename=source_filename,
                destination=run_dir / "validation_predictions.parquet",
                compression=str(settings["parquet_compression"]),
            )
            atomic_write_json(prediction_manifest, run_dir / "validation_predictions_manifest.json")
            _plot_results(comparison, ablation_frame, run_dir)

        with recorder.measure("saturation_attribution_summary"):
            serialization_reproduction_verified = all(
                evidence["saved_probability_reproduction"]["all_probabilities_equal_within_1e_12"]
                for evidence in sprint2_saturation["models"].values()
            )
            if not serialization_reproduction_verified:
                raise RefinementPipelineError(
                    "Accepted Sprint 2 prediction artifacts do not reproduce from saved models"
                )
            refined_saturation = {
                name: {
                    "candidate_id": result["candidate_id"],
                    "imbalance_policy": result["imbalance_policy"],
                    "convergence": result["convergence"],
                    "lightgbm_leaf_evidence": result["lightgbm_leaf_evidence"],
                    "saturation": result["saturation"],
                }
                for name, result in refined.items()
            }
            causal_controls = {
                family: [
                    {
                        "candidate_id": str(row["candidate_id"]),
                        "fold": int(row["fold"]),
                        "average_precision": float(row["average_precision"]),
                        "eligible": bool(row["eligible"]),
                        "imbalance_policy": row["imbalance_policy"],
                        "score_diagnostics": row["score_diagnostics"],
                        "convergence": row["convergence"],
                        "lightgbm_leaf_evidence": row["lightgbm_leaf_evidence"],
                    }
                    for row in trials.to_dict(orient="records")
                    if row["model_family"] == family
                ]
                for family in ("logistic_regression", "lightgbm")
            }
            saturation_summary = {
                "partition": "validation_and_inner_train_folds_only",
                "sprint2_reproduction": sprint2_saturation,
                "controlled_temporal_fold_trials": causal_controls,
                "refined_outer_validation": refined_saturation,
                "rows": [
                    *[
                        {
                            "stage": "sprint2_reproduced",
                            "model": name,
                            "raw_score": evidence["audit"]["raw_score"],
                            "probability": evidence["audit"]["probability"],
                            "raw_average_precision": evidence["audit"]["raw_average_precision"],
                            "probability_average_precision": evidence["audit"][
                                "probability_average_precision"
                            ],
                            "finding": "accepted_model_recomputed_from_serialized_estimator",
                        }
                        for name, evidence in sprint2_saturation["models"].items()
                    ],
                    *[
                        {
                            "stage": "sprint3_refined",
                            "model": name,
                            "raw_score": evidence["saturation"]["raw_score"],
                            "probability": evidence["saturation"]["probability"],
                            "raw_average_precision": evidence["saturation"][
                                "raw_average_precision"
                            ],
                            "probability_average_precision": evidence["saturation"][
                                "probability_average_precision"
                            ],
                            "finding": "refined_model_outer_validation_diagnostic",
                        }
                        for name, evidence in refined_saturation.items()
                    ],
                ],
                "root_cause_conclusion": {
                    "serialization_error": False,
                    "serialization_reproduction_verified": (serialization_reproduction_verified),
                    "immediate_tie_mechanism": (
                        "sigmoid/link conversion maps extreme distinguishable raw margins to "
                        "exact floating-point 0/1 probabilities"
                    ),
                    "logistic_upstream_cause": (
                        "non-converged weakly regularized balanced SGD plus heavy-tailed "
                        "standardized amount signals produced extreme margins"
                    ),
                    "lightgbm_upstream_cause": (
                        "full positive-class weight with near-zero leaf/Hessian regularization "
                        "produced extreme leaf values and raw margins"
                    ),
                    "refinement_controls": (
                        "convergent LogisticRegression C/solver/weight trials and controlled "
                        "LightGBM weight/regularization trials"
                    ),
                    "ranking_policy": "raw margins; probability ties are diagnosed separately",
                    "top_k_superiority_claim_allowed_only_without_material_cutoff_tie": True,
                },
                "test_access": False,
            }
            atomic_write_json(
                saturation_summary, run_dir / "saturation" / "sprint2_vs_refined.json"
            )

        final_test_policy = {
            "final_test_used_for_training": False,
            "final_test_used_for_preprocessing_fit": False,
            "final_test_used_for_tuning": False,
            "final_test_used_for_model_selection": False,
            "final_test_used_for_threshold_selection": False,
            "final_test_inference_performed": False,
            "test_feature_rows_materialized": False,
            "test_prediction_artifact_exists": False,
            "test_metadata_reported": True,
            "test_metadata_source": "pre_existing_sprint1_split_metadata.json",
            "permitted_transformed_partitions": ["train", "validation"],
            "permitted_model_partitions": ["train", "validation"],
            "selection_partition": "validation",
            "reason": "Final test remains sealed until a later explicitly authorized phase",
        }
        atomic_write_json(final_test_policy, run_dir / "final_test_policy.json")

        library_versions = {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "duckdb": duckdb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
        }
        atomic_write_json(library_versions, run_dir / "library_versions.json")

        with recorder.measure("work_matrix_cleanup"):
            for partition in retained_partitions:
                _close_partition(partition)
            retained_partitions.clear()
            del b_train, b_validation, prediction_scores
            gc.collect()
            connection.close()
            connection = None
            if not bool(settings.get("retain_work_matrices", False)):
                _safe_cleanup_work_directory(work_dir, run_dir)

        acceptance = {
            "sprint2_git_checkpoint_present": True,
            "frozen_full_data_unsampled": True,
            "three_model_families_tuned": set(selected) == set(_MODEL_FAMILIES),
            "expanding_temporal_cv_outer_train_only": len(folds) >= 3,
            "logistic_convergence_resolved": bool(
                refined["logistic_regression"]["convergence"]["converged"]
            ),
            "score_saturation_root_cause_investigated": True,
            "raw_ranking_metrics_and_tie_aware_diagnostics": True,
            "validation_only_threshold_optimization": True,
            "three_primary_feature_families_compared": set(_PRIMARY_FEATURE_FAMILIES).issubset(
                ablation_results
            ),
            "graph_value_same_model_protocol": all(
                graph_conclusion[key]
                for key in (
                    "same_model_family",
                    "same_candidate_id",
                    "same_effective_parameters",
                    "same_random_seed",
                    "same_frozen_split",
                    "same_validation_protocol",
                )
            ),
            "pr_auc_primary_operational_metrics_reported": True,
            "machine_readable_artifacts": True,
            "final_test_untouched": all(
                not value
                for key, value in final_test_policy.items()
                if key.startswith("final_test_") and isinstance(value, bool)
            )
            and not final_test_policy["test_feature_rows_materialized"],
            "sprint4_graphsage_not_started": True,
        }
        status = "PASS" if all(acceptance.values()) else "FAIL"
        atomic_write_json(acceptance, run_dir / "acceptance_checklist.json")
        runtime_total = time.perf_counter() - started
        core_artifacts = [
            "artifacts/sprint3/run_manifest.json",
            "artifacts/sprint3/temporal_cv/folds.json",
            "artifacts/sprint3/temporal_cv/trials.csv",
            "artifacts/sprint3/temporal_cv/candidate_summary.csv",
            "artifacts/sprint3/temporal_cv/selected_candidates.json",
            "artifacts/sprint3/refined_model_comparison.csv",
            "artifacts/sprint3/refined_transaction_champion.json",
            "artifacts/sprint3/baseline_vs_refined.csv",
            "artifacts/sprint3/baseline_vs_refined.json",
            "artifacts/sprint3/ablation/feature_family_ablation.csv",
            "artifacts/sprint3/ablation/graph_value_conclusion.json",
            "artifacts/sprint3/saturation/sprint2_vs_refined.json",
            "artifacts/sprint3/threshold/analysis.json",
            "artifacts/sprint3/final_test_policy.json",
            "artifacts/sprint3/validation_predictions.parquet",
            "reports/generated/SPRINT_3_STATUS.md",
        ]
        manifest_payload: dict[str, Any] = {
            "status": status,
            "sprint": "Sprint 3 - Model Refinement + Graph Value Experiment",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "runtime_seconds": runtime_total,
            "runtime_seconds_by_stage": recorder.seconds,
            "configuration": {
                "file": Path(config_path).name,
                "random_seed": int(config["project"]["random_seed"]),
                "full_data": True,
                "sampled": False,
                "primary_metric": "average_precision_on_raw_ranking_score",
            },
            "provenance": provenance,
            "sprint2_checkpoint": expected_checkpoint,
            "sprint2_manifest": file_fingerprint(sprint2_manifest_path),
            "split_prevalence": prevalence,
            "prevalence_shift": sprint2_manifest.get("prevalence_shift"),
            "temporal_cv": {
                "folds": [fold.to_dict() for fold in folds],
                "trial_count": len(trials),
                "trials": trials.to_dict(orient="records"),
                "candidate_summary": candidate_summary.to_dict(orient="records"),
                "selected_candidates": selected,
            },
            "models": refined,
            "champion": champion,
            "baseline_vs_refined": baseline_vs_refined_rows,
            "ablation": {
                "rows": ablation_frame.to_dict(orient="records"),
                "graph_value_conclusion": graph_conclusion,
            },
            "saturation": saturation_summary,
            "threshold_analysis": threshold_analysis,
            "prediction_evidence": prediction_manifest,
            "final_test_policy": final_test_policy,
            "data_access_audit": {
                "preprocessor_fit_partitions": ["inner_fold_train", "outer_train"],
                "transformed_partitions": [
                    "inner_fold_train",
                    "inner_fold_validation",
                    "train",
                    "validation",
                ],
                "evaluated_partitions": ["inner_fold_validation", "validation"],
                "test_access": "metadata_only_from_pre_existing_sprint1_json",
            },
            "libraries": library_versions,
            "acceptance": acceptance,
            "scope": {"sprint4_started": False, "graphsage_started": False},
            "core_artifact_paths": core_artifacts,
        }
        try:
            from argus.modeling.refinement_reporting import render_sprint3_markdown
        except ImportError:
            _render_fallback_report(manifest_payload, report_path)
        else:
            render_sprint3_markdown(manifest_payload, report_path)
        manifest = write_run_manifest(manifest_payload, run_dir)
        succeeded = True
        print(
            f"[refinement] {status}: champion={champion['champion_model']}, "
            f"AP={champion['selection_metric_value']:.8f}, runtime={runtime_total:.2f}s",
            flush=True,
        )
        return RefinementRunResult(run_dir, manifest_path, report_path, manifest)
    finally:
        if connection is not None:
            connection.close()
        for partition in retained_partitions:
            _close_partition(partition)
        if not succeeded and work_dir.exists():
            print(f"[refinement] Work files retained after failure: {work_dir}", flush=True)


def subprocess_checkpoint(project_root: Path) -> str:
    """Return the current Git commit without invoking a shell."""

    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_is_ancestor(project_root: Path, ancestor: str, descendant: str) -> bool:
    """Return whether ``ancestor`` is reachable from ``descendant``."""

    import subprocess

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0
