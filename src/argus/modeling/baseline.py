"""Executable full-data Sprint 2 transaction-baseline experiment.

The module consumes the immutable Sprint 1 feature and split artifacts. Learned
preprocessing is fitted on training rows, all three models see the same matrices,
and only validation predictions are evaluated. Test facts in the report come from
the already-published Sprint 1 split metadata; test feature rows are never
materialized and test inference is deliberately unavailable in this runner.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from argus.config import get_path, load_config
from argus.modeling.artifacts import (
    atomic_write_json,
    file_fingerprint,
    sha256_file,
    write_run_manifest,
)
from argus.modeling.metrics import evaluate_binary_predictions
from argus.modeling.models import (
    build_baseline_models,
    fit_baseline,
    json_safe_parameters,
    predict_positive_probability,
)
from argus.modeling.preprocessing import (
    FORBIDDEN_PREDICTORS,
    GRAPH_HISTORY_FEATURES,
    FeatureContract,
    FittedPreprocessor,
    iter_duckdb_transformed_batches,
)
from argus.modeling.reporting import (
    comparison_context_from_run,
    plot_metric_comparison,
    plot_precision_recall_curves,
    render_sprint2_markdown,
    write_comparison_table,
)
from argus.modeling.selection import select_transaction_baseline_champion

_WORK_PREFIX = ".argus_sprint2_work_"
_MODEL_PARTITIONS = frozenset({"train", "validation"})


class BaselinePipelineError(RuntimeError):
    """Raised when a Sprint 2 invariant or frozen input check fails."""


@dataclass(frozen=True)
class BaselineRunResult:
    """Paths and evidence for a completed Sprint 2 run."""

    run_dir: Path
    manifest_path: Path
    report_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PartitionMatrices:
    """Disk-backed transformed data for one allowed model partition."""

    partition: str
    matrix: np.memmap
    target: np.memmap
    source_row_numbers: np.memmap | None
    rows: int
    positive_labels: int
    batches: int


class _StageRecorder:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        print(f"[baseline] {name} ...", flush=True)
        started = time.perf_counter()
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - started
            self.seconds[name] = elapsed
            print(f"[baseline] {name} FAILED after {elapsed:.2f}s", flush=True)
            raise
        elapsed = time.perf_counter() - started
        self.seconds[name] = elapsed
        print(f"[baseline] {name} completed in {elapsed:.2f}s", flush=True)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselinePipelineError(f"{name} must be a mapping")
    return value


def _require_model_partition(partition: str) -> str:
    if partition not in _MODEL_PARTITIONS:
        raise BaselinePipelineError(
            "Sprint 2 model data access is restricted to train and validation; "
            f"requested={partition!r}"
        )
    return partition


def _feature_contract(settings: Mapping[str, Any]) -> FeatureContract:
    configured = _mapping(settings.get("features"), "baseline.features")
    graph_forbidden = tuple(configured.get("forbidden_graph_history", ()))
    if graph_forbidden != GRAPH_HISTORY_FEATURES:
        raise BaselinePipelineError(
            "Configured graph-history exclusion must exactly match the reviewed Sprint 1 family"
        )
    configured_forbidden = set(configured.get("forbidden_identity_or_target", ()))
    required_non_graph = set(FORBIDDEN_PREDICTORS).difference(GRAPH_HISTORY_FEATURES)
    if not required_non_graph.issubset(configured_forbidden | {"partition"}):
        missing = sorted(required_non_graph - configured_forbidden - {"partition"})
        raise BaselinePipelineError(f"Configured forbidden fields are missing: {missing}")
    return FeatureContract(
        numeric_features=tuple(configured["numeric"]),
        low_cardinality_categories=tuple(configured["one_hot_categorical"]),
        bank_frequency_categories=tuple(configured["frequency_categorical"]),
        missing_indicator_features=tuple(configured["missing_indicators"]),
        forbidden_predictors=FORBIDDEN_PREDICTORS,
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselinePipelineError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselinePipelineError(f"{label} must contain a JSON object")
    return payload


def _prevalence_from_sprint1_metadata(
    split_metadata: Mapping[str, Any], expected_rows: int
) -> dict[str, dict[str, Any]]:
    if split_metadata.get("strategy") != "chronological":
        raise BaselinePipelineError("Frozen split metadata is not chronological")
    if split_metadata.get("strict_boundaries_verified") is not True:
        raise BaselinePipelineError("Frozen split metadata lacks strict-boundary proof")
    if split_metadata.get("no_transaction_overlap_verified") is not True:
        raise BaselinePipelineError("Frozen split metadata lacks no-overlap proof")
    partitions = _mapping(split_metadata.get("partitions"), "split_metadata.partitions")
    if set(partitions) != {"train", "validation", "test"}:
        raise BaselinePipelineError("Frozen metadata must contain train, validation, and test")

    result: dict[str, dict[str, Any]] = {}
    for name in ("train", "validation", "test"):
        values = _mapping(partitions[name], f"split_metadata.partitions.{name}")
        rows = int(values["rows"])
        positives = int(values["positive_labels"])
        if rows <= 0 or positives < 0 or positives > rows:
            raise BaselinePipelineError(f"Invalid frozen prevalence counts for {name}")
        result[name] = {
            "rows": rows,
            "positive_labels": positives,
            "negative_labels": rows - positives,
            "positive_rate": positives / rows,
            "minimum_timestamp": str(values["minimum_timestamp"]),
            "maximum_timestamp": str(values["maximum_timestamp"]),
            "source": "pre_existing_sprint1_split_metadata",
        }
    if sum(item["rows"] for item in result.values()) != expected_rows:
        raise BaselinePipelineError("Frozen split rows do not reconcile to expected full rows")
    if not (
        datetime.fromisoformat(result["train"]["maximum_timestamp"])
        < datetime.fromisoformat(result["validation"]["minimum_timestamp"])
        <= datetime.fromisoformat(result["validation"]["maximum_timestamp"])
        < datetime.fromisoformat(result["test"]["minimum_timestamp"])
    ):
        raise BaselinePipelineError("Frozen chronological boundaries are not strictly ordered")
    return result


def _verify_frozen_inputs(
    *,
    settings: Mapping[str, Any],
    feature_table: Path,
    split_table: Path,
    split_metadata_path: Path,
    upstream_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    frozen = _mapping(settings.get("frozen_upstream"), "baseline.frozen_upstream")
    expected = {
        feature_table: str(frozen["feature_table_sha256"]),
        split_table: str(frozen["split_table_sha256"]),
        split_metadata_path: str(frozen["split_metadata_sha256"]),
    }
    fingerprints: dict[str, Any] = {}
    for path, expected_sha in expected.items():
        actual = sha256_file(path)
        if actual != expected_sha:
            raise BaselinePipelineError(
                f"Frozen Sprint 1 input hash mismatch for {path}: "
                f"expected={expected_sha}, actual={actual}"
            )
        fingerprints[path.name] = file_fingerprint(path)

    upstream = _load_json(upstream_manifest_path, "Sprint 1 full run manifest")
    if upstream.get("status") != "PASS":
        raise BaselinePipelineError("Sprint 1 upstream manifest status is not PASS")
    if int(upstream.get("features", {}).get("rows", -1)) != int(frozen["expected_rows"]):
        raise BaselinePipelineError("Sprint 1 feature row count differs from frozen expectation")
    split_metadata = _load_json(split_metadata_path, "Sprint 1 split metadata")
    prevalence = _prevalence_from_sprint1_metadata(split_metadata, int(frozen["expected_rows"]))
    provenance = {
        "dataset": upstream.get("provenance", {}).get("dataset"),
        "dataset_is_synthetic": upstream.get("provenance", {}).get("dataset_is_synthetic"),
        "raw_sources": {
            "transactions": upstream.get("provenance", {}).get("transactions", {}),
            "accounts": upstream.get("provenance", {}).get("accounts", {}),
        },
        "upstream_sprint": upstream.get("sprint"),
        "upstream_manifest": file_fingerprint(upstream_manifest_path),
        "frozen_inputs": fingerprints,
        "expected_full_rows": int(frozen["expected_rows"]),
    }
    return provenance, prevalence, upstream


def _configure_duckdb(
    connection: duckdb.DuckDBPyConnection,
    settings: Mapping[str, Any],
    spill_dir: Path,
) -> None:
    spill_dir.mkdir(parents=True, exist_ok=True)
    memory_limit = str(settings["memory_limit"]).replace("'", "''")
    max_temp = str(settings["max_temp_directory_size"]).replace("'", "''")
    spill_path = spill_dir.resolve().as_posix().replace("'", "''")
    connection.execute(f"SET memory_limit='{memory_limit}'")
    connection.execute(f"SET threads={int(settings['threads'])}")
    connection.execute(f"SET temp_directory='{spill_path}'")
    connection.execute(f"SET max_temp_directory_size='{max_temp}'")
    connection.execute("SET preserve_insertion_order=false")


def _materialize_partition(
    connection: duckdb.DuckDBPyConnection,
    preprocessor: FittedPreprocessor,
    feature_table: Path,
    split_table: Path,
    *,
    partition: str,
    expected_rows: int,
    expected_positives: int,
    batch_rows: int,
    work_dir: Path,
) -> PartitionMatrices:
    safe_partition = _require_model_partition(partition)
    feature_count = len(preprocessor.feature_names)
    matrix_path = work_dir / f"X_{safe_partition}.npy"
    target_path = work_dir / f"y_{safe_partition}.npy"
    source_path = work_dir / f"source_rows_{safe_partition}.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(expected_rows, feature_count)
    )
    target = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.int8, shape=(expected_rows,)
    )
    source_rows: np.memmap | None = None
    if safe_partition == "validation":
        source_rows = np.lib.format.open_memmap(
            source_path, mode="w+", dtype=np.int64, shape=(expected_rows,)
        )

    offset = 0
    batches = 0
    for batch in iter_duckdb_transformed_batches(
        connection,
        preprocessor,
        feature_table,
        split_table,
        safe_partition,
        batch_size=batch_rows,
    ):
        stop = offset + batch.rows
        if stop > expected_rows:
            raise BaselinePipelineError(
                f"{safe_partition} batches exceed frozen row count {expected_rows}"
            )
        matrix[offset:stop] = batch.matrix
        target[offset:stop] = batch.target
        if source_rows is not None:
            source_rows[offset:stop] = batch.source_row_numbers
        offset = stop
        batches += 1
        if batches == 1 or batches % 10 == 0:
            print(
                f"[baseline] {safe_partition} matrix: {offset:,}/{expected_rows:,} rows",
                flush=True,
            )
    matrix.flush()
    target.flush()
    if source_rows is not None:
        source_rows.flush()
    if offset != expected_rows:
        raise BaselinePipelineError(
            f"{safe_partition} matrix rows differ: expected={expected_rows}, actual={offset}"
        )
    actual_positives = int(np.count_nonzero(target == 1))
    if actual_positives != expected_positives:
        raise BaselinePipelineError(
            f"{safe_partition} labels differ from frozen metadata: "
            f"expected={expected_positives}, actual={actual_positives}"
        )
    if source_rows is not None and np.unique(source_rows).size != expected_rows:
        raise BaselinePipelineError("Validation source_row_number values are not unique")
    return PartitionMatrices(
        partition=safe_partition,
        matrix=matrix,
        target=target,
        source_row_numbers=source_rows,
        rows=expected_rows,
        positive_labels=actual_positives,
        batches=batches,
    )


def _atomic_joblib_dump(model: Any, destination: Path, compression: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(model, temporary, compress=compression)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_validation_predictions(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_row_numbers: np.ndarray,
    labels: np.ndarray,
    scores_by_model: Mapping[str, np.ndarray],
    source_filename: str,
    destination: Path,
    compression: str,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "source_row_number": np.asarray(source_row_numbers),
            "is_laundering": np.asarray(labels),
            **{f"score_{name}": np.asarray(scores) for name, scores in scores_by_model.items()},
        }
    )
    connection.register("argus_validation_predictions", frame)
    escaped_source = source_filename.replace("'", "''")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary_sql = temporary.resolve().as_posix().replace("'", "''")
    score_projection = ", ".join(
        f'CAST("score_{name}" AS DOUBLE) AS "score_{name}"' for name in sorted(scores_by_model)
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
                FROM argus_validation_predictions
                ORDER BY source_row_number
            ) TO '{temporary_sql}' (
                FORMAT PARQUET,
                COMPRESSION {compression.upper()}
            )
            """
        )
        os.replace(temporary, destination)
    finally:
        connection.unregister("argus_validation_predictions")
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
        "score_columns": [f"score_{name}" for name in sorted(scores_by_model)],
        "test_predictions_included": False,
    }


def _source_snapshot(project_root: Path) -> dict[str, Any]:
    included_roots = ("configs", "scripts", "src", "tests")
    files: list[Path] = []
    for root_name in included_roots:
        root = project_root / root_name
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not any(part.endswith(".egg-info") for part in path.parts)
            and not path.name.endswith((".pyc", ".pyo"))
        )
    files.extend(
        path
        for name in ("app.py", "pyproject.toml", "requirements.txt")
        if (path := project_root / name).is_file()
    )
    inventory = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(files), key=lambda item: item.relative_to(project_root).as_posix())
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = None
        status_lines = []
    return {
        "git_head": commit,
        "git_worktree_clean": not status_lines,
        "git_status_at_run": status_lines,
        "files": inventory,
    }


def _safe_cleanup_work_directory(work_dir: Path, run_dir: Path) -> None:
    resolved_work = work_dir.resolve()
    resolved_run = run_dir.resolve()
    if resolved_work.parent != resolved_run or not resolved_work.name.startswith(_WORK_PREFIX):
        raise BaselinePipelineError(f"Refusing to remove unexpected work directory: {work_dir}")
    if resolved_work.exists():
        shutil.rmtree(resolved_work)


def _close_memmap(array: np.memmap | None) -> None:
    """Flush and close a NumPy memmap explicitly for Windows cleanup."""

    if array is None:
        return
    array.flush()
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


def run_sprint2_baselines(
    config_path: str | Path = "configs/baseline.yaml",
) -> BaselineRunResult:
    """Run all fixed transaction baselines and select with validation AP only."""

    started = time.perf_counter()
    started_at = datetime.now(UTC)
    recorder = _StageRecorder()
    config = load_config(config_path)
    settings = _mapping(config.get("baseline"), "baseline")
    if settings.get("final_test_access") is not False:
        raise BaselinePipelineError("Final test access must remain disabled in Sprint 2")

    feature_table = get_path(config, "feature_table")
    split_table = get_path(config, "split_table")
    split_metadata_path = get_path(config, "split_metadata")
    upstream_manifest_path = get_path(config, "upstream_manifest")
    run_dir = get_path(config, "run_dir")
    report_path = get_path(config, "generated_report")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    work_dir = run_dir / f"{_WORK_PREFIX}{uuid.uuid4().hex}"
    work_dir.mkdir(parents=False, exist_ok=False)
    connection: duckdb.DuckDBPyConnection | None = None
    succeeded = False

    try:
        with recorder.measure("frozen_input_verification"):
            provenance, prevalence, upstream = _verify_frozen_inputs(
                settings=settings,
                feature_table=feature_table,
                split_table=split_table,
                split_metadata_path=split_metadata_path,
                upstream_manifest_path=upstream_manifest_path,
            )
            contract = _feature_contract(settings)

        with recorder.measure("source_snapshot"):
            source_snapshot = _source_snapshot(Path(config["_meta"]["project_root"]))
            atomic_write_json(source_snapshot, run_dir / "source_snapshot.json")
            atomic_write_json(config, run_dir / "resolved_config.json")
            atomic_write_json(prevalence, run_dir / "prevalence.json")

        with recorder.measure("duckdb_initialization"):
            connection = duckdb.connect(str(work_dir / "baseline.duckdb"))
            _configure_duckdb(connection, settings, work_dir / "spill")

        with recorder.measure("train_only_preprocessor_fit"):
            preprocessor = FittedPreprocessor.fit_duckdb(
                connection, feature_table, split_table, contract
            )
            if preprocessor.fitted_train_rows != prevalence["train"]["rows"]:
                raise BaselinePipelineError(
                    "Preprocessor fit row count is not the frozen train count"
                )
            preprocessing_dir = run_dir / "preprocessing"
            atomic_write_json(preprocessor.to_state_dict(), preprocessing_dir / "fitted_state.json")
            atomic_write_json(preprocessor.manifest(), preprocessing_dir / "manifest.json")
            atomic_write_json(contract.to_dict(), run_dir / "feature_contract.json")

        with recorder.measure("train_matrix_materialization"):
            train = _materialize_partition(
                connection,
                preprocessor,
                feature_table,
                split_table,
                partition="train",
                expected_rows=prevalence["train"]["rows"],
                expected_positives=prevalence["train"]["positive_labels"],
                batch_rows=int(settings["batch_rows"]),
                work_dir=work_dir,
            )

        with recorder.measure("validation_matrix_materialization"):
            validation = _materialize_partition(
                connection,
                preprocessor,
                feature_table,
                split_table,
                partition="validation",
                expected_rows=prevalence["validation"]["rows"],
                expected_positives=prevalence["validation"]["positive_labels"],
                batch_rows=int(settings["batch_rows"]),
                work_dir=work_dir,
            )
            assert validation.source_row_numbers is not None

        with recorder.measure("three_baseline_fit_and_validation"):
            built = build_baseline_models(
                _mapping(settings.get("models"), "baseline.models"),
                random_seed=int(config["project"]["random_seed"]),
                training_labels=train.target,
            )
            model_results: dict[str, dict[str, Any]] = {}
            scores_by_model: dict[str, np.memmap] = {}
            models_root = run_dir / "models"
            for model_name in ("logistic_regression", "random_forest", "lightgbm"):
                estimator, implementation, effective_parameters = built.pop(model_name)
                print(f"[baseline] fitting {model_name} ...", flush=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = fit_baseline(
                        model_name,
                        estimator,
                        implementation,
                        effective_parameters,
                        train.matrix,
                        train.target,
                    )
                scores, predict_seconds = predict_positive_probability(
                    fitted.estimator,
                    validation.matrix,
                    batch_rows=int(settings["batch_rows"]),
                )
                score_memmap = np.lib.format.open_memmap(
                    work_dir / f"scores_validation_{model_name}.npy",
                    mode="w+",
                    dtype=np.float64,
                    shape=(validation.rows,),
                )
                score_memmap[:] = scores
                score_memmap.flush()
                del scores
                metrics = evaluate_binary_predictions(
                    validation.target,
                    score_memmap,
                    validation.source_row_numbers,
                    threshold=float(settings["decision_threshold"]),
                    top_k_values=settings["top_k"],
                )
                model_dir = models_root / model_name
                model_path = model_dir / "model.joblib"
                _atomic_joblib_dump(
                    fitted.estimator,
                    model_path,
                    int(settings["model_serialization_compression"]),
                )
                result = {
                    "model": model_name,
                    "implementation": implementation,
                    "effective_parameters": json_safe_parameters(fitted.effective_parameters),
                    "class_imbalance_handling": (
                        "train_only_scale_pos_weight"
                        if model_name == "lightgbm"
                        else str(fitted.effective_parameters.get("class_weight"))
                    ),
                    "probability_calibration": "not_calibrated",
                    "evaluation_partition": "validation",
                    "threshold_basis": settings["threshold_basis"],
                    "runtime_seconds": {
                        "fit": fitted.fit_seconds,
                        "validation_predict": predict_seconds,
                        "fit_plus_validation_predict": fitted.fit_seconds + predict_seconds,
                    },
                    "warnings": [str(item.message) for item in caught],
                    "validation_metrics": metrics,
                    "model_artifact": model_path.relative_to(run_dir).as_posix(),
                    "test_metrics": None,
                    "test_predictions_generated": False,
                }
                atomic_write_json(result, model_dir / "metadata.json")
                atomic_write_json(metrics, model_dir / "validation_metrics.json")
                model_results[model_name] = result
                scores_by_model[model_name] = score_memmap
                print(
                    f"[baseline] {model_name}: validation AP={metrics['average_precision']:.8f}, "
                    f"fit={fitted.fit_seconds:.2f}s",
                    flush=True,
                )
                del fitted, estimator
                gc.collect()

        with recorder.measure("validation_comparison_and_champion"):
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
            comparison_context = comparison_context_from_run(
                provenance=provenance,
                feature_manifest=preprocessor.manifest(),
                configuration={"random_seed": int(config["project"]["random_seed"])},
                libraries=library_versions,
                feature_scope=str(settings["feature_scope"]),
            )
            champion = select_transaction_baseline_champion(
                {
                    name: float(result["validation_metrics"]["average_precision"])
                    for name, result in model_results.items()
                },
                partition="validation",
                test_metrics=None,
            )
            atomic_write_json(champion, run_dir / "transaction_baseline_champion.json")
            atomic_write_json(
                {
                    "evaluation_partition": "validation",
                    "primary_metric": "average_precision",
                    "accuracy_is_primary": False,
                    "provenance": comparison_context,
                    "model_results": model_results,
                },
                run_dir / "model_comparison.json",
            )
            comparison = write_comparison_table(
                model_results,
                run_dir / "model_comparison.csv",
                context=comparison_context,
            )
            plot_metric_comparison(comparison, run_dir / "figures" / "metric_comparison.png")
            plot_precision_recall_curves(
                validation.target,
                scores_by_model,
                run_dir / "figures" / "validation_precision_recall_curves.png",
            )

        with recorder.measure("validation_prediction_export"):
            source_filename = str(
                upstream.get("provenance", {})
                .get("transactions", {})
                .get("filename", "HI-Small_Trans.csv")
            )
            prediction_evidence = _write_validation_predictions(
                connection,
                source_row_numbers=validation.source_row_numbers,
                labels=validation.target,
                scores_by_model=scores_by_model,
                source_filename=source_filename,
                destination=run_dir / "validation_predictions.parquet",
                compression=str(settings["parquet_compression"]),
            )
            if (
                prediction_evidence["rows"] != validation.rows
                or prediction_evidence["unique_transaction_ids"] != validation.rows
                or prediction_evidence["positive_labels"] != validation.positive_labels
            ):
                raise BaselinePipelineError(
                    f"Validation prediction export verification failed: {prediction_evidence}"
                )
            atomic_write_json(prediction_evidence, run_dir / "validation_predictions_manifest.json")

        final_test_policy = {
            "final_test_used_for_training": False,
            "final_test_used_for_preprocessing_fit": False,
            "final_test_used_for_model_selection": False,
            "final_test_inference_performed": False,
            "test_feature_rows_materialized": False,
            "test_prediction_artifact_exists": False,
            "test_metadata_reported": True,
            "test_metadata_source": "pre_existing_sprint1_split_metadata.json",
            "permitted_model_partitions": sorted(_MODEL_PARTITIONS),
            "selection_partition": "validation",
            "selection_input": "validation_average_precision_only",
            "reason": "Final-test use follows model refinement/freeze in a later phase",
        }
        atomic_write_json(final_test_policy, run_dir / "final_test_policy.json")

        atomic_write_json(library_versions, run_dir / "library_versions.json")

        with recorder.measure("work_matrix_cleanup"):
            connection.close()
            connection = None
            for score_values in scores_by_model.values():
                _close_memmap(score_values)
            _close_memmap(train.matrix)
            _close_memmap(train.target)
            _close_memmap(train.source_row_numbers)
            _close_memmap(validation.matrix)
            _close_memmap(validation.target)
            _close_memmap(validation.source_row_numbers)
            del score_memmap
            del train, validation, scores_by_model, built
            gc.collect()
            if not bool(settings.get("retain_work_matrices", False)):
                _safe_cleanup_work_directory(work_dir, run_dir)

        runtime_total = time.perf_counter() - started
        core_artifacts = [
            "artifacts/sprint2/run_manifest.json",
            "artifacts/sprint2/model_comparison.json",
            "artifacts/sprint2/model_comparison.csv",
            "artifacts/sprint2/transaction_baseline_champion.json",
            "artifacts/sprint2/validation_predictions.parquet",
            "artifacts/sprint2/preprocessing/manifest.json",
            "artifacts/sprint2/final_test_policy.json",
            "artifacts/sprint2/figures/metric_comparison.png",
            "artifacts/sprint2/figures/validation_precision_recall_curves.png",
            "reports/generated/SPRINT_2_STATUS.md",
        ]
        manifest_payload: dict[str, Any] = {
            "status": "PASS",
            "sprint": "Sprint 2 - Baseline + Model Exploration",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "runtime_seconds": runtime_total,
            "runtime_seconds_by_stage": recorder.seconds,
            "configuration": {
                "file": Path(config_path).name,
                "random_seed": int(config["project"]["random_seed"]),
                "full_data": True,
                "sampled": False,
                "threshold": float(settings["decision_threshold"]),
                "threshold_basis": settings["threshold_basis"],
                "top_k": list(settings["top_k"]),
            },
            "provenance": provenance,
            "split_prevalence": prevalence,
            "prevalence_shift": {
                "train_to_validation_rate_ratio": (
                    prevalence["validation"]["positive_rate"] / prevalence["train"]["positive_rate"]
                ),
                "validation_to_test_rate_ratio": (
                    prevalence["test"]["positive_rate"] / prevalence["validation"]["positive_rate"]
                ),
                "interpretation": (
                    "Precision and fixed-threshold alert volume are prevalence-sensitive; "
                    "validation values are not assumed to transfer unchanged to test."
                ),
                "splits_rebalanced": False,
            },
            "feature_contract": preprocessor.manifest(),
            "models": model_results,
            "champion": champion,
            "prediction_evidence": prediction_evidence,
            "final_test_policy": final_test_policy,
            "libraries": library_versions,
            "data_access_audit": {
                "preprocessor_fit_partitions": ["train"],
                "transformed_partitions": ["train", "validation"],
                "evaluated_partitions": ["validation"],
                "test_access": "metadata_only_from_sprint1_json",
            },
            "acceptance": {
                "three_transaction_baselines_executed": True,
                "same_frozen_protocol": True,
                "validation_only_champion": True,
                "pr_auc_primary": True,
                "required_secondary_and_operational_metrics": True,
                "machine_readable_comparison": True,
                "train_fit_transform_discipline": True,
                "graph_history_excluded": True,
                "final_test_untouched_for_modeling": True,
                "sprint3_not_started": True,
            },
            "core_artifact_paths": core_artifacts,
        }
        manifest = write_run_manifest(
            manifest_payload,
            run_dir,
            exclude_paths=[
                path
                for path in run_dir.iterdir()
                if path.is_dir() and path.name.startswith(_WORK_PREFIX)
            ],
        )
        render_sprint2_markdown(
            model_results=model_results,
            champion=champion,
            prevalence=prevalence,
            runtime_seconds=recorder.seconds,
            artifact_paths=core_artifacts,
            destination=report_path,
            total_runtime_seconds=runtime_total,
        )
        succeeded = True
        print(
            f"[baseline] PASS: champion={champion['champion_model']}, "
            f"validation AP={champion['selection_metric_value']:.8f}, "
            f"runtime={runtime_total:.2f}s",
            flush=True,
        )
        return BaselineRunResult(run_dir, manifest_path, report_path, manifest)
    finally:
        if connection is not None:
            connection.close()
        if not succeeded and work_dir.exists():
            print(f"[baseline] Work files retained after failure: {work_dir}", flush=True)
