"""Read-only verification of persisted one-shot final-evaluation artifacts.

The verifier never imports estimators, rebuilds features, queries the raw/test
tables, or rewrites the run manifest. It recomputes every reported test metric
from the sealed prediction artifact and validates the immutable inventory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from argus.config import get_path, load_config
from argus.final_evaluation.contract import load_json_object, mapping
from argus.final_evaluation.evaluation import evaluate_frozen_models, stable_sigmoid
from argus.modeling.artifacts import atomic_write_json, sha256_file

_FROZEN_CHAMPION = "graph_enhanced_lightgbm"
_MODEL_NAMES = (
    "graph_enhanced_lightgbm",
    "refined_transaction_lightgbm",
    "graphsage_edge_classifier",
)
_POSITIVE_ACCEPTANCE_KEYS = {
    "explicit_final_test_authorization_recorded",
    "one_shot_access_receipt_created_before_test_query",
    "frozen_sprint4_checkpoint_and_artifact_hashes_verified",
    "graph_enhanced_lightgbm_frozen_as_pre_test_champion",
    "exact_train_fitted_preprocessors_reused",
    "exact_validation_selected_thresholds_reused",
    "full_untouched_test_partition_scored",
    "three_models_scored_in_same_authorized_run",
    "all_required_final_metrics_saved",
    "prevalence_shift_reported_without_rebalancing",
    "graphsage_uses_frozen_train_context_without_test_message_edges",
    "machine_readable_artifacts_saved",
    "streamlit_final_artifact_contract_saved",
}
_NEGATIVE_ACCEPTANCE_KEYS = {
    "test_used_for_selection_or_tuning",
    "post_test_tuning_or_retraining_performed",
    "unsupported_account_level_label_created",
}


class FinalEvaluationVerificationError(RuntimeError):
    """Raised when persisted final evidence is missing or inconsistent."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise FinalEvaluationVerificationError(message)


def _close(left: object, right: object, *, label: str, tolerance: float = 1e-12) -> None:
    if left is None or right is None:
        _assert(left is right, f"{label} differs: expected={left}, actual={right}")
        return
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        raise FinalEvaluationVerificationError(f"{label} is not numeric")
    try:
        matched = math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError) as exc:
        raise FinalEvaluationVerificationError(f"{label} is not numeric") from exc
    _assert(matched, f"{label} differs: expected={left}, actual={right}")


def _unique_rows(rows: object, *, name: str) -> list[Mapping[str, Any]]:
    _assert(isinstance(rows, list) and bool(rows), f"{name} must be a non-empty list")
    result = [mapping(row, f"{name} row") for row in rows]
    names = [str(row.get("model")) for row in result]
    _assert(len(names) == len(set(names)), f"{name} contains duplicate model rows")
    _assert(set(names) == set(_MODEL_NAMES), f"{name} model set changed")
    return result


def _verify_inventory(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = manifest.get("artifacts")
    _assert(isinstance(rows, list), "Run manifest artifact inventory is missing")
    paths: list[str] = []
    verified = 0
    for raw in rows:
        row = mapping(raw, "artifact inventory row")
        _assert(set(row) == {"path", "size_bytes", "sha256"}, "Inventory row schema changed")
        relative = str(row["path"])
        paths.append(relative)
        candidate = (run_dir / relative).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise FinalEvaluationVerificationError(
                f"Inventory path escapes final run directory: {relative}"
            ) from exc
        _assert(candidate.is_file(), f"Inventoried final artifact is missing: {relative}")
        _assert(candidate.stat().st_size == int(row["size_bytes"]), f"Size changed: {relative}")
        _assert(sha256_file(candidate) == row["sha256"], f"SHA-256 changed: {relative}")
        verified += 1
    _assert(len(paths) == len(set(paths)), "Run manifest contains duplicate artifact paths")
    _assert(
        verified == int(manifest.get("artifact_count_excluding_manifest", -1)),
        "Run manifest artifact count differs from its unique inventory",
    )
    required = {
        "FINAL_TEST_OPENED.json",
        "FINAL_TEST_COMPLETED.json",
        "freeze_contract.json",
        "final_test_predictions.parquet",
        "final_metrics.json",
        "final_metrics.csv",
        "final_model_comparison.json",
        "final_model_comparison.csv",
        "final_top_k_metrics.json",
        "final_top_k_metrics.csv",
        "final_pr_curves.csv",
        "confusion_matrices.csv",
        "frozen_threshold_details.json",
        "model_metadata.json",
        "test_identity_audit.json",
        "prevalence_shift.json",
        "product/final_test_summary.json",
    }
    missing = sorted(required.difference(paths))
    _assert(not missing, f"Required final artifacts are absent from inventory: {missing}")
    return {"status": "PASS", "unique_paths": len(paths), "required_paths": len(required)}


def _verify_receipts(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    freeze_path = run_dir / "freeze_contract.json"
    opened_path = run_dir / "FINAL_TEST_OPENED.json"
    completed_path = run_dir / "FINAL_TEST_COMPLETED.json"
    freeze = load_json_object(freeze_path, "freeze contract")
    opened = load_json_object(opened_path, "final-test opening receipt")
    completed = load_json_object(completed_path, "final-test completion receipt")
    _assert(
        freeze.get("schema") == "argus.final_evaluation.freeze_contract.v1",
        "Bad freeze schema",
    )
    _assert(freeze.get("created_before_final_test_access") is True, "Freeze was not pre-access")
    _assert(freeze.get("frozen_champion") == _FROZEN_CHAMPION, "Frozen champion changed")
    _assert(freeze.get("selection_partition") == "validation", "Selection was not validation")
    _assert(freeze.get("evaluation_partition") == "test", "Evaluation was not test")
    _assert(freeze.get("model_selection_locked_before_test") is True, "Model was not locked")
    _assert(freeze.get("thresholds_locked_before_test") is True, "Thresholds were not locked")
    _assert(freeze.get("final_test_rows_read") is False, "Freeze contract read test rows")
    _assert(freeze.get("final_test_labels_read") is False, "Freeze contract read test labels")
    _assert(
        opened.get("status") == "OPENED_FOR_ONE_SHOT_FINAL_EVALUATION",
        "Opening receipt status changed",
    )
    for key, expected in {
        "created_before_first_test_query": True,
        "explicit_user_authorization": True,
        "authorized_stage": "final_evaluation",
        "retry_permitted": False,
        "model_selection_used_test": False,
        "threshold_tuned_on_test": False,
    }.items():
        actual = opened.get(key)
        matched = actual is expected if isinstance(expected, bool) else actual == expected
        _assert(matched, f"Bad receipt field {key}")
    _assert(opened.get("frozen_champion") == _FROZEN_CHAMPION, "Opening champion changed")
    _assert(
        opened.get("selection_partition") == "validation",
        "Opening selection was not validation",
    )
    _assert(opened.get("evaluation_partition") == "test", "Opening evaluation was not test")
    _assert(
        mapping(opened.get("freeze_contract"), "opened.freeze_contract").get("sha256")
        == sha256_file(freeze_path),
        "Opening receipt is not bound to freeze contract",
    )
    _assert(completed.get("status") == "COMPLETED", "Completion receipt is not complete")
    _assert(completed.get("retry_permitted") is False, "Completion receipt permits retry")
    _assert(completed.get("tuning_after_test") is False, "Completion records tuning")
    _assert(completed.get("retraining_after_test") is False, "Completion records retraining")
    _assert(completed.get("frozen_champion") == _FROZEN_CHAMPION, "Completion champion changed")
    _assert(int(manifest.get("final_test_execution_count", -1)) == 1, "Execution count not one")
    _assert(manifest.get("final_test_reexecution_permitted") is False, "Manifest permits retry")
    return {
        "status": "PASS",
        "one_shot_access_count": 1,
        "freeze_sha256": sha256_file(freeze_path),
        "opened_sha256": sha256_file(opened_path),
        "completed_sha256": sha256_file(completed_path),
    }


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "final_test_predictions.parquet"
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY timestamp, source_row_number",
            [str(path.resolve())],
        ).fetch_df()
    finally:
        connection.close()
    return frame


def _verify_predictions(frame: pd.DataFrame, freeze: Mapping[str, Any]) -> dict[str, Any]:
    base = {
        "transaction_id",
        "source_row_number",
        "timestamp",
        "evaluation_partition",
        "is_laundering",
    }
    required = set(base)
    for name in _MODEL_NAMES:
        required.update({f"raw_score_{name}", f"probability_{name}", f"alert_{name}"})
    _assert(not frame.columns.duplicated().any(), "Final prediction schema has duplicate columns")
    _assert(set(frame.columns) == required, "Final prediction schema differs from exact contract")
    test = mapping(freeze["test_partition_metadata_only"], "freeze test partition")
    expected_rows = int(test["rows"])
    expected_positives = int(test["positives"])
    _assert(len(frame) == expected_rows, "Final prediction row count differs from frozen test")
    _assert(frame["transaction_id"].notna().all(), "Missing transaction IDs")
    _assert(frame["source_row_number"].notna().all(), "Missing source rows")
    _assert(
        frame["transaction_id"].nunique(dropna=False) == expected_rows,
        "Duplicate transaction IDs",
    )
    source_values = frame["source_row_number"].to_numpy()
    try:
        numeric_source_values = source_values.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise FinalEvaluationVerificationError("Source rows are not numeric integers") from exc
    _assert(np.isfinite(numeric_source_values).all(), "Source rows are not finite")
    _assert(
        np.equal(numeric_source_values, np.floor(numeric_source_values)).all(),
        "Source rows are not integers",
    )
    integer_source_values = numeric_source_values.astype(np.int64)
    _assert(np.unique(integer_source_values).size == expected_rows, "Duplicate source rows")
    _assert(set(frame["evaluation_partition"].astype(str)) == {"test"}, "Non-test prediction row")
    raw_labels = frame["is_laundering"].to_numpy()
    try:
        numeric_labels = raw_labels.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise FinalEvaluationVerificationError(
            "Prediction labels are not numeric binary values"
        ) from exc
    _assert(np.isfinite(numeric_labels).all(), "Prediction labels are not finite")
    _assert(np.isin(numeric_labels, (0.0, 1.0)).all(), "Prediction labels are not binary")
    labels = numeric_labels.astype(np.int8)
    _assert(int(labels.sum()) == expected_positives, "Prediction positive count changed")
    _assert(
        pd.Timestamp(frame["timestamp"].min()).isoformat() == test["minimum_timestamp"],
        "Final prediction minimum timestamp changed",
    )
    _assert(
        pd.Timestamp(frame["timestamp"].max()).isoformat() == test["maximum_timestamp"],
        "Final prediction maximum timestamp changed",
    )
    models = mapping(freeze["models"], "freeze.models")
    _assert(set(models) == set(_MODEL_NAMES), "Freeze model set changed")
    score_checks: dict[str, Any] = {}
    for name in _MODEL_NAMES:
        try:
            raw = frame[f"raw_score_{name}"].to_numpy(np.float64)
            probability = frame[f"probability_{name}"].to_numpy(np.float64)
        except (TypeError, ValueError) as exc:
            raise FinalEvaluationVerificationError(f"Non-numeric saved scores for {name}") from exc
        _assert(np.isfinite(raw).all(), f"Non-finite raw scores for {name}")
        _assert(np.isfinite(probability).all(), f"Non-finite probabilities for {name}")
        _assert(
            np.all((probability >= 0.0) & (probability <= 1.0)),
            f"Probability range for {name}",
        )
        maximum_error = float(np.max(np.abs(stable_sigmoid(raw) - probability)))
        _assert(maximum_error <= 1e-15, f"Raw/probability mismatch for {name}: {maximum_error}")
        threshold = float(mapping(models[name], f"freeze model {name}")["frozen_raw_threshold"])
        _assert(math.isfinite(threshold), f"Non-finite frozen threshold for {name}")
        expected_alert = raw >= threshold
        raw_alert = frame[f"alert_{name}"].to_numpy()
        _assert(
            all(isinstance(value, (bool, np.bool_)) for value in raw_alert),
            f"Saved alerts are not boolean for {name}",
        )
        saved_alert = raw_alert.astype(bool, copy=False)
        mismatches = int(np.count_nonzero(expected_alert != saved_alert))
        _assert(mismatches == 0, f"Frozen-threshold alert mismatch for {name}")
        score_checks[name] = {
            "finite_rows": expected_rows,
            "maximum_sigmoid_absolute_error": maximum_error,
            "alert_mismatches": mismatches,
        }
    return {
        "status": "PASS",
        "rows": expected_rows,
        "positive_count": expected_positives,
        "unique_transaction_ids": expected_rows,
        "unique_source_rows": expected_rows,
        "scores": score_checks,
    }


def _compare_metric_row(saved: Mapping[str, Any], computed: Mapping[str, Any], model: str) -> None:
    exact = {
        "model",
        "role",
        "partition",
        "evaluation_partition",
        "evaluation_role",
        "row_count",
        "positive_count",
        "negative_count",
        "threshold_source_partition",
        "threshold_comparison",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "alert_count",
        "alert_volume",
        "champion_frozen_before_test",
        "model_selection_used_test",
        "threshold_tuned_on_test",
        "feature_selection_used_test",
        "retrained_after_test",
    }
    numeric = {
        "positive_rate",
        "average_precision",
        "pr_auc",
        "roc_auc",
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "alert_rate",
        "validation_average_precision",
    }
    _assert(set(saved) == exact | numeric, f"Final metric row schema changed for {model}")
    for key in exact:
        _assert(saved[key] == computed[key], f"{model}.{key} changed")
    for key in numeric:
        _close(saved[key], computed[key], label=f"{model}.{key}")


def _verify_metrics(
    run_dir: Path,
    frame: pd.DataFrame,
    freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    models = mapping(freeze["models"], "freeze.models")
    raw_scores = {name: frame[f"raw_score_{name}"].to_numpy(np.float64) for name in _MODEL_NAMES}
    roles = {name: str(models[name]["role"]) for name in _MODEL_NAMES}
    thresholds = {name: float(models[name]["frozen_raw_threshold"]) for name in _MODEL_NAMES}
    validation_ap = {
        name: float(models[name]["validation_average_precision"]) for name in _MODEL_NAMES
    }
    computed, computed_top_k, _ = evaluate_frozen_models(
        frame["is_laundering"].to_numpy(np.int8),
        frame["source_row_number"].to_numpy(np.int64),
        raw_scores,
        frozen_thresholds=thresholds,
        model_roles=roles,
        validation_average_precision=validation_ap,
        top_k=tuple(int(value) for value in freeze["top_k"]),
        frozen_champion=_FROZEN_CHAMPION,
    )
    comparison_payload = load_json_object(
        run_dir / "final_model_comparison.json", "final comparison"
    )
    _assert(comparison_payload.get("partition") == "test", "Comparison is not test")
    _assert(
        comparison_payload.get("presentation_order")
        == "frozen_role_then_model_name_not_test_metric",
        "Comparison presentation order became a hidden test selector",
    )
    _assert(comparison_payload.get("frozen_champion") == _FROZEN_CHAMPION, "Champion changed")
    _assert(comparison_payload.get("champion_frozen_before_test") is True, "Champion not frozen")
    _assert(
        comparison_payload.get("test_metrics_used_for_model_selection") is False,
        "Test metrics selected a model",
    )
    _assert(
        comparison_payload.get("test_metrics_used_for_threshold_selection") is False,
        "Test metrics selected a threshold",
    )
    saved_rows = _unique_rows(comparison_payload.get("models"), name="final comparison")
    saved_by_model = {str(row["model"]): row for row in saved_rows}
    computed_by_model = {str(row["model"]): row for row in computed}
    for name in _MODEL_NAMES:
        _compare_metric_row(saved_by_model[name], computed_by_model[name], name)

    manifest_rows = _unique_rows(manifest.get("final_model_comparison"), name="manifest comparison")
    manifest_by_model = {str(row["model"]): row for row in manifest_rows}
    for name in _MODEL_NAMES:
        _compare_metric_row(manifest_by_model[name], computed_by_model[name], name)

    final_metrics = load_json_object(run_dir / "final_metrics.json", "champion metrics")
    _assert(final_metrics.get("frozen_champion") == _FROZEN_CHAMPION, "Metrics champion changed")
    _assert(final_metrics.get("champion_frozen_before_test") is True, "Champion not pre-frozen")
    _assert(final_metrics.get("model_selection_used_test") is False, "Test selected champion")
    _assert(final_metrics.get("threshold_tuned_on_test") is False, "Test tuned threshold")
    _assert(final_metrics.get("post_test_tuning_or_retraining") is False, "Post-test tuning")
    _compare_metric_row(
        mapping(final_metrics.get("metrics"), "final_metrics.metrics"),
        computed_by_model[_FROZEN_CHAMPION],
        _FROZEN_CHAMPION,
    )

    top_k_payload = load_json_object(run_dir / "final_top_k_metrics.json", "top-K metrics")
    saved_top_k = top_k_payload.get("rows")
    _assert(isinstance(saved_top_k, list), "Top-K rows are missing")
    _assert(len(saved_top_k) == len(computed_top_k), "Top-K row count changed")

    def keys(row: Mapping[str, Any]) -> tuple[str, int]:
        return str(row["model"]), int(row["requested_k"])

    saved_top_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for raw in saved_top_k:
        row = mapping(raw, "top-K row")
        key = keys(row)
        _assert(key not in saved_top_by_key, f"Duplicate Top-K row: {key}")
        saved_top_by_key[key] = row
    computed_top_by_key = {keys(row): row for row in computed_top_k}
    _assert(set(saved_top_by_key) == set(computed_top_by_key), "Top-K keys changed")
    for key, computed_row in computed_top_by_key.items():
        saved = saved_top_by_key[key]
        _assert(set(saved) == set(computed_row), f"Top-K schema changed: {key}")
        for field, expected in computed_row.items():
            if isinstance(expected, float) or expected is None:
                _close(saved[field], expected, label=f"top_k[{key}].{field}")
            else:
                _assert(saved[field] == expected, f"top_k[{key}].{field} changed")

    comparison_csv = pd.read_csv(run_dir / "final_model_comparison.csv")
    _assert(len(comparison_csv) == 3, "Final comparison CSV row count changed")
    _assert(not comparison_csv["model"].duplicated().any(), "Duplicate final CSV models")
    metrics_csv = pd.read_csv(run_dir / "final_metrics.csv")
    _assert(
        len(metrics_csv) == 1 and metrics_csv.iloc[0]["model"] == _FROZEN_CHAMPION,
        "Final metrics CSV must contain only the pre-frozen champion",
    )
    top_k_csv = pd.read_csv(run_dir / "final_top_k_metrics.csv")
    _assert(len(top_k_csv) == len(computed_top_k), "Top-K CSV row count changed")
    confusion = pd.read_csv(run_dir / "confusion_matrices.csv")
    _assert(set(confusion["model"]) == set(_MODEL_NAMES), "Confusion model set changed")
    return {
        "status": "PASS",
        "models_recomputed": list(_MODEL_NAMES),
        "metrics_recomputed": [
            "average_precision",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "alert_count",
            "confusion_matrix",
            "precision_at_k",
            "recall_at_k",
        ],
        "top_k_rows_recomputed": len(computed_top_k),
        "frozen_champion": _FROZEN_CHAMPION,
        "champion_test_metrics": computed_by_model[_FROZEN_CHAMPION],
    }


def _verify_protocol_artifacts(
    run_dir: Path,
    freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance = mapping(manifest.get("acceptance"), "manifest.acceptance")
    expected_keys = _POSITIVE_ACCEPTANCE_KEYS | _NEGATIVE_ACCEPTANCE_KEYS
    _assert(set(acceptance) == expected_keys, "Acceptance checklist key set changed")
    _assert(
        all(acceptance[key] is True for key in _POSITIVE_ACCEPTANCE_KEYS),
        "Positive acceptance failed",
    )
    _assert(
        all(acceptance[key] is False for key in _NEGATIVE_ACCEPTANCE_KEYS),
        "Negative acceptance failed",
    )
    post_test = mapping(manifest.get("post_test_actions"), "manifest.post_test_actions")
    _assert(
        post_test and all(value is False for value in post_test.values()),
        "Post-test action recorded",
    )
    graph = load_json_object(run_dir / "final_graph_inference_manifest.json", "graph audit")
    _assert(graph.get("message_context_partition") == "train", "Graph context is not train")
    _assert(graph.get("message_context_rows") == 300_000, "Graph context sample changed")
    _assert(graph.get("test_labels_used_for_graph_construction") is False, "Test labels in graph")
    _assert(
        graph.get("validation_edges_used_for_message_passing") is False,
        "Validation graph leak",
    )
    _assert(graph.get("test_edges_used_for_message_passing") is False, "Test graph leak")
    identity = load_json_object(run_dir / "test_identity_audit.json", "identity audit")
    _assert(identity.get("status") == "PASS", "In-run exact identity audit failed")
    _assert(
        identity.get("exact_split_membership_join_performed_inside_authorized_one_shot_run")
        is True,
        "No exact split join",
    )
    _assert(identity.get("raw_test_reopened_by_post_run_verifier") is False, "Raw test reopened")
    shift = load_json_object(run_dir / "prevalence_shift.json", "prevalence shift")
    validation = mapping(shift.get("validation"), "shift.validation")
    test = mapping(shift.get("test"), "shift.test")
    _close(
        validation["positive_rate"],
        freeze["validation_partition"]["positive_rate"],
        label="validation prevalence",
        tolerance=1e-18,
    )
    _close(
        test["positive_rate"],
        freeze["test_partition_metadata_only"]["positive_rate"],
        label="test prevalence",
        tolerance=1e-18,
    )
    _assert(shift.get("split_boundaries_changed") is False, "Split boundaries changed")
    _assert(shift.get("prevalence_rebalanced") is False, "Prevalence was rebalanced")
    summary = load_json_object(
        run_dir / "product" / "final_test_summary.json", "product final summary"
    )
    for key, expected in {
        "evaluation_partition": "test",
        "one_shot_final_evaluation": True,
        "final_test_opened": True,
        "frozen_champion_model": _FROZEN_CHAMPION,
        "champion_frozen_before_test": True,
        "test_used_for_model_selection": False,
        "tuning_after_test": False,
        "final_test_access_count": 1,
    }.items():
        actual = summary.get(key)
        matched = actual is expected if isinstance(expected, bool) else actual == expected
        _assert(matched, f"Bad product final summary field: {key}")
    return {
        "status": "PASS",
        "test_labels_used_for_training_preprocessing_graph_or_selection": False,
        "model_selection_partition": "validation",
        "threshold_selection_partition": "validation",
        "final_evaluation_partition": "test",
        "post_test_tuning_or_retraining": False,
        "exact_test_identity_join_recorded_in_one_shot_run": True,
        "saved_predictions_only_post_run_verification": True,
    }


def verify_final_evaluation(
    config_path: str | Path = "configs/final_evaluation.yaml",
) -> dict[str, Any]:
    """Verify final artifacts without changing files or reopening raw final-test data."""

    config = load_config(config_path)
    run_dir = get_path(config, "run_dir")
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json_object(manifest_path, "final run manifest")
    _assert(
        manifest.get("schema") == "argus.final_evaluation.run_manifest.v1",
        "Bad manifest schema",
    )
    _assert(manifest.get("sprint") == 5 and manifest.get("status") == "PASS", "Final run not PASS")
    _assert(manifest.get("frozen_champion") == _FROZEN_CHAMPION, "Manifest champion changed")
    inventory = _verify_inventory(run_dir, manifest)
    receipts = _verify_receipts(run_dir, manifest)
    freeze = load_json_object(run_dir / "freeze_contract.json", "freeze contract")
    frame = _load_predictions(run_dir)
    predictions = _verify_predictions(frame, freeze)
    metrics = _verify_metrics(run_dir, frame, freeze, manifest)
    protocol = _verify_protocol_artifacts(run_dir, freeze, manifest)
    return {
        "schema": "argus.final_evaluation.verification.v1",
        "status": "PASS",
        "mode": "read_only_saved_artifacts_no_raw_test_reopen_no_inference",
        "run_manifest_sha256": sha256_file(manifest_path),
        "inventory": inventory,
        "receipts": receipts,
        "predictions": predictions,
        "metrics": metrics,
        "test_leakage_verification": protocol,
    }


def publish_final_verification(
    config_path: str | Path = "configs/final_evaluation.yaml",
) -> dict[str, Any]:
    """Publish one verification report, then verify again without re-baselining."""

    result = verify_final_evaluation(config_path)
    config = load_config(config_path)
    run_dir = get_path(config, "run_dir")
    output = str(mapping(config["sprint5"]["outputs"], "outputs")["verification_report_json"])
    atomic_write_json(result, run_dir / output)
    second = verify_final_evaluation(config_path)
    _assert(
        second["run_manifest_sha256"] == result["run_manifest_sha256"],
        "Verifier mutated manifest",
    )
    return result


__all__ = [
    "FinalEvaluationVerificationError",
    "publish_final_verification",
    "verify_final_evaluation",
]
