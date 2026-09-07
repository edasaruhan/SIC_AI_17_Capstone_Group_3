"""Independent verification of saved Sprint 2 validation artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from argus.config import get_path, load_config
from argus.modeling.artifacts import (
    atomic_write_json,
    build_artifact_inventory,
    sha256_file,
    write_run_manifest,
)
from argus.modeling.metrics import evaluate_binary_predictions
from argus.modeling.reporting import PR_CURVE_MAX_PLOT_POINTS
from argus.modeling.selection import select_transaction_baseline_champion


class BaselineVerificationError(RuntimeError):
    """Raised when persisted Sprint 2 evidence does not reconcile."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineVerificationError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineVerificationError(f"JSON artifact is not an object: {path}")
    return payload


def _verify_saved_inventory(run_dir: Path, manifest: Mapping[str, Any]) -> int:
    inventory = manifest.get("artifacts")
    if not isinstance(inventory, list):
        raise BaselineVerificationError("Run manifest has no artifact inventory")
    if int(manifest.get("artifact_count_excluding_manifest", -1)) != len(inventory):
        raise BaselineVerificationError("Run manifest artifact count is inconsistent")
    for item in inventory:
        if not isinstance(item, Mapping):
            raise BaselineVerificationError("Run manifest inventory entry is invalid")
        path = run_dir / str(item["path"])
        if not path.is_file():
            raise BaselineVerificationError(f"Manifest artifact is missing: {path}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise BaselineVerificationError(f"Manifest artifact size differs: {path}")
        if sha256_file(path) != item["sha256"]:
            raise BaselineVerificationError(f"Manifest artifact SHA-256 differs: {path}")
    current_paths = {
        item["path"]
        for item in build_artifact_inventory(
            run_dir,
            exclude_paths=[run_dir / "run_manifest.json"],
        )
    }
    saved_paths = {str(item["path"]) for item in inventory}
    extra = current_paths - saved_paths
    # A previous verifier report is allowed because it is incorporated into the
    # refreshed manifest at the end of this function.
    if extra - {"verification_report.json"}:
        raise BaselineVerificationError(f"Unmanifested artifacts found: {sorted(extra)}")
    missing = saved_paths - current_paths
    if missing:
        raise BaselineVerificationError(
            f"Manifest entries absent from run directory: {sorted(missing)}"
        )
    return len(inventory)


def _verify_prediction_table(
    run_dir: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    prediction_path = run_dir / "validation_predictions.parquet"
    connection = duckdb.connect()
    try:
        prediction = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY source_row_number",
            [str(prediction_path.resolve())],
        ).fetch_df()
        connection.register("argus_saved_prediction_ids", prediction[["transaction_id"]])
        membership_rows = connection.execute(
            "SELECT split.partition, count(*) "
            "FROM argus_saved_prediction_ids prediction "
            "INNER JOIN read_parquet(?) split USING (transaction_id) "
            "GROUP BY split.partition ORDER BY split.partition",
            [str(get_path(config, "split_table").resolve())],
        ).fetchall()
        connection.unregister("argus_saved_prediction_ids")
    finally:
        connection.close()
    expected_columns = {
        "transaction_id",
        "source_row_number",
        "is_laundering",
        "score_logistic_regression",
        "score_random_forest",
        "score_lightgbm",
    }
    if set(prediction.columns) != expected_columns:
        raise BaselineVerificationError(
            f"Validation prediction schema differs: {sorted(prediction.columns)}"
        )
    expected_rows = int(manifest["split_prevalence"]["validation"]["rows"])
    expected_positives = int(manifest["split_prevalence"]["validation"]["positive_labels"])
    if len(prediction) != expected_rows:
        raise BaselineVerificationError("Validation prediction row count differs")
    if prediction["transaction_id"].nunique() != expected_rows:
        raise BaselineVerificationError("Validation prediction transaction IDs are not unique")
    if prediction["source_row_number"].nunique() != expected_rows:
        raise BaselineVerificationError("Validation prediction source rows are not unique")
    labels = prediction["is_laundering"].to_numpy(dtype=np.int8)
    if int(labels.sum()) != expected_positives:
        raise BaselineVerificationError("Validation prediction labels differ from frozen metadata")
    membership = {str(partition): int(rows) for partition, rows in membership_rows}
    if membership != {"validation": expected_rows}:
        raise BaselineVerificationError(
            f"Saved prediction identities are not exactly the frozen validation set: {membership}"
        )

    baseline = config["baseline"]
    recomputed: dict[str, dict[str, Any]] = {}
    for model_name in ("logistic_regression", "random_forest", "lightgbm"):
        scores = prediction[f"score_{model_name}"].to_numpy(dtype=np.float64)
        metrics = evaluate_binary_predictions(
            labels,
            scores,
            prediction["source_row_number"].to_numpy(dtype=np.int64),
            threshold=float(baseline["decision_threshold"]),
            top_k_values=baseline["top_k"],
        )
        saved = _load_object(run_dir / "models" / model_name / "validation_metrics.json")
        if metrics != saved:
            raise BaselineVerificationError(
                f"Recomputed validation metrics differ for {model_name}"
            )
        model = joblib.load(run_dir / "models" / model_name / "model.joblib")
        expected_class = {
            "logistic_regression": "SGDClassifier",
            "random_forest": "RandomForestClassifier",
            "lightgbm": "LGBMClassifier",
        }[model_name]
        if type(model).__name__ != expected_class:
            raise BaselineVerificationError(
                f"Saved {model_name} model class differs: {type(model).__name__}"
            )
        recomputed[model_name] = metrics

    source_prefixes = prediction["transaction_id"].str.extract(r"^(.*):row-\d+$", expand=False)
    if source_prefixes.isna().any() or source_prefixes.nunique() != 1:
        raise BaselineVerificationError("Validation transaction identity format is invalid")
    return {
        "rows": expected_rows,
        "positive_labels": expected_positives,
        "unique_transaction_ids": expected_rows,
        "unique_source_row_numbers": expected_rows,
        "source_filename": str(source_prefixes.iloc[0]),
        "evaluation_partition": "validation",
        "frozen_split_membership": membership,
        "test_rows_present": membership.get("test", 0) > 0,
    }, recomputed


def _verify_champion(
    run_dir: Path,
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    recomputed = select_transaction_baseline_champion(
        {name: float(values["average_precision"]) for name, values in metrics.items()},
        partition="validation",
        test_metrics=None,
    )
    saved = _load_object(run_dir / "transaction_baseline_champion.json")
    if recomputed != saved:
        raise BaselineVerificationError("Saved champion differs from validation-only recomputation")
    return recomputed


def _verify_comparison_artifacts(
    run_dir: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    champion: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify comparison values, provenance, and Top-K tie diagnostics."""

    comparison_payload = _load_object(run_dir / "model_comparison.json")
    frozen = manifest["provenance"]["frozen_inputs"]
    expected_context = {
        "dataset_name": manifest["provenance"]["dataset"],
        "raw_transaction_sha256": manifest["provenance"]["raw_sources"]["transactions"]["sha256"],
        "feature_table_sha256": frozen["transaction_features.parquet"]["sha256"],
        "split_manifest_sha256": frozen["split_manifest.parquet"]["sha256"],
        "split_metadata_sha256": frozen["split_metadata.json"]["sha256"],
        "feature_scope": config["baseline"]["feature_scope"],
        "preprocessing_state_sha256": manifest["feature_contract"]["state_sha256"],
        "random_seed": manifest["configuration"]["random_seed"],
        "final_test_used": False,
        "library_versions": manifest["libraries"],
    }
    if comparison_payload.get("provenance") != expected_context:
        raise BaselineVerificationError("Comparison JSON provenance differs from run inputs")
    if comparison_payload.get("evaluation_partition") != "validation":
        raise BaselineVerificationError("Comparison JSON is not validation-only")
    if comparison_payload.get("accuracy_is_primary") is not False:
        raise BaselineVerificationError("Comparison JSON incorrectly promotes accuracy")

    json_results = comparison_payload.get("model_results")
    if not isinstance(json_results, Mapping) or set(json_results) != set(metrics):
        raise BaselineVerificationError("Comparison JSON model set differs")
    for model_name, recomputed in metrics.items():
        result = json_results[model_name]
        if not isinstance(result, Mapping) or result.get("validation_metrics") != recomputed:
            raise BaselineVerificationError(
                f"Comparison JSON validation metrics differ for {model_name}"
            )
        for top_k in recomputed["top_k_metrics"]:
            required_tie_fields = {
                "cutoff_score",
                "cutoff_tie_group_size",
                "selected_from_cutoff_tie_group",
                "cutoff_tie_group_fully_included",
                "tie_break_material_at_cutoff",
            }
            if not required_tie_fields.issubset(top_k):
                raise BaselineVerificationError(
                    f"Top-K tie diagnostics are incomplete for {model_name}"
                )

    comparison = pd.read_csv(run_dir / "model_comparison.csv")
    context_columns = set(expected_context) - {"library_versions"}
    required_columns = {
        "model",
        "model_implementation",
        "model_library_version",
        "evaluation_partition",
        "pr_auc_average_precision",
        "roc_auc_secondary",
        *context_columns,
    }
    for requested_k in config["baseline"]["top_k"]:
        required_columns.update(
            {
                f"precision_at_{requested_k}",
                f"recall_at_{requested_k}",
                f"true_positives_at_{requested_k}",
                f"cutoff_score_at_{requested_k}",
                f"cutoff_tie_group_size_at_{requested_k}",
                f"selected_from_cutoff_tie_group_at_{requested_k}",
                f"tie_break_material_at_{requested_k}",
            }
        )
    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise BaselineVerificationError(
            f"Comparison CSV columns are incomplete: {sorted(missing_columns)}"
        )
    if len(comparison) != len(metrics) or set(comparison["model"]) != set(metrics):
        raise BaselineVerificationError("Comparison CSV model rows differ")
    if set(comparison["evaluation_partition"]) != {"validation"}:
        raise BaselineVerificationError("Comparison table includes a non-validation partition")
    if "accuracy" in comparison.columns:
        raise BaselineVerificationError("Accuracy must not be a primary comparison column")
    if comparison["final_test_used"].tolist() != [False] * len(comparison):
        raise BaselineVerificationError("Comparison CSV indicates final-test use")
    for key, expected in expected_context.items():
        if key in {"library_versions", "final_test_used"}:
            continue
        if set(comparison[key].astype(str)) != {str(expected)}:
            raise BaselineVerificationError(f"Comparison CSV provenance differs: {key}")
    library_versions = expected_context["library_versions"]
    expected_library = {
        "logistic_regression": library_versions["scikit_learn"],
        "random_forest": library_versions["scikit_learn"],
        "lightgbm": library_versions["lightgbm"],
    }
    for row in comparison.itertuples(index=False):
        model_name = str(row.model)
        json_result = json_results[model_name]
        if row.model_implementation != json_result["implementation"]:
            raise BaselineVerificationError(f"Comparison implementation differs for {model_name}")
        if str(row.model_library_version) != str(expected_library[model_name]):
            raise BaselineVerificationError(f"Comparison library version differs for {model_name}")
        if not np.isclose(
            float(row.pr_auc_average_precision),
            float(metrics[model_name]["average_precision"]),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise BaselineVerificationError(f"Comparison PR-AUC differs for {model_name}")
        top_k_by_requested = {
            int(item["requested_k"]): item for item in metrics[model_name]["top_k_metrics"]
        }
        for requested_k, top_k in top_k_by_requested.items():
            if int(getattr(row, f"cutoff_tie_group_size_at_{requested_k}")) != int(
                top_k["cutoff_tie_group_size"]
            ):
                raise BaselineVerificationError(
                    f"Comparison cutoff tie size differs for {model_name} at K={requested_k}"
                )
            if int(getattr(row, f"selected_from_cutoff_tie_group_at_{requested_k}")) != int(
                top_k["selected_from_cutoff_tie_group"]
            ):
                raise BaselineVerificationError(
                    f"Comparison selected tie rows differ for {model_name} at K={requested_k}"
                )
            if bool(getattr(row, f"tie_break_material_at_{requested_k}")) is not bool(
                top_k["tie_break_material_at_cutoff"]
            ):
                raise BaselineVerificationError(
                    f"Comparison tie-break flag differs for {model_name} at K={requested_k}"
                )
    if comparison.iloc[0]["model"] != champion["champion_model"]:
        raise BaselineVerificationError("Comparison ordering disagrees with champion")
    return {
        "partition": "validation",
        "provenance_embedded": True,
        "top_k_cutoff_tie_diagnostics": True,
        "models": len(comparison),
    }


def verify_sprint2_run(
    config_path: str | Path = "configs/baseline.yaml",
) -> dict[str, Any]:
    """Verify hashes, predictions, metrics, models, champion, and test policy."""

    config = load_config(config_path)
    run_dir = get_path(config, "run_dir")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _load_object(manifest_path)
    if manifest.get("status") not in {"PASS", "FAIL_QUALITY_CHECKS"}:
        raise BaselineVerificationError("Sprint 2 run manifest has no verifiable core run")
    plot_evidence = manifest.get("derived_artifacts", {}).get("precision_recall_figure", {})
    if plot_evidence != {
        "numeric_metrics_use_all_validation_rows": True,
        "display_only_decimation": "deterministic_even_index_with_endpoints",
        "maximum_points_per_model": PR_CURVE_MAX_PLOT_POINTS,
    }:
        raise BaselineVerificationError("Precision-recall visualization policy is incomplete")

    verified_inventory_count = _verify_saved_inventory(run_dir, manifest)
    prediction_evidence, recomputed_metrics = _verify_prediction_table(run_dir, manifest, config)
    champion = _verify_champion(run_dir, recomputed_metrics)
    final_test_policy = _load_object(run_dir / "final_test_policy.json")
    forbidden_true = [
        "final_test_used_for_training",
        "final_test_used_for_preprocessing_fit",
        "final_test_used_for_model_selection",
        "final_test_inference_performed",
        "test_feature_rows_materialized",
        "test_prediction_artifact_exists",
    ]
    if any(final_test_policy.get(key) is not False for key in forbidden_true):
        raise BaselineVerificationError("Saved final-test access policy is not closed")
    unexpected_test_predictions = sorted(run_dir.rglob("*test*prediction*"))
    if unexpected_test_predictions:
        raise BaselineVerificationError(
            f"Unexpected test prediction artifacts exist: {unexpected_test_predictions}"
        )

    comparison_evidence = _verify_comparison_artifacts(
        run_dir,
        manifest,
        config,
        recomputed_metrics,
        champion,
    )

    report = {
        "status": "PASS",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "method": (
            "SHA-256/size inventory; validation Parquet identity/count checks; full metric "
            "recomputation from saved scores; model deserialization; validation-only champion "
            "reselection; final-test policy and artifact-name checks"
        ),
        "saved_inventory_entries_verified": verified_inventory_count,
        "validation_predictions": prediction_evidence,
        "validation_metrics_recomputed": {
            name: {
                "average_precision": values["average_precision"],
                "roc_auc": values["roc_auc"],
            }
            for name, values in recomputed_metrics.items()
        },
        "champion_recomputed": champion,
        "saved_models_deserialized": [
            "logistic_regression",
            "random_forest",
            "lightgbm",
        ],
        "comparison_partition": comparison_evidence["partition"],
        "comparison_provenance_verified": comparison_evidence["provenance_embedded"],
        "top_k_cutoff_tie_diagnostics_verified": comparison_evidence[
            "top_k_cutoff_tie_diagnostics"
        ],
        "precision_recall_numeric_metrics_use_all_rows": True,
        "precision_recall_display_maximum_points_per_model": PR_CURVE_MAX_PLOT_POINTS,
        "accuracy_primary": False,
        "final_test_policy_verified_closed": True,
        "test_prediction_artifacts_found": 0,
    }
    atomic_write_json(report, run_dir / "verification_report.json")
    manifest["post_run_verification"] = report
    refreshed = write_run_manifest(manifest, run_dir)
    return {
        **report,
        "refreshed_artifact_count_excluding_manifest": refreshed[
            "artifact_count_excluding_manifest"
        ],
    }
