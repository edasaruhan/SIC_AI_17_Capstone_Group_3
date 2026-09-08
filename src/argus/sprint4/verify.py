"""Independent verification of persisted Sprint 4 validation-only evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

from argus.app.artifacts import load_dashboard_artifacts
from argus.config import get_path, load_config
from argus.gnn.model import GraphSAGEEdgeClassifier
from argus.modeling.artifacts import (
    atomic_write_json,
    build_artifact_inventory,
    sha256_file,
    write_run_manifest,
)
from argus.modeling.metrics import compute_average_precision
from argus.product.schemas import validate_case_payload
from argus.sprint4.reporting import render_sprint4_report


class Sprint4VerificationError(RuntimeError):
    """Raised when saved Sprint 4 evidence is missing or contradictory."""


_FINAL_TEST_FALSE_FIELDS = (
    "feature_rows_loaded",
    "feature_transform_called",
    "graph_nodes_or_edges_constructed",
    "model_predictions_generated",
    "metrics_computed",
    "model_selection_or_tuning_used_test",
)
_MODEL_SCORE_COLUMNS = {
    "graphsage_edge_classifier": "raw_score_graphsage_edge_classifier",
    "refined_transaction_lightgbm": "raw_score_refined_transaction_lightgbm",
    "graph_enhanced_lightgbm": "raw_score_graph_enhanced_lightgbm",
}


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sprint4VerificationError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Sprint4VerificationError(f"JSON artifact must contain an object: {path}")
    return payload


def _assert_close(actual: object, expected: object, *, label: str) -> None:
    try:
        matches = math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15)
    except (TypeError, ValueError) as exc:
        raise Sprint4VerificationError(f"{label} is not numeric") from exc
    if not matches:
        raise Sprint4VerificationError(
            f"{label} mismatch: saved={actual!r}, recomputed={expected!r}"
        )


def validate_final_test_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Require an explicit sealed test and false access flags."""

    if policy.get("partition") != "test" or policy.get("status") != "SEALED":
        raise Sprint4VerificationError("Final test policy must identify a sealed test partition")
    for key in _FINAL_TEST_FALSE_FIELDS:
        if policy.get(key) is not False:
            raise Sprint4VerificationError(f"Final test policy {key} must be explicitly false")
    if policy.get("metadata_counts_inherited_from_sprint1") is not True:
        raise Sprint4VerificationError("Only inherited test metadata counts may be reported")
    return {
        "status": "PASS",
        "false_access_flags_verified": len(_FINAL_TEST_FALSE_FIELDS),
        "final_test_opened": False,
    }


def validate_case_collection(payload: Mapping[str, Any], *, expected_count: int) -> dict[str, Any]:
    """Validate strict case schemas, evidence counts, and validation provenance."""

    if payload.get("partition") != "validation" or payload.get("test_rows_included") is not False:
        raise Sprint4VerificationError("Cases must be validation-only")
    if payload.get("target_fields_included") is not False:
        raise Sprint4VerificationError("Cases must explicitly exclude target fields")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise Sprint4VerificationError(
            f"Case count mismatch: expected={expected_count}, actual={len(cases or [])}"
        )
    encoded = json.dumps(payload, ensure_ascii=False).lower()
    if '"is_laundering"' in encoded:
        raise Sprint4VerificationError("Case collection contains transaction targets")
    identifiers: set[str] = set()
    observed_counts: list[int] = []
    fallback_count = 0
    for case in cases:
        if not isinstance(case, Mapping):
            raise Sprint4VerificationError("Every saved case must be an object")
        validate_case_payload(case)
        case_id = str(case["case_id"])
        if case_id in identifiers:
            raise Sprint4VerificationError(f"Duplicate case ID: {case_id}")
        identifiers.add(case_id)
        observed_counts.append(len(case["observed_evidence"]))
        fallback_count += case["analyst_note"]["mode"] == "deterministic_fallback"
        if case["network"].get("directed") is not True:
            raise Sprint4VerificationError(f"Case {case_id} does not preserve graph direction")
    if min(observed_counts) < 3:
        raise Sprint4VerificationError("At least one case has fewer than three observed facts")
    return {
        "status": "PASS",
        "case_count": len(cases),
        "minimum_observed_evidence": min(observed_counts),
        "deterministic_fallback_notes": fallback_count,
        "all_case_schemas_valid": True,
        "target_fields_found": 0,
    }


def validate_manifest_inventory(
    manifest_inventory: Sequence[Mapping[str, Any]],
    actual_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require exact path, size, and SHA-256 agreement."""

    saved = {str(item.get("path")): dict(item) for item in manifest_inventory}
    actual = {str(item.get("path")): dict(item) for item in actual_inventory}
    if saved != actual:
        missing = sorted(set(saved).difference(actual))
        extra = sorted(set(actual).difference(saved))
        changed = sorted(
            path for path in set(saved).intersection(actual) if saved[path] != actual[path]
        )
        raise Sprint4VerificationError(
            f"Artifact inventory mismatch: missing={missing}, extra={extra}, changed={changed}"
        )
    return {
        "status": "PASS",
        "artifact_count": len(actual),
        "path_size_sha256_exact": True,
    }


def _verify_predictions(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    expected_rows: int,
    expected_positives: int,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    description = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path.resolve())]
    ).fetch_df()
    columns = set(description["column_name"].astype(str))
    required = {
        "transaction_id",
        "source_row_number",
        "evaluation_partition",
        "is_laundering",
        *_MODEL_SCORE_COLUMNS.values(),
        "probability_graphsage_edge_classifier",
        "probability_refined_transaction_lightgbm",
        "probability_graph_enhanced_lightgbm",
    }
    missing = sorted(required.difference(columns))
    if missing:
        raise Sprint4VerificationError(f"Prediction artifact columns missing: {missing}")
    counts = connection.execute(
        """
        SELECT
            count(*), count(DISTINCT transaction_id), count(DISTINCT source_row_number),
            sum(is_laundering), count(DISTINCT evaluation_partition),
            min(evaluation_partition),
            count(*) FILTER (
                WHERE NOT isfinite(raw_score_graphsage_edge_classifier)
                   OR NOT isfinite(probability_graphsage_edge_classifier)
                   OR NOT isfinite(raw_score_refined_transaction_lightgbm)
                   OR NOT isfinite(raw_score_graph_enhanced_lightgbm)
            )
        FROM read_parquet(?)
        """,
        [str(path.resolve())],
    ).fetchone()
    if tuple(int(counts[index]) for index in range(4)) != (
        expected_rows,
        expected_rows,
        expected_rows,
        expected_positives,
    ):
        raise Sprint4VerificationError(f"Prediction row identity/count mismatch: {counts}")
    if int(counts[4]) != 1 or counts[5] != "validation" or int(counts[6]) != 0:
        raise Sprint4VerificationError("Predictions are not finite validation-only scores")

    frame = connection.execute(
        """
        SELECT source_row_number, is_laundering,
               raw_score_graphsage_edge_classifier,
               raw_score_refined_transaction_lightgbm,
               raw_score_graph_enhanced_lightgbm
        FROM read_parquet(?) ORDER BY source_row_number
        """,
        [str(path.resolve())],
    ).fetch_df()
    labels = frame["is_laundering"].to_numpy(np.int8)
    recomputed: dict[str, float] = {}
    saved_by_name = {str(row["model"]): row for row in manifest_rows}
    if set(saved_by_name) != set(_MODEL_SCORE_COLUMNS):
        raise Sprint4VerificationError("Manifest comparison does not contain exactly three models")
    for name, column in _MODEL_SCORE_COLUMNS.items():
        metric = compute_average_precision(labels, frame[column].to_numpy())
        recomputed[name] = float(metric)
        _assert_close(
            saved_by_name[name]["average_precision"], metric, label=f"{name} average precision"
        )
        if saved_by_name[name].get("evaluation_partition") != "validation":
            raise Sprint4VerificationError(f"{name} comparison is not validation-only")
        if saved_by_name[name].get("test_metrics_used") is not False:
            raise Sprint4VerificationError(f"{name} reports test metric use")
    return {
        "status": "PASS",
        "rows": expected_rows,
        "positive_labels": expected_positives,
        "partition": "validation",
        "models_recomputed": recomputed,
        "nonfinite_rows": 0,
    }


def verify_sprint4_run(
    config_path: str | Path = "configs/sprint4.yaml",
    *,
    refresh_manifest: bool = True,
) -> dict[str, Any]:
    """Independently recompute core Sprint 4 artifact facts without model fitting."""

    config = load_config(config_path)
    settings = config["sprint4"]
    frozen = settings["frozen_references"]
    run_dir = get_path(config, "run_dir")
    sprint3_dir = get_path(config, "sprint3_run_dir")
    report_path = get_path(config, "generated_report")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _load_object(manifest_path)
    status = str(manifest.get("status", ""))
    if manifest.get("sprint") != 4 or status not in {"PASS", "FAIL_QUALITY_CHECKS"}:
        raise Sprint4VerificationError("Sprint 4 manifest status is not verifiable")
    failed_acceptance = sorted(
        key for key, passed in manifest.get("acceptance", {}).items() if passed is not True
    )
    recoverable_quality_failure = status == "FAIL_QUALITY_CHECKS" and failed_acceptance == [
        "automated_quality_checks"
    ]
    if failed_acceptance and not recoverable_quality_failure:
        raise Sprint4VerificationError(
            f"Sprint 4 manifest has failed acceptance items: {failed_acceptance}"
        )
    node_design = manifest.get("node_feature_design", {})
    expected_node_features = [
        *settings["node_features"]["structural_features"],
        *settings["node_features"]["deterministic_identity_features"],
    ]
    if (
        node_design.get("features") != expected_node_features
        or node_design.get("monetary_node_aggregates_used") is not False
        or node_design.get("amount_aggregation_policy")
        != settings["node_features"]["amount_aggregation_policy"]
    ):
        raise Sprint4VerificationError("Node features violate the cross-currency safeguard")
    score_semantics = manifest.get("score_semantics", {})
    if (
        score_semantics.get("graphsage_ranking_score") != "raw_logit"
        or score_semantics.get("graphsage_sigmoid_score_calibrated") is not False
    ):
        raise Sprint4VerificationError("GraphSAGE score calibration semantics are missing")

    actual_inventory = build_artifact_inventory(
        run_dir, exclude_paths=[manifest_path], exclude_names={".gitkeep"}
    )
    inventory_evidence = validate_manifest_inventory(manifest["artifacts"], actual_inventory)
    sprint3_manifest = sprint3_dir / "run_manifest.json"
    sprint3_sha = sha256_file(sprint3_manifest)
    if sprint3_sha != frozen["sprint3_manifest_sha256"]:
        raise Sprint4VerificationError("Frozen Sprint 3 manifest hash changed after Sprint 4")

    final_policy = _load_object(run_dir / "final_test_policy.json")
    final_test_evidence = validate_final_test_policy(final_policy)
    connection = duckdb.connect()
    try:
        prediction_evidence = _verify_predictions(
            connection,
            run_dir / "validation_predictions.parquet",
            expected_rows=int(frozen["validation_rows"]),
            expected_positives=int(frozen["validation_positives"]),
            manifest_rows=manifest["model_comparison"],
        )
        top_ids = (
            connection.execute(
                """
            SELECT transaction_id FROM read_parquet(?)
            ORDER BY raw_score_graphsage_edge_classifier DESC, source_row_number ASC
            LIMIT ?
            """,
                [
                    str((run_dir / "validation_predictions.parquet").resolve()),
                    int(settings["case_builder"]["maximum_cases"]),
                ],
            )
            .fetch_df()["transaction_id"]
            .astype(str)
            .tolist()
        )
    finally:
        connection.close()

    case_payload = _load_object(run_dir / "product" / "cases.json")
    case_evidence = validate_case_collection(
        case_payload, expected_count=int(settings["case_builder"]["maximum_cases"])
    )
    case_ids = [str(item["transaction_id"]) for item in case_payload["cases"]]
    if case_ids != top_ids:
        raise Sprint4VerificationError("Saved cases are not the deterministic top GraphSAGE rows")
    case_evidence["deterministic_top_score_rows_verified"] = True
    if any(
        "uncalibrated" not in str(item["model_evidence"].get("score_name", ""))
        for item in case_payload["cases"]
    ):
        raise Sprint4VerificationError("Case scores are not labeled as uncalibrated")

    checkpoint = torch.load(
        run_dir / "model" / "graphsage.pt", map_location="cpu", weights_only=True
    )
    model = GraphSAGEEdgeClassifier.from_checkpoint(checkpoint)
    if model.config.transaction_feature_dim <= 0:
        raise Sprint4VerificationError("GraphSAGE checkpoint lacks transaction features")
    embeddings = np.load(run_dir / "model" / "inference_node_embeddings.npy", mmap_mode="r")
    node_count = int(
        _load_object(run_dir / "model" / "inference_graph_manifest.json")["node_count"]
    )
    if embeddings.shape != (node_count, model.config.embedding_dim):
        raise Sprint4VerificationError(
            f"Embedding shape mismatch: {embeddings.shape}, expected nodes={node_count}"
        )
    if not np.isfinite(embeddings).all():
        raise Sprint4VerificationError("Saved node embeddings contain non-finite values")
    del embeddings, model

    shap = _load_object(run_dir / "product" / "tree_shap_explanations.json")
    gnn = _load_object(run_dir / "product" / "graphsage_explanations.json")
    if (
        shap.get("method") != "lightgbm_native_pred_contrib_treeshap"
        or float(shap.get("maximum_additivity_absolute_error", math.inf)) > 1e-8
        or shap.get("test_rows_used") is not False
    ):
        raise Sprint4VerificationError("TreeSHAP artifact failed method/additivity safeguards")
    if (
        gnn.get("method") != "local_gradient_x_input_sensitivity"
        or gnn.get("claim_scope") != "sensitivity_not_shap"
        or gnn.get("test_rows_used") is not False
        or any(
            item.get("calibrated_probability") is not False for item in gnn.get("explanations", [])
        )
    ):
        raise Sprint4VerificationError("GNN explanation is mislabeled or uses test rows")
    dashboard = load_dashboard_artifacts(run_dir)
    if len(dashboard.cases) != len(case_payload["cases"]):
        raise Sprint4VerificationError("Streamlit loader case count differs from case artifact")

    verification = {
        "status": "PASS",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "verification_mode": "saved_artifacts_only_no_training",
        "artifact_inventory": inventory_evidence,
        "frozen_sprint3": {
            "expected_manifest_sha256": frozen["sprint3_manifest_sha256"],
            "actual_manifest_sha256": sprint3_sha,
            "match": True,
        },
        "predictions": prediction_evidence,
        "final_test": final_test_evidence,
        "cases": case_evidence,
        "graphsage": {
            "checkpoint_restored": True,
            "target_unit": "transaction_edge",
            "unsupported_account_label_created": False,
            "inference_node_count": node_count,
            "embedding_dimension": int(checkpoint["config"]["embedding_dim"]),
            "embedding_shape_verified": True,
            "uncalibrated_sigmoid_label_verified": True,
            "cross_currency_node_amount_aggregation_excluded": True,
        },
        "explanations": {
            "tree_shap_additivity_verified": True,
            "gnn_sensitivity_not_shap_label_verified": True,
        },
        "streamlit_saved_artifact_load": "PASS",
    }
    if refresh_manifest:
        atomic_write_json(verification, run_dir / "verification_report.json")
        manifest["post_run_verification"] = verification
        if "artifacts/sprint4/verification_report.json" not in manifest["core_artifact_paths"]:
            manifest["core_artifact_paths"].append("artifacts/sprint4/verification_report.json")
        render_sprint4_report(manifest, run_dir / "SPRINT_4_STATUS.md")
        refreshed = write_run_manifest(manifest, run_dir)
        render_sprint4_report(refreshed, report_path)
        verification["refreshed_artifact_count_excluding_manifest"] = refreshed[
            "artifact_count_excluding_manifest"
        ]
    return verification


__all__ = [
    "Sprint4VerificationError",
    "validate_case_collection",
    "validate_final_test_policy",
    "validate_manifest_inventory",
    "verify_sprint4_run",
]
