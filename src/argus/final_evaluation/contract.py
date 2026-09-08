"""Immutable pre-test contract and exclusive final-test access gate."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.modeling.artifacts import file_fingerprint, sha256_file
from argus.modeling.baseline import _source_snapshot
from argus.modeling.refinement import git_is_ancestor, subprocess_checkpoint


class FinalEvaluationContractError(RuntimeError):
    """Raised before final-test access when a frozen dependency has changed."""


def mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalEvaluationContractError(f"{name} must be a mapping")
    return value


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalEvaluationContractError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalEvaluationContractError(f"{label} must contain a JSON object")
    return payload


def project_path(project_root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FinalEvaluationContractError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _verify_file_reference(
    project_root: Path,
    payload: Mapping[str, Any],
    name: str,
) -> tuple[Path, dict[str, Any]]:
    path = project_path(project_root, payload.get("path"), f"{name}.path")
    expected = payload.get("sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise FinalEvaluationContractError(
            f"Frozen file hash changed for {name}: expected={expected}, actual={actual}"
        )
    return path, file_fingerprint(path, relative_to=project_root)


def _verify_direct_file(
    project_root: Path,
    payload: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    name: str,
) -> tuple[Path, dict[str, Any]]:
    path = project_path(project_root, payload.get(path_key), f"{name}.{path_key}")
    expected = payload.get(sha_key)
    actual = sha256_file(path)
    if actual != expected:
        raise FinalEvaluationContractError(
            f"Frozen file hash changed for {name}: expected={expected}, actual={actual}"
        )
    return path, file_fingerprint(path, relative_to=project_root)


def _threshold_from_artifact(payload: Mapping[str, Any], model_name: str) -> float:
    selected: object
    if "models" in payload:
        models = mapping(payload["models"], "thresholds.models")
        selected = mapping(models.get(model_name), f"thresholds.models.{model_name}")
    else:
        selected = payload
    summaries = mapping(mapping(selected, "threshold artifact").get("summaries"), "summaries")
    primary = mapping(
        summaries.get("predeclared_joint_primary_constraint"),
        "predeclared_joint_primary_constraint",
    )
    point = mapping(primary.get("operating_point"), "operating_point")
    threshold = float(point["threshold"])
    if not math.isfinite(threshold):
        raise FinalEvaluationContractError("Frozen threshold artifact contains a non-finite value")
    if (
        selected.get("partition") != "validation"
        or selected.get("test_partition_used") is not False
    ):
        raise FinalEvaluationContractError("Frozen operating threshold is not validation-only")
    return threshold


def build_freeze_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every predeclared dependency without reading a final-test row."""

    meta = mapping(config.get("_meta"), "_meta")
    project_root = Path(str(meta["project_root"])).resolve()
    settings = mapping(config.get("sprint5"), "sprint5")
    authorization = mapping(settings.get("authorization"), "sprint5.authorization")
    if authorization.get("explicit_user_authorization") is not True:
        raise FinalEvaluationContractError("Explicit Sprint 5 final-test authorization is absent")
    if authorization.get("authorized_stage") != "final_evaluation":
        raise FinalEvaluationContractError("Authorization is not scoped to final_evaluation")

    frozen = mapping(settings.get("frozen_references"), "sprint5.frozen_references")
    fingerprints: dict[str, Any] = {}
    resolved_references: dict[str, Path] = {}
    for key in (
        "sprint3_manifest",
        "sprint4_manifest",
        "full_manifest",
        "split_manifest",
        "feature_store",
        "sprint4_thresholds",
        "sprint4_model_comparison",
    ):
        path, fingerprint = _verify_file_reference(
            project_root,
            mapping(frozen.get(key), f"sprint5.frozen_references.{key}"),
            f"sprint5.frozen_references.{key}",
        )
        resolved_references[key] = path
        fingerprints[key] = fingerprint

    head = subprocess_checkpoint(project_root)
    checkpoint = str(frozen["sprint4_checkpoint"])
    if head != checkpoint and not git_is_ancestor(project_root, checkpoint, head):
        raise FinalEvaluationContractError("The accepted Sprint 4 checkpoint is not in Git history")
    sprint3_manifest = load_json_object(
        resolved_references["sprint3_manifest"], "Sprint 3 manifest"
    )
    sprint4_manifest = load_json_object(
        resolved_references["sprint4_manifest"], "Sprint 4 manifest"
    )
    if sprint3_manifest.get("status") != "PASS" or not all(
        mapping(sprint3_manifest.get("acceptance"), "Sprint 3 acceptance").values()
    ):
        raise FinalEvaluationContractError("Frozen Sprint 3 evidence is not accepted")
    if sprint4_manifest.get("status") != "PASS" or not all(
        mapping(sprint4_manifest.get("acceptance"), "Sprint 4 acceptance").values()
    ):
        raise FinalEvaluationContractError("Frozen Sprint 4 evidence is not accepted")
    data_access = mapping(sprint4_manifest.get("data_access_audit"), "Sprint 4 data access")
    if any(
        data_access.get(key) is not False
        for key in (
            "test_feature_rows_loaded",
            "test_labels_loaded",
            "test_predictions_generated",
            "test_metrics_computed",
        )
    ):
        raise FinalEvaluationContractError("Sprint 4 no longer proves that final test was sealed")

    comparison_payload = load_json_object(
        resolved_references["sprint4_model_comparison"], "Sprint 4 model comparison"
    )
    comparison_rows = comparison_payload.get("models")
    if not isinstance(comparison_rows, list) or len(comparison_rows) != 3:
        raise FinalEvaluationContractError("Sprint 4 comparison must contain exactly three models")
    if comparison_payload.get("validation_leader") != "graph_enhanced_lightgbm":
        raise FinalEvaluationContractError(
            "Frozen validation leader is not graph_enhanced_lightgbm"
        )
    comparison_by_model: dict[str, Mapping[str, Any]] = {}
    for row in comparison_rows:
        item = mapping(row, "Sprint 4 comparison row")
        name = str(item.get("model"))
        if name in comparison_by_model:
            raise FinalEvaluationContractError(f"Duplicate Sprint 4 model row: {name}")
        comparison_by_model[name] = item
    if sum(item.get("is_validation_leader") is True for item in comparison_by_model.values()) != 1:
        raise FinalEvaluationContractError("Sprint 4 must have exactly one validation leader")

    split_metadata_path = project_path(
        project_root,
        mapping(config.get("paths"), "paths").get("split_metadata"),
        "paths.split_metadata",
    )
    split_metadata = load_json_object(split_metadata_path, "Sprint 1 split metadata")
    expected_split_sha = mapping(
        mapping(config.get("baseline"), "baseline").get("frozen_upstream"),
        "baseline.frozen_upstream",
    ).get("split_metadata_sha256")
    if sha256_file(split_metadata_path) != expected_split_sha:
        raise FinalEvaluationContractError("Sprint 1 split metadata hash changed")
    if (
        split_metadata.get("strategy") != "chronological"
        or split_metadata.get("strict_boundaries_verified") is not True
        or split_metadata.get("no_transaction_overlap_verified") is not True
    ):
        raise FinalEvaluationContractError("Sprint 1 split proof is incomplete")
    split_partitions = mapping(split_metadata.get("partitions"), "split_metadata.partitions")
    for config_key, partition_name in (
        ("validation_partition", "validation"),
        ("test_partition", "test"),
    ):
        declared = mapping(frozen.get(config_key), f"sprint5.frozen_references.{config_key}")
        observed = mapping(split_partitions.get(partition_name), f"split.{partition_name}")
        exact = {
            "rows": int(observed["rows"]),
            "positives": int(observed["positive_labels"]),
            "negatives": int(observed["rows"]) - int(observed["positive_labels"]),
            "positive_rate": int(observed["positive_labels"]) / int(observed["rows"]),
            "minimum_timestamp": str(observed["minimum_timestamp"]),
            "maximum_timestamp": str(observed["maximum_timestamp"]),
        }
        for key, actual in exact.items():
            expected = declared.get(key)
            if isinstance(actual, float):
                matched = math.isclose(float(expected), actual, rel_tol=0.0, abs_tol=1e-18)
            else:
                matched = expected == actual
            if not matched:
                raise FinalEvaluationContractError(
                    f"Frozen {partition_name} metadata changed for {key}: "
                    f"expected={expected}, actual={actual}"
                )

    models = mapping(settings.get("models"), "sprint5.models")
    expected_models = {
        "graph_enhanced_lightgbm",
        "refined_transaction_lightgbm",
        "graphsage_edge_classifier",
    }
    if set(models) != expected_models:
        raise FinalEvaluationContractError("Final model set differs from the predeclared set")
    model_contracts: dict[str, Any] = {}
    for model_name in sorted(models):
        model = mapping(models[model_name], f"sprint5.models.{model_name}")
        model_path, model_fingerprint = _verify_direct_file(
            project_root,
            model,
            path_key="model_path",
            sha_key="model_sha256",
            name=f"sprint5.models.{model_name}.model",
        )
        state_path, state_fingerprint = _verify_direct_file(
            project_root,
            model,
            path_key="preprocessor_state_path",
            sha_key="preprocessor_state_sha256",
            name=f"sprint5.models.{model_name}.preprocessor_state",
        )
        manifest_path, manifest_fingerprint = _verify_direct_file(
            project_root,
            model,
            path_key="preprocessor_manifest_path",
            sha_key="preprocessor_manifest_sha256",
            name=f"sprint5.models.{model_name}.preprocessor_manifest",
        )
        threshold_path, threshold_fingerprint = _verify_direct_file(
            project_root,
            model,
            path_key="threshold_source_path",
            sha_key="threshold_source_sha256",
            name=f"sprint5.models.{model_name}.threshold_source",
        )
        preprocessor_manifest = load_json_object(manifest_path, f"{model_name} preprocessor")
        # GraphSAGE consumes the frozen transaction+temporal/history transform,
        # then appends sender/receiver embeddings inside the already-fitted edge
        # classifier.  Its overall feature-family label therefore intentionally
        # differs from the referenced tabular preprocessor's family label.
        expected_preprocessor_family = (
            "transaction_temporal_history"
            if model_name == "graphsage_edge_classifier"
            else model.get("feature_family")
        )
        if (
            preprocessor_manifest.get("fit_scope") != "train_only"
            or int(preprocessor_manifest.get("fitted_train_rows", -1)) != 3_554_957
            or preprocessor_manifest.get("feature_family") != expected_preprocessor_family
        ):
            raise FinalEvaluationContractError(f"Unsafe frozen preprocessor for {model_name}")
        threshold_payload = load_json_object(threshold_path, f"{model_name} threshold source")
        artifact_name = model_name if "models" in threshold_payload else model_name
        actual_threshold = _threshold_from_artifact(threshold_payload, artifact_name)
        expected_threshold = float(model["threshold"])
        if not math.isclose(actual_threshold, expected_threshold, rel_tol=0.0, abs_tol=0.0):
            raise FinalEvaluationContractError(
                f"Frozen threshold changed for {model_name}: "
                f"expected={expected_threshold}, actual={actual_threshold}"
            )
        comparison = comparison_by_model[model_name]
        if not math.isclose(
            float(comparison["average_precision"]),
            float(model["validation_average_precision"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise FinalEvaluationContractError(f"Frozen validation AP changed for {model_name}")
        model_contracts[model_name] = {
            "role": model["role"],
            "evaluate_on_final_test": model["evaluate_on_final_test"],
            "feature_family": model["feature_family"],
            "preprocessor_feature_family": expected_preprocessor_family,
            "model": model_fingerprint,
            "preprocessor_state": state_fingerprint,
            "preprocessor_manifest": manifest_fingerprint,
            "threshold_source": threshold_fingerprint,
            "frozen_raw_threshold": expected_threshold,
            "validation_average_precision": float(model["validation_average_precision"]),
            "resolved_paths": {
                "model": str(model_path),
                "preprocessor_state": str(state_path),
                "preprocessor_manifest": str(manifest_path),
                "threshold_source": str(threshold_path),
            },
        }

    graph_settings = mapping(settings.get("graphsage_inference"), "sprint5.graphsage_inference")
    graph_context, graph_context_fingerprint = _verify_direct_file(
        project_root,
        graph_settings,
        path_key="context_source_path",
        sha_key="context_source_sha256",
        name="sprint5.graphsage_inference.context_source",
    )
    inference_manifest_path, inference_manifest_fingerprint = _verify_direct_file(
        project_root,
        graph_settings,
        path_key="inference_manifest_path",
        sha_key="inference_manifest_sha256",
        name="sprint5.graphsage_inference.inference_manifest",
    )
    training_manifest_path, training_manifest_fingerprint = _verify_direct_file(
        project_root,
        graph_settings,
        path_key="training_graph_manifest_path",
        sha_key="training_graph_manifest_sha256",
        name="sprint5.graphsage_inference.training_graph_manifest",
    )
    inference_manifest = load_json_object(inference_manifest_path, "Sprint 4 graph manifest")
    if (
        graph_settings.get("message_context_partitions") != ["train"]
        or graph_settings.get("append_test_endpoint_identities") is not True
        or graph_settings.get("test_endpoint_identity_features_only") is not True
        or graph_settings.get("validation_edges_used_for_message_passing") is not False
        or graph_settings.get("test_edges_used_for_message_passing") is not False
        or graph_settings.get("test_labels_used_for_graph_construction") is not False
        or inference_manifest.get("message_context_partition") != "train"
        or int(inference_manifest.get("context_edge_count", -1))
        != int(graph_settings["context_max_edges"])
    ):
        raise FinalEvaluationContractError("GraphSAGE final-inference leakage contract changed")

    source_snapshot = _source_snapshot(project_root)
    return {
        "schema": "argus.final_evaluation.freeze_contract.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "created_before_final_test_access": True,
        "authorization": dict(authorization),
        "sprint4_checkpoint": checkpoint,
        "current_git_head": head,
        "sprint4_checkpoint_is_ancestor": True,
        "source_snapshot": source_snapshot,
        "frozen_references": fingerprints,
        "split_metadata": file_fingerprint(split_metadata_path, relative_to=project_root),
        "validation_partition": dict(frozen["validation_partition"]),
        "test_partition_metadata_only": dict(frozen["test_partition"]),
        "frozen_champion": "graph_enhanced_lightgbm",
        "selection_partition": "validation",
        "evaluation_partition": "test",
        "models": model_contracts,
        "graphsage_inference": {
            "message_context_partitions": ["train"],
            "context": graph_context_fingerprint,
            "inference_manifest": inference_manifest_fingerprint,
            "training_graph_manifest": training_manifest_fingerprint,
            "resolved_paths": {
                "context": str(graph_context),
                "inference_manifest": str(inference_manifest_path),
                "training_graph_manifest": str(training_manifest_path),
            },
            "append_test_endpoint_identities": True,
            "validation_edges_used_for_message_passing": False,
            "test_edges_used_for_message_passing": False,
            "test_labels_used_for_graph_construction": False,
        },
        "top_k": list(settings["top_k"]),
        "ranking_tie_break": settings["ranking_tie_break"],
        "prohibitions": dict(mapping(settings["prohibitions"], "sprint5.prohibitions")),
        "final_test_rows_read": False,
        "final_test_labels_read": False,
        "model_selection_locked_before_test": True,
        "thresholds_locked_before_test": True,
    }


def create_exclusive_access_receipt(
    receipt_path: Path,
    *,
    freeze_contract_path: Path,
    freeze_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically consume the one allowed final-test run before its first query."""

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "argus.final_test_access_receipt.v1",
        "status": "OPENED_FOR_ONE_SHOT_FINAL_EVALUATION",
        "opened_at_utc": datetime.now(UTC).isoformat(),
        "created_before_first_test_query": True,
        "explicit_user_authorization": True,
        "authorized_stage": "final_evaluation",
        "freeze_contract": file_fingerprint(freeze_contract_path),
        "frozen_champion": freeze_contract["frozen_champion"],
        "selection_partition": "validation",
        "evaluation_partition": "test",
        "retry_permitted": False,
        "model_selection_used_test": False,
        "threshold_tuned_on_test": False,
    }
    encoded = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(receipt_path, flags)
    except FileExistsError as exc:
        raise FinalEvaluationContractError(
            "Final-test access receipt already exists; the one-shot run cannot be retried"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Never remove the receipt after O_EXCL succeeds: even a failed write/run
        # consumes the authorization conservatively.
        raise
    return payload


__all__ = [
    "FinalEvaluationContractError",
    "build_freeze_contract",
    "create_exclusive_access_receipt",
    "load_json_object",
    "mapping",
    "project_path",
]
