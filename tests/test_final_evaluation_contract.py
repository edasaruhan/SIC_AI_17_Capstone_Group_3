from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from argus.config import load_config
from argus.final_evaluation import contract
from argus.modeling.artifacts import sha256_file


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _path(root: Path, value: str) -> Path:
    return root / Path(value)


def _threshold_payload(threshold: float) -> dict[str, Any]:
    return {
        "partition": "validation",
        "test_partition_used": False,
        "summaries": {
            "predeclared_joint_primary_constraint": {"operating_point": {"threshold": threshold}}
        },
    }


def _synthetic_freeze_config(tmp_path: Path) -> dict[str, Any]:
    config = deepcopy(load_config("configs/final_evaluation.yaml", resolve_paths=False))
    config["_meta"]["project_root"] = str(tmp_path)
    settings = config["sprint5"]
    frozen = settings["frozen_references"]

    sprint3_manifest = {
        "status": "PASS",
        "acceptance": {"artifacts_complete": True, "test_sealed": True},
    }
    sprint4_manifest = {
        "status": "PASS",
        "acceptance": {"artifacts_complete": True, "test_sealed": True},
        "data_access_audit": {
            "test_feature_rows_loaded": False,
            "test_labels_loaded": False,
            "test_predictions_generated": False,
            "test_metrics_computed": False,
        },
    }
    comparison = {
        "validation_leader": "graph_enhanced_lightgbm",
        "models": [
            {
                "model": name,
                "average_precision": model["validation_average_precision"],
                "is_validation_leader": name == "graph_enhanced_lightgbm",
            }
            for name, model in settings["models"].items()
        ],
    }
    split_metadata = {
        "strategy": "chronological",
        "strict_boundaries_verified": True,
        "no_transaction_overlap_verified": True,
        "partitions": {
            name: {
                "rows": declared["rows"],
                "positive_labels": declared["positives"],
                "minimum_timestamp": declared["minimum_timestamp"],
                "maximum_timestamp": declared["maximum_timestamp"],
            }
            for name, declared in (
                ("validation", frozen["validation_partition"]),
                ("test", frozen["test_partition"]),
            )
        },
    }

    reference_payloads = {
        "sprint3_manifest": sprint3_manifest,
        "sprint4_manifest": sprint4_manifest,
        "full_manifest": {"status": "PASS"},
        "split_manifest": b"synthetic split manifest",
        "feature_store": b"synthetic feature store",
        "sprint4_model_comparison": comparison,
    }
    for key, payload in reference_payloads.items():
        _write(_path(tmp_path, frozen[key]["path"]), payload)

    split_metadata_path = _path(tmp_path, config["paths"]["split_metadata"])
    _write(split_metadata_path, split_metadata)
    config["baseline"]["frozen_upstream"]["split_metadata_sha256"] = sha256_file(
        split_metadata_path
    )

    shared_thresholds = {
        "models": {
            name: _threshold_payload(float(model["threshold"]))
            for name, model in settings["models"].items()
        }
    }
    written_thresholds: set[Path] = set()
    for name, model in settings["models"].items():
        model_path = _path(tmp_path, model["model_path"])
        state_path = _path(tmp_path, model["preprocessor_state_path"])
        manifest_path = _path(tmp_path, model["preprocessor_manifest_path"])
        threshold_path = _path(tmp_path, model["threshold_source_path"])
        _write(model_path, f"synthetic model: {name}".encode())
        _write(state_path, {"fit_scope": "train_only"})
        expected_family = (
            "transaction_temporal_history"
            if name == "graphsage_edge_classifier"
            else model["feature_family"]
        )
        _write(
            manifest_path,
            {
                "fit_scope": "train_only",
                "fitted_train_rows": 3_554_957,
                "feature_family": expected_family,
            },
        )
        if threshold_path not in written_thresholds:
            payload = (
                shared_thresholds
                if name == "graphsage_edge_classifier"
                else _threshold_payload(float(model["threshold"]))
            )
            _write(threshold_path, payload)
            written_thresholds.add(threshold_path)
        for path_key, sha_key, path in (
            ("model_path", "model_sha256", model_path),
            ("preprocessor_state_path", "preprocessor_state_sha256", state_path),
            ("preprocessor_manifest_path", "preprocessor_manifest_sha256", manifest_path),
            ("threshold_source_path", "threshold_source_sha256", threshold_path),
        ):
            assert model[path_key]
            model[sha_key] = sha256_file(path)

    graph = settings["graphsage_inference"]
    graph_context = _path(tmp_path, graph["context_source_path"])
    inference_manifest = _path(tmp_path, graph["inference_manifest_path"])
    training_manifest = _path(tmp_path, graph["training_graph_manifest_path"])
    _write(graph_context, b"synthetic train-only graph context")
    _write(
        inference_manifest,
        {
            "message_context_partition": "train",
            "context_edge_count": graph["context_max_edges"],
        },
    )
    _write(training_manifest, {"partition": "train"})
    for sha_key, path in (
        ("context_source_sha256", graph_context),
        ("inference_manifest_sha256", inference_manifest),
        ("training_graph_manifest_sha256", training_manifest),
    ):
        graph[sha_key] = sha256_file(path)

    for key in (
        "sprint3_manifest",
        "sprint4_manifest",
        "full_manifest",
        "split_manifest",
        "feature_store",
        "sprint4_thresholds",
        "sprint4_model_comparison",
    ):
        frozen[key]["sha256"] = sha256_file(_path(tmp_path, frozen[key]["path"]))
    return config


def _patch_source_and_git(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> None:
    checkpoint = config["sprint5"]["frozen_references"]["sprint4_checkpoint"]
    monkeypatch.setattr(contract, "subprocess_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(contract, "_source_snapshot", lambda _: {"files": []})


def test_freeze_contract_reconciles_all_frozen_inputs_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_freeze_config(tmp_path)
    _patch_source_and_git(monkeypatch, config)

    result = contract.build_freeze_contract(config)

    assert result["schema"] == "argus.final_evaluation.freeze_contract.v1"
    assert result["created_before_final_test_access"] is True
    assert result["frozen_champion"] == "graph_enhanced_lightgbm"
    assert result["selection_partition"] == "validation"
    assert result["evaluation_partition"] == "test"
    assert result["final_test_rows_read"] is False
    assert result["final_test_labels_read"] is False
    assert set(result["models"]) == {
        "graph_enhanced_lightgbm",
        "refined_transaction_lightgbm",
        "graphsage_edge_classifier",
    }
    assert result["models"]["graphsage_edge_classifier"]["preprocessor_feature_family"] == (
        "transaction_temporal_history"
    )
    assert result["graphsage_inference"]["message_context_partitions"] == ["train"]


def test_freeze_contract_rejects_threshold_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_freeze_config(tmp_path)
    _patch_source_and_git(monkeypatch, config)
    config["sprint5"]["models"]["graph_enhanced_lightgbm"]["threshold"] += 1.0

    with pytest.raises(contract.FinalEvaluationContractError, match="threshold changed"):
        contract.build_freeze_contract(config)


def test_freeze_contract_rejects_graph_message_passing_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_freeze_config(tmp_path)
    _patch_source_and_git(monkeypatch, config)
    config["sprint5"]["graphsage_inference"]["validation_edges_used_for_message_passing"] = True

    with pytest.raises(contract.FinalEvaluationContractError, match="leakage contract"):
        contract.build_freeze_contract(config)


def test_freeze_contract_rejects_unaccepted_sprint4_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_freeze_config(tmp_path)
    _patch_source_and_git(monkeypatch, config)
    reference = config["sprint5"]["frozen_references"]["sprint4_manifest"]
    path = _path(tmp_path, reference["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["acceptance"]["artifacts_complete"] = False
    _write(path, payload)
    reference["sha256"] = sha256_file(path)

    with pytest.raises(contract.FinalEvaluationContractError, match="not accepted"):
        contract.build_freeze_contract(config)


def test_freeze_contract_requires_authorization_and_accepted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_freeze_config(tmp_path)
    config["sprint5"]["authorization"]["explicit_user_authorization"] = False
    with pytest.raises(contract.FinalEvaluationContractError, match="authorization is absent"):
        contract.build_freeze_contract(config)

    config["sprint5"]["authorization"]["explicit_user_authorization"] = True
    monkeypatch.setattr(contract, "subprocess_checkpoint", lambda _: "new-head")
    monkeypatch.setattr(contract, "git_is_ancestor", lambda *_: False)
    with pytest.raises(contract.FinalEvaluationContractError, match="not in Git history"):
        contract.build_freeze_contract(config)


def test_contract_helpers_reject_invalid_mapping_json_path_hash_and_threshold(
    tmp_path: Path,
) -> None:
    with pytest.raises(contract.FinalEvaluationContractError, match="must be a mapping"):
        contract.mapping([], "payload")
    with pytest.raises(contract.FinalEvaluationContractError, match="non-empty path"):
        contract.project_path(tmp_path, "", "artifact")
    with pytest.raises(FileNotFoundError, match="was not found"):
        contract.load_json_object(tmp_path / "missing.json", "Missing artifact")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    with pytest.raises(contract.FinalEvaluationContractError, match="Could not read"):
        contract.load_json_object(invalid, "Invalid artifact")

    array = tmp_path / "array.json"
    _write(array, [1, 2, 3])
    with pytest.raises(contract.FinalEvaluationContractError, match="JSON object"):
        contract.load_json_object(array, "Array artifact")

    frozen = tmp_path / "frozen.bin"
    frozen.write_bytes(b"frozen")
    with pytest.raises(contract.FinalEvaluationContractError, match="hash changed"):
        contract._verify_file_reference(
            tmp_path,
            {"path": "frozen.bin", "sha256": "0" * 64},
            "frozen",
        )

    unsafe = _threshold_payload(float("inf"))
    with pytest.raises(contract.FinalEvaluationContractError, match="non-finite"):
        contract._threshold_from_artifact(unsafe, "model")
    unsafe = _threshold_payload(0.5)
    unsafe["test_partition_used"] = True
    with pytest.raises(contract.FinalEvaluationContractError, match="validation-only"):
        contract._threshold_from_artifact(unsafe, "model")
