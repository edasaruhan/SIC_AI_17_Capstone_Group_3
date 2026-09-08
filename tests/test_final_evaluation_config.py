from __future__ import annotations

from copy import deepcopy

import pytest

from argus.config import ConfigError, load_config, validate_config


@pytest.fixture
def final_config() -> dict:
    return load_config("configs/final_evaluation.yaml", resolve_paths=False)


def test_final_config_loads_with_pre_test_frozen_champion(final_config: dict) -> None:
    sprint5 = final_config["sprint5"]
    assert sprint5["final_test_access"] is True
    assert sprint5["selection_partition"] == "validation"
    assert sprint5["evaluation_partition"] == "test"
    assert sprint5["authorization"]["authorized_stage"] == "final_evaluation"
    assert sprint5["one_shot_gate"]["freeze_contract_path"] == "freeze_contract.json"
    assert sprint5["one_shot_gate"]["receipt_path"] == "FINAL_TEST_OPENED.json"
    assert sprint5["outputs"]["failure_marker_json"] == "FINAL_EVALUATION_FAILED.json"
    assert sprint5["models"]["graph_enhanced_lightgbm"]["role"] == "frozen_champion"
    assert sprint5["models"]["graph_enhanced_lightgbm"]["threshold"] == pytest.approx(
        -4.3019702136515985
    )
    assert final_config["sprint4"]["final_test_access"] is False


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("protocol_version",), 2, "protocol_version"),
        (("authorization", "explicit_user_authorization"), False, "explicit_user_authorization"),
        (("final_test_access",), False, "final_test_access"),
        (("selection_partition",), "test", "selection_partition"),
        (("evaluation_partition",), "validation", "evaluation_partition"),
        (("ranking_tie_break",), "arbitrary", "ranking_tie_break"),
        (("top_k",), [100, 1000], "top_k"),
        (("one_shot_gate", "refuse_if_receipt_exists"), False, "refuse_if_receipt_exists"),
        (("one_shot_gate", "create_receipt_before_test_read"), False, "create_receipt"),
        (("prohibitions", "test_informed_model_selection"), True, "model_selection"),
        (
            ("prohibitions", "model_retraining_during_or_after_final_evaluation"),
            True,
            "model_retraining",
        ),
    ],
)
def test_final_config_rejects_protocol_mutations(
    final_config: dict, path: tuple[str, ...], value: object, match: str
) -> None:
    changed = deepcopy(final_config)
    target = changed["sprint5"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=match):
        validate_config(changed)


@pytest.mark.parametrize(
    ("model_name", "key", "value"),
    [
        ("graph_enhanced_lightgbm", "role", "comparator"),
        ("graph_enhanced_lightgbm", "threshold", -4.0),
        ("graph_enhanced_lightgbm", "model_sha256", "0" * 64),
        ("graph_enhanced_lightgbm", "feature_family", "transaction_only"),
        ("refined_transaction_lightgbm", "threshold", -3.0),
        ("graphsage_edge_classifier", "threshold", 0.5),
        ("graphsage_edge_classifier", "evaluate_on_final_test", False),
    ],
)
def test_final_config_rejects_frozen_model_mutations(
    final_config: dict, model_name: str, key: str, value: object
) -> None:
    changed = deepcopy(final_config)
    changed["sprint5"]["models"][model_name][key] = value
    with pytest.raises(ConfigError):
        validate_config(changed)


@pytest.mark.parametrize(
    ("partition", "key", "value"),
    [
        ("validation_partition", "rows", 761750),
        ("validation_partition", "positives", 761),
        ("test_partition", "rows", 761638),
        ("test_partition", "positives", 1560),
        ("test_partition", "positive_rate", 0.5),
    ],
)
def test_final_config_rejects_partition_mutations(
    final_config: dict, partition: str, key: str, value: object
) -> None:
    changed = deepcopy(final_config)
    changed["sprint5"]["frozen_references"][partition][key] = value
    with pytest.raises(ConfigError):
        validate_config(changed)


def test_final_config_rejects_upstream_hash_or_path_mutation(final_config: dict) -> None:
    for key, field, value in (
        ("sprint4_manifest", "sha256", "0" * 64),
        ("split_manifest", "path", "artifacts/other/split.parquet"),
        ("feature_store", "sha256", "not-a-hash"),
    ):
        changed = deepcopy(final_config)
        changed["sprint5"]["frozen_references"][key][field] = value
        with pytest.raises(ConfigError):
            validate_config(changed)


def test_final_graphsage_contract_uses_only_frozen_train_context(final_config: dict) -> None:
    inference = final_config["sprint5"]["graphsage_inference"]
    assert inference["message_context_partitions"] == ["train"]
    assert inference["append_test_endpoint_identities"] is True
    assert inference["test_endpoint_identity_features_only"] is True
    assert inference["validation_edges_used_for_message_passing"] is False
    assert inference["test_edges_used_for_message_passing"] is False
    assert inference["test_labels_used_for_graph_construction"] is False

    changed = deepcopy(final_config)
    changed["sprint5"]["graphsage_inference"]["message_context_partitions"] = [
        "train",
        "validation",
    ]
    with pytest.raises(ConfigError, match="message_context_partitions"):
        validate_config(changed)


def test_final_config_rejects_unknown_keys(final_config: dict) -> None:
    changed = deepcopy(final_config)
    changed["sprint5"]["allow_retry"] = True
    with pytest.raises(ConfigError, match="unexpected=\\['allow_retry'\\]"):
        validate_config(changed)
