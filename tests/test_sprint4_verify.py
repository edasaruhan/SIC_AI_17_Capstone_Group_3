from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from argus.product.cases import build_case_from_frames
from argus.sprint4.verify import (
    Sprint4VerificationError,
    validate_case_collection,
    validate_final_test_policy,
    validate_manifest_inventory,
)


def _policy() -> dict[str, object]:
    return {
        "partition": "test",
        "status": "SEALED",
        "metadata_counts_inherited_from_sprint1": True,
        "feature_rows_loaded": False,
        "feature_transform_called": False,
        "graph_nodes_or_edges_constructed": False,
        "model_predictions_generated": False,
        "metrics_computed": False,
        "model_selection_or_tuning_used_test": False,
    }


def _case_collection() -> dict[str, object]:
    transaction = {
        "transaction_id": "TX-VALIDATION-1",
        "timestamp": "2022-09-08T10:00:00",
        "from_node_id": "001::A",
        "to_node_id": "002::B",
        "amount_paid": 100.0,
        "amount_received": 99.0,
        "payment_currency": "US Dollar",
        "receiving_currency": "US Dollar",
        "payment_format": "Wire",
        "partition": "validation",
    }
    neighborhood = pd.DataFrame(
        {
            "transaction_id": ["OLD-1"],
            "timestamp": ["2022-09-07T10:00:00"],
            "from_node_id": ["001::A"],
            "to_node_id": ["003::C"],
            "amount_paid": [20.0],
            "payment_currency": ["US Dollar"],
            "payment_format": ["Wire"],
            "partition": ["train"],
        }
    )
    case = build_case_from_frames(
        transaction,
        neighborhood,
        model_score=0.8,
        model_name="graphsage_edge_classifier",
        threshold=0.5,
        rank=1,
        feature_contributions={"transaction::amount": 0.2},
    )
    return {
        "partition": "validation",
        "test_rows_included": False,
        "target_fields_included": False,
        "cases": [case.to_dict()],
    }


def test_final_test_policy_requires_every_access_path_false() -> None:
    result = validate_final_test_policy(_policy())
    assert result["status"] == "PASS"
    assert result["final_test_opened"] is False


def test_final_test_policy_rejects_inference() -> None:
    changed = _policy()
    changed["model_predictions_generated"] = True
    with pytest.raises(Sprint4VerificationError, match="model_predictions_generated"):
        validate_final_test_policy(changed)


def test_case_collection_validates_real_schema_and_evidence_minimum() -> None:
    result = validate_case_collection(_case_collection(), expected_count=1)
    assert result["case_count"] == 1
    assert result["minimum_observed_evidence"] >= 3
    assert result["deterministic_fallback_notes"] == 1


def test_case_collection_rejects_target_field() -> None:
    changed = deepcopy(_case_collection())
    changed["cases"][0]["is_laundering"] = 1
    with pytest.raises(Sprint4VerificationError, match="targets"):
        validate_case_collection(changed, expected_count=1)


def test_manifest_inventory_requires_exact_hash_size_and_path() -> None:
    inventory = [{"path": "a.json", "size_bytes": 2, "sha256": "abc"}]
    assert validate_manifest_inventory(inventory, deepcopy(inventory))["status"] == "PASS"
    changed = [{"path": "a.json", "size_bytes": 3, "sha256": "abc"}]
    with pytest.raises(Sprint4VerificationError, match="changed"):
        validate_manifest_inventory(inventory, changed)
