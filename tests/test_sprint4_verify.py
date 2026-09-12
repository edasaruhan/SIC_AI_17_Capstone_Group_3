from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from argus.modeling.metrics import compute_average_precision
from argus.product.cases import build_case_from_frames
from argus.sprint4.verify import (
    Sprint4VerificationError,
    _assert_close,
    _load_object,
    _verify_predictions,
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


def _write_prediction_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.unlink(missing_ok=True)
    connection = duckdb.connect()
    try:
        connection.register("prediction_source", frame)
        connection.execute("COPY prediction_source TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["TX-1", "TX-2", "TX-3", "TX-4"],
            "source_row_number": [1, 2, 3, 4],
            "evaluation_partition": ["validation"] * 4,
            "is_laundering": [0, 1, 0, 1],
            "raw_score_graphsage_edge_classifier": [-1.0, 2.0, -0.5, 1.0],
            "raw_score_refined_transaction_lightgbm": [-2.0, 1.5, -1.0, 2.5],
            "raw_score_graph_enhanced_lightgbm": [-1.5, 3.0, -0.25, 2.0],
            "probability_graphsage_edge_classifier": [0.2, 0.8, 0.3, 0.7],
            "probability_refined_transaction_lightgbm": [0.1, 0.9, 0.2, 0.85],
            "probability_graph_enhanced_lightgbm": [0.15, 0.95, 0.4, 0.9],
        }
    )


def _comparison_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    labels = frame["is_laundering"].to_numpy(np.int8)
    return [
        {
            "model": name,
            "average_precision": compute_average_precision(labels, frame[column].to_numpy()),
            "evaluation_partition": "validation",
            "test_metrics_used": False,
        }
        for name, column in (
            ("graphsage_edge_classifier", "raw_score_graphsage_edge_classifier"),
            (
                "refined_transaction_lightgbm",
                "raw_score_refined_transaction_lightgbm",
            ),
            ("graph_enhanced_lightgbm", "raw_score_graph_enhanced_lightgbm"),
        )
    ]


def test_prediction_verification_recomputes_all_three_validation_models(tmp_path: Path) -> None:
    frame = _prediction_frame()
    path = tmp_path / "validation_predictions.parquet"
    _write_prediction_parquet(frame, path)
    connection = duckdb.connect()
    try:
        result = _verify_predictions(
            connection,
            path,
            expected_rows=4,
            expected_positives=2,
            manifest_rows=_comparison_rows(frame),
        )
    finally:
        connection.close()

    assert result["status"] == "PASS"
    assert result["rows"] == 4
    assert result["partition"] == "validation"
    assert set(result["models_recomputed"]) == {
        "graphsage_edge_classifier",
        "refined_transaction_lightgbm",
        "graph_enhanced_lightgbm",
    }


def test_prediction_verification_rejects_schema_partition_and_metric_drift(
    tmp_path: Path,
) -> None:
    frame = _prediction_frame()
    path = tmp_path / "validation_predictions.parquet"

    _write_prediction_parquet(frame.drop(columns="probability_graphsage_edge_classifier"), path)
    connection = duckdb.connect()
    try:
        with pytest.raises(Sprint4VerificationError, match="columns missing"):
            _verify_predictions(
                connection,
                path,
                expected_rows=4,
                expected_positives=2,
                manifest_rows=_comparison_rows(frame),
            )
    finally:
        connection.close()

    changed_partition = frame.copy()
    changed_partition.loc[0, "evaluation_partition"] = "test"
    _write_prediction_parquet(changed_partition, path)
    connection = duckdb.connect()
    try:
        with pytest.raises(Sprint4VerificationError, match="validation-only"):
            _verify_predictions(
                connection,
                path,
                expected_rows=4,
                expected_positives=2,
                manifest_rows=_comparison_rows(frame),
            )
    finally:
        connection.close()

    _write_prediction_parquet(frame, path)
    rows = _comparison_rows(frame)
    rows[0]["average_precision"] = 0.0
    connection = duckdb.connect()
    try:
        with pytest.raises(Sprint4VerificationError, match="average precision mismatch"):
            _verify_predictions(
                connection,
                path,
                expected_rows=4,
                expected_positives=2,
                manifest_rows=rows,
            )
    finally:
        connection.close()


def test_sprint4_verification_helpers_reject_invalid_json_and_numbers(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    with pytest.raises(Sprint4VerificationError, match="Could not read JSON"):
        _load_object(invalid)

    array = tmp_path / "array.json"
    array.write_text("[]\n", encoding="utf-8")
    with pytest.raises(Sprint4VerificationError, match="must contain an object"):
        _load_object(array)

    with pytest.raises(Sprint4VerificationError, match="not numeric"):
        _assert_close("not-a-number", 1.0, label="saved metric")
