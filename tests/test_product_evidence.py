from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from argus.product.evidence import EvidenceEngineError, build_evidence_bundle
from argus.product.schemas import ProductSchemaError, validate_evidence_payload


def _transaction(**updates: object) -> pd.Series:
    values: dict[str, object] = {
        "transaction_id": "HI-Small_Trans.csv:row-42",
        "timestamp": pd.Timestamp("2022-09-08 12:00:00"),
        "from_node_id": "1::SENDER",
        "to_node_id": "2::RECEIVER",
        "amount_paid": np.float64(1250.5),
        "amount_received": np.float64(1250.5),
        "payment_currency": "US Dollar",
        "receiving_currency": "US Dollar",
        "payment_format": "Wire",
        "partition": "validation",
        "is_laundering": 1,
        "pair_previous_transfer_count": np.int64(2),
        "sender_prior_fan_out_degree": np.int64(3),
        "receiver_prior_fan_in_degree": np.int64(4),
        "log_amount_paid": np.float64(7.1313),
    }
    values.update(updates)
    return pd.Series(values)


def _neighborhood() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["p1", "p2", "p3", "same-time", "future"],
            "timestamp": pd.to_datetime(
                [
                    "2022-09-08 10:00",
                    "2022-09-08 11:00",
                    "2022-09-08 11:30",
                    "2022-09-08 12:00",
                    "2022-09-08 13:00",
                ]
            ),
            "from_node_id": [
                "1::SENDER",
                "1::SENDER",
                "3::OTHER",
                "1::SENDER",
                "1::SENDER",
            ],
            "to_node_id": [
                "2::RECEIVER",
                "4::OTHER",
                "1::SENDER",
                "2::RECEIVER",
                "2::RECEIVER",
            ],
            "partition": ["train", "validation", "validation", "validation", "validation"],
        }
    )


def test_evidence_bundle_separates_observations_and_model_evidence() -> None:
    bundle = build_evidence_bundle(
        _transaction(),
        _neighborhood(),
        model_score=np.float64(3.25),
        model_name="graphsage_edge_classifier",
        threshold=2.5,
        rank=7,
        feature_contributions={
            "sender_prior_fan_out_degree": 0.25,
            "log_amount_paid": -1.5,
        },
    )
    payload = bundle.to_dict()

    assert len(payload["observed_evidence"]) == 7
    assert all(item["basis"] == "observed" for item in payload["observed_evidence"])
    assert payload["model_evidence"]["basis"] == "model"
    assert payload["model_evidence"]["unit_of_analysis"] == "transaction"
    assert payload["model_evidence"]["threshold_exceeded"] is True
    assert [item["feature"] for item in payload["model_evidence"]["feature_contributions"]] == [
        "log_amount_paid",
        "sender_prior_fan_out_degree",
    ]
    assert payload["model_evidence"]["feature_contributions"][0][
        "observed_feature_value"
    ] == pytest.approx(7.1313)
    assert "is_laundering" not in json.dumps(payload, allow_nan=False)
    validate_evidence_payload(payload)


def test_neighborhood_evidence_uses_strictly_prior_rows_only() -> None:
    bundle = build_evidence_bundle(
        _transaction(),
        _neighborhood(),
        model_score=0.75,
        model_name="saved-model",
    )
    facts = {fact.evidence_id: fact for fact in bundle.observed_evidence}

    assert facts["obs-prior-pair-activity"].values == {
        "prior_directed_transfer_count": 1,
        "future_or_same_timestamp_rows_used": 0,
    }
    assert facts["obs-prior-sender-neighborhood"].values == {
        "prior_outgoing_transactions": 2,
        "prior_incoming_transactions": 1,
        "prior_unique_outgoing_counterparties": 2,
        "prior_unique_incoming_counterparties": 1,
    }


def test_evidence_is_strict_json_safe_and_accepts_one_row_frame() -> None:
    bundle = build_evidence_bundle(
        _transaction().to_frame().T,
        _neighborhood(),
        model_score=np.float32(0.5),
        model_name="model",
    )
    encoded = json.dumps(bundle.to_dict(), allow_nan=False, sort_keys=True)
    assert "HI-Small_Trans.csv:row-42" in encoded


@pytest.mark.parametrize(
    ("transaction", "neighborhood", "message"),
    [
        (_transaction(payment_format=""), _neighborhood(), "payment_format"),
        (_transaction(amount_paid=np.inf), _neighborhood(), "amount_paid"),
        (_transaction(partition="test"), _neighborhood(), "outside the allowed"),
        (
            _transaction(),
            _neighborhood().assign(
                partition=["train", "validation", "test", "validation", "validation"]
            ),
            "forbidden closed partition",
        ),
        (_transaction(), _neighborhood().drop(columns="timestamp"), "missing required columns"),
    ],
)
def test_evidence_rejects_invalid_or_closed_test_inputs(
    transaction: pd.Series,
    neighborhood: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(EvidenceEngineError, match=message):
        build_evidence_bundle(
            transaction,
            neighborhood,
            model_score=0.5,
            model_name="model",
        )


def test_saved_evidence_validator_rejects_insufficient_or_mixed_basis() -> None:
    payload = build_evidence_bundle(
        _transaction(), _neighborhood(), model_score=0.5, model_name="model"
    ).to_dict()
    too_short = deepcopy(payload)
    too_short["observed_evidence"] = too_short["observed_evidence"][:2]
    with pytest.raises(ProductSchemaError, match="at least 3"):
        validate_evidence_payload(too_short)

    mixed = deepcopy(payload)
    mixed["observed_evidence"][0]["basis"] = "model"
    with pytest.raises(ProductSchemaError, match="basis='observed'"):
        validate_evidence_payload(mixed)

    hidden_target = deepcopy(payload)
    hidden_target["observed_evidence"][0]["values"]["account_fraud_label"] = 1
    with pytest.raises(ProductSchemaError, match="target/split key"):
        validate_evidence_payload(hidden_target)

    contradictory_threshold = deepcopy(payload)
    contradictory_threshold["model_evidence"]["threshold"] = 0.1
    contradictory_threshold["model_evidence"]["threshold_exceeded"] = False
    with pytest.raises(ProductSchemaError, match="contradicts"):
        validate_evidence_payload(contradictory_threshold)


def test_target_or_nonfinite_model_attribution_is_rejected() -> None:
    with pytest.raises(ProductSchemaError, match="forbidden target"):
        build_evidence_bundle(
            _transaction(),
            _neighborhood(),
            model_score=0.5,
            model_name="model",
            feature_contributions={"is_laundering": 2.0},
        )
    with pytest.raises(ProductSchemaError, match="finite"):
        build_evidence_bundle(
            _transaction(),
            _neighborhood(),
            model_score=0.5,
            model_name="model",
            feature_contributions={"log_amount_paid": np.nan},
        )
