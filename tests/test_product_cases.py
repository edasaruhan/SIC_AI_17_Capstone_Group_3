from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
import pytest

from argus.product.cases import (
    CaseBuilderError,
    build_case_from_frames,
    build_investigation_cases,
)
from argus.product.schemas import ProductSchemaError, validate_case_payload


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["tx-b", "tx-a", "tx-c"],
            "timestamp": pd.to_datetime(
                ["2022-09-08 12:00", "2022-09-08 12:01", "2022-09-08 12:02"]
            ),
            "from_node_id": ["1::A", "2::B", "3::C"],
            "to_node_id": ["2::B", "3::C", "1::A"],
            "amount_paid": [100.0, 200.0, 300.0],
            "amount_received": [100.0, 200.0, 300.0],
            "payment_currency": ["US Dollar"] * 3,
            "receiving_currency": ["US Dollar"] * 3,
            "payment_format": ["Wire", "ACH", "Cheque"],
            "partition": ["validation"] * 3,
            # The builder must not read or serialize this target.
            "is_laundering": [1, 0, 1],
        }
    )


def _neighborhood() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["old-1", "old-2"],
            "timestamp": pd.to_datetime(["2022-09-08 10:00", "2022-09-08 11:00"]),
            "from_node_id": ["1::A", "3::C"],
            "to_node_id": ["2::B", "2::B"],
            "partition": ["train", "validation"],
        }
    )


def test_case_is_deterministic_guarded_and_contains_real_evidence() -> None:
    first = build_case_from_frames(
        _transactions().iloc[[0]],
        _neighborhood(),
        model_score=0.91,
        model_name="graphsage_edge_classifier",
        threshold=0.8,
        rank=1,
    )
    second = build_case_from_frames(
        _transactions().iloc[0],
        _neighborhood(),
        model_score=0.91,
        model_name="graphsage_edge_classifier",
        threshold=0.8,
        rank=1,
    )
    payload = first.to_dict()

    assert first.to_dict() == second.to_dict()
    assert first.case_id.startswith("ARGUS-")
    assert payload["status"] == "pending_human_review"
    assert payload["priority"] == {
        "basis": "model",
        "band": "elevated_review_priority",
        "threshold_exceeded": True,
    }
    assert len(payload["observed_evidence"]) >= 3
    assert payload["analyst_note"]["mode"] == "deterministic_fallback"
    assert "trained analyst review" in payload["analyst_note"]["text"]
    assert payload["guardrails"] == {
        "human_review_required": True,
        "automatic_action_authorized": False,
        "account_level_fraud_label_inferred": False,
    }
    assert payload["network"]["directed"] is True
    assert payload["network"]["multigraph"] is True
    assert payload["network"]["input_prior_edge_count"] == 2
    assert payload["network"]["saved_prior_edge_count"] == 2
    assert len(payload["network"]["edges"]) == 3
    assert payload["network"]["edges"][-1]["is_focal"] is True
    assert len(payload["transactions"]) == 3
    assert "is_laundering" not in json.dumps(payload)
    validate_case_payload(payload)

    unsafe = deepcopy(payload)
    unsafe["analyst_note"]["text"] = "Freeze this account immediately."
    with pytest.raises(ProductSchemaError, match="accusation or account-action"):
        validate_case_payload(unsafe)


def test_batch_case_builder_ranks_ties_deterministically_and_applies_threshold() -> None:
    scores = pd.DataFrame(
        {
            "transaction_id": ["tx-b", "tx-c", "tx-a"],
            "raw_score": [0.9, 0.7, 0.9],
            "partition": ["validation"] * 3,
        }
    )
    cases = build_investigation_cases(
        _transactions(),
        _neighborhood(),
        scores,
        model_name="saved-graphsage",
        score_column="raw_score",
        threshold=0.8,
        top_n=10,
        contributions_by_transaction={
            "tx-a": {"amount_paid": 1.0},
            "tx-b": {"amount_paid": 0.5},
        },
    )

    assert [case.transaction_id for case in cases] == ["tx-a", "tx-b"]
    assert [case.evidence.model_evidence.rank for case in cases] == [1, 2]
    assert all(case.evidence.model_evidence.threshold_exceeded for case in cases)


def test_batch_case_builder_rejects_closed_test_or_unmatched_score_rows() -> None:
    test_scores = pd.DataFrame({"transaction_id": ["tx-a"], "score": [1.0], "partition": ["test"]})
    with pytest.raises(CaseBuilderError, match="closed final-test"):
        build_investigation_cases(
            _transactions(),
            _neighborhood(),
            test_scores,
            model_name="model",
            score_column="score",
        )

    unmatched = pd.DataFrame({"transaction_id": ["absent"], "score": [1.0]})
    with pytest.raises(CaseBuilderError, match="lack matching transactions"):
        build_investigation_cases(
            _transactions(),
            _neighborhood(),
            unmatched,
            model_name="model",
            score_column="score",
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_batch_case_builder_rejects_nonfinite_scores(score: float) -> None:
    scores = pd.DataFrame({"transaction_id": ["tx-a"], "score": [score]})
    with pytest.raises(CaseBuilderError, match="non-finite"):
        build_investigation_cases(
            _transactions(),
            _neighborhood(),
            scores,
            model_name="model",
            score_column="score",
        )
