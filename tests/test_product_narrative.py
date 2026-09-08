from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from argus.product.evidence import build_evidence_bundle
from argus.product.narrative import generate_analyst_note


def _bundle():
    transaction = pd.Series(
        {
            "transaction_id": "tx-1",
            "timestamp": "2022-09-08 12:00",
            "from_node_id": "1::A",
            "to_node_id": "2::B",
            "amount_paid": 10.0,
            "amount_received": 10.0,
            "payment_currency": "Euro",
            "receiving_currency": "Euro",
            "payment_format": "ACH",
            "partition": "validation",
        }
    )
    neighborhood = pd.DataFrame(columns=["timestamp", "from_node_id", "to_node_id", "partition"])
    return build_evidence_bundle(
        transaction,
        neighborhood,
        model_score=0.8,
        model_name="saved-model",
        threshold=0.7,
    )


class SafeGenerator:
    def __init__(self) -> None:
        self.received: Mapping[str, Any] | None = None

    def summarize_evidence(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        self.received = evidence
        return {
            "summary": (
                "The saved facts support prioritization but do not establish wrongdoing. "
                "The context requires verification through trained analyst review."
            ),
            "cited_observed_evidence_ids": [
                "obs-transaction-value",
                "obs-transfer-route",
                "obs-time-channel",
            ],
        }


def test_optional_llm_receives_only_structured_evidence_and_citations() -> None:
    client = SafeGenerator()
    note = generate_analyst_note(_bundle(), llm=client)

    assert note.mode == "optional_llm"
    assert note.fallback_reason is None
    assert client.received is not None
    assert set(client.received) == {
        "schema_version",
        "unit_of_analysis",
        "transaction_id",
        "focal_timestamp",
        "observed_evidence",
        "model_evidence",
        "limitations",
    }
    assert "obs-time-channel" in note.text
    assert "no automatic account action" in note.text


class RaisingGenerator:
    def summarize_evidence(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError("secret-token-must-not-leak")


class UnsafeGenerator:
    def summarize_evidence(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "summary": (
                "This actor is guilty. Freeze the account despite human review and uncertainty."
            ),
            "cited_observed_evidence_ids": [
                "obs-transaction-value",
                "obs-transfer-route",
                "obs-time-channel",
            ],
        }


class UngroundedGenerator:
    def summarize_evidence(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "summary": ("This does not establish wrongdoing and requires trained analyst review."),
            "cited_observed_evidence_ids": ["obs-transaction-value", "invented", "also-invented"],
        }


class MalformedCitationGenerator:
    def summarize_evidence(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "summary": ("This does not establish wrongdoing and requires trained analyst review."),
            "cited_observed_evidence_ids": [{"not": "hashable"}, "second", "third"],
        }


def test_llm_error_safely_falls_back_without_exposing_exception_text() -> None:
    note = generate_analyst_note(_bundle(), llm=RaisingGenerator())
    assert note.mode == "deterministic_fallback"
    assert note.fallback_reason == "llm_error_RuntimeError"
    assert "secret-token" not in note.text
    assert "secret-token" not in (note.fallback_reason or "")


def test_unsafe_or_ungrounded_llm_output_is_discarded() -> None:
    for client in (UnsafeGenerator(), UngroundedGenerator(), MalformedCitationGenerator()):
        note = generate_analyst_note(_bundle(), llm=client)
        assert note.mode == "deterministic_fallback"
        assert note.fallback_reason == "unsafe_or_invalid_llm_output"
        assert "guilty" not in note.text.lower()
        assert "freeze the account" not in note.text.lower()


def test_no_llm_path_is_complete_and_explicitly_uncertain() -> None:
    note = generate_analyst_note(_bundle())
    assert note.mode == "deterministic_fallback"
    assert note.fallback_reason == "llm_not_configured"
    assert "do not establish wrongdoing" in note.text
    assert "requires verification" in note.text
    assert "trained analyst review" in note.text
