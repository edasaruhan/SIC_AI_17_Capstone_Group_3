"""Strict, JSON-safe schemas for saved ARGUS investigation cases."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"
MINIMUM_OBSERVED_EVIDENCE = 3
_FORBIDDEN_EVIDENCE_FIELDS = {
    "account_fraud_label",
    "account_level_label",
    "fraud_label",
    "is_laundering",
    "label",
    "target",
    "partition",
}
_ALLOWED_SCOPES = {
    "focal_transaction",
    "strictly_prior_supplied_neighborhood",
    "focal_transaction_strictly_prior_feature",
}
_UNSAFE_NOTE_PATTERNS = (
    r"\bguilty\b",
    r"\bcriminal\b",
    r"\bfraudster\b",
    r"\bmoney[ -]launderer\b",
    r"\bfreez\w*\b.{0,30}\baccount\b",
    r"\b(?:block|close|seize)\w*\b.{0,30}\baccount\b",
    r"\baccount\b.{0,30}\b(?:freez|block|close|seize)\w*\b",
    r"\bsuçlu\b",
    r"\bkara para akl(?:ıyor|amıştır)\b",
    r"\bhesab\w*\b.{0,30}\b(?:dondur|bloke|kapat)\w*\b",
)


class ProductSchemaError(ValueError):
    """Raised when a case could be ambiguous, unsafe, or non-serializable."""


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductSchemaError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ProductSchemaError(f"{field_name} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProductSchemaError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ProductSchemaError(f"{field_name} must be a finite number")
    return number


def to_json_safe(value: object, *, path: str = "$") -> Any:
    """Convert supported tabular scalars and reject lossy/stringified values.

    Missing scalar values become JSON ``null``. NaN and infinity are rejected
    because generated files use strict JSON rather than JavaScript's
    non-standard numeric extensions.
    """

    if value is None:
        return None
    if isinstance(value, Enum):
        return to_json_safe(value.value, path=path)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductSchemaError(f"{path} contains NaN or infinity")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ProductSchemaError(f"{path} contains a non-string or empty object key")
            converted[key] = to_json_safe(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]

    # numpy and pandas scalar objects expose ``item``. Converting through it
    # preserves their actual scalar type without accepting arbitrary objects via
    # a misleading string representation.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
        except (TypeError, ValueError):
            pass
        else:
            if item is not value:
                return to_json_safe(item, path=path)

    # pandas.NA/NaT do not have a useful ``item`` in every pandas release.
    type_name = type(value).__name__
    if type_name in {"NAType", "NaTType"}:
        return None
    raise ProductSchemaError(f"{path} has unsupported JSON value type {type_name}")


def _assert_strict_json(payload: Mapping[str, Any]) -> None:
    try:
        json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ProductSchemaError("payload is not strict JSON serializable") from exc


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field_name: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ProductSchemaError(f"{field_name} contains a non-string key")
    actual = set(value)
    if actual != expected:
        raise ProductSchemaError(
            f"{field_name} keys differ from schema; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _reject_forbidden_keys(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_EVIDENCE_FIELDS:
                raise ProductSchemaError(f"target/split key is forbidden at {path}.{key}")
            _reject_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, path=f"{path}[{index}]")


def _validate_note_safety(text: str) -> None:
    normalized = " ".join(text.split()).casefold()
    if any(
        re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _UNSAFE_NOTE_PATTERNS
    ):
        raise ProductSchemaError("analyst note contains an accusation or account-action language")


def _validate_source_fields(source_fields: Sequence[str]) -> tuple[str, ...]:
    if not source_fields:
        raise ProductSchemaError("observed evidence must cite at least one source field")
    normalized = tuple(_required_text(value, field_name="source_field") for value in source_fields)
    if len(set(normalized)) != len(normalized):
        raise ProductSchemaError("observed evidence source_fields must be unique")
    forbidden = sorted(field for field in normalized if field.lower() in _FORBIDDEN_EVIDENCE_FIELDS)
    if forbidden:
        raise ProductSchemaError(f"target/split fields cannot be observed evidence: {forbidden}")
    return normalized


@dataclass(frozen=True)
class ObservedEvidenceFact:
    """One directly observed or deterministically counted, non-model fact."""

    evidence_id: str
    kind: str
    statement: str
    values: Mapping[str, Any]
    source_fields: tuple[str, ...]
    scope: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_id", _required_text(self.evidence_id, field_name="evidence_id")
        )
        object.__setattr__(self, "kind", _required_text(self.kind, field_name="kind"))
        object.__setattr__(
            self, "statement", _required_text(self.statement, field_name="statement")
        )
        if self.scope not in _ALLOWED_SCOPES:
            raise ProductSchemaError(f"unsupported observed evidence scope: {self.scope!r}")
        object.__setattr__(self, "source_fields", _validate_source_fields(self.source_fields))
        if not isinstance(self.values, Mapping) or not self.values:
            raise ProductSchemaError("observed evidence values must be a non-empty mapping")
        _reject_forbidden_keys(self.values, path="$.observed_evidence.values")
        object.__setattr__(self, "values", to_json_safe(dict(self.values), path="$.values"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": "observed",
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "statement": self.statement,
            "values": dict(self.values),
            "source_fields": list(self.source_fields),
            "scope": self.scope,
        }


@dataclass(frozen=True)
class ModelContribution:
    """One model-attribution value; never represented as an observed fact."""

    feature: str
    contribution: float
    observed_feature_value: Any = None

    def __post_init__(self) -> None:
        feature = _required_text(self.feature, field_name="feature")
        if feature.lower() in _FORBIDDEN_EVIDENCE_FIELDS:
            raise ProductSchemaError(f"forbidden target/split model feature: {feature}")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(
            self,
            "contribution",
            _finite_float(self.contribution, field_name=f"contribution[{feature}]"),
        )
        object.__setattr__(
            self,
            "observed_feature_value",
            to_json_safe(self.observed_feature_value, path=f"$.feature_values.{feature}"),
        )

    def to_dict(self) -> dict[str, Any]:
        direction = "increases_score" if self.contribution > 0 else "decreases_score"
        if self.contribution == 0:
            direction = "neutral"
        return {
            "feature": self.feature,
            "contribution": self.contribution,
            "direction": direction,
            "observed_feature_value": self.observed_feature_value,
        }


@dataclass(frozen=True)
class ModelEvidence:
    """Transaction-level model prioritization evidence and optional attribution."""

    model_name: str
    score_name: str
    score: float
    threshold: float | None = None
    rank: int | None = None
    feature_contributions: tuple[ModelContribution, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_name", _required_text(self.model_name, field_name="model_name")
        )
        object.__setattr__(
            self, "score_name", _required_text(self.score_name, field_name="score_name")
        )
        object.__setattr__(self, "score", _finite_float(self.score, field_name="score"))
        if self.threshold is not None:
            object.__setattr__(
                self,
                "threshold",
                _finite_float(self.threshold, field_name="threshold"),
            )
        if self.rank is not None:
            if isinstance(self.rank, bool):
                raise ProductSchemaError("rank must be null or a positive integer")
            try:
                normalized_rank = int(self.rank)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProductSchemaError("rank must be null or a positive integer") from exc
            try:
                exact_rank = float(self.rank) == normalized_rank
            except (TypeError, ValueError):
                exact_rank = False
            if not exact_rank or normalized_rank < 1:
                raise ProductSchemaError("rank must be null or a positive integer")
            object.__setattr__(self, "rank", normalized_rank)
        contributions = tuple(self.feature_contributions)
        if any(not isinstance(item, ModelContribution) for item in contributions):
            raise ProductSchemaError("feature_contributions must contain ModelContribution items")
        object.__setattr__(self, "feature_contributions", contributions)
        features = [item.feature for item in contributions]
        if len(set(features)) != len(features):
            raise ProductSchemaError("model feature contributions must have unique feature names")

    @property
    def threshold_exceeded(self) -> bool | None:
        if self.threshold is None:
            return None
        return self.score >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": "model",
            "unit_of_analysis": "transaction",
            "model_name": self.model_name,
            "score_name": self.score_name,
            "score": self.score,
            "threshold": self.threshold,
            "threshold_exceeded": self.threshold_exceeded,
            "rank": self.rank,
            "feature_contributions": [item.to_dict() for item in self.feature_contributions],
            "interpretation": (
                "Model output prioritizes a transaction for human review; it is not an "
                "account-level label or a finding of wrongdoing."
            ),
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """Complete separation of observed facts from transaction-model evidence."""

    transaction_id: str
    focal_timestamp: str
    observed_evidence: tuple[ObservedEvidenceFact, ...]
    model_evidence: ModelEvidence
    limitations: tuple[str, ...] = (
        "The supplied neighborhood may be a bounded view rather than the complete network.",
        "Model scores and feature attributions are associational, not causal.",
    )
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transaction_id",
            _required_text(self.transaction_id, field_name="transaction_id"),
        )
        object.__setattr__(
            self,
            "focal_timestamp",
            _required_text(self.focal_timestamp, field_name="focal_timestamp"),
        )
        observed = tuple(self.observed_evidence)
        if any(not isinstance(item, ObservedEvidenceFact) for item in observed):
            raise ProductSchemaError("observed_evidence must contain ObservedEvidenceFact items")
        object.__setattr__(self, "observed_evidence", observed)
        if len(observed) < MINIMUM_OBSERVED_EVIDENCE:
            raise ProductSchemaError(
                f"a case requires at least {MINIMUM_OBSERVED_EVIDENCE} observed evidence facts"
            )
        identifiers = [fact.evidence_id for fact in observed]
        if len(set(identifiers)) != len(identifiers):
            raise ProductSchemaError("observed evidence IDs must be unique within a case")
        normalized_limitations = tuple(
            _required_text(item, field_name="limitation") for item in self.limitations
        )
        if not normalized_limitations:
            raise ProductSchemaError("at least one evidence limitation is required")
        object.__setattr__(self, "limitations", normalized_limitations)
        _assert_strict_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unit_of_analysis": "transaction",
            "transaction_id": self.transaction_id,
            "focal_timestamp": self.focal_timestamp,
            "observed_evidence": [fact.to_dict() for fact in self.observed_evidence],
            "model_evidence": self.model_evidence.to_dict(),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class AnalystNote:
    """A constrained case note plus auditable generation mode."""

    text: str
    mode: str
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _required_text(self.text, field_name="analyst_note.text"))
        _validate_note_safety(self.text)
        if self.mode not in {"deterministic_fallback", "optional_llm"}:
            raise ProductSchemaError(f"unsupported analyst note mode: {self.mode!r}")
        if self.fallback_reason is not None:
            object.__setattr__(
                self,
                "fallback_reason",
                _required_text(self.fallback_reason, field_name="fallback_reason"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class InvestigationCase:
    """Saved transaction-level investigation case; never an enforcement decision."""

    case_id: str
    evidence: EvidenceBundle
    analyst_note: AnalystNote
    priority_band: str
    network: Mapping[str, Any] = field(default_factory=dict)
    transactions: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required_text(self.case_id, field_name="case_id"))
        if self.priority_band not in {"elevated_review_priority", "ranked_review_candidate"}:
            raise ProductSchemaError(f"unsupported priority_band: {self.priority_band!r}")
        if not isinstance(self.network, Mapping):
            raise ProductSchemaError("case network must be a mapping")
        network = to_json_safe(dict(self.network), path="$.network")
        _reject_forbidden_keys(network, path="$.network")
        object.__setattr__(self, "network", network)
        transactions = tuple(
            to_json_safe(dict(item), path=f"$.transactions[{index}]")
            if isinstance(item, Mapping)
            else None
            for index, item in enumerate(self.transactions)
        )
        if any(item is None for item in transactions):
            raise ProductSchemaError("case transactions must contain mappings")
        _reject_forbidden_keys(transactions, path="$.transactions")
        object.__setattr__(self, "transactions", transactions)
        _assert_strict_json(self.to_dict())

    @property
    def transaction_id(self) -> str:
        return self.evidence.transaction_id

    def to_dict(self) -> dict[str, Any]:
        model = self.evidence.model_evidence
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "transaction_id": self.transaction_id,
            "focal_timestamp": self.evidence.focal_timestamp,
            "status": "pending_human_review",
            "priority": {
                "basis": "model",
                "band": self.priority_band,
                "threshold_exceeded": model.threshold_exceeded,
            },
            "observed_evidence": [fact.to_dict() for fact in self.evidence.observed_evidence],
            "model_evidence": model.to_dict(),
            "network": dict(self.network),
            "transactions": [dict(item) for item in self.transactions],
            "limitations": list(self.evidence.limitations),
            "analyst_note": self.analyst_note.to_dict(),
            "guardrails": {
                "human_review_required": True,
                "automatic_action_authorized": False,
                "account_level_fraud_label_inferred": False,
            },
        }


def validate_evidence_payload(payload: Mapping[str, Any]) -> None:
    """Validate a decoded saved evidence bundle without trusting its producer."""

    if not isinstance(payload, Mapping):
        raise ProductSchemaError("evidence payload must be a mapping")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "unit_of_analysis",
            "transaction_id",
            "focal_timestamp",
            "observed_evidence",
            "model_evidence",
            "limitations",
        },
        field_name="evidence payload",
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProductSchemaError("unsupported or missing evidence schema_version")
    if payload.get("unit_of_analysis") != "transaction":
        raise ProductSchemaError("evidence unit_of_analysis must be transaction")
    _required_text(payload.get("transaction_id"), field_name="transaction_id")
    _required_text(payload.get("focal_timestamp"), field_name="focal_timestamp")
    observed = payload.get("observed_evidence")
    if not isinstance(observed, list) or len(observed) < MINIMUM_OBSERVED_EVIDENCE:
        raise ProductSchemaError(
            f"evidence payload requires at least {MINIMUM_OBSERVED_EVIDENCE} observed facts"
        )
    evidence_ids: list[str] = []
    for index, fact in enumerate(observed):
        if not isinstance(fact, Mapping) or fact.get("basis") != "observed":
            raise ProductSchemaError(f"observed_evidence[{index}] must have basis='observed'")
        _require_exact_keys(
            fact,
            {"basis", "evidence_id", "kind", "statement", "values", "source_fields", "scope"},
            field_name=f"observed_evidence[{index}]",
        )
        evidence_ids.append(_required_text(fact.get("evidence_id"), field_name="evidence_id"))
        _required_text(fact.get("kind"), field_name="kind")
        _required_text(fact.get("statement"), field_name="statement")
        values = fact.get("values")
        if not isinstance(values, Mapping) or not values:
            raise ProductSchemaError(f"observed_evidence[{index}].values must be a mapping")
        _reject_forbidden_keys(values, path=f"$.observed_evidence[{index}].values")
        source_fields = fact.get("source_fields")
        if not isinstance(source_fields, list):
            raise ProductSchemaError(f"observed_evidence[{index}].source_fields must be a list")
        _validate_source_fields(source_fields)
        if fact.get("scope") not in _ALLOWED_SCOPES:
            raise ProductSchemaError(f"observed_evidence[{index}] has an unsupported scope")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ProductSchemaError("observed evidence IDs must be unique")

    model = payload.get("model_evidence")
    if not isinstance(model, Mapping) or model.get("basis") != "model":
        raise ProductSchemaError("model_evidence must have basis='model'")
    _require_exact_keys(
        model,
        {
            "basis",
            "unit_of_analysis",
            "model_name",
            "score_name",
            "score",
            "threshold",
            "threshold_exceeded",
            "rank",
            "feature_contributions",
            "interpretation",
        },
        field_name="model_evidence",
    )
    if model.get("unit_of_analysis") != "transaction":
        raise ProductSchemaError("model evidence cannot contain an account-level target")
    _required_text(model.get("model_name"), field_name="model_name")
    _required_text(model.get("score_name"), field_name="score_name")
    score = _finite_float(model.get("score"), field_name="score")
    threshold_value = model.get("threshold")
    threshold = (
        None if threshold_value is None else _finite_float(threshold_value, field_name="threshold")
    )
    expected_exceeded = None if threshold is None else score >= threshold
    if model.get("threshold_exceeded") is not expected_exceeded:
        raise ProductSchemaError("model threshold_exceeded contradicts score/threshold")
    rank = model.get("rank")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
        raise ProductSchemaError("model rank must be null or a positive integer")
    contributions = model.get("feature_contributions")
    if not isinstance(contributions, list):
        raise ProductSchemaError("model feature_contributions must be a list")
    contribution_features: list[str] = []
    for item in contributions:
        if not isinstance(item, Mapping):
            raise ProductSchemaError("each model contribution must be a mapping")
        _require_exact_keys(
            item,
            {"feature", "contribution", "direction", "observed_feature_value"},
            field_name="model contribution",
        )
        feature = _required_text(item.get("feature"), field_name="feature")
        contribution_features.append(feature)
        if feature.lower() in _FORBIDDEN_EVIDENCE_FIELDS:
            raise ProductSchemaError(f"forbidden model feature: {feature}")
        contribution = _finite_float(
            item.get("contribution"), field_name=f"contribution[{feature}]"
        )
        expected_direction = "neutral"
        if contribution > 0:
            expected_direction = "increases_score"
        elif contribution < 0:
            expected_direction = "decreases_score"
        if item.get("direction") != expected_direction:
            raise ProductSchemaError(
                f"model contribution direction contradicts value for {feature!r}"
            )
    if len(set(contribution_features)) != len(contribution_features):
        raise ProductSchemaError("model contribution feature names must be unique")
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise ProductSchemaError("evidence payload requires limitations")
    _assert_strict_json(to_json_safe(dict(payload)))


def validate_case_payload(payload: Mapping[str, Any]) -> None:
    """Validate a saved case artifact and its non-automated-action guardrails."""

    if not isinstance(payload, Mapping):
        raise ProductSchemaError("case payload must be a mapping")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "case_id",
            "transaction_id",
            "focal_timestamp",
            "status",
            "priority",
            "observed_evidence",
            "model_evidence",
            "network",
            "transactions",
            "limitations",
            "analyst_note",
            "guardrails",
        },
        field_name="case payload",
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProductSchemaError("unsupported or missing case schema_version")
    _required_text(payload.get("case_id"), field_name="case_id")
    if payload.get("status") != "pending_human_review":
        raise ProductSchemaError("case status must remain pending_human_review")
    priority = payload.get("priority")
    if not isinstance(priority, Mapping) or priority.get("basis") != "model":
        raise ProductSchemaError("case priority must be explicitly model-based")
    _require_exact_keys(
        priority,
        {"basis", "band", "threshold_exceeded"},
        field_name="case priority",
    )
    if priority.get("band") not in {"elevated_review_priority", "ranked_review_candidate"}:
        raise ProductSchemaError("case priority band is invalid")
    guardrails = payload.get("guardrails")
    expected_guardrails = {
        "human_review_required": True,
        "automatic_action_authorized": False,
        "account_level_fraud_label_inferred": False,
    }
    if guardrails != expected_guardrails:
        raise ProductSchemaError("case human-review guardrails are absent or weakened")
    network = payload.get("network")
    if not isinstance(network, Mapping):
        raise ProductSchemaError("case network must be a mapping")
    _reject_forbidden_keys(network, path="$.network")
    if network:
        required_network_keys = {
            "directed",
            "multigraph",
            "scope",
            "selection_method",
            "input_prior_edge_count",
            "saved_prior_edge_count",
            "nodes",
            "edges",
        }
        _require_exact_keys(network, required_network_keys, field_name="case network")
        if network.get("directed") is not True or network.get("multigraph") is not True:
            raise ProductSchemaError("case network must preserve a directed multigraph")
        if not isinstance(network.get("nodes"), list) or not isinstance(network.get("edges"), list):
            raise ProductSchemaError("case network nodes and edges must be lists")
    transactions = payload.get("transactions")
    if not isinstance(transactions, list):
        raise ProductSchemaError("case transactions must be a list")
    if any(not isinstance(item, Mapping) for item in transactions):
        raise ProductSchemaError("case transactions must contain objects")
    _reject_forbidden_keys(transactions, path="$.transactions")
    note = payload.get("analyst_note")
    if not isinstance(note, Mapping):
        raise ProductSchemaError("case analyst_note must be a mapping")
    _require_exact_keys(
        note,
        {"text", "mode", "fallback_reason"},
        field_name="analyst_note",
    )
    note_text = _required_text(note.get("text"), field_name="analyst_note.text")
    _validate_note_safety(note_text)
    if note.get("mode") not in {"deterministic_fallback", "optional_llm"}:
        raise ProductSchemaError("case analyst_note mode is invalid")
    evidence_view = {
        "schema_version": payload.get("schema_version"),
        "unit_of_analysis": "transaction",
        "transaction_id": payload.get("transaction_id"),
        "focal_timestamp": payload.get("focal_timestamp"),
        "observed_evidence": payload.get("observed_evidence"),
        "model_evidence": payload.get("model_evidence"),
        "limitations": payload.get("limitations"),
    }
    validate_evidence_payload(evidence_view)
    if priority.get("threshold_exceeded") is not payload["model_evidence"].get(
        "threshold_exceeded"
    ):
        raise ProductSchemaError("case priority contradicts model threshold evidence")
    _assert_strict_json(to_json_safe(dict(payload)))


__all__ = [
    "AnalystNote",
    "EvidenceBundle",
    "InvestigationCase",
    "MINIMUM_OBSERVED_EVIDENCE",
    "ModelContribution",
    "ModelEvidence",
    "ObservedEvidenceFact",
    "ProductSchemaError",
    "SCHEMA_VERSION",
    "to_json_safe",
    "validate_case_payload",
    "validate_evidence_payload",
]
