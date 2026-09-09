"""Deterministic evidence extraction from transactions and graph neighborhoods."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from argus.product.schemas import (
    EvidenceBundle,
    ModelContribution,
    ModelEvidence,
    ObservedEvidenceFact,
    ProductSchemaError,
)

_REQUIRED_TRANSACTION_FIELDS = {
    "transaction_id",
    "timestamp",
    "from_node_id",
    "to_node_id",
    "amount_paid",
    "amount_received",
    "payment_currency",
    "receiving_currency",
    "payment_format",
}
_REQUIRED_NEIGHBORHOOD_FIELDS = {"timestamp", "from_node_id", "to_node_id"}
_PRIOR_FEATURE_FIELDS = (
    "sender_previous_transaction_count",
    "sender_previous_outgoing_amount",
    "receiver_previous_transaction_count",
    "receiver_previous_incoming_amount",
    "sender_prior_fan_out_degree",
    "sender_prior_fan_in_degree",
    "receiver_prior_fan_out_degree",
    "receiver_prior_fan_in_degree",
    "pair_previous_transfer_count",
)


class EvidenceEngineError(ProductSchemaError):
    """Raised when source data cannot support a defensible evidence bundle."""


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _record_from_transaction(
    transaction: pd.DataFrame | pd.Series | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(transaction, pd.DataFrame):
        if len(transaction) != 1:
            raise EvidenceEngineError("transaction DataFrame must contain exactly one row")
        return transaction.iloc[0].to_dict()
    if isinstance(transaction, pd.Series):
        return transaction.to_dict()
    if isinstance(transaction, Mapping):
        return dict(transaction)
    raise EvidenceEngineError("transaction must be a one-row DataFrame, Series, or mapping")


def _required_source_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if _is_missing_scalar(value):
        raise EvidenceEngineError(f"transaction field {field!r} is missing")
    text = str(value).strip()
    if not text:
        raise EvidenceEngineError(f"transaction field {field!r} is empty")
    return text


def _amount(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool):
        raise EvidenceEngineError(
            f"transaction field {field!r} must be a finite non-negative number"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceEngineError(
            f"transaction field {field!r} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise EvidenceEngineError(
            f"transaction field {field!r} must be a finite non-negative number"
        )
    return number


def _timestamp(value: Any, *, field: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvidenceEngineError(f"{field} must contain parseable timestamps") from exc
    if isinstance(parsed, pd.DatetimeIndex):
        raise EvidenceEngineError(f"{field} must be scalar")
    result = pd.Timestamp(parsed)
    if pd.isna(result):
        raise EvidenceEngineError(f"{field} must contain parseable timestamps")
    return result


def _format_number(number: float | int) -> str:
    if float(number).is_integer():
        return f"{int(number):,}"
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _strictly_prior_neighborhood(
    neighborhood: pd.DataFrame,
    *,
    focal_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(neighborhood, pd.DataFrame):
        raise EvidenceEngineError("neighborhood must be a pandas DataFrame")
    missing = sorted(_REQUIRED_NEIGHBORHOOD_FIELDS.difference(neighborhood.columns))
    if missing:
        raise EvidenceEngineError(f"neighborhood is missing required columns: {missing}")
    result = neighborhood.copy()
    parsed = pd.to_datetime(result["timestamp"], errors="coerce")
    if parsed.isna().any():
        raise EvidenceEngineError(
            f"neighborhood timestamp contains {int(parsed.isna().sum())} unparseable value(s)"
        )
    for endpoint in ("from_node_id", "to_node_id"):
        values = result[endpoint].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise EvidenceEngineError(f"neighborhood field {endpoint!r} contains empty values")
        result[endpoint] = values
    result["timestamp"] = parsed
    try:
        prior_mask = result["timestamp"].lt(focal_timestamp)
    except TypeError as exc:
        raise EvidenceEngineError(
            "transaction and neighborhood timestamps must use compatible timezone semantics"
        ) from exc
    return result.loc[prior_mask].copy()


def _base_observations(record: Mapping[str, Any]) -> tuple[ObservedEvidenceFact, ...]:
    transaction_id = _required_source_text(record, "transaction_id")
    sender = _required_source_text(record, "from_node_id")
    receiver = _required_source_text(record, "to_node_id")
    paid_currency = _required_source_text(record, "payment_currency")
    received_currency = _required_source_text(record, "receiving_currency")
    payment_format = _required_source_text(record, "payment_format")
    focal_timestamp = _timestamp(record.get("timestamp"), field="transaction timestamp")
    amount_paid = _amount(record, "amount_paid")
    amount_received = _amount(record, "amount_received")

    return (
        ObservedEvidenceFact(
            evidence_id="obs-transaction-value",
            kind="transaction_value",
            statement=(
                f"The sender paid {_format_number(amount_paid)} {paid_currency}; the receiver "
                f"recorded {_format_number(amount_received)} {received_currency}."
            ),
            values={
                "amount_paid": amount_paid,
                "payment_currency": paid_currency,
                "amount_received": amount_received,
                "receiving_currency": received_currency,
            },
            source_fields=(
                "amount_paid",
                "payment_currency",
                "amount_received",
                "receiving_currency",
            ),
            scope="focal_transaction",
        ),
        ObservedEvidenceFact(
            evidence_id="obs-transfer-route",
            kind="directed_transfer_route",
            statement=f"The observed directed route is {sender} to {receiver}.",
            values={
                "from_node_id": sender,
                "to_node_id": receiver,
                "self_transfer": sender == receiver,
            },
            source_fields=("from_node_id", "to_node_id"),
            scope="focal_transaction",
        ),
        ObservedEvidenceFact(
            evidence_id="obs-time-channel",
            kind="transaction_time_and_channel",
            statement=(
                f"Transaction {transaction_id} was recorded at {focal_timestamp.isoformat()} "
                f"using {payment_format}."
            ),
            values={
                "transaction_id": transaction_id,
                "timestamp": focal_timestamp.isoformat(),
                "payment_format": payment_format,
            },
            source_fields=("transaction_id", "timestamp", "payment_format"),
            scope="focal_transaction",
        ),
    )


def _neighborhood_observations(
    record: Mapping[str, Any],
    prior: pd.DataFrame,
) -> tuple[ObservedEvidenceFact, ...]:
    sender = str(record["from_node_id"]).strip()
    receiver = str(record["to_node_id"]).strip()
    pair = prior[prior["from_node_id"].eq(sender) & prior["to_node_id"].eq(receiver)]

    sender_outgoing = prior[prior["from_node_id"].eq(sender)]
    sender_incoming = prior[prior["to_node_id"].eq(sender)]
    receiver_outgoing = prior[prior["from_node_id"].eq(receiver)]
    receiver_incoming = prior[prior["to_node_id"].eq(receiver)]

    return (
        ObservedEvidenceFact(
            evidence_id="obs-prior-pair-activity",
            kind="strictly_prior_directed_pair_activity",
            statement=(
                f"The saved case context contains {len(pair):,} earlier transfer(s) on "
                "the same directed sender-to-receiver route."
            ),
            values={
                "prior_directed_transfer_count": len(pair),
                "future_or_same_timestamp_rows_used": 0,
            },
            source_fields=("timestamp", "from_node_id", "to_node_id"),
            scope="strictly_prior_supplied_neighborhood",
        ),
        ObservedEvidenceFact(
            evidence_id="obs-prior-sender-neighborhood",
            kind="strictly_prior_sender_neighborhood",
            statement=(
                f"Before the focal time, the saved case context contains "
                f"{len(sender_outgoing):,} outgoing and {len(sender_incoming):,} incoming "
                "sender transaction(s)."
            ),
            values={
                "prior_outgoing_transactions": len(sender_outgoing),
                "prior_incoming_transactions": len(sender_incoming),
                "prior_unique_outgoing_counterparties": int(
                    sender_outgoing["to_node_id"].nunique()
                ),
                "prior_unique_incoming_counterparties": int(
                    sender_incoming["from_node_id"].nunique()
                ),
            },
            source_fields=("timestamp", "from_node_id", "to_node_id"),
            scope="strictly_prior_supplied_neighborhood",
        ),
        ObservedEvidenceFact(
            evidence_id="obs-prior-receiver-neighborhood",
            kind="strictly_prior_receiver_neighborhood",
            statement=(
                f"Before the focal time, the saved case context contains "
                f"{len(receiver_incoming):,} incoming and {len(receiver_outgoing):,} outgoing "
                "receiver transaction(s)."
            ),
            values={
                "prior_incoming_transactions": len(receiver_incoming),
                "prior_outgoing_transactions": len(receiver_outgoing),
                "prior_unique_incoming_counterparties": int(
                    receiver_incoming["from_node_id"].nunique()
                ),
                "prior_unique_outgoing_counterparties": int(
                    receiver_outgoing["to_node_id"].nunique()
                ),
            },
            source_fields=("timestamp", "from_node_id", "to_node_id"),
            scope="strictly_prior_supplied_neighborhood",
        ),
    )


def _prior_feature_observation(record: Mapping[str, Any]) -> ObservedEvidenceFact | None:
    available: dict[str, Any] = {}
    for field in _PRIOR_FEATURE_FIELDS:
        if field not in record:
            continue
        value = record[field]
        if _is_missing_scalar(value):
            continue
        if isinstance(value, bool):
            raise EvidenceEngineError(f"strictly-prior feature {field!r} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise EvidenceEngineError(f"strictly-prior feature {field!r} must be numeric") from exc
        if not math.isfinite(number) or number < 0:
            raise EvidenceEngineError(
                f"strictly-prior feature {field!r} must be finite and non-negative"
            )
        available[field] = int(number) if number.is_integer() else number
    if not available:
        return None
    names = ", ".join(sorted(available))
    return ObservedEvidenceFact(
        evidence_id="obs-strictly-prior-features",
        kind="precomputed_strictly_prior_features",
        statement=f"Saved leakage-safe history fields are available for review: {names}.",
        values=available,
        source_fields=tuple(sorted(available)),
        scope="focal_transaction_strictly_prior_feature",
    )


def _model_contributions(
    values: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    transaction: Mapping[str, Any],
    maximum: int,
) -> tuple[ModelContribution, ...]:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 50:
        raise EvidenceEngineError("max_feature_contributions must be an integer from 1 through 50")
    if values is None:
        return ()

    records: list[tuple[str, Any, Any]] = []
    if isinstance(values, Mapping):
        for feature, contribution in values.items():
            if not isinstance(feature, str):
                raise EvidenceEngineError("feature contribution names must be strings")
            records.append((feature, contribution, transaction.get(feature)))
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                raise EvidenceEngineError(f"feature_contributions[{index}] must be a mapping")
            if "feature" not in item or "contribution" not in item:
                raise EvidenceEngineError(
                    f"feature_contributions[{index}] requires feature and contribution"
                )
            feature = str(item["feature"])
            records.append(
                (
                    feature,
                    item["contribution"],
                    item.get("observed_feature_value", transaction.get(feature)),
                )
            )
    else:
        raise EvidenceEngineError("feature_contributions must be a mapping or sequence of mappings")

    contributions = [
        ModelContribution(
            feature=feature,
            contribution=contribution,
            observed_feature_value=observed_value,
        )
        for feature, contribution, observed_value in records
    ]
    contributions.sort(key=lambda item: (-abs(item.contribution), item.feature))
    return tuple(contributions[:maximum])


def build_evidence_bundle(
    transaction: pd.DataFrame | pd.Series | Mapping[str, Any],
    neighborhood: pd.DataFrame,
    *,
    model_score: float,
    model_name: str,
    threshold: float | None = None,
    rank: int | None = None,
    score_name: str = "raw_ranking_score",
    feature_contributions: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    max_feature_contributions: int = 5,
    allowed_focal_partitions: Sequence[str] | None = ("validation",),
    forbidden_neighborhood_partitions: Sequence[str] = ("test",),
) -> EvidenceBundle:
    """Build transaction-level evidence without reading a target or future edge.

    The neighborhood may include current/future rows, but only rows strictly
    earlier than the focal timestamp are counted. Its facts explicitly retain
    the scope ``strictly_prior_supplied_neighborhood`` so a sampled neighborhood
    cannot be presented as the complete graph.
    """

    record = _record_from_transaction(transaction)
    missing = sorted(_REQUIRED_TRANSACTION_FIELDS.difference(record))
    if missing:
        raise EvidenceEngineError(f"transaction is missing required columns: {missing}")
    if "partition" in record and allowed_focal_partitions is not None:
        partition = _required_source_text(record, "partition").lower()
        allowed = {str(item).strip().lower() for item in allowed_focal_partitions}
        if partition not in allowed:
            raise EvidenceEngineError(
                f"focal partition {partition!r} is outside the allowed product scope "
                f"{sorted(allowed)}"
            )
    if "partition" in neighborhood:
        present = set(neighborhood["partition"].dropna().astype("string").str.strip().str.lower())
        forbidden = {str(item).strip().lower() for item in forbidden_neighborhood_partitions}
        blocked = sorted(present.intersection(forbidden))
        if blocked:
            raise EvidenceEngineError(
                f"neighborhood contains forbidden closed partition(s): {blocked}"
            )
    focal_timestamp = _timestamp(record["timestamp"], field="transaction timestamp")
    prior = _strictly_prior_neighborhood(neighborhood, focal_timestamp=focal_timestamp)

    observed = [*_base_observations(record), *_neighborhood_observations(record, prior)]
    prior_features = _prior_feature_observation(record)
    if prior_features is not None:
        observed.append(prior_features)

    model = ModelEvidence(
        model_name=model_name,
        score_name=score_name,
        score=model_score,
        threshold=threshold,
        rank=rank,
        feature_contributions=_model_contributions(
            feature_contributions,
            transaction=record,
            maximum=max_feature_contributions,
        ),
    )
    return EvidenceBundle(
        transaction_id=_required_source_text(record, "transaction_id"),
        focal_timestamp=focal_timestamp.isoformat(),
        observed_evidence=tuple(observed),
        model_evidence=model,
    )


__all__ = ["EvidenceEngineError", "build_evidence_bundle"]
