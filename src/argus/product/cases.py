"""Deterministic transaction-case construction from saved score artifacts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from argus.product.evidence import EvidenceEngineError, build_evidence_bundle
from argus.product.narrative import EvidenceNoteGenerator, generate_analyst_note
from argus.product.schemas import EvidenceBundle, InvestigationCase, validate_case_payload


class CaseBuilderError(EvidenceEngineError):
    """Raised when source frames cannot produce an auditable case queue."""


def _case_id(transaction_id: str) -> str:
    digest = hashlib.sha256(transaction_id.encode("utf-8")).hexdigest()[:16].upper()
    return f"ARGUS-{digest}"


def build_investigation_case(
    evidence: EvidenceBundle,
    *,
    llm: EvidenceNoteGenerator | None = None,
    network: Mapping[str, Any] | None = None,
    transactions: Sequence[Mapping[str, Any]] = (),
) -> InvestigationCase:
    """Turn validated evidence into a deterministic human-review case."""

    note = generate_analyst_note(evidence, llm=llm)
    priority_band = (
        "elevated_review_priority"
        if evidence.model_evidence.threshold_exceeded is True
        else "ranked_review_candidate"
    )
    case = InvestigationCase(
        case_id=_case_id(evidence.transaction_id),
        evidence=evidence,
        analyst_note=note,
        priority_band=priority_band,
        network={} if network is None else network,
        transactions=tuple(transactions),
    )
    validate_case_payload(case.to_dict())
    return case


def build_case_from_frames(
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
    max_neighborhood_edges: int = 100,
    llm: EvidenceNoteGenerator | None = None,
) -> InvestigationCase:
    """Build one case directly from one transaction row and graph neighborhood."""

    evidence = build_evidence_bundle(
        transaction,
        neighborhood,
        model_score=model_score,
        model_name=model_name,
        threshold=threshold,
        rank=rank,
        score_name=score_name,
        feature_contributions=feature_contributions,
        max_feature_contributions=max_feature_contributions,
    )
    network, transactions = _build_display_context(
        transaction,
        neighborhood,
        evidence=evidence,
        maximum_prior_edges=max_neighborhood_edges,
    )
    return build_investigation_case(
        evidence,
        llm=llm,
        network=network,
        transactions=transactions,
    )


def _transaction_record(
    transaction: pd.DataFrame | pd.Series | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(transaction, pd.DataFrame):
        if len(transaction) != 1:
            raise CaseBuilderError("transaction DataFrame must contain exactly one row")
        return transaction.iloc[0].to_dict()
    if isinstance(transaction, pd.Series):
        return transaction.to_dict()
    return dict(transaction)


def _edge_record(row: Mapping[str, Any], *, is_focal: bool) -> dict[str, Any]:
    amount = row.get("amount_paid")
    if amount is not None:
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            amount = None
        else:
            if not math.isfinite(amount):
                amount = None
    timestamp = pd.Timestamp(row["timestamp"]).isoformat()
    return {
        "from_node_id": str(row["from_node_id"]),
        "to_node_id": str(row["to_node_id"]),
        "transaction_id": None if row.get("transaction_id") is None else str(row["transaction_id"]),
        "timestamp": timestamp,
        "amount": amount,
        "currency": None if row.get("payment_currency") is None else str(row["payment_currency"]),
        "payment_format": None if row.get("payment_format") is None else str(row["payment_format"]),
        "is_focal": is_focal,
    }


def _build_display_context(
    transaction: pd.DataFrame | pd.Series | Mapping[str, Any],
    neighborhood: pd.DataFrame,
    *,
    evidence: EvidenceBundle,
    maximum_prior_edges: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if (
        isinstance(maximum_prior_edges, bool)
        or not isinstance(maximum_prior_edges, int)
        or not 1 <= maximum_prior_edges <= 1_000
    ):
        raise CaseBuilderError("max_neighborhood_edges must be an integer from 1 through 1000")
    record = _transaction_record(transaction)
    sender = str(record["from_node_id"])
    receiver = str(record["to_node_id"])
    focal_timestamp = pd.Timestamp(evidence.focal_timestamp)
    local_mask = neighborhood["from_node_id"].astype("string").isin(
        {sender, receiver}
    ) | neighborhood["to_node_id"].astype("string").isin({sender, receiver})
    local = neighborhood.loc[local_mask].copy()
    local["timestamp"] = pd.to_datetime(local["timestamp"], errors="raise")
    prior = local[local["timestamp"].lt(focal_timestamp)].copy()
    input_prior_edges = len(prior)
    sort_columns = ["timestamp"]
    if "transaction_id" in prior:
        sort_columns.append("transaction_id")
    prior = prior.sort_values(sort_columns, kind="mergesort").tail(maximum_prior_edges)
    prior_records = [_edge_record(row, is_focal=False) for row in prior.to_dict("records")]
    focal = _edge_record(record, is_focal=True)
    edges = [*prior_records, focal]

    node_ids = sorted(
        {str(edge[endpoint]) for edge in edges for endpoint in ("from_node_id", "to_node_id")}
    )
    nodes: list[dict[str, Any]] = []
    for node_id in node_ids:
        if node_id == sender == receiver:
            role = "focal_sender_receiver"
        elif node_id == sender:
            role = "focal_sender"
        elif node_id == receiver:
            role = "focal_receiver"
        else:
            role = "strictly_prior_context"
        nodes.append(
            {
                "node_id": node_id,
                "role": role,
                "is_seed": node_id in {sender, receiver},
                "case_in_degree": sum(edge["to_node_id"] == node_id for edge in edges),
                "case_out_degree": sum(edge["from_node_id"] == node_id for edge in edges),
            }
        )
    network = {
        "directed": True,
        "multigraph": True,
        "scope": "focal_edge_plus_strictly_prior_incident_neighborhood",
        "selection_method": "most_recent_strictly_prior_incident_edges",
        "input_prior_edge_count": input_prior_edges,
        "saved_prior_edge_count": len(prior_records),
        "nodes": nodes,
        "edges": edges,
    }
    return network, tuple(edges)


def _validate_case_frames(
    transactions: pd.DataFrame,
    model_scores: pd.DataFrame,
    *,
    transaction_id_column: str,
    score_column: str,
) -> None:
    if not isinstance(transactions, pd.DataFrame) or not isinstance(model_scores, pd.DataFrame):
        raise CaseBuilderError("transactions and model_scores must be pandas DataFrames")
    for name, frame, required in (
        ("transactions", transactions, {transaction_id_column}),
        ("model_scores", model_scores, {transaction_id_column, score_column}),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise CaseBuilderError(f"{name} is missing required columns: {missing}")
    if transactions[transaction_id_column].duplicated().any():
        raise CaseBuilderError("transactions contains duplicate transaction IDs")
    if model_scores[transaction_id_column].duplicated().any():
        raise CaseBuilderError("model_scores contains duplicate transaction IDs")
    for name, frame in (("transactions", transactions), ("model_scores", model_scores)):
        if "partition" in frame:
            partitions = set(frame["partition"].dropna().astype("string").str.strip().str.lower())
            if "test" in partitions:
                raise CaseBuilderError(f"{name} contains closed final-test rows")


def build_investigation_cases(
    transactions: pd.DataFrame,
    neighborhood: pd.DataFrame,
    model_scores: pd.DataFrame,
    *,
    model_name: str,
    score_column: str = "raw_ranking_score",
    transaction_id_column: str = "transaction_id",
    threshold: float | None = None,
    top_n: int = 10,
    score_name: str | None = None,
    contributions_by_transaction: Mapping[str, Mapping[str, Any] | Sequence[Mapping[str, Any]]]
    | None = None,
    max_feature_contributions: int = 5,
    max_neighborhood_edges: int = 100,
    llm: EvidenceNoteGenerator | None = None,
) -> tuple[InvestigationCase, ...]:
    """Create a deterministic top-score case queue from saved validation outputs.

    Ranking is descending by the supplied model score and then transaction ID,
    making score ties reproducible. If a threshold is supplied, only rows meeting
    it are emitted. The transaction target is never read and final-test rows are
    rejected whenever partition provenance is present.
    """

    _validate_case_frames(
        transactions,
        model_scores,
        transaction_id_column=transaction_id_column,
        score_column=score_column,
    )
    if not isinstance(neighborhood, pd.DataFrame):
        raise CaseBuilderError("neighborhood must be a pandas DataFrame")
    if "partition" in neighborhood:
        partitions = set(
            neighborhood["partition"].dropna().astype("string").str.strip().str.lower()
        )
        if "test" in partitions:
            raise CaseBuilderError("neighborhood contains closed final-test rows")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise CaseBuilderError("top_n must be a positive integer")
    if threshold is not None:
        try:
            threshold_number = float(threshold)
        except (TypeError, ValueError) as exc:
            raise CaseBuilderError("threshold must be null or finite") from exc
        if not math.isfinite(threshold_number):
            raise CaseBuilderError("threshold must be null or finite")
    else:
        threshold_number = None

    scores = model_scores[[transaction_id_column, score_column]].copy()
    scores[transaction_id_column] = scores[transaction_id_column].astype("string")
    scores[score_column] = pd.to_numeric(scores[score_column], errors="coerce")
    if (
        scores[transaction_id_column].isna().any()
        or scores[transaction_id_column].str.strip().eq("").any()
    ):
        raise CaseBuilderError("model_scores contains empty transaction IDs")
    if scores[score_column].isna().any() or (~scores[score_column].map(math.isfinite)).any():
        raise CaseBuilderError("model_scores contains non-finite scores")
    scores = scores.sort_values(
        [score_column, transaction_id_column],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scores["_queue_rank"] = range(1, len(scores) + 1)
    if threshold_number is not None:
        scores = scores[scores[score_column].ge(threshold_number)]
    selected = scores.head(top_n)

    indexed_transactions = transactions.set_index(transaction_id_column, drop=False)
    missing_ids = sorted(
        set(selected[transaction_id_column].astype(str)).difference(
            set(indexed_transactions.index.astype(str))
        )
    )
    if missing_ids:
        preview = missing_ids[:3]
        raise CaseBuilderError(f"selected score rows lack matching transactions: {preview}")

    cases: list[InvestigationCase] = []
    for _, score_row in selected.iterrows():
        score_values = score_row.to_dict()
        transaction_id = str(score_values[transaction_id_column])
        transaction = indexed_transactions.loc[transaction_id]
        if isinstance(transaction, pd.DataFrame):
            raise CaseBuilderError(f"transaction ID is not unique: {transaction_id}")
        sender = str(transaction.get("from_node_id", ""))
        receiver = str(transaction.get("to_node_id", ""))
        if {"from_node_id", "to_node_id"}.issubset(neighborhood.columns):
            local_mask = neighborhood["from_node_id"].astype("string").isin(
                {sender, receiver}
            ) | neighborhood["to_node_id"].astype("string").isin({sender, receiver})
            local_neighborhood = neighborhood.loc[local_mask]
        else:
            local_neighborhood = neighborhood
        contributions = (
            contributions_by_transaction.get(transaction_id)
            if contributions_by_transaction is not None
            else None
        )
        cases.append(
            build_case_from_frames(
                transaction,
                local_neighborhood,
                model_score=float(score_values[score_column]),
                model_name=model_name,
                threshold=threshold_number,
                rank=int(score_values["_queue_rank"]),
                score_name=score_name or score_column,
                feature_contributions=contributions,
                max_feature_contributions=max_feature_contributions,
                max_neighborhood_edges=max_neighborhood_edges,
                llm=llm,
            )
        )
    return tuple(cases)


__all__ = [
    "CaseBuilderError",
    "build_case_from_frames",
    "build_investigation_case",
    "build_investigation_cases",
]
