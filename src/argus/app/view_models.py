"""Pure presentation projections for the ARGUS analyst workspace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

_STATUS_LABELS = {
    "pending_human_review": "Pending review",
    "pending_review": "Pending review",
    "in_review": "In review",
    "escalated": "Escalated",
    "false_positive": "False positive",
    "closed": "Closed",
}
_PRIORITY_LABELS = {
    "elevated_review_priority": "Elevated",
    "high_priority": "High",
    "high": "High",
    "elevated": "Elevated",
    "medium": "Medium",
    "low": "Low",
}
_PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "high_priority": 1,
    "elevated": 1,
    "elevated_review_priority": 1,
    "medium": 2,
    "low": 3,
}
_QUEUE_COLUMNS = (
    "case_id",
    "priority",
    "priority_label",
    "priority_rank",
    "pattern",
    "pattern_label",
    "account_ids",
    "account_search",
    "account_count",
    "transaction_count",
    "total_flow",
    "currency_context",
    "last_activity",
    "status",
    "status_label",
    "research_score",
    "search_text",
)


def canonical_token(value: Any) -> str:
    """Return a stable token for labels and filters."""

    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def humanize(value: Any) -> str:
    """Turn a stored token into readable interface copy."""

    if value is None:
        return "Not available"
    try:
        if pd.isna(value):
            return "Not available"
    except (TypeError, ValueError):
        pass
    return str(value).replace("_", " ").strip().title() or "Not available"


def display_status(value: Any) -> str:
    return _STATUS_LABELS.get(canonical_token(value), humanize(value))


def display_priority(value: Any) -> str:
    return _PRIORITY_LABELS.get(canonical_token(value), humanize(value))


def display_pattern(value: Any) -> str:
    if canonical_token(value) == "high_graphsage_transaction_score":
        return "Elevated network ranking"
    return humanize(value)


def _compact_amount(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "Not available"
    absolute = abs(float(numeric))
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if absolute >= scale:
            return f"{float(numeric) / scale:,.2f}{suffix}"
    return f"{float(numeric):,.2f}"


def _display_total_flow(value: Any, currency: str) -> str:
    if currency == "Mixed currencies":
        return currency
    compact = _compact_amount(value)
    return f"{compact} · {currency}" if compact != "Not available" else compact


def _case_transactions(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = case.get("transactions", [])
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    return []


def _case_nodes(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    network = case.get("network", {})
    if not isinstance(network, dict):
        return []
    records = network.get("nodes", [])
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    return []


def case_account_ids(case: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect only account identifiers already present in the saved case."""

    values: set[str] = set()
    for node in _case_nodes(case):
        value = node.get("node_id", node.get("id"))
        if value is not None:
            values.add(str(value))
    for transaction in _case_transactions(case):
        for key in ("from_node_id", "to_node_id", "source", "target", "sender", "receiver"):
            value = transaction.get(key)
            if value is not None:
                values.add(str(value))
    for key in ("sender_id", "receiver_id"):
        value = case.get(key)
        if value is not None:
            values.add(str(value))
    return tuple(sorted(values))


def case_latest_activity(case: Mapping[str, Any]) -> pd.Timestamp | pd.NaT:
    """Return the latest timestamp already saved with a case."""

    values = [record.get("timestamp") for record in _case_transactions(case)]
    values.append(case.get("focal_timestamp"))
    timestamps = pd.to_datetime(pd.Series(values, dtype="object"), errors="coerce", utc=True)
    return timestamps.max() if timestamps.notna().any() else pd.NaT


def case_currency_context(case: Mapping[str, Any]) -> str:
    """Describe saved currencies without implying cross-currency conversion."""

    currencies = {
        str(record.get("currency", record.get("payment_currency"))).strip()
        for record in _case_transactions(case)
        if record.get("currency", record.get("payment_currency")) not in (None, "")
    }
    if not currencies:
        return "Currency not available"
    if len(currencies) == 1:
        return next(iter(currencies))
    return "Mixed currencies"


def _base_case_status(case: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    return str(case.get("status", row.get("status", "pending_human_review")))


def _workflow_status(
    workflow: Mapping[str, Mapping[str, Any]] | None,
    case_id: str,
    fallback: str,
) -> str:
    if not workflow:
        return fallback
    state = workflow.get(case_id)
    if not isinstance(state, Mapping):
        return fallback
    return str(state.get("status", fallback))


def build_queue_view(
    queue: pd.DataFrame,
    cases: Mapping[str, Mapping[str, Any]],
    workflow: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Create an in-memory business worklist without mutating saved inputs."""

    records: list[dict[str, Any]] = []
    for source in queue.to_dict(orient="records"):
        case_id = str(source.get("case_id", ""))
        case = cases.get(case_id, {})
        accounts = case_account_ids(case)
        priority = str(source.get("priority", "not_available"))
        pattern = str(source.get("major_pattern", "not_available"))
        base_status = _base_case_status(case, source)
        status = _workflow_status(workflow, case_id, base_status)
        account_count = source.get("account_count")
        if account_count is None:
            account_count = len(accounts)
        transaction_count = source.get("transaction_count")
        if transaction_count is None:
            transaction_count = len(_case_transactions(case))
        record = {
            "case_id": case_id,
            "priority": priority,
            "priority_label": display_priority(priority),
            "priority_rank": _PRIORITY_ORDER.get(canonical_token(priority), 9),
            "pattern": pattern,
            "pattern_label": display_pattern(pattern),
            "account_ids": accounts,
            "account_search": " ".join(accounts),
            "account_count": account_count,
            "transaction_count": transaction_count,
            "total_flow": source.get("total_flow"),
            "currency_context": case_currency_context(case),
            "last_activity": case_latest_activity(case),
            "status": status,
            "status_label": display_status(status),
            "research_score": source.get("risk_score"),
        }
        record["search_text"] = " ".join(
            (case_id, record["pattern_label"], record["account_search"])
        ).casefold()
        records.append(record)
    return pd.DataFrame.from_records(records, columns=_QUEUE_COLUMNS)


def display_queue(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only analyst-facing columns with stable formatting."""

    displayed = pd.DataFrame(index=frame.index)
    displayed["Priority"] = frame["priority_label"]
    displayed["Case ID"] = frame["case_id"]
    displayed["Review Signal"] = frame["pattern_label"]
    displayed["Accounts"] = pd.to_numeric(frame["account_count"], errors="coerce").astype("Int64")
    displayed["Transfers"] = pd.to_numeric(frame["transaction_count"], errors="coerce").astype(
        "Int64"
    )
    displayed["Total Flow"] = [
        _display_total_flow(value, currency)
        for value, currency in zip(frame["total_flow"], frame["currency_context"], strict=False)
    ]
    timestamps = pd.to_datetime(frame["last_activity"], errors="coerce", utc=True)
    displayed["Last Activity"] = timestamps.dt.strftime("%d %b %y · %H:%M").fillna("Not available")
    displayed["Status"] = frame["status_label"]
    return displayed.reset_index(drop=True)


def filter_queue_view(
    frame: pd.DataFrame,
    *,
    query: str = "",
    priorities: list[str] | None = None,
    statuses: list[str] | None = None,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter the worklist using analyst-facing values."""

    filtered = frame.copy()
    if query.strip():
        filtered = filtered[filtered["search_text"].str.contains(query.casefold(), regex=False)]
    if priorities is not None:
        filtered = filtered[filtered["priority_label"].isin(priorities)]
    if statuses is not None:
        filtered = filtered[filtered["status_label"].isin(statuses)]
    if patterns is not None:
        filtered = filtered[filtered["pattern_label"].isin(patterns)]
    return filtered


def sort_queue_view(frame: pd.DataFrame, sort_label: str) -> pd.DataFrame:
    """Apply stable business sorting without exposing raw field names."""

    if sort_label == "Latest activity":
        return frame.sort_values("last_activity", ascending=False, kind="mergesort")
    if sort_label == "Total flow":
        return frame.sort_values("total_flow", ascending=False, kind="mergesort")
    if sort_label == "Transfers":
        return frame.sort_values("transaction_count", ascending=False, kind="mergesort")
    return frame.sort_values(
        ["priority_rank", "last_activity"], ascending=[True, False], kind="mergesort"
    )
