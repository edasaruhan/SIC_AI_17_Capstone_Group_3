"""Canonical, traceable preprocessing for IBM AML transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from argus.data.load import RAW_TRANSACTION_COLUMNS
from argus.data.validate import validate_raw_schema, validate_transactions

DEFAULT_COLUMN_MAPPING = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}


class PreprocessingError(ValueError):
    """Raised when raw values cannot be converted without ambiguity."""


def _normalize_bank_ids(series: pd.Series, column: str) -> pd.Series:
    values = series.astype("string").str.strip()
    invalid = values.eq("") | ~values.str.fullmatch(r"\d+")
    if invalid.any():
        examples = values[invalid].head(3).tolist()
        raise PreprocessingError(f"{column} has empty/non-decimal bank IDs: {examples}")
    normalized = values.str.lstrip("0").replace("", "0")
    return normalized.astype("string")


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def _source_name(frame: pd.DataFrame) -> str:
    value = frame.attrs.get("source_name")
    if value:
        return Path(str(value)).name
    return "in_memory_transactions.csv"


def preprocess_transactions(
    raw: pd.DataFrame,
    *,
    column_mapping: dict[str, str] | None = None,
    drop_exact_duplicates: bool = False,
) -> pd.DataFrame:
    """Return a deterministically ordered canonical transaction table.

    Exact duplicates are preserved by default and always receive distinct source
    identities. Setting ``drop_exact_duplicates=True`` is an explicit, auditable
    policy that retains the first physical occurrence only.
    """

    raw_report = validate_raw_schema(raw)
    raw_report.raise_for_errors()
    mapping = dict(column_mapping or DEFAULT_COLUMN_MAPPING)
    required_mapping_keys = set(RAW_TRANSACTION_COLUMNS)
    if set(mapping) != required_mapping_keys:
        missing = sorted(required_mapping_keys.difference(mapping))
        unexpected = sorted(set(mapping).difference(required_mapping_keys))
        raise PreprocessingError(
            f"Column mapping must cover the exact raw schema; missing={missing}, "
            f"unexpected={unexpected}"
        )

    source_attrs: dict[str, Any] = dict(raw.attrs)
    result = raw.copy()
    if "source_row_number" not in result:
        result.insert(0, "source_row_number", range(2, len(result) + 2))
    source_rows = pd.to_numeric(result["source_row_number"], errors="coerce")
    if source_rows.isna().any() or source_rows.duplicated().any():
        raise PreprocessingError("source_row_number must be numeric and unique")
    result["source_row_number"] = source_rows.astype("int64")
    result = result.rename(columns=mapping)

    timestamp_text = result["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(timestamp_text):
        parsed_timestamps = pd.to_datetime(timestamp_text, errors="coerce")
    else:
        parsed_timestamps = pd.to_datetime(timestamp_text, format="%Y/%m/%d %H:%M", errors="coerce")
        remaining = parsed_timestamps.isna()
        if remaining.any():
            parsed_timestamps.loc[remaining] = pd.to_datetime(
                timestamp_text.loc[remaining], format="mixed", errors="coerce"
            )
    result["timestamp"] = parsed_timestamps

    for column in ("from_account", "to_account"):
        result[column] = _normalize_text(result[column]).str.upper()
    for column in ("receiving_currency", "payment_currency", "payment_format"):
        result[column] = _normalize_text(result[column])

    result["from_bank_raw"] = _normalize_text(result["from_bank"])
    result["to_bank_raw"] = _normalize_text(result["to_bank"])
    result["from_bank"] = _normalize_bank_ids(result["from_bank_raw"], "from_bank")
    result["to_bank"] = _normalize_bank_ids(result["to_bank_raw"], "to_bank")
    result["from_node_id"] = result["from_bank"] + "::" + result["from_account"]
    result["to_node_id"] = result["to_bank"] + "::" + result["to_account"]

    result["amount_received"] = pd.to_numeric(result["amount_received"], errors="coerce")
    result["amount_paid"] = pd.to_numeric(result["amount_paid"], errors="coerce")
    target = pd.to_numeric(result["is_laundering"], errors="coerce")
    if target.isna().any() or (~target.isin([0, 1])).any():
        invalid_count = int((target.isna() | ~target.isin([0, 1])).sum())
        raise PreprocessingError(f"is_laundering contains {invalid_count} value(s) outside 0/1")
    result["is_laundering"] = target.astype("Int8")

    source_name = _source_name(raw)
    result.insert(0, "source_file", source_name)
    result.insert(
        0,
        "transaction_id",
        result["source_row_number"].map(lambda row: f"{source_name}:row-{row}"),
    )

    exact_event_columns = [mapping[column] for column in RAW_TRANSACTION_COLUMNS]
    duplicate_mask = result.duplicated(subset=exact_event_columns, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if drop_exact_duplicates:
        result = result.loc[~duplicate_mask].copy()

    result = result.sort_values(
        ["timestamp", "source_row_number"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    result.attrs.update(source_attrs)
    result.attrs.update(
        {
            "source_name": source_name,
            "exact_duplicate_rows_observed": duplicate_count,
            "exact_duplicate_policy": (
                "drop_after_first" if drop_exact_duplicates else "preserve_all"
            ),
        }
    )
    canonical_report = validate_transactions(result)
    canonical_report.raise_for_errors()
    return result
