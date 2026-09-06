"""Auditable validation reports for raw and canonical transaction data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from argus.data.load import RAW_TRANSACTION_COLUMNS

Severity = Literal["error", "warning", "info"]

CANONICAL_REQUIRED_COLUMNS = (
    "transaction_id",
    "source_row_number",
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "from_node_id",
    "to_node_id",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
)


class DataValidationError(ValueError):
    """Raised when a validation report contains one or more errors."""


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable validation observation."""

    code: str
    severity: Severity
    message: str
    count: int = 0
    columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Validation result that separates blocking errors from audit warnings."""

    stage: str
    row_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
            raise DataValidationError(f"{self.stage} validation failed: {details}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "row_count": self.row_count,
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
            "statistics": self.statistics,
        }


def _missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return sorted(set(required).difference(frame.columns))


def validate_raw_schema(frame: pd.DataFrame) -> ValidationReport:
    """Validate the positional pandas representation of the 11-field raw schema."""

    report = ValidationReport(stage="raw_schema", row_count=len(frame))
    missing = _missing_columns(frame, RAW_TRANSACTION_COLUMNS)
    if missing:
        report.issues.append(
            ValidationIssue(
                code="missing_required_columns",
                severity="error",
                message=f"Missing required raw columns: {missing}",
                count=len(missing),
                columns=tuple(missing),
            )
        )
    if frame.empty:
        report.issues.append(
            ValidationIssue(
                code="empty_dataset",
                severity="error",
                message="The transaction table has no rows",
            )
        )

    available = [column for column in RAW_TRANSACTION_COLUMNS if column in frame]
    empty_counts: dict[str, int] = {}
    for column in available:
        series = frame[column].astype("string")
        empty_counts[column] = int(series.isna().sum() + series.str.strip().eq("").sum())
    report.statistics["empty_values_by_column"] = empty_counts
    critical_empty = {name: count for name, count in empty_counts.items() if count}
    if critical_empty:
        report.issues.append(
            ValidationIssue(
                code="missing_or_empty_values",
                severity="error",
                message=f"Raw required fields contain empty values: {critical_empty}",
                count=sum(critical_empty.values()),
                columns=tuple(critical_empty),
            )
        )
    return report


def validate_transactions(frame: pd.DataFrame) -> ValidationReport:
    """Validate canonical values without mutating or deleting repeated events."""

    report = ValidationReport(stage="canonical_transactions", row_count=len(frame))
    missing = _missing_columns(frame, CANONICAL_REQUIRED_COLUMNS)
    if missing:
        report.issues.append(
            ValidationIssue(
                code="missing_required_columns",
                severity="error",
                message=f"Missing canonical columns: {missing}",
                count=len(missing),
                columns=tuple(missing),
            )
        )
        return report
    if frame.empty:
        report.issues.append(
            ValidationIssue("empty_dataset", "error", "The canonical table has no rows")
        )
        return report

    missing_counts = {column: int(frame[column].isna().sum()) for column in frame.columns}
    report.statistics["missing_values_by_column"] = missing_counts
    required_missing = {
        column: missing_counts[column]
        for column in CANONICAL_REQUIRED_COLUMNS
        if missing_counts[column]
    }
    if required_missing:
        report.issues.append(
            ValidationIssue(
                "missing_values",
                "error",
                f"Canonical required fields contain missing values: {required_missing}",
                sum(required_missing.values()),
                tuple(required_missing),
            )
        )

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    report.statistics["invalid_timestamp_count"] = invalid_timestamps
    if invalid_timestamps:
        report.issues.append(
            ValidationIssue(
                "invalid_timestamp",
                "error",
                "Some timestamps cannot be parsed",
                invalid_timestamps,
                ("timestamp",),
            )
        )

    target = pd.to_numeric(frame["is_laundering"], errors="coerce")
    invalid_labels = int((target.isna() | ~target.isin([0, 1])).sum())
    report.statistics["label_counts"] = {
        str(key): int(value) for key, value in target.value_counts(dropna=False).items()
    }
    if invalid_labels:
        report.issues.append(
            ValidationIssue(
                "invalid_target",
                "error",
                "is_laundering must contain only 0 or 1",
                invalid_labels,
                ("is_laundering",),
            )
        )

    for column in ("amount_received", "amount_paid"):
        amount = pd.to_numeric(frame[column], errors="coerce")
        invalid = int(amount.isna().sum() + (~np.isfinite(amount.fillna(0))).sum())
        negative = int(amount.lt(0).sum())
        report.statistics[f"{column}_invalid_count"] = invalid
        report.statistics[f"{column}_negative_count"] = negative
        if invalid or negative:
            report.issues.append(
                ValidationIssue(
                    "invalid_amount",
                    "error",
                    f"{column} contains non-numeric, non-finite, or negative values",
                    invalid + negative,
                    (column,),
                )
            )

    identifier_columns = (
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "from_node_id",
        "to_node_id",
        "transaction_id",
    )
    empty_identifiers = {
        column: int(frame[column].astype("string").str.strip().eq("").sum())
        for column in identifier_columns
    }
    report.statistics["empty_identifiers_by_column"] = empty_identifiers
    if any(empty_identifiers.values()):
        report.issues.append(
            ValidationIssue(
                "empty_identifier",
                "error",
                f"Identifier fields contain empty values: {empty_identifiers}",
                sum(empty_identifiers.values()),
                tuple(name for name, count in empty_identifiers.items() if count),
            )
        )

    duplicate_ids = int(frame["transaction_id"].duplicated().sum())
    report.statistics["duplicate_transaction_id_count"] = duplicate_ids
    if duplicate_ids:
        report.issues.append(
            ValidationIssue(
                "duplicate_transaction_id",
                "error",
                "Generated transaction identifiers are not unique",
                duplicate_ids,
                ("transaction_id",),
            )
        )

    event_columns = [
        "timestamp",
        "from_bank_raw" if "from_bank_raw" in frame else "from_bank",
        "from_account",
        "to_bank_raw" if "to_bank_raw" in frame else "to_bank",
        "to_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    ]
    exact_duplicates = int(frame.duplicated(subset=event_columns, keep="first").sum())
    report.statistics["exact_duplicate_rows_after_first"] = exact_duplicates
    if exact_duplicates:
        report.issues.append(
            ValidationIssue(
                "exact_duplicate_events_preserved",
                "warning",
                "Exact event duplicates are reported and remain preserved",
                exact_duplicates,
                tuple(event_columns),
            )
        )
    return report
