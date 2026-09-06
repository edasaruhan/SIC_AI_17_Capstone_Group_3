"""Loader and referential checks for the optional HI-Small accounts table."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ACCOUNT_COLUMN_MAPPING = {
    "Bank Name": "bank_name",
    "Bank ID": "bank_id",
    "Account Number": "account_number",
    "Entity ID": "entity_id",
    "Entity Name": "entity_name",
}


class AccountDataError(ValueError):
    """Raised when the companion accounts file is structurally invalid."""


def normalize_bank_id(value: object) -> str:
    """Normalize numeric bank IDs without losing their identity.

    The transaction file zero-pads bank identifiers while the companion account
    file does not. Integer-like normalization is therefore required for a valid
    cross-file join. The raw source columns remain available in each table.
    """

    text = str(value).strip()
    if not text or not text.isdecimal():
        raise AccountDataError(f"Bank ID must contain decimal digits: {value!r}")
    return text.lstrip("0") or "0"


def composite_node_id(bank_id: object, account_id: object) -> str:
    """Create a globally stable bank-account identifier."""

    account = str(account_id).strip().upper()
    if not account:
        raise AccountDataError("Account ID must not be empty")
    return f"{normalize_bank_id(bank_id)}::{account}"


def load_ibm_accounts(path: str | Path) -> pd.DataFrame:
    """Load and validate the HI-Small companion accounts table."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"HI-Small accounts file not found: {source}")

    frame = pd.read_csv(source, dtype="string", keep_default_na=False)
    missing = sorted(set(ACCOUNT_COLUMN_MAPPING).difference(frame.columns))
    unexpected = sorted(set(frame.columns).difference(ACCOUNT_COLUMN_MAPPING))
    if missing or unexpected:
        raise AccountDataError(
            f"Invalid accounts schema; missing={missing}, unexpected={unexpected}"
        )

    result = frame.rename(columns=ACCOUNT_COLUMN_MAPPING).copy()
    result.insert(0, "source_row_number", range(2, len(result) + 2))
    for column in ACCOUNT_COLUMN_MAPPING.values():
        result[column] = result[column].astype("string").str.strip()
        if result[column].eq("").any():
            raise AccountDataError(f"Accounts column {column!r} contains empty values")

    result.insert(2, "raw_bank_id", result["bank_id"].copy())
    result["bank_id"] = result["bank_id"].map(normalize_bank_id).astype("string")
    result["account_number"] = result["account_number"].str.upper()
    result["node_id"] = [
        composite_node_id(bank, account)
        for bank, account in zip(result["bank_id"], result["account_number"], strict=True)
    ]
    if result["node_id"].duplicated().any():
        duplicate_count = int(result["node_id"].duplicated(keep=False).sum())
        raise AccountDataError(
            f"Accounts table has {duplicate_count} rows with duplicate composite IDs"
        )
    return result


@dataclass(frozen=True)
class AccountReferenceReport:
    """Aggregate-only referential evidence; entity names are never emitted."""

    account_rows: int
    unique_account_nodes: int
    transaction_nodes: int
    matched_transaction_nodes: int
    unmatched_transaction_nodes: int
    unmatched_sender_rows: int
    unmatched_receiver_rows: int
    unused_account_nodes: int

    @property
    def valid(self) -> bool:
        return self.unmatched_transaction_nodes == 0

    def to_dict(self) -> dict[str, int | bool]:
        return {**asdict(self), "valid": self.valid}


def validate_account_references(
    transactions: pd.DataFrame, accounts: pd.DataFrame
) -> AccountReferenceReport:
    """Check transaction endpoints against companion account composite IDs."""

    required_transactions = {"from_node_id", "to_node_id"}
    missing = sorted(required_transactions.difference(transactions.columns))
    if missing:
        raise AccountDataError(f"Transaction table lacks node columns: {missing}")
    if "node_id" not in accounts:
        raise AccountDataError("Accounts table lacks canonical node_id")

    account_nodes = set(accounts["node_id"].astype(str))
    sender = transactions["from_node_id"].astype(str)
    receiver = transactions["to_node_id"].astype(str)
    transaction_nodes = set(sender).union(receiver)
    matched = transaction_nodes.intersection(account_nodes)
    return AccountReferenceReport(
        account_rows=len(accounts),
        unique_account_nodes=len(account_nodes),
        transaction_nodes=len(transaction_nodes),
        matched_transaction_nodes=len(matched),
        unmatched_transaction_nodes=len(transaction_nodes - account_nodes),
        unmatched_sender_rows=int((~sender.isin(account_nodes)).sum()),
        unmatched_receiver_rows=int((~receiver.isin(account_nodes)).sum()),
        unused_account_nodes=len(account_nodes - transaction_nodes),
    )
