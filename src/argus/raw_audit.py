"""Streaming, full-file audit for the supplied IBM AML HI-Small CSV files."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from argus.data.accounts import composite_node_id

TRANSACTION_HEADER = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)
TRANSACTION_POSITION_LABELS = (
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
)
ACCOUNT_HEADER = (
    "Bank Name",
    "Bank ID",
    "Account Number",
    "Entity ID",
    "Entity Name",
)


class RawAuditError(ValueError):
    """Raised when a raw file cannot be audited as the expected CSV shape."""


def _decode_row(raw_line: bytes, expected_width: int) -> list[str]:
    """Decode the quote-free IBM source row without altering its values."""

    line = raw_line.rstrip(b"\r\n")
    if b'"' in line:
        raise RawAuditError(
            "Streaming audit expects the published quote-free HI-Small layout; "
            "use the canonical pandas loader for general CSV quoting"
        )
    try:
        values = [part.decode("utf-8") for part in line.split(b",")]
    except UnicodeDecodeError as exc:
        raise RawAuditError(f"Raw CSV is not valid UTF-8 at byte {exc.start}") from exc
    if len(values) != expected_width:
        raise RawAuditError(
            f"Expected {expected_width} fields but found {len(values)} in row: {line[:120]!r}"
        )
    return values


def _source_metadata(path: Path, digest: Any) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _confirm_duplicate_rows(path: Path, candidate_hashes: set[bytes]) -> tuple[int, int]:
    """Confirm candidate hashes with exact bytes in a second streaming pass."""

    if not candidate_hashes:
        return 0, 0
    exact_counts: Counter[bytes] = Counter()
    with path.open("rb", buffering=1024 * 1024) as handle:
        next(handle)
        for raw_line in handle:
            row = raw_line.rstrip(b"\r\n")
            digest = hashlib.blake2b(row, digest_size=16).digest()
            if digest in candidate_hashes:
                exact_counts[row] += 1
    duplicate_groups = sum(1 for count in exact_counts.values() if count > 1)
    duplicate_rows_after_first = sum(max(0, count - 1) for count in exact_counts.values())
    return duplicate_groups, duplicate_rows_after_first


def audit_accounts(path: str | Path) -> tuple[dict[str, Any], set[str]]:
    """Audit every accounts row and return aggregate facts plus composite keys."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Accounts file not found: {source}")

    file_digest = hashlib.sha256()
    seen_rows: set[bytes] = set()
    duplicate_hashes: set[bytes] = set()
    composite_keys: set[str] = set()
    account_numbers: Counter[str] = Counter()
    bank_ids: set[str] = set()
    entity_ids: set[str] = set()
    empty_counts = [0] * len(ACCOUNT_HEADER)
    whitespace_counts = [0] * len(ACCOUNT_HEADER)
    data_rows = 0

    with source.open("rb", buffering=1024 * 1024) as handle:
        raw_header = next(handle, b"")
        file_digest.update(raw_header)
        header = tuple(_decode_row(raw_header, len(ACCOUNT_HEADER)))
        if header != ACCOUNT_HEADER:
            raise RawAuditError(f"Unexpected accounts header: {header}")

        for raw_line in handle:
            file_digest.update(raw_line)
            row_bytes = raw_line.rstrip(b"\r\n")
            row_hash = hashlib.blake2b(row_bytes, digest_size=16).digest()
            if row_hash in seen_rows:
                duplicate_hashes.add(row_hash)
            else:
                seen_rows.add(row_hash)

            values = _decode_row(raw_line, len(ACCOUNT_HEADER))
            data_rows += 1
            for index, value in enumerate(values):
                empty_counts[index] += int(value == "")
                whitespace_counts[index] += int(value != value.strip())
            _, bank_id, account_number, entity_id, _ = values
            key = composite_node_id(bank_id, account_number)
            if key in composite_keys:
                raise RawAuditError(f"Duplicate composite account key: {key}")
            composite_keys.add(key)
            account_numbers[account_number] += 1
            bank_ids.add(bank_id.lstrip("0") or "0")
            entity_ids.add(entity_id)

    duplicate_groups, duplicate_rows = _confirm_duplicate_rows(source, duplicate_hashes)
    report = {
        "source": _source_metadata(source, file_digest),
        "data_rows": data_rows,
        "columns": list(ACCOUNT_HEADER),
        "empty_by_column": dict(zip(ACCOUNT_HEADER, empty_counts, strict=True)),
        "surrounding_whitespace_by_column": dict(
            zip(ACCOUNT_HEADER, whitespace_counts, strict=True)
        ),
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_rows_after_first": duplicate_rows,
        "unique_composite_nodes": len(composite_keys),
        "unique_account_numbers": len(account_numbers),
        "account_number_collision_groups": sum(
            1 for count in account_numbers.values() if count > 1
        ),
        "unique_banks": len(bank_ids),
        "unique_entities": len(entity_ids),
    }
    return report, composite_keys


def audit_transactions(
    path: str | Path,
    *,
    account_keys: set[str] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """Audit every transaction row, including exact duplicates and references."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Transactions file not found: {source}")

    file_digest = hashlib.sha256()
    seen_rows: set[bytes] = set()
    duplicate_hashes: set[bytes] = set()
    transaction_nodes: set[str] = set()
    labels: Counter[str] = Counter()
    payment_formats: set[str] = set()
    payment_currencies: set[str] = set()
    receiving_currencies: set[str] = set()
    empty_counts = [0] * len(TRANSACTION_HEADER)
    whitespace_counts = [0] * len(TRANSACTION_HEADER)
    invalid_timestamp_count = 0
    invalid_amount_counts = {"Amount Received": 0, "Amount Paid": 0}
    nonfinite_amount_counts = {"Amount Received": 0, "Amount Paid": 0}
    negative_amount_counts = {"Amount Received": 0, "Amount Paid": 0}
    zero_amount_counts = {"Amount Received": 0, "Amount Paid": 0}
    amount_min: dict[str, float | None] = {"Amount Received": None, "Amount Paid": None}
    amount_max: dict[str, float | None] = {"Amount Received": None, "Amount Paid": None}
    min_timestamp: datetime | None = None
    max_timestamp: datetime | None = None
    unmatched_sender_rows = 0
    unmatched_receiver_rows = 0
    data_rows = 0

    with source.open("rb", buffering=1024 * 1024) as handle:
        raw_header = next(handle, b"")
        file_digest.update(raw_header)
        header = tuple(_decode_row(raw_header, len(TRANSACTION_HEADER)))
        if header != TRANSACTION_HEADER:
            raise RawAuditError(f"Unexpected transactions header: {header}")

        for raw_line in handle:
            file_digest.update(raw_line)
            row_bytes = raw_line.rstrip(b"\r\n")
            row_hash = hashlib.blake2b(row_bytes, digest_size=16).digest()
            if row_hash in seen_rows:
                duplicate_hashes.add(row_hash)
            else:
                seen_rows.add(row_hash)

            values = _decode_row(raw_line, len(TRANSACTION_HEADER))
            data_rows += 1
            for index, value in enumerate(values):
                empty_counts[index] += int(value == "")
                whitespace_counts[index] += int(value != value.strip())

            timestamp_text, from_bank, from_account, to_bank, to_account = values[:5]
            try:
                timestamp = datetime.strptime(timestamp_text, "%Y/%m/%d %H:%M")
            except ValueError:
                invalid_timestamp_count += 1
            else:
                min_timestamp = (
                    timestamp if min_timestamp is None else min(min_timestamp, timestamp)
                )
                max_timestamp = (
                    timestamp if max_timestamp is None else max(max_timestamp, timestamp)
                )

            for name, index in (("Amount Received", 5), ("Amount Paid", 7)):
                try:
                    amount = float(values[index])
                except ValueError:
                    invalid_amount_counts[name] += 1
                    continue
                if not math.isfinite(amount):
                    nonfinite_amount_counts[name] += 1
                    continue
                negative_amount_counts[name] += int(amount < 0)
                zero_amount_counts[name] += int(amount == 0)
                amount_min[name] = (
                    amount if amount_min[name] is None else min(amount_min[name], amount)
                )
                amount_max[name] = (
                    amount if amount_max[name] is None else max(amount_max[name], amount)
                )

            labels[values[10]] += 1
            receiving_currencies.add(values[6])
            payment_currencies.add(values[8])
            payment_formats.add(values[9])
            sender_key = composite_node_id(from_bank, from_account)
            receiver_key = composite_node_id(to_bank, to_account)
            transaction_nodes.add(sender_key)
            transaction_nodes.add(receiver_key)
            if account_keys is not None:
                unmatched_sender_rows += int(sender_key not in account_keys)
                unmatched_receiver_rows += int(receiver_key not in account_keys)

    duplicate_groups, duplicate_rows = _confirm_duplicate_rows(source, duplicate_hashes)
    positive_count = labels.get("1", 0)
    report: dict[str, Any] = {
        "source": _source_metadata(source, file_digest),
        "data_rows": data_rows,
        "columns": list(TRANSACTION_HEADER),
        "empty_by_column": dict(zip(TRANSACTION_POSITION_LABELS, empty_counts, strict=True)),
        "surrounding_whitespace_by_column": dict(
            zip(TRANSACTION_POSITION_LABELS, whitespace_counts, strict=True)
        ),
        "timestamp": {
            "invalid_count": invalid_timestamp_count,
            "minimum": min_timestamp.isoformat() if min_timestamp else None,
            "maximum": max_timestamp.isoformat() if max_timestamp else None,
        },
        "labels": dict(sorted(labels.items())),
        "laundering_rate": positive_count / data_rows if data_rows else None,
        "amounts": {
            name: {
                "invalid_count": invalid_amount_counts[name],
                "nonfinite_count": nonfinite_amount_counts[name],
                "negative_count": negative_amount_counts[name],
                "zero_count": zero_amount_counts[name],
                "minimum": amount_min[name],
                "maximum": amount_max[name],
            }
            for name in ("Amount Received", "Amount Paid")
        },
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_rows_after_first": duplicate_rows,
        "unique_transaction_nodes": len(transaction_nodes),
        "payment_formats": sorted(payment_formats),
        "payment_currencies": sorted(payment_currencies),
        "receiving_currencies": sorted(receiving_currencies),
    }
    if account_keys is not None:
        report["account_references"] = {
            "unmatched_sender_rows": unmatched_sender_rows,
            "unmatched_receiver_rows": unmatched_receiver_rows,
            "matched_all_rows": unmatched_sender_rows == 0 and unmatched_receiver_rows == 0,
        }
    return report, transaction_nodes


def audit_hi_small_files(
    transactions_path: str | Path,
    accounts_path: str | Path,
    *,
    dataset_label: str = "IBM AML-Data HI-Small",
    dataset_is_synthetic: bool = True,
    locally_generated_fixture: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable, full-file audit with no sampled statistics."""

    accounts, account_keys = audit_accounts(accounts_path)
    transactions, transaction_nodes = audit_transactions(
        transactions_path, account_keys=account_keys
    )
    return {
        "provenance": {
            "dataset": dataset_label,
            "scope": "full_source_files",
            "generated_by": "argus.raw_audit.audit_hi_small_files",
            "dataset_is_synthetic": dataset_is_synthetic,
            "locally_generated_fixture": locally_generated_fixture,
        },
        "transactions": transactions,
        "accounts": accounts,
        "cross_file": {
            "transaction_nodes": len(transaction_nodes),
            "matched_transaction_nodes": len(transaction_nodes & account_keys),
            "unmatched_transaction_nodes": len(transaction_nodes - account_keys),
            "unused_account_nodes": len(account_keys - transaction_nodes),
        },
    }


def write_audit(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write an audit report atomically enough for a local single-process run."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
