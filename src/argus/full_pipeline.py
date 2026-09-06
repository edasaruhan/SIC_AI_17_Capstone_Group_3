"""Disk-backed Sprint 1 pipeline for the complete IBM AML HI-Small files.

The ordinary :mod:`argus.pipeline` implementation deliberately favors simple,
inspectable pandas operations for the quick development run.  Materializing its
intermediate copies for five million rows is not safe on a modest workstation,
however.  This module keeps the same scientific definitions while using DuckDB
for external sorting, window calculations, joins, and Parquet export.

The central temporal invariant is explicit: a transaction at timestamp ``t``
may use state from timestamps strictly earlier than ``t`` only.  Calculations
therefore happen on one row per entity/timestamp batch before being joined back
to individual transactions.  Duration windows are closed on the lower bound and
open on the current timestamp: ``[t - window, t)``.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import platform
import re
import shutil
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from argus.config import get_path, load_config

RAW_TRANSACTION_HEADER = (
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

# DuckDB makes duplicate CSV headers unique by appending ``_1``.
DUCKDB_TRANSACTION_COLUMNS = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account_1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)

ACCOUNT_HEADER = ("Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name")

CANONICAL_COLUMNS = (
    "transaction_id",
    "source_file",
    "source_row_number",
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
    "from_bank_raw",
    "to_bank_raw",
    "from_node_id",
    "to_node_id",
)

TRANSACTION_FEATURE_COLUMNS = (
    "log_amount_paid",
    "log_amount_received",
    "same_bank",
    "currency_match",
    "amount_difference_same_currency",
    "amount_ratio_same_currency",
)

TIME_FEATURE_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
    "sender_seconds_since_previous",
    "receiver_seconds_since_previous",
)

GRAPH_FEATURE_COLUMNS = (
    "sender_prior_fan_out_degree",
    "sender_prior_fan_in_degree",
    "receiver_prior_fan_out_degree",
    "receiver_prior_fan_in_degree",
    "pair_previous_transfer_count",
)

_SIZE_SETTING = re.compile(r"\s*\d+(?:\.\d+)?\s*(?:kb|mb|gb|tb)\s*", re.IGNORECASE)
_WINDOW_SETTING = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(us|ms|s|min|h|d|w)\s*", re.IGNORECASE)
_WORK_PREFIX = ".argus_full_work_"


class FullPipelineError(RuntimeError):
    """Raised when a full-data invariant or output verification fails."""


@dataclass(frozen=True)
class FullPipelineRunResult:
    """Location and saved evidence for one successful out-of-core run."""

    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


class _StageRecorder:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        print(f"[full] {name} ...", flush=True)
        started = time.perf_counter()
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - started
            self.seconds[name] = elapsed
            print(f"[full] {name} FAILED after {elapsed:.2f}s", flush=True)
            raise
        elapsed = time.perf_counter() - started
        self.seconds[name] = elapsed
        print(f"[full] {name} completed in {elapsed:.2f}s", flush=True)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_path(path: Path) -> str:
    return _sql_literal(path.resolve().as_posix())


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_exact_header(path: Path, expected: Sequence[str], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV was not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        try:
            actual = tuple(next(csv.reader(handle)))
        except StopIteration as exc:
            raise FullPipelineError(f"{label} CSV is empty: {path}") from exc
    if actual != tuple(expected):
        raise FullPipelineError(
            f"{label} CSV header differs from the exact expected schema; "
            f"expected={list(expected)}, actual={list(actual)}"
        )


def _validated_size_setting(value: object, name: str) -> str:
    text = str(value)
    if _SIZE_SETTING.fullmatch(text) is None:
        raise FullPipelineError(f"full_pipeline.{name} must look like '1GB', not {value!r}")
    return re.sub(r"\s+", "", text).upper()


def _full_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("full_pipeline")
    if not isinstance(raw, Mapping):
        raise FullPipelineError("configs/full.yaml must define a full_pipeline mapping")
    engine = str(raw.get("engine", "duckdb")).casefold()
    if engine != "duckdb":
        raise FullPipelineError(
            "The out-of-core full pipeline requires full_pipeline.engine=duckdb"
        )
    threads = raw.get("threads", 2)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise FullPipelineError("full_pipeline.threads must be a positive integer")
    compression = str(raw.get("parquet_compression", "zstd")).casefold()
    if compression not in {"zstd", "snappy", "gzip", "lz4", "brotli", "uncompressed"}:
        raise FullPipelineError(f"Unsupported Parquet compression: {compression!r}")
    sample_rows = raw.get("eda_sample_rows", 100_000)
    if isinstance(sample_rows, bool) or not isinstance(sample_rows, int) or sample_rows <= 0:
        raise FullPipelineError("full_pipeline.eda_sample_rows must be a positive integer")
    return {
        "engine": engine,
        "memory_limit": _validated_size_setting(raw.get("memory_limit", "1GB"), "memory_limit"),
        "threads": threads,
        "max_temp_directory_size": _validated_size_setting(
            raw.get("max_temp_directory_size", "100GB"), "max_temp_directory_size"
        ),
        "parquet_compression": compression,
        "eda_sample_rows": sample_rows,
        "run_eda": bool(raw.get("run_eda", True)),
    }


def _assert_full_contract(config: Mapping[str, Any]) -> None:
    project = config.get("project", {})
    if not isinstance(project, Mapping) or project.get("mode") != "full":
        raise FullPipelineError("run_full_sprint1_pipeline requires project.mode=full")
    data = config.get("data", {})
    sampling = data.get("sampling", {}) if isinstance(data, Mapping) else {}
    if not isinstance(sampling, Mapping) or sampling.get("enabled", False):
        raise FullPipelineError("Full feature engineering cannot use data.sampling.enabled=true")


def _configure_connection(
    connection: duckdb.DuckDBPyConnection,
    settings: Mapping[str, Any],
    spill_dir: Path,
) -> None:
    spill_dir.mkdir(parents=True, exist_ok=False)
    connection.execute(f"SET memory_limit={_sql_literal(settings['memory_limit'])}")
    connection.execute(
        f"SET max_temp_directory_size={_sql_literal(settings['max_temp_directory_size'])}"
    )
    connection.execute(f"SET temp_directory={_sql_path(spill_dir)}")
    connection.execute("SET preserve_insertion_order=true")
    connection.execute("SET enable_progress_bar=false")
    connection.execute("SET threads=1")


def _ingest_csvs(
    connection: duckdb.DuckDBPyConnection,
    transaction_path: Path,
    accounts_path: Path,
) -> tuple[int, int]:
    # A single scanner thread plus insertion-order preservation makes the hidden
    # rowid an auditable physical-row identity.  CSV row 1 is the header, hence
    # source_row_number = rowid + 2.
    connection.execute(
        f"""
        CREATE TABLE raw_transactions AS
        SELECT *
        FROM read_csv(
            {_sql_path(transaction_path)},
            header=true,
            all_varchar=true,
            parallel=false,
            strict_mode=true,
            null_padding=false,
            ignore_errors=false
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE raw_accounts AS
        SELECT *
        FROM read_csv(
            {_sql_path(accounts_path)},
            header=true,
            all_varchar=true,
            parallel=false,
            strict_mode=true,
            null_padding=false,
            ignore_errors=false
        )
        """
    )
    transaction_columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info('raw_transactions')").fetchall()
    )
    account_columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info('raw_accounts')").fetchall()
    )
    if transaction_columns != DUCKDB_TRANSACTION_COLUMNS:
        raise FullPipelineError(
            "DuckDB transaction schema changed unexpectedly after exact-header validation: "
            f"{transaction_columns}"
        )
    if account_columns != ACCOUNT_HEADER:
        raise FullPipelineError(
            "DuckDB account schema changed unexpectedly after exact-header validation: "
            f"{account_columns}"
        )
    transaction_stats = connection.execute(
        "SELECT count(*), min(rowid), max(rowid) FROM raw_transactions"
    ).fetchone()
    account_stats = connection.execute(
        "SELECT count(*), min(rowid), max(rowid) FROM raw_accounts"
    ).fetchone()
    assert transaction_stats is not None and account_stats is not None
    transaction_rows = int(transaction_stats[0])
    account_rows = int(account_stats[0])
    if transaction_rows <= 0 or transaction_stats[1:] != (0, transaction_rows - 1):
        raise FullPipelineError("Transaction rowid is not contiguous from zero after ingestion")
    if account_rows <= 0 or account_stats[1:] != (0, account_rows - 1):
        raise FullPipelineError("Account rowid is not contiguous from zero after ingestion")
    return transaction_rows, account_rows


def _empty_count_select(columns: Sequence[str]) -> str:
    return ",\n".join(
        "sum(CASE WHEN "
        f"{_quote_identifier(column)} IS NULL OR trim({_quote_identifier(column)}) = '' "
        f"THEN 1 ELSE 0 END) AS empty_{index}"
        for index, column in enumerate(columns)
    )


def _validate_raw_transactions(
    connection: duckdb.DuckDBPyConnection, row_count: int
) -> dict[str, Any]:
    empty_values = connection.execute(
        f"SELECT {_empty_count_select(DUCKDB_TRANSACTION_COLUMNS)} FROM raw_transactions"
    ).fetchone()
    assert empty_values is not None
    report_names = list(RAW_TRANSACTION_HEADER)
    report_names[4] = "Account.1"
    empty_by_column = {
        name: int(count) for name, count in zip(report_names, empty_values, strict=True)
    }
    checks = connection.execute(
        """
        SELECT
            sum(CASE WHEN try_strptime(trim("Timestamp"), '%Y/%m/%d %H:%M') IS NULL
                THEN 1 ELSE 0 END),
            sum(CASE WHEN NOT regexp_full_match(trim("From Bank"), '[0-9]+')
                THEN 1 ELSE 0 END),
            sum(CASE WHEN NOT regexp_full_match(trim("To Bank"), '[0-9]+')
                THEN 1 ELSE 0 END),
            sum(CASE WHEN try_cast(trim("Amount Received") AS DOUBLE) IS NULL
                         OR NOT isfinite(try_cast(trim("Amount Received") AS DOUBLE))
                         OR try_cast(trim("Amount Received") AS DOUBLE) < 0
                THEN 1 ELSE 0 END),
            sum(CASE WHEN try_cast(trim("Amount Paid") AS DOUBLE) IS NULL
                         OR NOT isfinite(try_cast(trim("Amount Paid") AS DOUBLE))
                         OR try_cast(trim("Amount Paid") AS DOUBLE) < 0
                THEN 1 ELSE 0 END),
            sum(CASE WHEN try_cast(trim("Is Laundering") AS INTEGER) IS NULL
                         OR try_cast(trim("Is Laundering") AS INTEGER) NOT IN (0, 1)
                THEN 1 ELSE 0 END)
        FROM raw_transactions
        """
    ).fetchone()
    assert checks is not None
    check_names = (
        "invalid_timestamp_count",
        "invalid_from_bank_count",
        "invalid_to_bank_count",
        "invalid_amount_received_count",
        "invalid_amount_paid_count",
        "invalid_target_count",
    )
    diagnostics = {name: int(value) for name, value in zip(check_names, checks, strict=True)}
    errors = {
        **{key: value for key, value in empty_by_column.items() if value},
        **{key: value for key, value in diagnostics.items() if value},
    }
    if errors:
        raise FullPipelineError(f"Full raw transaction validation failed: {errors}")

    grouped_columns = ", ".join(_quote_identifier(name) for name in DUCKDB_TRANSACTION_COLUMNS)
    duplicate_row = connection.execute(
        f"""
        SELECT coalesce(sum(group_size - 1), 0)
        FROM (
            SELECT count(*) AS group_size
            FROM raw_transactions
            GROUP BY {grouped_columns}
            HAVING count(*) > 1
        ) duplicate_groups
        """
    ).fetchone()
    assert duplicate_row is not None
    return {
        "stage": "raw_schema",
        "row_count": row_count,
        "valid": True,
        "error_count": 0,
        "warning_count": 0,
        "issues": [],
        "statistics": {
            "empty_values_by_column": empty_by_column,
            **diagnostics,
            "exact_duplicate_rows_after_first": int(duplicate_row[0]),
        },
    }


def _validate_raw_accounts(connection: duckdb.DuckDBPyConnection, row_count: int) -> dict[str, Any]:
    empty_values = connection.execute(
        f"SELECT {_empty_count_select(ACCOUNT_HEADER)} FROM raw_accounts"
    ).fetchone()
    assert empty_values is not None
    empty_by_column = {
        name: int(count) for name, count in zip(ACCOUNT_HEADER, empty_values, strict=True)
    }
    invalid_bank = int(
        connection.execute(
            """
            SELECT count(*)
            FROM raw_accounts
            WHERE NOT regexp_full_match(trim("Bank ID"), '[0-9]+')
            """
        ).fetchone()[0]
    )
    errors = {key: value for key, value in empty_by_column.items() if value}
    if invalid_bank:
        errors["invalid_bank_id_count"] = invalid_bank
    if errors:
        raise FullPipelineError(f"Full raw accounts validation failed: {errors}")
    return {
        "stage": "raw_accounts_schema",
        "row_count": row_count,
        "valid": True,
        "error_count": 0,
        "warning_count": 0,
        "issues": [],
        "statistics": {
            "empty_values_by_column": empty_by_column,
            "invalid_bank_id_count": invalid_bank,
        },
    }


def _clean_text(expression: str) -> str:
    return f"regexp_replace(trim({expression}), '\\s+', ' ', 'g')"


def _normalized_bank(expression: str) -> str:
    return f"coalesce(nullif(regexp_replace({expression}, '^0+', ''), ''), '0')"


def _export_canonical(
    connection: duckdb.DuckDBPyConnection,
    destination: Path,
    source_name: str,
    compression: str,
) -> None:
    from_bank_raw = _clean_text(_quote_identifier("From Bank"))
    to_bank_raw = _clean_text(_quote_identifier("To Bank"))
    from_account = f"upper({_clean_text(_quote_identifier('Account'))})"
    to_account = f"upper({_clean_text(_quote_identifier('Account_1'))})"
    query = f"""
        WITH normalized AS (
            SELECT
                cast(rowid + 2 AS BIGINT) AS source_row_number,
                strptime(trim("Timestamp"), '%Y/%m/%d %H:%M') AS timestamp,
                {from_bank_raw} AS from_bank_raw,
                {from_account} AS from_account,
                {to_bank_raw} AS to_bank_raw,
                {to_account} AS to_account,
                cast(trim("Amount Received") AS DOUBLE) AS amount_received,
                {_clean_text(_quote_identifier("Receiving Currency"))} AS receiving_currency,
                cast(trim("Amount Paid") AS DOUBLE) AS amount_paid,
                {_clean_text(_quote_identifier("Payment Currency"))} AS payment_currency,
                {_clean_text(_quote_identifier("Payment Format"))} AS payment_format,
                cast(trim("Is Laundering") AS TINYINT) AS is_laundering
            FROM raw_transactions
        ), canonical AS (
            SELECT
                {_sql_literal(source_name)} || ':row-' || cast(source_row_number AS VARCHAR)
                    AS transaction_id,
                {_sql_literal(source_name)} AS source_file,
                source_row_number,
                timestamp,
                {_normalized_bank("from_bank_raw")} AS from_bank,
                from_account,
                {_normalized_bank("to_bank_raw")} AS to_bank,
                to_account,
                amount_received,
                receiving_currency,
                amount_paid,
                payment_currency,
                payment_format,
                is_laundering,
                from_bank_raw,
                to_bank_raw,
                {_normalized_bank("from_bank_raw")} || '::' || from_account AS from_node_id,
                {_normalized_bank("to_bank_raw")} || '::' || to_account AS to_node_id
            FROM normalized
        )
        SELECT *
        FROM canonical
        ORDER BY timestamp, source_row_number
    """
    _copy_parquet(connection, query, destination, compression)
    connection.execute(
        f"CREATE VIEW canonical AS SELECT * FROM read_parquet({_sql_path(destination)})"
    )


def _canonical_report(
    connection: duckdb.DuckDBPyConnection,
    expected_rows: int,
    exact_duplicate_rows: int,
) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT
            count(*),
            count(DISTINCT transaction_id),
            min(source_row_number),
            max(source_row_number),
            min(timestamp),
            max(timestamp),
            sum(CASE WHEN is_laundering = 0 THEN 1 ELSE 0 END),
            sum(CASE WHEN is_laundering = 1 THEN 1 ELSE 0 END),
            sum(CASE WHEN amount_paid IS NULL OR NOT isfinite(amount_paid) OR amount_paid < 0
                THEN 1 ELSE 0 END),
            sum(CASE WHEN amount_received IS NULL OR NOT isfinite(amount_received)
                         OR amount_received < 0 THEN 1 ELSE 0 END)
        FROM canonical
        """
    ).fetchone()
    assert row is not None
    actual_rows, unique_ids = int(row[0]), int(row[1])
    if actual_rows != expected_rows or unique_ids != expected_rows:
        raise FullPipelineError(
            f"Canonical row/identity mismatch: rows={actual_rows}, unique_ids={unique_ids}, "
            f"expected={expected_rows}"
        )
    if (int(row[2]), int(row[3])) != (2, expected_rows + 1):
        raise FullPipelineError("Canonical source_row_number no longer covers every physical row")
    if int(row[8]) or int(row[9]):
        raise FullPipelineError("Canonical amount validation failed after conversion")
    return {
        "stage": "canonical_transactions",
        "row_count": actual_rows,
        "valid": True,
        "error_count": 0,
        "warning_count": 1 if exact_duplicate_rows else 0,
        "issues": (
            [
                {
                    "code": "exact_duplicate_events_preserved",
                    "severity": "warning",
                    "message": "Exact event duplicates are reported and remain preserved",
                    "count": exact_duplicate_rows,
                }
            ]
            if exact_duplicate_rows
            else []
        ),
        "statistics": {
            "duplicate_transaction_id_count": actual_rows - unique_ids,
            "exact_duplicate_rows_after_first": exact_duplicate_rows,
            "timestamp_minimum": row[4].isoformat(),
            "timestamp_maximum": row[5].isoformat(),
            "label_counts": {"0": int(row[6]), "1": int(row[7])},
            "amount_paid_invalid_count": int(row[8]),
            "amount_received_invalid_count": int(row[9]),
        },
    }


def _account_reference_report(
    connection: duckdb.DuckDBPyConnection,
    account_rows: int,
) -> dict[str, Any]:
    bank = _normalized_bank(_clean_text(_quote_identifier("Bank ID")))
    account = f"upper({_clean_text(_quote_identifier('Account Number'))})"
    connection.execute(
        f"""
        CREATE TABLE account_nodes AS
        SELECT {bank} || '::' || {account} AS node_id
        FROM raw_accounts
        """
    )
    account_counts = connection.execute(
        "SELECT count(*), count(DISTINCT node_id) FROM account_nodes"
    ).fetchone()
    assert account_counts is not None
    unique_account_nodes = int(account_counts[1])
    if int(account_counts[0]) != unique_account_nodes:
        raise FullPipelineError(
            f"Accounts table contains {int(account_counts[0]) - unique_account_nodes} "
            "duplicate composite node IDs"
        )
    connection.execute(
        """
        CREATE TABLE transaction_nodes AS
        SELECT from_node_id AS node_id FROM canonical
        UNION
        SELECT to_node_id AS node_id FROM canonical
        """
    )
    transaction_nodes = int(
        connection.execute("SELECT count(*) FROM transaction_nodes").fetchone()[0]
    )
    matched = int(
        connection.execute(
            """
            SELECT count(*)
            FROM transaction_nodes transaction_node
            INNER JOIN account_nodes account USING (node_id)
            """
        ).fetchone()[0]
    )
    unmatched_sender_rows = int(
        connection.execute(
            """
            SELECT count(*)
            FROM canonical transaction
            WHERE NOT EXISTS (
                SELECT 1 FROM account_nodes account
                WHERE account.node_id = transaction.from_node_id
            )
            """
        ).fetchone()[0]
    )
    unmatched_receiver_rows = int(
        connection.execute(
            """
            SELECT count(*)
            FROM canonical transaction
            WHERE NOT EXISTS (
                SELECT 1 FROM account_nodes account
                WHERE account.node_id = transaction.to_node_id
            )
            """
        ).fetchone()[0]
    )
    unmatched_nodes = transaction_nodes - matched
    if unmatched_nodes or unmatched_sender_rows or unmatched_receiver_rows:
        raise FullPipelineError(
            "Canonical transaction endpoints do not all exist in the accounts table: "
            f"unmatched_nodes={unmatched_nodes}, sender_rows={unmatched_sender_rows}, "
            f"receiver_rows={unmatched_receiver_rows}"
        )
    report = {
        "account_rows": account_rows,
        "unique_account_nodes": unique_account_nodes,
        "transaction_nodes": transaction_nodes,
        "matched_transaction_nodes": matched,
        "unmatched_transaction_nodes": unmatched_nodes,
        "unmatched_sender_rows": unmatched_sender_rows,
        "unmatched_receiver_rows": unmatched_receiver_rows,
        "unused_account_nodes": unique_account_nodes - matched,
        "valid": True,
    }
    connection.execute("DROP TABLE transaction_nodes")
    connection.execute("DROP TABLE account_nodes")
    connection.execute("DROP TABLE raw_accounts")
    return report


def _timestamp_literal(value: datetime) -> str:
    return f"TIMESTAMP {_sql_literal(value.isoformat(sep=' '))}"


def _split_expression_and_request(
    connection: duckdb.DuckDBPyConnection,
    split_config: Mapping[str, Any],
    groups: Sequence[tuple[datetime, int, int]],
) -> tuple[str, dict[str, Any]]:
    if len(groups) < 3:
        raise FullPipelineError(
            "At least three distinct timestamps are required for a three-way split"
        )
    train_end = split_config.get("train_end")
    validation_end = split_config.get("validation_end")
    if train_end is not None or validation_end is not None:
        if train_end is None or validation_end is None:
            raise FullPipelineError(
                "split.train_end and split.validation_end must be provided together"
            )
        train_boundary = connection.execute(
            "SELECT cast(? AS TIMESTAMP)", [str(train_end)]
        ).fetchone()[0]
        validation_boundary = connection.execute(
            "SELECT cast(? AS TIMESTAMP)", [str(validation_end)]
        ).fetchone()[0]
        if train_boundary >= validation_boundary:
            raise FullPipelineError("split.train_end must be strictly earlier than validation_end")
        expression = (
            f"CASE WHEN timestamp <= {_timestamp_literal(train_boundary)} THEN 'train' "
            f"WHEN timestamp <= {_timestamp_literal(validation_boundary)} THEN 'validation' "
            "ELSE 'test' END"
        )
        requested = {
            "mode": "explicit_boundaries",
            "train_end": train_boundary.isoformat(),
            "validation_end": validation_boundary.isoformat(),
        }
        return expression, requested

    train_fraction = float(split_config["train_fraction"])
    validation_fraction = float(split_config["validation_fraction"])
    test_fraction = float(split_config["test_fraction"])
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise FullPipelineError("Chronological split fractions must be positive and sum to 1.0")
    total_rows = sum(row_count for _, row_count, _ in groups)
    cumulative: list[int] = []
    running = 0
    for _, row_count, _ in groups:
        running += row_count
        cumulative.append(running)
    group_total = len(groups)
    train_groups = bisect.bisect_left(cumulative, total_rows * train_fraction) + 1
    validation_groups = (
        bisect.bisect_left(cumulative, total_rows * (train_fraction + validation_fraction)) + 1
    )
    train_groups = min(max(train_groups, 1), group_total - 2)
    validation_groups = min(max(validation_groups, train_groups + 1), group_total - 1)
    validation_start = groups[train_groups][0]
    test_start = groups[validation_groups][0]
    expression = (
        f"CASE WHEN timestamp < {_timestamp_literal(validation_start)} THEN 'train' "
        f"WHEN timestamp < {_timestamp_literal(test_start)} THEN 'validation' "
        "ELSE 'test' END"
    )
    requested = {
        "mode": "fractions_snapped_to_timestamp_groups",
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }
    return expression, requested


def _build_split_metadata(
    connection: duckdb.DuckDBPyConnection,
    split_config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    raw_groups = connection.execute(
        """
        SELECT timestamp, count(*) AS rows, cast(sum(is_laundering) AS BIGINT) AS positives
        FROM canonical
        GROUP BY timestamp
        ORDER BY timestamp
        """
    ).fetchall()
    groups = [(row[0], int(row[1]), int(row[2])) for row in raw_groups]
    expression, requested = _split_expression_and_request(connection, split_config, groups)
    summaries = connection.execute(
        f"""
        WITH assigned AS (
            SELECT timestamp, is_laundering, {expression} AS partition
            FROM canonical
        )
        SELECT
            partition,
            count(*) AS rows,
            min(timestamp) AS minimum_timestamp,
            max(timestamp) AS maximum_timestamp,
            count(DISTINCT timestamp) AS unique_timestamps,
            cast(sum(is_laundering) AS BIGINT) AS positive_labels
        FROM assigned
        GROUP BY partition
        """
    ).fetchall()
    by_name = {
        str(row[0]): {
            "rows": int(row[1]),
            "minimum_timestamp": row[2].isoformat(),
            "maximum_timestamp": row[3].isoformat(),
            "unique_timestamps": int(row[4]),
            "positive_labels": int(row[5]),
        }
        for row in summaries
    }
    if set(by_name) != {"train", "validation", "test"}:
        raise FullPipelineError(f"Chronological split produced empty partitions: {sorted(by_name)}")
    if not (
        by_name["train"]["maximum_timestamp"]
        < by_name["validation"]["minimum_timestamp"]
        < by_name["test"]["minimum_timestamp"]
    ):
        raise FullPipelineError("Chronological split timestamps are not strictly separated")
    total_rows = sum(partition["rows"] for partition in by_name.values())
    metadata = {
        "strategy": "chronological",
        "timestamp_groups_kept_intact": True,
        "strict_boundaries_verified": True,
        "no_transaction_overlap_verified": True,
        "requested": requested,
        "total_rows": total_rows,
        "partitions": {name: by_name[name] for name in ("train", "validation", "test")},
    }
    return expression, metadata


def _window_spec(value: object) -> tuple[str, str]:
    match = _WINDOW_SETTING.fullmatch(str(value))
    if match is None:
        raise FullPipelineError(f"Unsupported fixed history window: {value!r}")
    magnitude, abbreviation = match.groups()
    abbreviation = abbreviation.casefold()
    unit = {
        "us": "microsecond",
        "ms": "millisecond",
        "s": "second",
        "min": "minute",
        "h": "hour",
        "d": "day",
        "w": "week",
    }[abbreviation]
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    return slug, f"INTERVAL {_sql_literal(f'{magnitude} {unit}')}"


def _history_columns(windows: Sequence[tuple[str, str]]) -> tuple[list[str], list[str]]:
    sender = [
        "sender_previous_transaction_count",
        "sender_previous_outgoing_amount",
        "sender_previous_unique_counterparties",
    ]
    receiver = [
        "receiver_previous_transaction_count",
        "receiver_previous_incoming_amount",
        "receiver_previous_unique_counterparties",
    ]
    for slug, _ in windows:
        sender.extend([f"sender_burst_count_{slug}", f"sender_rolling_outgoing_amount_{slug}"])
        receiver.extend(
            [f"receiver_burst_count_{slug}", f"receiver_rolling_incoming_amount_{slug}"]
        )
    return sender, receiver


def _build_role_history(
    connection: duckdb.DuckDBPyConnection,
    *,
    prefix: str,
    entity_column: str,
    counterparty_column: str,
    amount_column: str,
    volume_label: str,
    windows: Sequence[tuple[str, str]],
) -> None:
    batch_table = f"{prefix}_batches"
    new_table = f"{prefix}_new_counterparties"
    history_table = f"{prefix}_history"
    connection.execute(
        f"""
        CREATE TABLE {batch_table} AS
        SELECT
            {entity_column} AS entity_id,
            timestamp,
            cast(count(*) AS BIGINT) AS batch_transaction_count,
            cast(sum({amount_column}) AS DOUBLE) AS batch_amount
        FROM canonical
        GROUP BY {entity_column}, timestamp
        ORDER BY entity_id, timestamp
        """
    )
    connection.execute(
        f"""
        CREATE TABLE {new_table} AS
        SELECT entity_id, first_seen AS timestamp, cast(count(*) AS BIGINT) AS new_counterparties
        FROM (
            SELECT
                {entity_column} AS entity_id,
                {counterparty_column} AS counterparty_id,
                min(timestamp) AS first_seen
            FROM canonical
            GROUP BY {entity_column}, {counterparty_column}
        ) first_seen_pairs
        GROUP BY entity_id, first_seen
        ORDER BY entity_id, timestamp
        """
    )
    rolling_expressions: list[str] = []
    for slug, interval in windows:
        frame = (
            f"PARTITION BY b.entity_id ORDER BY b.timestamp RANGE BETWEEN {interval} PRECEDING "
            "AND INTERVAL '1 microsecond' PRECEDING"
        )
        rolling_expressions.extend(
            [
                "cast(coalesce(sum(b.batch_transaction_count) over "
                f"({frame}), 0) AS BIGINT) AS {prefix}_burst_count_{slug}",
                "cast(coalesce(sum(b.batch_amount) over "
                f"({frame}), 0.0) AS DOUBLE) AS {prefix}_rolling_{volume_label}_{slug}",
            ]
        )
    rolling_sql = ",\n                ".join(rolling_expressions)
    connection.execute(
        f"""
        CREATE TABLE {history_table} AS
        SELECT
            b.entity_id,
            b.timestamp,
            epoch(
                b.timestamp - lag(b.timestamp) OVER (
                    PARTITION BY b.entity_id ORDER BY b.timestamp
                )
            ) AS seconds_since_previous,
            cast(coalesce(sum(b.batch_transaction_count) OVER (
                PARTITION BY b.entity_id ORDER BY b.timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS BIGINT) AS {prefix}_previous_transaction_count,
            cast(coalesce(sum(b.batch_amount) OVER (
                PARTITION BY b.entity_id ORDER BY b.timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0.0) AS DOUBLE) AS {prefix}_previous_{volume_label},
            cast(coalesce(sum(coalesce(n.new_counterparties, 0)) OVER (
                PARTITION BY b.entity_id ORDER BY b.timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS BIGINT) AS {prefix}_previous_unique_counterparties,
            {rolling_sql}
        FROM {batch_table} b
        LEFT JOIN {new_table} n USING (entity_id, timestamp)
        ORDER BY b.entity_id, b.timestamp
        """
    )


def _build_graph_history(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE node_timestamps AS
        SELECT from_node_id AS node_id, timestamp FROM canonical
        UNION
        SELECT to_node_id AS node_id, timestamp FROM canonical
        """
    )
    connection.execute(
        """
        CREATE TABLE node_degree_history AS
        SELECT
            q.node_id,
            q.timestamp,
            cast(coalesce(sum(coalesce(out_update.new_counterparties, 0)) OVER (
                PARTITION BY q.node_id ORDER BY q.timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS BIGINT) AS prior_fan_out_degree,
            cast(coalesce(sum(coalesce(in_update.new_counterparties, 0)) OVER (
                PARTITION BY q.node_id ORDER BY q.timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS BIGINT) AS prior_fan_in_degree
        FROM node_timestamps q
        LEFT JOIN sender_new_counterparties out_update
            ON q.node_id = out_update.entity_id AND q.timestamp = out_update.timestamp
        LEFT JOIN receiver_new_counterparties in_update
            ON q.node_id = in_update.entity_id AND q.timestamp = in_update.timestamp
        ORDER BY q.node_id, q.timestamp
        """
    )
    connection.execute(
        """
        CREATE TABLE pair_history AS
        WITH pair_batches AS (
            SELECT
                from_node_id,
                to_node_id,
                timestamp,
                cast(count(*) AS BIGINT) AS pair_batch_count
            FROM canonical
            GROUP BY from_node_id, to_node_id, timestamp
        )
        SELECT
            from_node_id,
            to_node_id,
            timestamp,
            cast(coalesce(sum(pair_batch_count) OVER (
                PARTITION BY from_node_id, to_node_id ORDER BY timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS BIGINT) AS pair_previous_transfer_count
        FROM pair_batches
        ORDER BY from_node_id, to_node_id, timestamp
        """
    )


def _feature_query(
    sender_history_columns: Sequence[str],
    receiver_history_columns: Sequence[str],
) -> str:
    canonical = ",\n            ".join(
        f"c.{_quote_identifier(column)}" for column in CANONICAL_COLUMNS
    )
    sender = ",\n            ".join(
        f"sh.{_quote_identifier(column)}" for column in sender_history_columns
    )
    receiver = ",\n            ".join(
        f"rh.{_quote_identifier(column)}" for column in receiver_history_columns
    )
    return f"""
        SELECT
            {canonical},
            ln(1.0 + c.amount_paid) AS log_amount_paid,
            ln(1.0 + c.amount_received) AS log_amount_received,
            cast(c.from_bank = c.to_bank AS TINYINT) AS same_bank,
            cast(upper(trim(c.payment_currency)) = upper(trim(c.receiving_currency)) AS TINYINT)
                AS currency_match,
            CASE
                WHEN upper(trim(c.payment_currency)) = upper(trim(c.receiving_currency))
                THEN c.amount_paid - c.amount_received
                ELSE NULL
            END AS amount_difference_same_currency,
            CASE
                WHEN upper(trim(c.payment_currency)) = upper(trim(c.receiving_currency))
                     AND c.amount_received > 0
                THEN c.amount_paid / c.amount_received
                ELSE NULL
            END AS amount_ratio_same_currency,
            cast(extract(hour FROM c.timestamp) AS TINYINT) AS hour,
            cast(extract(isodow FROM c.timestamp) - 1 AS TINYINT) AS day_of_week,
            cast(extract(isodow FROM c.timestamp) >= 6 AS TINYINT) AS is_weekend,
            sh.seconds_since_previous AS sender_seconds_since_previous,
            rh.seconds_since_previous AS receiver_seconds_since_previous,
            {sender},
            {receiver},
            sender_degree.prior_fan_out_degree AS sender_prior_fan_out_degree,
            sender_degree.prior_fan_in_degree AS sender_prior_fan_in_degree,
            receiver_degree.prior_fan_out_degree AS receiver_prior_fan_out_degree,
            receiver_degree.prior_fan_in_degree AS receiver_prior_fan_in_degree,
            pair.pair_previous_transfer_count
        FROM canonical c
        INNER JOIN sender_history sh
            ON c.from_node_id = sh.entity_id AND c.timestamp = sh.timestamp
        INNER JOIN receiver_history rh
            ON c.to_node_id = rh.entity_id AND c.timestamp = rh.timestamp
        INNER JOIN node_degree_history sender_degree
            ON c.from_node_id = sender_degree.node_id AND c.timestamp = sender_degree.timestamp
        INNER JOIN node_degree_history receiver_degree
            ON c.to_node_id = receiver_degree.node_id AND c.timestamp = receiver_degree.timestamp
        INNER JOIN pair_history pair
            ON c.from_node_id = pair.from_node_id
            AND c.to_node_id = pair.to_node_id
            AND c.timestamp = pair.timestamp
    """


def _copy_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    destination: Path,
    compression: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        connection.execute(
            f"COPY ({query}) TO {_sql_path(temporary)} "
            f"(FORMAT PARQUET, COMPRESSION {compression.upper()})"
        )
        temporary.replace(destination)
    finally:
        # DuckDB can leave a zero-byte or partial file after an OOM.  Only the
        # exact UUID path owned by this call is eligible for cleanup.
        temporary.unlink(missing_ok=True)


def _core_output_paths(tables_dir: Path) -> dict[str, Path]:
    return {
        "clean": tables_dir / "transactions_clean.parquet",
        "features": tables_dir / "transaction_features.parquet",
        "split": tables_dir / "split_manifest.parquet",
        "graph_edges": tables_dir / "graph_edges.parquet",
    }


def _export_split_and_graph_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    outputs: Mapping[str, Path],
    split_expression: str,
    compression: str,
) -> None:
    _copy_parquet(
        connection,
        f"""
        SELECT transaction_id, timestamp, {split_expression} AS partition
        FROM canonical
        ORDER BY timestamp, transaction_id
        """,
        outputs["split"],
        compression,
    )
    _copy_parquet(
        connection,
        """
        SELECT
            transaction_id,
            timestamp,
            from_node_id,
            to_node_id,
            amount_paid,
            amount_received,
            is_laundering
        FROM canonical
        ORDER BY timestamp, source_row_number
        """,
        outputs["graph_edges"],
        compression,
    )


def _verify_materialized_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    expected_rows: int,
    expected_columns: Sequence[str],
    sender_history_columns: Sequence[str],
    receiver_history_columns: Sequence[str],
) -> dict[str, Any]:
    actual_columns = tuple(row[0] for row in connection.execute("DESCRIBE feature_rows").fetchall())
    if actual_columns != tuple(expected_columns):
        raise FullPipelineError(
            "Materialized feature schema/order mismatch: "
            f"expected={list(expected_columns)}, actual={list(actual_columns)}"
        )

    required_columns = (
        *CANONICAL_COLUMNS,
        "log_amount_paid",
        "log_amount_received",
        "same_bank",
        "currency_match",
        "hour",
        "day_of_week",
        "is_weekend",
        *sender_history_columns,
        *receiver_history_columns,
        *GRAPH_FEATURE_COLUMNS,
    )
    nonnegative_columns = (
        "amount_paid",
        "amount_received",
        "log_amount_paid",
        "log_amount_received",
        *sender_history_columns,
        *receiver_history_columns,
        *GRAPH_FEATURE_COLUMNS,
    )
    required_null_predicate = " OR ".join(
        f"{_quote_identifier(column)} IS NULL" for column in required_columns
    )
    invalid_nonnegative_predicate = " OR ".join(
        "("
        + _quote_identifier(column)
        + " IS NULL OR NOT isfinite(cast("
        + _quote_identifier(column)
        + " AS DOUBLE)) OR "
        + _quote_identifier(column)
        + " < 0)"
        for column in nonnegative_columns
    )
    row = connection.execute(
        f"""
        SELECT
            count(*) AS rows,
            count(DISTINCT transaction_id) AS unique_ids,
            count(*) FILTER (WHERE {required_null_predicate}) AS required_null_rows,
            count(*) FILTER (WHERE
                is_laundering NOT IN (0, 1)
                OR same_bank NOT IN (0, 1)
                OR currency_match NOT IN (0, 1)
                OR is_weekend NOT IN (0, 1)
                OR hour NOT BETWEEN 0 AND 23
                OR day_of_week NOT BETWEEN 0 AND 6
            ) AS invalid_domain_rows,
            count(*) FILTER (WHERE {invalid_nonnegative_predicate})
                AS invalid_nonnegative_rows,
            count(*) FILTER (WHERE
                (sender_seconds_since_previous IS NOT NULL AND (
                    NOT isfinite(sender_seconds_since_previous)
                    OR sender_seconds_since_previous <= 0
                ))
                OR (receiver_seconds_since_previous IS NOT NULL AND (
                    NOT isfinite(receiver_seconds_since_previous)
                    OR receiver_seconds_since_previous <= 0
                ))
            ) AS invalid_time_gap_rows,
            count(*) FILTER (WHERE
                (amount_difference_same_currency IS NOT NULL
                    AND NOT isfinite(amount_difference_same_currency))
                OR (amount_ratio_same_currency IS NOT NULL AND (
                    NOT isfinite(amount_ratio_same_currency)
                    OR amount_ratio_same_currency < 0
                ))
            ) AS invalid_optional_numeric_rows
        FROM feature_rows
        """
    ).fetchone()
    assert row is not None
    report = {
        "status": "PASS",
        "rows": int(row[0]),
        "unique_transaction_ids": int(row[1]),
        "columns": len(actual_columns),
        "required_null_rows": int(row[2]),
        "invalid_domain_rows": int(row[3]),
        "invalid_nonnegative_rows": int(row[4]),
        "invalid_time_gap_rows": int(row[5]),
        "invalid_optional_numeric_rows": int(row[6]),
    }
    failures = {
        key: value
        for key, value in report.items()
        if key not in {"status", "rows", "unique_transaction_ids", "columns"} and value
    }
    if report["rows"] != expected_rows:
        failures["rows"] = report["rows"]
    if report["unique_transaction_ids"] != expected_rows:
        failures["unique_transaction_ids"] = report["unique_transaction_ids"]
    if failures:
        raise FullPipelineError(f"Materialized feature invariants failed: {failures}")
    return report


def _verify_outputs(
    connection: duckdb.DuckDBPyConnection,
    outputs: Mapping[str, Path],
    expected_rows: int,
    expected_feature_columns: Sequence[str],
    split_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {key: _sql_path(value) for key, value in outputs.items()}
    counts: dict[str, int] = {}
    unique_counts: dict[str, int] = {}
    for name in ("clean", "features", "split", "graph_edges"):
        row = connection.execute(
            f"""
            SELECT count(*), count(DISTINCT transaction_id)
            FROM read_parquet({paths[name]})
            """
        ).fetchone()
        assert row is not None
        counts[name] = int(row[0])
        unique_counts[name] = int(row[1])
        if counts[name] != expected_rows or unique_counts[name] != expected_rows:
            raise FullPipelineError(
                f"{name} output row/identity mismatch: rows={counts[name]}, "
                f"unique_ids={unique_counts[name]}, expected={expected_rows}"
            )

    for name in ("features", "split", "graph_edges"):
        mismatch = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM (
                    (SELECT transaction_id FROM read_parquet({paths["clean"]})
                     EXCEPT SELECT transaction_id FROM read_parquet({paths[name]}))
                    UNION ALL
                    (SELECT transaction_id FROM read_parquet({paths[name]})
                     EXCEPT SELECT transaction_id FROM read_parquet({paths["clean"]}))
                ) differences
                """
            ).fetchone()[0]
        )
        if mismatch:
            raise FullPipelineError(f"{name} output has {mismatch} transaction identity mismatches")

    feature_columns = tuple(
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({paths['features']})"
        ).fetchall()
    )
    if feature_columns != tuple(expected_feature_columns):
        raise FullPipelineError(
            "Feature Parquet schema/order mismatch: "
            f"expected={list(expected_feature_columns)}, actual={list(feature_columns)}"
        )
    split_rows = connection.execute(
        f"""
        SELECT
            split.partition,
            count(*),
            min(split.timestamp),
            max(split.timestamp),
            sum(clean.is_laundering)
        FROM read_parquet({paths["split"]}) split
        INNER JOIN read_parquet({paths["clean"]}) clean USING (transaction_id)
        GROUP BY split.partition
        """
    ).fetchall()
    saved_split = {str(row[0]): row for row in split_rows}
    for name, expected in split_metadata["partitions"].items():
        actual = saved_split.get(name)
        if actual is None or int(actual[1]) != int(expected["rows"]):
            raise FullPipelineError(f"Saved split row count disagrees for {name}")
        if actual[2].isoformat() != expected["minimum_timestamp"]:
            raise FullPipelineError(f"Saved split minimum timestamp disagrees for {name}")
        if actual[3].isoformat() != expected["maximum_timestamp"]:
            raise FullPipelineError(f"Saved split maximum timestamp disagrees for {name}")
        if int(actual[4]) != int(expected["positive_labels"]):
            raise FullPipelineError(f"Saved split positive-label count disagrees for {name}")
    return {
        "status": "PASS",
        "method": "DuckDB counts, distinct counts, EXCEPT identity checks, and split aggregates",
        "no_python_identifier_sets": True,
        "rows": counts,
        "unique_transaction_ids": unique_counts,
        "feature_columns": len(feature_columns),
        "strict_chronology": True,
        "no_row_overlap": True,
    }


def _run_full_eda(
    connection: duckdb.DuckDBPyConnection,
    *,
    enabled: bool,
    canonical_path: Path,
    output_dir: Path,
    sample_size: int,
    top_n: int,
    random_seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not enabled:
        return {
            "status": "SKIPPED_BY_CONFIG",
            "full_exact": False,
            "sample_descriptive_only": False,
        }
    try:
        from argus.full_eda import generate_full_eda_artifacts
    except ModuleNotFoundError as exc:
        if exc.name == "argus.full_eda":
            raise FullPipelineError(
                "Full EDA implementation is unavailable; refusing to label the full run PASS"
            ) from exc
        raise
    return generate_full_eda_artifacts(
        connection,
        canonical_path,
        output_dir,
        sample_size=sample_size,
        top_n=top_n,
        random_seed=random_seed,
        provenance=provenance,
    )


def _artifact_inventory(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if path.resolve() == manifest_path.resolve() or path.name == ".gitkeep":
            continue
        if any(part.startswith(_WORK_PREFIX) for part in relative.parts):
            continue
        inventory.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def _safe_cleanup_work_directory(work_dir: Path, run_dir: Path) -> None:
    resolved_work = work_dir.resolve()
    resolved_run = run_dir.resolve()
    if resolved_work.parent != resolved_run or not resolved_work.name.startswith(_WORK_PREFIX):
        raise FullPipelineError(f"Refusing to remove unexpected work directory: {resolved_work}")
    if resolved_work.exists():
        shutil.rmtree(resolved_work)


def run_full_sprint1_pipeline(
    config_path: str | Path = "configs/full.yaml",
) -> FullPipelineRunResult:
    """Run exact full-data Sprint 1 processing without materializing pandas copies."""

    started = time.perf_counter()
    started_at = datetime.now(UTC)
    recorder = _StageRecorder()
    config = load_config(config_path)
    _assert_full_contract(config)
    settings = _full_settings(config)
    transaction_source = get_path(config, "raw_data")
    account_source = get_path(config, "raw_accounts")
    run_dir = get_path(config, "run_dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = run_dir / "tables"
    metadata_dir = run_dir / "metadata"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"

    _read_exact_header(transaction_source, RAW_TRANSACTION_HEADER, "HI-Small transaction")
    _read_exact_header(account_source, ACCOUNT_HEADER, "HI-Small accounts")

    work_dir = run_dir / f"{_WORK_PREFIX}{uuid.uuid4().hex}"
    work_dir.mkdir(parents=False, exist_ok=False)
    work_database = work_dir / "pipeline.duckdb"
    spill_dir = work_dir / "spill"
    connection: duckdb.DuckDBPyConnection | None = None
    pipeline_succeeded = False

    try:
        with recorder.measure("source_fingerprints"):
            source_provenance = {
                "dataset": "IBM AML-Data HI-Small",
                "dataset_is_synthetic": True,
                "locally_generated_fixture": False,
                "analysis_scope": "full transaction file",
                "transactions": _source_fingerprint(transaction_source),
                "accounts": _source_fingerprint(account_source),
            }

        with recorder.measure("duckdb_initialization"):
            connection = duckdb.connect(str(work_database))
            _configure_connection(connection, settings, spill_dir)

        with recorder.measure("single_threaded_ingestion"):
            transaction_rows, account_rows = _ingest_csvs(
                connection, transaction_source, account_source
            )
            connection.execute(f"SET threads={int(settings['threads'])}")
            # Physical source identity is now fixed in each raw table's rowid.
            # Every material output has an explicit ORDER BY, so retaining global
            # insertion order for blocking full-data operators only wastes memory.
            connection.execute("SET preserve_insertion_order=false")

        with recorder.measure("raw_validation"):
            raw_report = _validate_raw_transactions(connection, transaction_rows)
            raw_accounts_report = _validate_raw_accounts(connection, account_rows)

        clean_path = tables_dir / "transactions_clean.parquet"
        with recorder.measure("canonical_external_sort_and_export"):
            _export_canonical(
                connection,
                clean_path,
                transaction_source.name,
                settings["parquet_compression"],
            )
            canonical_report = _canonical_report(
                connection,
                transaction_rows,
                raw_report["statistics"]["exact_duplicate_rows_after_first"],
            )
            connection.execute("DROP TABLE raw_transactions")

        with recorder.measure("full_account_reference_check"):
            account_report = _account_reference_report(connection, account_rows)

        with recorder.measure("chronological_split"):
            split_config = config["split"]
            split_expression, split_metadata = _build_split_metadata(connection, split_config)
            if split_metadata["total_rows"] != transaction_rows:
                raise FullPipelineError("Split metadata does not cover all canonical rows")
            _write_json(split_metadata, metadata_dir / "split_metadata.json")

        configured_windows = tuple(config["features"]["history_windows"])
        windows = tuple(_window_spec(window) for window in configured_windows)
        sender_history_columns, receiver_history_columns = _history_columns(windows)

        with recorder.measure("sender_history_features"):
            _build_role_history(
                connection,
                prefix="sender",
                entity_column="from_node_id",
                counterparty_column="to_node_id",
                amount_column="amount_paid",
                volume_label="outgoing_amount",
                windows=windows,
            )

        with recorder.measure("receiver_history_features"):
            _build_role_history(
                connection,
                prefix="receiver",
                entity_column="to_node_id",
                counterparty_column="from_node_id",
                amount_column="amount_received",
                volume_label="incoming_amount",
                windows=windows,
            )

        with recorder.measure("directed_graph_history_features"):
            _build_graph_history(connection)

        feature_query = _feature_query(sender_history_columns, receiver_history_columns)
        expected_feature_columns = (
            *CANONICAL_COLUMNS,
            *TRANSACTION_FEATURE_COLUMNS,
            *TIME_FEATURE_COLUMNS,
            *sender_history_columns,
            *receiver_history_columns,
            *GRAPH_FEATURE_COLUMNS,
        )
        with recorder.measure("feature_join_materialization"):
            connection.execute(f"CREATE TABLE feature_rows AS {feature_query}")

        with recorder.measure("materialized_feature_verification"):
            materialized_feature_report = _verify_materialized_features(
                connection,
                expected_rows=transaction_rows,
                expected_columns=expected_feature_columns,
                sender_history_columns=sender_history_columns,
                receiver_history_columns=receiver_history_columns,
            )

        outputs = _core_output_paths(tables_dir)
        with recorder.measure("sorted_feature_export"):
            _copy_parquet(
                connection,
                "SELECT * FROM feature_rows ORDER BY timestamp, source_row_number",
                outputs["features"],
                settings["parquet_compression"],
            )
            connection.execute("DROP TABLE feature_rows")

        with recorder.measure("split_and_graph_exports"):
            _export_split_and_graph_tables(
                connection,
                outputs=outputs,
                split_expression=split_expression,
                compression=settings["parquet_compression"],
            )

        project = config["project"]
        eda_config = config.get("eda", {})
        with recorder.measure("full_and_sampled_eda"):
            eda_manifest = _run_full_eda(
                connection,
                enabled=settings["run_eda"],
                canonical_path=clean_path,
                output_dir=run_dir / "eda",
                sample_size=settings["eda_sample_rows"],
                top_n=int(eda_config.get("top_n", 20)),
                random_seed=int(project["random_seed"]),
                provenance=source_provenance,
            )

        with recorder.measure("sql_output_verification"):
            verification = _verify_outputs(
                connection,
                outputs,
                transaction_rows,
                expected_feature_columns,
                split_metadata,
            )

        _write_json(raw_report, run_dir / "raw_validation_report.json")
        _write_json(raw_accounts_report, run_dir / "raw_accounts_validation_report.json")
        _write_json(canonical_report, run_dir / "canonical_validation_report.json")
        _write_json(account_report, run_dir / "account_reference_report.json")
        _write_json(config, run_dir / "resolved_config.json")
        _write_json(verification, run_dir / "verification_report.json")

        connection.close()
        connection = None
        with recorder.measure("work_file_cleanup"):
            _safe_cleanup_work_directory(work_dir, run_dir)

        history_columns = [*sender_history_columns, *receiver_history_columns]
        manifest: dict[str, Any] = {
            "status": "PASS",
            "sprint": "Sprint 1 - Repository Foundation + Data Proof",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "runtime_seconds": 0.0,
            "runtime_seconds_by_stage": recorder.seconds,
            "python": {
                "version": sys.version.split()[0],
                "implementation": platform.python_implementation(),
            },
            "engine": {
                "name": "DuckDB",
                "version": duckdb.__version__,
                "persistent_temporary_database": True,
                "external_spill_enabled": True,
                "memory_limit": settings["memory_limit"],
                "threads_after_ingestion": settings["threads"],
                "ingestion_threads": 1,
                "preserve_insertion_order_during_ingestion": True,
                "preserve_insertion_order_during_processing": False,
                "source_row_number_definition": "raw table rowid + 2",
                "max_temp_directory_size": settings["max_temp_directory_size"],
                "parquet_compression": settings["parquet_compression"],
                "work_files_cleaned_after_success": True,
            },
            "provenance": source_provenance,
            "configuration": {
                "file": Path(config_path).name,
                "mode": project.get("mode"),
                "random_seed": int(project["random_seed"]),
                "loaded_transaction_rows": transaction_rows,
                "loaded_account_rows": account_rows,
                "feature_engineering_sampled": False,
            },
            "validation": {
                "raw_transactions": raw_report,
                "raw_accounts": raw_accounts_report,
                "canonical": canonical_report,
                "accounts": account_report,
                "materialized_features": materialized_feature_report,
                "saved_outputs": verification,
            },
            "preprocessing": {
                "canonical_rows": transaction_rows,
                "exact_duplicate_rows_observed": raw_report["statistics"][
                    "exact_duplicate_rows_after_first"
                ],
                "exact_duplicate_policy": "preserve_all",
                "external_sort": ["timestamp", "source_row_number"],
                "raw_file_order_assumed_chronological": False,
                "composite_node_delimiter": "::",
                "bank_id_normalization": "strip leading zeroes; all-zero becomes 0",
            },
            "split": split_metadata,
            "features": {
                "rows": transaction_rows,
                "columns": len(expected_feature_columns),
                "transaction": list(TRANSACTION_FEATURE_COLUMNS),
                "time": list(TIME_FEATURE_COLUMNS),
                "history": history_columns,
                "graph_history": list(GRAPH_FEATURE_COLUMNS),
                "strictly_prior_timestamp_batches": True,
                "rolling_interval": "[t-window, t)",
                "sampled": False,
                "join_materialized_before_sort": True,
                "materialized_table_dropped_after_sorted_export": True,
                "parquet_order": ["timestamp", "source_row_number"],
            },
            "eda": {
                "status": eda_manifest.get("status"),
                "feature_engineering_sampled": False,
                "sample_rows_requested": settings["eda_sample_rows"],
                "full_exact": eda_manifest.get("scopes", {}).get("full_exact"),
                "sample_descriptive_only": eda_manifest.get("scopes", {}).get(
                    "sample_descriptive_only"
                ),
                "manifest_path": "eda/full_eda_manifest.json" if settings["run_eda"] else None,
            },
            "models": "NOT_IN_SPRINT_1",
            "artifacts": [],
        }
        with recorder.measure("artifact_hash_inventory"):
            manifest["artifacts"] = _artifact_inventory(run_dir, manifest_path)
            manifest["artifact_count_excluding_manifest"] = len(manifest["artifacts"])
        manifest["runtime_seconds_by_stage"] = recorder.seconds
        manifest["runtime_seconds"] = time.perf_counter() - started
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(manifest, manifest_path)
        pipeline_succeeded = True
        print(
            f"[full] PASS: {transaction_rows:,} rows, {len(expected_feature_columns)} columns, "
            f"{manifest['runtime_seconds']:.2f}s",
            flush=True,
        )
        return FullPipelineRunResult(run_dir, manifest_path, manifest)
    finally:
        if connection is not None:
            connection.close()
        if not pipeline_succeeded and work_dir.exists():
            print(
                f"[full] Work files retained after failure for diagnosis: {work_dir}",
                flush=True,
            )
