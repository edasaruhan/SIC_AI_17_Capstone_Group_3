from __future__ import annotations

import csv
import re
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from argus.config import load_config
from argus.data.load import load_ibm_aml
from argus.data.preprocess import preprocess_transactions
from argus.features.graph import add_graph_history_features
from argus.features.temporal import add_history_features, add_time_features
from argus.features.transaction import add_transaction_features
from argus.full_pipeline import _copy_parquet, run_full_sprint1_pipeline

TRANSACTION_HEADER = [
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
]

ACCOUNT_HEADER = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]


def _write_unordered_fixture(transactions: Path, accounts: Path) -> None:
    # Physical order is intentionally non-chronological.  The repeated 03:00
    # transfer is an exact source event duplicate but must retain two identities.
    rows = [
        ("2024/01/01 03:00", "001", "A", "002", "B", "40", "USD", "40", "USD", "Wire", 1),
        ("2024/01/01 00:00", "001", "A", "002", "B", "10", "USD", "10", "USD", "ACH", 0),
        ("2024/01/01 01:00", "001", "A", "002", "B", "30", "USD", "30", "USD", "ACH", 0),
        ("2024/01/01 00:00", "001", "A", "003", "C", "20", "USD", "20", "USD", "ACH", 0),
        ("2024/01/01 02:00", "002", "B", "001", "A", "15", "EUR", "16", "USD", "Cheque", 0),
        ("2024/01/01 03:00", "001", "A", "002", "B", "40", "USD", "40", "USD", "Wire", 1),
        ("2024/01/02 00:00", "001", "A", "004", "D", "50", "USD", "50", "USD", "ACH", 0),
        ("2024/01/01 04:00", "003", "C", "001", "A", "12", "USD", "12", "USD", "Cash", 0),
        ("2024/01/08 00:00", "001", "A", "002", "B", "60", "USD", "60", "USD", "Wire", 0),
        ("2024/01/02 01:00", "002", "B", "003", "C", "22", "USD", "22", "USD", "ACH", 0),
        ("2024/01/03 00:00", "004", "D", "001", "A", "18", "USD", "18", "USD", "ACH", 0),
        ("2024/01/04 00:00", "001", "A", "003", "C", "25", "USD", "25", "USD", "ACH", 0),
        ("2024/01/05 00:00", "003", "C", "002", "B", "27", "USD", "27", "USD", "Wire", 0),
        ("2024/01/09 00:00", "002", "B", "004", "D", "31", "USD", "31", "USD", "ACH", 0),
        ("2024/01/10 00:00", "001", "A", "002", "B", "35", "USD", "35", "USD", "ACH", 0),
    ]
    with transactions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TRANSACTION_HEADER)
        writer.writerows(rows)
    with accounts.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(ACCOUNT_HEADER)
        for index, (bank, account) in enumerate(
            (("1", "A"), ("2", "B"), ("3", "C"), ("4", "D")), start=1
        ):
            writer.writerow([f"Bank {bank}", bank, account, f"entity-{index}", f"Entity {index}"])


def _test_config(tmp_path: Path, transactions: Path, accounts: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "full_test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "extends": str(project_root / "configs" / "full.yaml"),
                "paths": {
                    "raw_data": str(transactions),
                    "raw_accounts": str(accounts),
                    "run_dir": str(tmp_path / "artifacts"),
                },
                "full_pipeline": {
                    "engine": "duckdb",
                    "memory_limit": "256MB",
                    "threads": 1,
                    "max_temp_directory_size": "1GB",
                    "parquet_compression": "zstd",
                    "eda_sample_rows": 10,
                    "run_eda": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _pandas_reference(transactions: Path, config_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    raw = load_ibm_aml(transactions)
    canonical = preprocess_transactions(raw, column_mapping=dict(config["data"]["column_mapping"]))
    featured = add_transaction_features(canonical)
    featured = add_time_features(featured)
    featured = add_history_features(featured, windows=tuple(config["features"]["history_windows"]))
    return add_graph_history_features(featured)


def _read_parquet(path: Path) -> pd.DataFrame:
    with duckdb.connect(":memory:") as connection:
        return connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY timestamp, source_row_number", [str(path)]
        ).df()


def test_full_pipeline_matches_pandas_reference_and_preserves_temporal_contract(
    tmp_path: Path,
) -> None:
    transactions = tmp_path / "unordered_transactions.csv"
    accounts = tmp_path / "accounts.csv"
    _write_unordered_fixture(transactions, accounts)
    config_path = _test_config(tmp_path, transactions, accounts)

    result = run_full_sprint1_pipeline(config_path)
    expected = _pandas_reference(transactions, config_path)
    actual = _read_parquet(result.run_dir / "tables" / "transaction_features.parquet")

    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )

    first_batch = actual.loc[
        actual["timestamp"].eq(pd.Timestamp("2024-01-01 00:00")) & actual["from_node_id"].eq("1::A")
    ]
    assert len(first_batch) == 2
    assert first_batch["sender_previous_transaction_count"].eq(0).all()
    assert first_batch["sender_prior_fan_out_degree"].eq(0).all()

    duplicate_batch = actual.loc[
        actual["timestamp"].eq(pd.Timestamp("2024-01-01 03:00"))
        & actual["from_node_id"].eq("1::A")
        & actual["to_node_id"].eq("2::B")
    ]
    assert len(duplicate_batch) == 2
    assert duplicate_batch["sender_previous_transaction_count"].eq(3).all()
    assert duplicate_batch["pair_previous_transfer_count"].eq(2).all()

    one_hour_boundary = actual.loc[actual["transaction_id"].eq(f"{transactions.name}:row-4")].iloc[
        0
    ]
    assert one_hour_boundary["sender_burst_count_1h"] == 2

    assert actual["transaction_id"].is_unique
    assert result.manifest["preprocessing"]["exact_duplicate_rows_observed"] == 1
    assert result.manifest["features"]["rows"] == len(actual) == 15
    assert result.manifest["features"]["columns"] == 52
    assert result.manifest["features"]["sampled"] is False
    assert result.manifest["features"]["strictly_prior_timestamp_batches"] is True
    assert result.manifest["features"]["join_materialized_before_sort"] is True
    assert result.manifest["features"]["materialized_table_dropped_after_sorted_export"] is True
    assert result.manifest["features"]["parquet_order"] == [
        "timestamp",
        "source_row_number",
    ]

    materialized = result.manifest["validation"]["materialized_features"]
    assert materialized == {
        "status": "PASS",
        "rows": 15,
        "unique_transaction_ids": 15,
        "columns": 52,
        "required_null_rows": 0,
        "invalid_domain_rows": 0,
        "invalid_nonnegative_rows": 0,
        "invalid_time_gap_rows": 0,
        "invalid_optional_numeric_rows": 0,
    }
    stage_runtimes = result.manifest["runtime_seconds_by_stage"]
    assert stage_runtimes["feature_join_materialization"] >= 0
    assert stage_runtimes["materialized_feature_verification"] >= 0
    assert stage_runtimes["sorted_feature_export"] >= 0
    assert stage_runtimes["split_and_graph_exports"] >= 0

    clean = _read_parquet(result.run_dir / "tables" / "transactions_clean.parquet")
    with duckdb.connect(":memory:") as connection:
        physical_feature_order = connection.execute(
            "SELECT timestamp, source_row_number FROM read_parquet(?)",
            [str(result.run_dir / "tables" / "transaction_features.parquet")],
        ).fetchall()
        split = connection.execute(
            "SELECT * FROM read_parquet(?)",
            [str(result.run_dir / "tables" / "split_manifest.parquet")],
        ).df()
        graph = connection.execute(
            "SELECT * FROM read_parquet(?)",
            [str(result.run_dir / "tables" / "graph_edges.parquet")],
        ).df()
    expected_ids = set(clean["transaction_id"])
    assert set(actual["transaction_id"]) == expected_ids
    assert set(split["transaction_id"]) == expected_ids
    assert set(graph["transaction_id"]) == expected_ids
    assert split["transaction_id"].is_unique
    assert split.groupby("timestamp")["partition"].nunique().max() == 1
    assert physical_feature_order == sorted(physical_feature_order)
    assert sum(
        partition["rows"] for partition in result.manifest["split"]["partitions"].values()
    ) == len(actual)
    assert not list(result.run_dir.rglob("*.tmp.parquet"))
    assert not list(result.run_dir.glob(".argus_full_work_*"))


def test_atomic_parquet_copy_preserves_previous_output_and_cleans_partial_temp(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "existing.parquet"
    destination.write_bytes(b"previous-valid-output")

    class PartialCopyConnection:
        def execute(self, query: str) -> None:
            match = re.search(r"\bTO '((?:''|[^'])+)'", query)
            assert match is not None
            temporary = Path(match.group(1).replace("''", "'"))
            temporary.write_bytes(b"partial-output")
            raise RuntimeError("simulated COPY failure")

    with pytest.raises(RuntimeError, match="simulated COPY failure"):
        _copy_parquet(
            PartialCopyConnection(),  # type: ignore[arg-type]
            "SELECT 1 AS value",
            destination,
            "zstd",
        )

    assert destination.read_bytes() == b"previous-valid-output"
    assert not list(tmp_path.glob(".existing.parquet.*.tmp.parquet"))
