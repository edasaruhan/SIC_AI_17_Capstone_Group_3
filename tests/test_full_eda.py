from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from argus.full_eda import generate_full_eda_artifacts


def _canonical_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i:02d}" for i in range(12)],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-02 00:00",
                    "2024-01-02 01:00",
                    "2024-01-02 02:00",
                    "2024-01-03 00:00",
                    "2024-01-03 01:00",
                    "2024-01-03 02:00",
                    "2024-01-04 00:00",
                    "2024-01-04 01:00",
                    "2024-01-04 02:00",
                ]
            ),
            "from_bank": ["1", "1", "1", "2", "2", "3", "3", "4", "4", "5", "5", "6"],
            "to_bank": ["2", "2", "2", "1", "3", "2", "4", "3", "5", "4", "6", "5"],
            "from_node_id": ["A", "A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F"],
            "to_node_id": ["B", "B", "B", "A", "C", "B", "D", "C", "E", "D", "F", "E"],
            "amount_paid": [10.0, 20.0, 30.0, 5.0, 8.0, 7.0, 9.0, 4.0, 6.0, 3.0, 2.0, 1.0],
            "amount_received": [10.0, 20.0, 30.0, 5.0, 8.0, 7.0, 9.0, 4.0, 6.0, 3.0, 2.0, 1.0],
            "payment_format": ["ACH", "Wire"] * 6,
            "payment_currency": ["US Dollar"] * 8 + ["Euro"] * 4,
            "receiving_currency": ["US Dollar"] * 8 + ["Euro"] * 4,
            "is_laundering": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        }
    )


def _write_parquet(connection: duckdb.DuckDBPyConnection, frame: pd.DataFrame, path: Path) -> None:
    connection.from_df(frame).write_parquet(str(path))


@pytest.fixture
def full_eda_runs(tmp_path: Path) -> tuple[pd.DataFrame, dict, dict, Path, Path]:
    frame = _canonical_fixture()
    first_source = tmp_path / "ordered.parquet"
    second_source = tmp_path / "reversed.parquet"
    first_output = tmp_path / "ordered_eda"
    second_output = tmp_path / "reversed_eda"
    connection = duckdb.connect()
    try:
        _write_parquet(connection, frame, first_source)
        _write_parquet(connection, frame.iloc[::-1].reset_index(drop=True), second_source)
        first = generate_full_eda_artifacts(
            connection,
            first_source,
            first_output,
            sample_size=5,
            top_n=3,
            provenance={"fixture": "ordered"},
        )
        second = generate_full_eda_artifacts(
            connection,
            second_source,
            second_output,
            sample_size=5,
            top_n=3,
            provenance={"fixture": "reversed"},
        )
    finally:
        connection.close()
    return frame, first, second, first_output, second_output


def test_full_exact_tables_cover_the_population(full_eda_runs) -> None:
    frame, manifest, _, output, _ = full_eda_runs
    tables = output / "full_exact" / "tables"

    classes = pd.read_csv(tables / "class_distribution.csv")
    activity = pd.read_csv(tables / "activity_over_time.csv")
    payment_formats = pd.read_csv(tables / "payment_format_distribution.csv")
    currencies = pd.read_csv(tables / "currency_distribution.csv")
    banks = pd.read_csv(tables / "bank_activity.csv")
    degrees = pd.read_csv(tables / "node_degree_counterparties.csv")
    repeated = pd.read_csv(tables / "repeated_sender_receiver_pairs.csv")

    assert int(classes["count"].sum()) == len(frame)
    assert int(activity["transactions"].sum()) == len(frame)
    assert int(payment_formats["count"].sum()) == len(frame)
    assert currencies.groupby("role")["count"].sum().to_dict() == {
        "payment": len(frame),
        "receiving": len(frame),
    }
    assert banks.groupby("role")["count"].sum().to_dict() == {
        "receiver": len(frame),
        "sender": len(frame),
    }
    assert int(degrees["in_edge_count"].sum()) == len(frame)
    assert int(degrees["out_edge_count"].sum()) == len(frame)
    assert repeated.iloc[0][["from_node_id", "to_node_id", "transfer_count"]].tolist() == [
        "A",
        "B",
        3,
    ]
    assert manifest["population_rows"] == len(frame)


def test_stable_hash_sample_is_independent_of_parquet_row_order(full_eda_runs) -> None:
    _, first, second, first_output, second_output = full_eda_runs
    first_scope = first["scopes"]["sample_descriptive_only"]
    second_scope = second["scopes"]["sample_descriptive_only"]
    first_ids = pd.read_csv(first_output / "sample_descriptive" / "sample_transaction_ids.csv")
    second_ids = pd.read_csv(second_output / "sample_descriptive" / "sample_transaction_ids.csv")

    assert first_scope["actual_rows"] == 5
    assert second_scope["actual_rows"] == 5
    assert first_scope["sampling_order_sql"] == "hash(transaction_id), transaction_id"
    assert first_ids.to_dict("records") == second_ids.to_dict("records")
    assert (
        first_scope["sample_transaction_id_sha256"] == second_scope["sample_transaction_id_sha256"]
    )


def test_manifests_truthfully_separate_exact_and_sampled_scopes(full_eda_runs) -> None:
    frame, manifest, _, output, _ = full_eda_runs
    exact_manifest = json.loads(
        (output / "full_exact" / "full_exact_manifest.json").read_text(encoding="utf-8")
    )
    sampled_profile = json.loads(
        (output / "sample_descriptive" / "dataset_profile.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "PASS"
    assert manifest["models"] == "NOT_IN_SPRINT_1"
    assert manifest["scopes"]["full_exact"]["rows"] == len(frame)
    assert exact_manifest["scope"] == "full_exact"
    assert all(item["scope"] == "full_exact" for item in exact_manifest["artifacts"])
    assert sampled_profile["provenance"]["artifact_scope"] == "sample_descriptive_only"
    rationale = manifest["scopes"]["sample_descriptive_only"]["sampling_rationale"]
    assert rationale["selection_independent_of_target"] is True
    assert rationale["full_exact_policy"].startswith("All feasible tabular aggregates")
    assert rationale["sampled_operations"] == [
        "plotting",
        "NetworkX component analysis",
        "NetworkX local-subgraph analysis",
    ]
    assert "not full-population graph estimates" in rationale["interpretation"]
    assert sampled_profile["provenance"]["sampling_rationale"] == rationale
    assert manifest["sampling_rationale"] == rationale
    assert sampled_profile["analysis_scope"]["input_rows"] == 5
    assert sampled_profile["analysis_scope"]["graph_edges_analyzed"] == 5
    assert {item["scope"] for item in manifest["artifacts"]} == {
        "full_exact",
        "sample_descriptive_only",
    }
