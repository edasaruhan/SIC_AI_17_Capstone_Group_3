from __future__ import annotations

import json

import pandas as pd

from argus.eda import generate_eda_artifacts


def _canonical_frame() -> pd.DataFrame:
    rows = 18
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(rows)],
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="10min"),
            "from_bank": [str(i % 3) for i in range(rows)],
            "to_bank": [str((i + 1) % 3) for i in range(rows)],
            "from_node_id": [f"{i % 3}::A{i % 5}" for i in range(rows)],
            "to_node_id": [f"{(i + 1) % 3}::B{i % 4}" for i in range(rows)],
            "amount_paid": [float(i + 1) for i in range(rows)],
            "amount_received": [float(i + 1) for i in range(rows)],
            "payment_format": ["ACH" if i % 2 else "Wire" for i in range(rows)],
            "payment_currency": ["US Dollar"] * rows,
            "receiving_currency": ["US Dollar"] * rows,
            "is_laundering": [1 if i == 7 else 0 for i in range(rows)],
        }
    )


def test_eda_writes_real_machine_readable_and_png_artifacts(tmp_path) -> None:
    output = tmp_path / "eda"

    manifest = generate_eda_artifacts(
        _canonical_frame(),
        output,
        graph_max_edges=None,
        provenance={"dataset": "synthetic unit-test fixture", "synthetic_fixture": True},
    )

    assert manifest["status"] == "PASS"
    assert (output / "dataset_profile.json").is_file()
    assert (output / "tables" / "class_distribution.csv").is_file()
    assert (output / "tables" / "degree_distribution.csv").is_file()
    assert (output / "figures" / "class_distribution.png").stat().st_size > 0
    assert (output / "figures" / "suspicious_local_subgraph.png").stat().st_size > 0
    profile = json.loads((output / "dataset_profile.json").read_text(encoding="utf-8"))
    assert profile["dataset"]["rows"] == len(_canonical_frame())
    assert profile["dataset"]["positive_labels"] == 1
    assert profile["suspicious_subgraph"]["seed_transaction_id"] == "t7"
