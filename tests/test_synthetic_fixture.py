from __future__ import annotations

import csv

from scripts.generate_synthetic_fixture import generate_fixture


def test_fixture_is_deterministic_and_explicitly_synthetic(tmp_path) -> None:
    transactions = tmp_path / "synthetic_transactions.csv"
    accounts = tmp_path / "synthetic_accounts.csv"

    generate_fixture(transactions, accounts, rows=12)
    first_bytes = transactions.read_bytes()
    generate_fixture(transactions, accounts, rows=12)

    assert transactions.read_bytes() == first_bytes
    with transactions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0].count("Account") == 2
    assert len(rows) == 13
    assert "Synthetic" in accounts.read_text(encoding="utf-8")
