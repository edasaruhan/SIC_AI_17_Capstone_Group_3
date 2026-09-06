from __future__ import annotations

from argus.raw_audit import audit_hi_small_files
from scripts.generate_synthetic_fixture import generate_fixture


def test_full_file_audit_is_code_generated_from_fixture(tmp_path) -> None:
    transactions = tmp_path / "transactions.csv"
    accounts = tmp_path / "accounts.csv"
    generate_fixture(transactions, accounts, rows=24)

    report = audit_hi_small_files(
        transactions,
        accounts,
        dataset_label="synthetic IBM-shaped test fixture",
        dataset_is_synthetic=True,
        locally_generated_fixture=True,
    )

    assert report["provenance"]["dataset_is_synthetic"] is True
    assert report["provenance"]["locally_generated_fixture"] is True
    assert report["transactions"]["data_rows"] == 24
    assert report["transactions"]["timestamp"]["invalid_count"] == 0
    assert report["transactions"]["labels"] == {"0": 24}
    assert report["accounts"]["data_rows"] > 0
    assert report["cross_file"]["unmatched_transaction_nodes"] == 0
