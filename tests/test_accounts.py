from __future__ import annotations

import pandas as pd
import pytest

from argus.data.accounts import (
    AccountDataError,
    load_ibm_accounts,
    validate_account_references,
)


def test_accounts_normalize_zero_padded_transaction_bank_ids(tmp_path) -> None:
    source = tmp_path / "accounts.csv"
    pd.DataFrame(
        {
            "Bank Name": ["Bank A", "Bank B"],
            "Bank ID": ["10", "1"],
            "Account Number": ["ABC", "ABC"],
            "Entity ID": ["E1", "E2"],
            "Entity Name": ["Entity 1", "Entity 2"],
        }
    ).to_csv(source, index=False)

    accounts = load_ibm_accounts(source)
    transactions = pd.DataFrame({"from_node_id": ["10::ABC"], "to_node_id": ["1::ABC"]})
    report = validate_account_references(transactions, accounts)

    assert accounts["node_id"].tolist() == ["10::ABC", "1::ABC"]
    assert report.valid
    assert report.unmatched_sender_rows == 0
    assert report.unmatched_receiver_rows == 0


def test_account_number_alone_is_not_used_as_global_key(tmp_path) -> None:
    source = tmp_path / "accounts.csv"
    pd.DataFrame(
        {
            "Bank Name": ["Bank A", "Bank B"],
            "Bank ID": ["001", "002"],
            "Account Number": ["SAME", "SAME"],
            "Entity ID": ["E1", "E2"],
            "Entity Name": ["Entity 1", "Entity 2"],
        }
    ).to_csv(source, index=False)

    accounts = load_ibm_accounts(source)

    assert accounts["node_id"].is_unique
    assert set(accounts["node_id"]) == {"1::SAME", "2::SAME"}


def test_duplicate_composite_account_is_rejected(tmp_path) -> None:
    source = tmp_path / "accounts.csv"
    pd.DataFrame(
        {
            "Bank Name": ["Bank A", "Bank A"],
            "Bank ID": ["001", "1"],
            "Account Number": ["ABC", "ABC"],
            "Entity ID": ["E1", "E1"],
            "Entity Name": ["Entity 1", "Entity 1"],
        }
    ).to_csv(source, index=False)

    with pytest.raises(AccountDataError, match="duplicate composite"):
        load_ibm_accounts(source)
