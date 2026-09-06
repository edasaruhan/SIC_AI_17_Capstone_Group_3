from __future__ import annotations

import pandas as pd
import pytest

from argus.data.preprocess import PreprocessingError, preprocess_transactions


def _raw_rows() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Timestamp": ["2022/09/01 00:01", "2022/09/01 00:00"],
            "From Bank": ["010", "001"],
            "Account": ["same", "same"],
            "To Bank": ["001", "010"],
            "Account.1": ["other", "other"],
            "Amount Received": ["10.0", "20.0"],
            "Receiving Currency": [" US Dollar ", "Euro"],
            "Amount Paid": ["10.0", "20.0"],
            "Payment Currency": ["US Dollar", "Euro"],
            "Payment Format": ["ACH", "Wire"],
            "Is Laundering": ["0", "1"],
        }
    )
    frame.attrs["source_name"] = "fixture.csv"
    return frame


def test_preprocessing_normalizes_composite_ids_and_order() -> None:
    result = preprocess_transactions(_raw_rows())

    assert result["timestamp"].is_monotonic_increasing
    assert result["from_bank"].tolist() == ["1", "10"]
    assert result["from_bank_raw"].tolist() == ["001", "010"]
    assert result["from_node_id"].tolist() == ["1::SAME", "10::SAME"]
    assert result["from_node_id"].nunique() == 2
    assert result["transaction_id"].is_unique
    assert result["is_laundering"].dtype == "Int8"


def test_exact_duplicate_events_are_reported_but_preserved() -> None:
    row = _raw_rows().iloc[[0]]
    duplicated = pd.concat([row, row], ignore_index=True)
    duplicated.attrs["source_name"] = "fixture.csv"

    result = preprocess_transactions(duplicated)

    assert len(result) == 2
    assert result["transaction_id"].nunique() == 2
    assert result.attrs["exact_duplicate_rows_observed"] == 1
    assert result.attrs["exact_duplicate_policy"] == "preserve_all"


def test_explicit_duplicate_drop_is_auditable() -> None:
    row = _raw_rows().iloc[[0]]
    duplicated = pd.concat([row, row], ignore_index=True)

    result = preprocess_transactions(duplicated, drop_exact_duplicates=True)

    assert len(result) == 1
    assert result.attrs["exact_duplicate_rows_observed"] == 1
    assert result.attrs["exact_duplicate_policy"] == "drop_after_first"


def test_invalid_target_is_rejected_before_nullable_integer_cast() -> None:
    raw = _raw_rows()
    raw.loc[0, "Is Laundering"] = "unknown"

    with pytest.raises(PreprocessingError, match="outside 0/1"):
        preprocess_transactions(raw)
