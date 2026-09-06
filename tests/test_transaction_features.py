from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argus.features.transaction import TransactionFeatureError, add_transaction_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount_paid": [100.0, 50.0],
            "amount_received": [80.0, 25.0],
            "from_bank": ["A", "A"],
            "to_bank": ["A", "B"],
            "payment_currency": ["USD", "USD"],
            "receiving_currency": [" usd ", "EUR"],
        }
    )


def test_amount_comparison_only_when_currency_matches() -> None:
    result = add_transaction_features(_frame())

    assert result["currency_match"].tolist() == [1, 0]
    assert result["same_bank"].tolist() == [1, 0]
    assert result.loc[0, "amount_difference_same_currency"] == 20.0
    assert result.loc[0, "amount_ratio_same_currency"] == 1.25
    assert np.isnan(result.loc[1, "amount_difference_same_currency"])
    assert np.isnan(result.loc[1, "amount_ratio_same_currency"])


def test_negative_amount_is_rejected_not_clipped() -> None:
    frame = _frame()
    frame.loc[0, "amount_paid"] = -1

    with pytest.raises(TransactionFeatureError, match="non-negative"):
        add_transaction_features(frame)
