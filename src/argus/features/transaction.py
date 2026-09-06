"""Transaction-local features that do not depend on future observations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TransactionFeatureError(ValueError):
    """Raised when canonical inputs cannot safely produce transaction features."""


_REQUIRED_COLUMNS = {
    "amount_paid",
    "amount_received",
    "from_bank",
    "to_bank",
    "payment_currency",
    "receiving_currency",
}


def add_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with deterministic transaction-local features.

    Amount differences and ratios are only computed when both amounts use the
    same currency. Treating cross-currency values as directly comparable would
    require an exchange-rate table that is outside Sprint 1.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(transactions.columns))
    if missing:
        raise TransactionFeatureError(
            f"Cannot build transaction features; missing columns: {missing}"
        )

    result = transactions.copy()
    for column in ("amount_paid", "amount_received"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any() or (values < 0).any():
            raise TransactionFeatureError(
                f"Column {column!r} must contain finite, non-negative numbers"
            )
        result[column] = values.astype("float64")

    payment_currency = result["payment_currency"].astype("string").str.strip().str.upper()
    receiving_currency = result["receiving_currency"].astype("string").str.strip().str.upper()
    same_currency = payment_currency.eq(receiving_currency)

    result["log_amount_paid"] = np.log1p(result["amount_paid"])
    result["log_amount_received"] = np.log1p(result["amount_received"])
    result["same_bank"] = (
        result["from_bank"].astype("string").str.strip()
        == result["to_bank"].astype("string").str.strip()
    ).astype("int8")
    result["currency_match"] = same_currency.astype("int8")

    result["amount_difference_same_currency"] = np.where(
        same_currency,
        result["amount_paid"] - result["amount_received"],
        np.nan,
    )
    comparable_nonzero = same_currency & result["amount_received"].gt(0)
    result["amount_ratio_same_currency"] = np.where(
        comparable_nonzero,
        result["amount_paid"] / result["amount_received"],
        np.nan,
    )
    return result
