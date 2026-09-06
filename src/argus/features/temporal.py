"""Time and strictly-prior account-history feature engineering."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd


class TemporalFeatureError(ValueError):
    """Raised when temporal features cannot be computed safely."""


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise TemporalFeatureError(f"Missing required columns: {missing}")


def _validated_timestamps(frame: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        bad_count = int(timestamps.isna().sum())
        raise TemporalFeatureError(f"timestamp contains {bad_count} unparseable value(s)")
    return timestamps


def _restore_row_order(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("_argus_row_order", kind="mergesort").drop(columns="_argus_row_order")


def add_time_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and previous-event-gap features.

    Previous timestamps are calculated on distinct entity/timestamp pairs. As a
    result, transactions sharing a timestamp see the same strictly-prior state
    and cannot leak into one another through arbitrary row ordering.
    """

    _require_columns(transactions, {"timestamp", "from_node_id", "to_node_id"})
    result = transactions.copy()
    result["timestamp"] = _validated_timestamps(result)
    result["_argus_row_order"] = range(len(result))

    result["hour"] = result["timestamp"].dt.hour.astype("int8")
    result["day_of_week"] = result["timestamp"].dt.dayofweek.astype("int8")
    result["is_weekend"] = result["day_of_week"].ge(5).astype("int8")

    for entity_column, prefix in (
        ("from_node_id", "sender"),
        ("to_node_id", "receiver"),
    ):
        timeline = (
            result[[entity_column, "timestamp"]]
            .drop_duplicates()
            .sort_values([entity_column, "timestamp"], kind="mergesort")
        )
        previous_column = f"{prefix}_previous_timestamp"
        timeline[previous_column] = timeline.groupby(entity_column, sort=False, observed=True)[
            "timestamp"
        ].shift(1)
        result = result.merge(
            timeline,
            on=[entity_column, "timestamp"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        result[f"{prefix}_seconds_since_previous"] = (
            result["timestamp"] - result[previous_column]
        ).dt.total_seconds()
        result = result.drop(columns=previous_column)

    return _restore_row_order(result)


def _window_slug(window: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(window).strip().lower()).strip("_")
    if not slug:
        raise TemporalFeatureError(f"Invalid history window: {window!r}")
    return slug


def _parse_window(window: str) -> pd.Timedelta:
    """Parse explicit fixed-duration windows without ambiguous month units."""

    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(ns|us|ms|s|min|h|d|w)\s*",
        str(window),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise TemporalFeatureError(
            f"Invalid fixed history window {window!r}; examples: 30min, 1h, 7d"
        )
    magnitude, unit = match.groups()
    return pd.Timedelta(float(magnitude), unit=unit.lower())


def _add_role_history(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    counterparty_column: str,
    amount_column: str,
    prefix: str,
    volume_label: str,
    windows: Sequence[str],
) -> pd.DataFrame:
    grouped = (
        frame.groupby([entity_column, "timestamp"], sort=True, observed=True)
        .agg(
            _batch_transaction_count=(entity_column, "size"),
            _batch_amount=(amount_column, "sum"),
        )
        .reset_index()
        .sort_values([entity_column, "timestamp"], kind="mergesort")
    )

    grouped[f"{prefix}_previous_transaction_count"] = (
        grouped.groupby(entity_column, sort=False, observed=True)[
            "_batch_transaction_count"
        ].cumsum()
        - grouped["_batch_transaction_count"]
    ).astype("int64")
    grouped[f"{prefix}_previous_{volume_label}"] = (
        grouped.groupby(entity_column, sort=False, observed=True)["_batch_amount"].cumsum()
        - grouped["_batch_amount"]
    ).astype("float64")

    first_seen = (
        frame.groupby([entity_column, counterparty_column], sort=False, observed=True)["timestamp"]
        .min()
        .rename("_first_seen")
        .reset_index()
    )
    new_counterparties = (
        first_seen.groupby([entity_column, "_first_seen"], sort=False, observed=True)
        .size()
        .rename("_new_counterparties")
        .reset_index()
        .rename(columns={"_first_seen": "timestamp"})
    )
    grouped = grouped.merge(
        new_counterparties,
        on=[entity_column, "timestamp"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    grouped["_new_counterparties"] = grouped["_new_counterparties"].fillna(0).astype("int64")
    grouped[f"{prefix}_previous_unique_counterparties"] = (
        grouped.groupby(entity_column, sort=False, observed=True)["_new_counterparties"].cumsum()
        - grouped["_new_counterparties"]
    ).astype("int64")

    history_columns = [
        f"{prefix}_previous_transaction_count",
        f"{prefix}_previous_{volume_label}",
        f"{prefix}_previous_unique_counterparties",
    ]
    for window in windows:
        duration = _parse_window(window)
        if duration <= pd.Timedelta(0, unit="ns"):
            raise TemporalFeatureError(f"History window must be positive: {window!r}")

        slug = _window_slug(window)
        rolling = (
            grouped.groupby(entity_column, sort=False, observed=True)
            .rolling(str(window), on="timestamp", closed="left")[
                [
                    "_batch_transaction_count",
                    "_batch_amount",
                ]
            ]
            .sum()
            .reset_index()
            .rename(
                columns={
                    "_batch_transaction_count": f"{prefix}_burst_count_{slug}",
                    "_batch_amount": f"{prefix}_rolling_{volume_label}_{slug}",
                }
            )
        )
        rolling_columns = [
            f"{prefix}_burst_count_{slug}",
            f"{prefix}_rolling_{volume_label}_{slug}",
        ]
        grouped = grouped.merge(
            rolling[[entity_column, "timestamp", *rolling_columns]],
            on=[entity_column, "timestamp"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        grouped[f"{prefix}_burst_count_{slug}"] = (
            grouped[f"{prefix}_burst_count_{slug}"].fillna(0).astype("int64")
        )
        grouped[f"{prefix}_rolling_{volume_label}_{slug}"] = grouped[
            f"{prefix}_rolling_{volume_label}_{slug}"
        ].fillna(0.0)
        history_columns.extend(rolling_columns)

    return frame.merge(
        grouped[[entity_column, "timestamp", *history_columns]],
        on=[entity_column, "timestamp"],
        how="left",
        validate="many_to_one",
        sort=False,
    )


def add_history_features(
    transactions: pd.DataFrame,
    *,
    windows: Sequence[str] = ("1h", "24h"),
) -> pd.DataFrame:
    """Add strictly-prior sender and receiver history features.

    Computation happens at entity/timestamp batch granularity. This guarantees
    that an event at time ``t`` never observes itself or any peer event at the
    same timestamp. The implementation uses groupby/cumulative/rolling kernels,
    not a Python loop over transaction rows.
    """

    _require_columns(
        transactions,
        {
            "timestamp",
            "from_node_id",
            "to_node_id",
            "amount_paid",
            "amount_received",
        },
    )
    result = transactions.copy()
    result["timestamp"] = _validated_timestamps(result)
    result["amount_paid"] = pd.to_numeric(result["amount_paid"], errors="raise")
    result["amount_received"] = pd.to_numeric(result["amount_received"], errors="raise")
    result["_argus_row_order"] = range(len(result))

    result = _add_role_history(
        result,
        entity_column="from_node_id",
        counterparty_column="to_node_id",
        amount_column="amount_paid",
        prefix="sender",
        volume_label="outgoing_amount",
        windows=windows,
    )
    result = _add_role_history(
        result,
        entity_column="to_node_id",
        counterparty_column="from_node_id",
        amount_column="amount_received",
        prefix="receiver",
        volume_label="incoming_amount",
        windows=windows,
    )
    return _restore_row_order(result)
