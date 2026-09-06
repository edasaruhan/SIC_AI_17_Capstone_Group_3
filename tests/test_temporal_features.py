from __future__ import annotations

import pandas as pd

from argus.features.temporal import add_history_features, add_time_features


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 03:00:00",
                ]
            ),
            "from_node_id": ["A", "A", "A", "A"],
            "to_node_id": ["B", "C", "D", "B"],
            "amount_paid": [10.0, 20.0, 30.0, 40.0],
            "amount_received": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_same_timestamp_events_share_strictly_prior_state() -> None:
    result = add_history_features(_transactions(), windows=("2h",))

    assert result.loc[:1, "sender_previous_transaction_count"].tolist() == [0, 0]
    assert result.loc[:1, "sender_previous_outgoing_amount"].tolist() == [0.0, 0.0]
    assert result.loc[:1, "sender_previous_unique_counterparties"].tolist() == [0, 0]
    assert result.loc[2, "sender_previous_transaction_count"] == 2
    assert result.loc[2, "sender_previous_outgoing_amount"] == 30.0
    assert result.loc[2, "sender_previous_unique_counterparties"] == 2
    assert result.loc[2, "sender_burst_count_2h"] == 2
    # The interval is [t - 2h, t), so the event exactly two hours earlier is
    # included while the current event is excluded.
    assert result.loc[3, "sender_burst_count_2h"] == 1


def test_time_since_previous_is_strict_across_timestamp_batches() -> None:
    result = add_time_features(_transactions())

    assert result.loc[:1, "sender_seconds_since_previous"].isna().all()
    assert result.loc[2, "sender_seconds_since_previous"] == 3600.0
    assert result.loc[3, "sender_seconds_since_previous"] == 7200.0


def test_appending_future_event_does_not_change_past_features() -> None:
    original = _transactions()
    prior = add_history_features(original, windows=("24h",))
    future = pd.concat(
        [
            original,
            pd.DataFrame(
                {
                    "transaction_id": ["future"],
                    "timestamp": pd.to_datetime(["2025-01-01"]),
                    "from_node_id": ["A"],
                    "to_node_id": ["Z"],
                    "amount_paid": [999.0],
                    "amount_received": [999.0],
                }
            ),
        ],
        ignore_index=True,
    )
    recomputed = add_history_features(future, windows=("24h",)).iloc[: len(original)]

    feature_columns = [column for column in prior if column.startswith(("sender_", "receiver_"))]
    pd.testing.assert_frame_equal(
        prior[feature_columns].reset_index(drop=True),
        recomputed[feature_columns].reset_index(drop=True),
    )
