from __future__ import annotations

import pandas as pd

from argus.features.graph import add_graph_history_features
from argus.features.temporal import add_history_features


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00"]
            ),
            "from_node_id": ["A", "A", "B", "A"],
            "to_node_id": ["B", "C", "A", "B"],
            "amount_paid": [10.0, 20.0, 30.0, 40.0],
            "amount_received": [10.0, 20.0, 30.0, 40.0],
            "is_laundering": [0, 1, 0, 1],
        }
    )


def test_features_do_not_depend_on_target_values() -> None:
    original = _events()
    changed_labels = original.copy()
    changed_labels["is_laundering"] = 1 - changed_labels["is_laundering"]

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        return add_graph_history_features(add_history_features(frame, windows=("24h",)))

    left = build(original).drop(columns="is_laundering")
    right = build(changed_labels).drop(columns="is_laundering")
    pd.testing.assert_frame_equal(left, right)
