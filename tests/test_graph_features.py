from __future__ import annotations

import pandas as pd

from argus.features.graph import add_graph_history_features


def test_graph_features_are_directed_and_strictly_prior() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4", "t5"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 03:00",
                ]
            ),
            "from_node_id": ["A", "A", "A", "B", "A"],
            "to_node_id": ["B", "C", "B", "A", "D"],
        }
    )

    result = add_graph_history_features(frame)

    assert result.loc[:1, "sender_prior_fan_out_degree"].tolist() == [0, 0]
    assert result.loc[2, "sender_prior_fan_out_degree"] == 2
    assert result.loc[2, "pair_previous_transfer_count"] == 1
    assert result.loc[3, "sender_prior_fan_out_degree"] == 0
    assert result.loc[3, "receiver_prior_fan_out_degree"] == 2
    assert result.loc[4, "sender_prior_fan_in_degree"] == 1
    assert result.loc[4, "sender_prior_fan_out_degree"] == 2
