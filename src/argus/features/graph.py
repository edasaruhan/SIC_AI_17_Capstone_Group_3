"""Strictly-prior directed fan-in/fan-out graph features."""

from __future__ import annotations

import pandas as pd


class GraphFeatureError(ValueError):
    """Raised when directed graph-history features cannot be generated."""


def add_graph_history_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add directed degree and repeated-pair facts known strictly before ``t``.

    Repeated transfers remain individual rows. Degree increments only when a new
    directed neighbor is first observed, while ``pair_previous_transfer_count``
    counts all earlier multiedges between the ordered sender/receiver pair.
    """

    required = {"timestamp", "from_node_id", "to_node_id"}
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise GraphFeatureError(f"Missing required columns: {missing}")

    result = transactions.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    if result["timestamp"].isna().any():
        raise GraphFeatureError("timestamp contains unparseable values")
    result["_argus_row_order"] = range(len(result))

    # Build direction-specific frames explicitly so the grouped node/neighbor
    # columns cannot alias one another.
    out_edges = result[["from_node_id", "to_node_id", "timestamp"]].rename(
        columns={"from_node_id": "node_id", "to_node_id": "neighbor_id"}
    )
    in_edges = result[["to_node_id", "from_node_id", "timestamp"]].rename(
        columns={"to_node_id": "node_id", "from_node_id": "neighbor_id"}
    )

    def timeline_for(edges: pd.DataFrame, name: str) -> pd.DataFrame:
        first_edges = (
            edges.groupby(["node_id", "neighbor_id"], sort=False, observed=True)["timestamp"]
            .min()
            .rename("_first_seen")
            .reset_index()
        )
        updates = (
            first_edges.groupby(["node_id", "_first_seen"], sort=False, observed=True)
            .size()
            .rename("_new_neighbors")
            .reset_index()
            .rename(columns={"_first_seen": "timestamp"})
        )
        queries = pd.concat(
            [
                result[["from_node_id", "timestamp"]].rename(columns={"from_node_id": "node_id"}),
                result[["to_node_id", "timestamp"]].rename(columns={"to_node_id": "node_id"}),
            ],
            ignore_index=True,
        ).drop_duplicates()
        timeline = queries.merge(
            updates,
            on=["node_id", "timestamp"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        timeline["_new_neighbors"] = timeline["_new_neighbors"].fillna(0).astype("int64")
        timeline = timeline.sort_values(["node_id", "timestamp"], kind="mergesort")
        timeline[name] = (
            timeline.groupby("node_id", sort=False, observed=True)["_new_neighbors"].cumsum()
            - timeline["_new_neighbors"]
        ).astype("int64")
        return timeline[["node_id", "timestamp", name]]

    out_timeline = timeline_for(out_edges, "prior_fan_out_degree")
    in_timeline = timeline_for(in_edges, "prior_fan_in_degree")

    for endpoint, label in (("from_node_id", "sender"), ("to_node_id", "receiver")):
        result = result.merge(
            out_timeline.rename(
                columns={
                    "node_id": endpoint,
                    "prior_fan_out_degree": f"{label}_prior_fan_out_degree",
                }
            ),
            on=[endpoint, "timestamp"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        result = result.merge(
            in_timeline.rename(
                columns={
                    "node_id": endpoint,
                    "prior_fan_in_degree": f"{label}_prior_fan_in_degree",
                }
            ),
            on=[endpoint, "timestamp"],
            how="left",
            validate="many_to_one",
            sort=False,
        )

    pair_batches = (
        result.groupby(
            ["from_node_id", "to_node_id", "timestamp"],
            sort=True,
            observed=True,
        )
        .size()
        .rename("_pair_batch_count")
        .reset_index()
        .sort_values(["from_node_id", "to_node_id", "timestamp"], kind="mergesort")
    )
    pair_batches["pair_previous_transfer_count"] = (
        pair_batches.groupby(["from_node_id", "to_node_id"], sort=False, observed=True)[
            "_pair_batch_count"
        ].cumsum()
        - pair_batches["_pair_batch_count"]
    ).astype("int64")
    result = result.merge(
        pair_batches[
            [
                "from_node_id",
                "to_node_id",
                "timestamp",
                "pair_previous_transfer_count",
            ]
        ],
        on=["from_node_id", "to_node_id", "timestamp"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    degree_columns = [
        "sender_prior_fan_out_degree",
        "sender_prior_fan_in_degree",
        "receiver_prior_fan_out_degree",
        "receiver_prior_fan_in_degree",
        "pair_previous_transfer_count",
    ]
    result[degree_columns] = result[degree_columns].fillna(0).astype("int64")
    return result.sort_values("_argus_row_order", kind="mergesort").drop(columns="_argus_row_order")
