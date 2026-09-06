from __future__ import annotations

import networkx as nx
import pandas as pd

from argus.graph.build import (
    build_transaction_graph,
    compute_graph_summary,
    extract_suspicious_local_subgraph,
)


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "from_node_id": ["A", "A", "B", "C"],
            "to_node_id": ["B", "B", "A", "D"],
            "amount_paid": [10.0, 20.0, 30.0, 40.0],
            "is_laundering": [0, 1, 0, 0],
        }
    )


def test_directed_multigraph_preserves_direction_and_repeated_edges() -> None:
    graph = build_transaction_graph(_edges())

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_edges("A", "B") == 2
    assert graph.number_of_edges("B", "A") == 1
    assert graph.number_of_edges() == len(_edges())
    summary = compute_graph_summary(graph)
    assert summary["unique_directed_pairs"] == 3
    assert summary["repeated_edge_occurrences_after_first"] == 1


def test_suspicious_subgraph_uses_actual_positive_edge() -> None:
    graph = build_transaction_graph(_edges())

    subgraph, evidence = extract_suspicious_local_subgraph(graph)

    assert evidence["available"] is True
    assert evidence["seed_transaction_id"] == "t2"
    assert subgraph.has_edge("A", "B", key="t2")
