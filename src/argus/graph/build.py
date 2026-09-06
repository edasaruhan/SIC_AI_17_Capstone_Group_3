"""Construct interpretable directed multigraphs without collapsing transfers."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


class GraphBuildError(ValueError):
    """Raised when canonical transfers cannot form a directed graph."""


def sample_transaction_frame(
    transactions: pd.DataFrame,
    *,
    max_edges: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return all rows or a deterministic uniform graph-analysis sample."""

    if max_edges is None or len(transactions) <= max_edges:
        return transactions.copy()
    if isinstance(max_edges, bool) or not isinstance(max_edges, int) or max_edges <= 0:
        raise GraphBuildError("max_edges must be null or a positive integer")
    sampled = transactions.sample(n=max_edges, random_state=random_seed, replace=False)
    order = [column for column in ("timestamp", "transaction_id") if column in sampled]
    return sampled.sort_values(order, kind="mergesort").reset_index(drop=True)


def build_transaction_graph(
    transactions: pd.DataFrame,
    *,
    max_edges: int | None = None,
    random_seed: int = 42,
) -> nx.MultiDiGraph:
    """Build a directed multigraph with one graph edge per transaction row."""

    required = {"from_node_id", "to_node_id"}
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise GraphBuildError(f"Missing graph endpoint columns: {missing}")
    sampled = sample_transaction_frame(transactions, max_edges=max_edges, random_seed=random_seed)
    graph = nx.MultiDiGraph(
        graph_sample_method=(
            "all_input_rows" if len(sampled) == len(transactions) else "uniform_without_replacement"
        ),
        input_rows=len(transactions),
        sampled_rows=len(sampled),
        random_seed=random_seed,
    )
    attribute_columns = [
        column
        for column in (
            "transaction_id",
            "source_row_number",
            "timestamp",
            "amount_paid",
            "amount_received",
            "payment_currency",
            "receiving_currency",
            "payment_format",
            "is_laundering",
        )
        if column in sampled
    ]
    for row_number, row in enumerate(sampled.itertuples(index=False), start=1):
        values = row._asdict()
        source = str(values["from_node_id"])
        target = str(values["to_node_id"])
        if not source or not target:
            raise GraphBuildError("Graph endpoint identifiers must not be empty")
        attributes: dict[str, Any] = {column: values[column] for column in attribute_columns}
        if "timestamp" in attributes and hasattr(attributes["timestamp"], "isoformat"):
            attributes["timestamp"] = attributes["timestamp"].isoformat()
        for column, value in list(attributes.items()):
            if pd.isna(value):
                attributes[column] = None
            elif hasattr(value, "item"):
                attributes[column] = value.item()
        key = str(values.get("transaction_id") or f"edge-{row_number}")
        if graph.has_edge(source, target, key=key):
            key = f"{key}#{row_number}"
        graph.add_edge(source, target, key=key, **attributes)
    return graph


def compute_graph_summary(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Compute reproducible directed/multiedge and component facts."""

    if not isinstance(graph, nx.MultiDiGraph) or not graph.is_directed():
        raise GraphBuildError("Expected a directed NetworkX MultiDiGraph")
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    unique_pairs = len({(source, target) for source, target in graph.edges()})
    weak_components = list(nx.weakly_connected_components(graph)) if node_count else []
    strong_components = list(nx.strongly_connected_components(graph)) if node_count else []
    in_degrees = [degree for _, degree in graph.in_degree()]
    out_degrees = [degree for _, degree in graph.out_degree()]
    return {
        "directed": True,
        "multigraph": True,
        "nodes": node_count,
        "edges": edge_count,
        "unique_directed_pairs": unique_pairs,
        "repeated_edge_occurrences_after_first": edge_count - unique_pairs,
        "self_loop_edges": nx.number_of_selfloops(graph),
        "weakly_connected_components": len(weak_components),
        "largest_weak_component_nodes": max(map(len, weak_components), default=0),
        "strongly_connected_components": len(strong_components),
        "largest_strong_component_nodes": max(map(len, strong_components), default=0),
        "maximum_in_degree": max(in_degrees, default=0),
        "maximum_out_degree": max(out_degrees, default=0),
        "mean_in_degree": edge_count / node_count if node_count else 0.0,
        "mean_out_degree": edge_count / node_count if node_count else 0.0,
        "graph_sample_method": graph.graph.get("graph_sample_method"),
        "input_rows": graph.graph.get("input_rows"),
        "sampled_rows": graph.graph.get("sampled_rows"),
        "random_seed": graph.graph.get("random_seed"),
    }


def extract_suspicious_local_subgraph(
    graph: nx.MultiDiGraph,
    *,
    radius: int = 1,
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    """Return a deterministic local graph around the first observed positive edge."""

    if radius < 1:
        raise GraphBuildError("radius must be at least 1")
    candidates: list[tuple[str, str, str, dict[str, Any]]] = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        if int(data.get("is_laundering") or 0) == 1:
            candidates.append((str(source), str(target), str(key), data))
    if not candidates:
        empty = nx.MultiDiGraph()
        return empty, {"available": False, "reason": "no_positive_label_in_graph_sample"}
    candidates.sort(key=lambda item: (str(item[3].get("timestamp", "")), item[2]))
    source, target, key, data = candidates[0]
    undirected = graph.to_undirected(as_view=True)
    nodes = set(nx.single_source_shortest_path_length(undirected, source, cutoff=radius))
    nodes.update(nx.single_source_shortest_path_length(undirected, target, cutoff=radius))
    subgraph = graph.subgraph(nodes).copy()
    return subgraph, {
        "available": True,
        "seed_transaction_id": data.get("transaction_id", key),
        "seed_source": source,
        "seed_target": target,
        "radius": radius,
        "nodes": subgraph.number_of_nodes(),
        "edges": subgraph.number_of_edges(),
        "interpretation": "labeled-positive local evidence for human review; not a guilt finding",
    }
