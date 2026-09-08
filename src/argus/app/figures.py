"""Deterministic Plotly figures built exclusively from saved case artifacts."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go


def _case_transactions(case: dict[str, Any]) -> list[dict[str, Any]]:
    transactions = case.get("transactions", [])
    return transactions if isinstance(transactions, list) else []


def _network_records(case: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    network = case.get("network", {})
    if not isinstance(network, dict):
        network = {}
    nodes = network.get("nodes", case.get("nodes", []))
    edges = network.get("edges", case.get("edges", []))
    nodes = nodes if isinstance(nodes, list) else []
    edges = edges if isinstance(edges, list) else []
    if edges:
        return nodes, edges

    derived_edges = []
    for transaction in _case_transactions(case):
        source = transaction.get(
            "source", transaction.get("from_node_id", transaction.get("sender"))
        )
        target = transaction.get(
            "target", transaction.get("to_node_id", transaction.get("receiver"))
        )
        if source is None or target is None:
            continue
        derived_edges.append(
            {
                "source": source,
                "target": target,
                "transaction_id": transaction.get("transaction_id"),
                "amount": transaction.get("amount", transaction.get("amount_paid")),
                "suspicious": transaction.get("suspicious", transaction.get("flagged", False)),
                "timestamp": transaction.get("timestamp"),
            }
        )
    return nodes, derived_edges


def build_network_figure(case: dict[str, Any]) -> go.Figure | None:
    """Create a deterministic directed network with explicit arrowheads."""

    nodes, edges = _network_records(case)
    node_ids = {
        str(value)
        for edge in edges
        for value in (
            edge.get("source", edge.get("from_node_id", edge.get("sender"))),
            edge.get("target", edge.get("to_node_id", edge.get("receiver"))),
        )
        if value is not None
    }
    node_ids.update(
        str(node.get("id", node.get("node_id")))
        for node in nodes
        if isinstance(node, dict) and node.get("id", node.get("node_id")) is not None
    )
    if not node_ids or not edges:
        return None

    ordered = sorted(node_ids)
    count = len(ordered)
    positions = {
        node_id: (
            math.cos((2 * math.pi * index / count) - math.pi / 2),
            math.sin((2 * math.pi * index / count) - math.pi / 2),
        )
        for index, node_id in enumerate(ordered)
    }
    node_metadata = {
        str(node.get("id", node.get("node_id"))): node
        for node in nodes
        if isinstance(node, dict) and node.get("id", node.get("node_id")) is not None
    }

    figure = go.Figure()
    annotations = []
    for edge in edges[:150]:
        source = edge.get("source", edge.get("from_node_id", edge.get("sender")))
        target = edge.get("target", edge.get("to_node_id", edge.get("receiver")))
        if source is None or target is None:
            continue
        source_id, target_id = str(source), str(target)
        if source_id not in positions or target_id not in positions:
            continue
        x0, y0 = positions[source_id]
        x1, y1 = positions[target_id]
        focal = bool(edge.get("is_focal", False))
        prioritized = bool(edge.get("suspicious", edge.get("is_high_risk", False)))
        color = "#f59e0b" if focal else "#ef4444" if prioritized else "#64748b"
        width = 3.0 if focal else 2.5 if prioritized else 1.2
        amount = edge.get("amount", edge.get("amount_paid", "N/A"))
        transaction_id = edge.get("transaction_id", "N/A")
        edge_role = "focal transaction" if focal else "strictly-prior context"
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"color": color, "width": width},
                hovertemplate=(
                    f"{source_id} → {target_id}<br>Transaction: {transaction_id}"
                    f"<br>Amount: {amount}<br>Role: {edge_role}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        annotations.append(
            {
                "ax": x0,
                "ay": y0,
                "x": x1,
                "y": y1,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 3,
                "arrowsize": 1.1,
                "arrowwidth": 1.5,
                "arrowcolor": color,
                "opacity": 0.8,
            }
        )

    seed_nodes = {str(case.get("sender_id", "")), str(case.get("receiver_id", ""))}
    node_colors = []
    hover = []
    for node_id in ordered:
        metadata = node_metadata.get(node_id, {})
        is_seed = node_id in seed_nodes or bool(metadata.get("is_seed", False))
        node_colors.append("#f59e0b" if is_seed else "#0f766e")
        hover.append(
            f"Account: {node_id}<br>Role: {metadata.get('role', 'network account')}"
            f"<br>Uncalibrated score: {metadata.get('risk_score', 'N/A')}"
        )
    figure.add_trace(
        go.Scatter(
            x=[positions[node_id][0] for node_id in ordered],
            y=[positions[node_id][1] for node_id in ordered],
            mode="markers+text",
            text=[node_id if len(node_id) <= 18 else f"{node_id[:15]}…" for node_id in ordered],
            textposition="bottom center",
            hovertext=hover,
            hoverinfo="text",
            marker={
                "size": 22,
                "color": node_colors,
                "line": {"width": 2, "color": "#f8fafc"},
            },
            showlegend=False,
        )
    )
    figure.update_layout(
        annotations=annotations,
        title="Directed transaction neighborhood",
        height=520,
        margin={"l": 15, "r": 15, "t": 55, "b": 15},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
    )
    return figure


def build_timeline_figure(case: dict[str, Any]) -> go.Figure | None:
    """Create a transaction timeline from saved timestamps and amounts."""

    records = _case_transactions(case)
    if not records:
        network = case.get("network", {})
        records = network.get("edges", []) if isinstance(network, dict) else []
    if not records:
        return None
    frame = pd.DataFrame(records)
    timestamp_column = next(
        (column for column in ("timestamp", "focal_timestamp", "time") if column in frame), None
    )
    if timestamp_column is None:
        return None
    frame["_timestamp"] = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    frame = frame.dropna(subset=["_timestamp"]).sort_values("_timestamp", kind="mergesort")
    if frame.empty:
        return None
    amount_column = next(
        (column for column in ("amount", "amount_paid", "total_amount") if column in frame), None
    )
    if amount_column is None:
        frame["_amount"] = 1.0
        y_title = "Transaction sequence"
    else:
        frame["_amount"] = pd.to_numeric(frame[amount_column], errors="coerce").fillna(0.0)
        y_title = "Amount"
    prioritized = frame.get("suspicious", frame.get("flagged", pd.Series(False, index=frame.index)))
    focal = frame.get("is_focal", pd.Series(False, index=frame.index)).map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    prioritized = prioritized.map(lambda value: bool(value) if pd.notna(value) else False)
    colors = [
        "#f59e0b" if bool(is_focal) else "#ef4444" if bool(is_prioritized) else "#0f766e"
        for is_focal, is_prioritized in zip(focal, prioritized, strict=False)
    ]
    transaction_ids = frame.get("transaction_id", pd.Series("N/A", index=frame.index)).astype(str)
    figure = go.Figure(
        go.Scatter(
            x=frame["_timestamp"],
            y=frame["_amount"],
            mode="lines+markers",
            marker={"size": 10, "color": colors},
            line={"color": "#94a3b8", "width": 1.5},
            text=transaction_ids,
            hovertemplate="%{x}<br>Amount: %{y:,.2f}<br>Transaction: %{text}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Case transaction timeline",
        height=330,
        margin={"l": 15, "r": 15, "t": 55, "b": 15},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Timestamp (UTC)",
        yaxis_title=y_title,
    )
    return figure


def build_model_metric_figure(
    comparison: pd.DataFrame, *, partition_label: str = "Validation"
) -> go.Figure:
    """Compare only metrics that are present in the saved comparison artifact."""

    metric_columns = [
        column
        for column in ("pr_auc", "recall_at_k", "precision_at_k", "fpr")
        if column in comparison and comparison[column].notna().any()
    ]
    figure = go.Figure()
    labels = comparison["model"].astype(str)
    if "version" in comparison:
        labels = comparison["version"].fillna(comparison["model"]).astype(str)
    for metric in metric_columns:
        figure.add_trace(
            go.Bar(name=metric.replace("_", " ").upper(), x=labels, y=comparison[metric])
        )
    figure.update_layout(
        title=f"{partition_label} model metrics",
        barmode="group",
        yaxis_title="Metric value",
        height=430,
        margin={"l": 15, "r": 15, "t": 55, "b": 90},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def build_pr_curve_figure(
    curves: pd.DataFrame, *, partition_label: str = "Validation"
) -> go.Figure:
    """Render saved PR-curve points; never reconstruct curves from aggregate metrics."""

    figure = go.Figure()
    for model, frame in curves.groupby("model", sort=False):
        ordered = frame.sort_values("recall", kind="mergesort")
        figure.add_trace(
            go.Scatter(
                x=ordered["recall"],
                y=ordered["precision"],
                name=str(model),
                mode="lines",
            )
        )
    figure.update_layout(
        title=f"Saved {partition_label.lower()} precision–recall curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=430,
        margin={"l": 15, "r": 15, "t": 55, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure
