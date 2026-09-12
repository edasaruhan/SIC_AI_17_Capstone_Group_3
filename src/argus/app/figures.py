"""Deterministic Plotly figures built exclusively from saved case artifacts."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go

_COLORS = {
    "navy": "#123047",
    "teal": "#0B6B66",
    "teal_soft": "#4F8F8A",
    "amber": "#C97918",
    "red": "#B54747",
    "slate": "#73838F",
    "muted": "#A7B4BA",
    "paper": "#FBFCFA",
    "grid": "#DCE5E3",
    "white": "#FFFFFF",
}

_MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "graph_enhanced_lightgbm": "Graph-enhanced LightGBM",
    "refined_transaction_lightgbm": "Refined transaction LightGBM",
    "graphsage_edge_classifier": "GraphSAGE (research comparator)",
    "graphsage": "GraphSAGE (research comparator)",
}

_MODEL_COLORS = {
    "graph_enhanced_lightgbm": _COLORS["teal"],
    "refined_transaction_lightgbm": _COLORS["navy"],
    "graphsage_edge_classifier": _COLORS["slate"],
    "graphsage": _COLORS["slate"],
    "lightgbm": _COLORS["navy"],
}

_NODE_STYLES = {
    "focal_sender": (_COLORS["navy"], "diamond", 29, "Focal sender"),
    "focal_receiver": (_COLORS["teal"], "square", 29, "Focal receiver"),
    "focal_account": (_COLORS["amber"], "diamond", 29, "Focal account"),
    "context": (_COLORS["slate"], "circle", 21, "Network context"),
}

_EDGE_STYLES = {
    "focal": (_COLORS["amber"], 4.0, "Focal transfer"),
    "prioritized": (_COLORS["red"], 2.8, "Saved prioritized transfer"),
    "context": (_COLORS["teal_soft"], 1.6, "Earlier case context"),
}


def _display_model_name(value: Any) -> str:
    canonical = "_".join(str(value).strip().lower().replace("-", " ").split())
    return _MODEL_LABELS.get(canonical, str(value).replace("_", " ").strip().title())


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
                "currency": transaction.get("currency", transaction.get("payment_currency")),
                "suspicious": transaction.get("suspicious", transaction.get("flagged", False)),
                "is_focal": transaction.get("is_focal", False),
                "timestamp": transaction.get("timestamp"),
            }
        )
    return nodes, derived_edges


def _edge_endpoints(edge: dict[str, Any]) -> tuple[str | None, str | None]:
    source = edge.get("source", edge.get("from_node_id", edge.get("sender")))
    target = edge.get("target", edge.get("to_node_id", edge.get("receiver")))
    return (
        str(source) if source is not None else None,
        str(target) if target is not None else None,
    )


def _focal_accounts(
    case: dict[str, Any],
    edges: list[dict[str, Any]],
    node_metadata: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    source = case.get("sender_id")
    target = case.get("receiver_id")
    focal_edge = next((edge for edge in edges if bool(edge.get("is_focal", False))), None)
    if focal_edge is not None:
        edge_source, edge_target = _edge_endpoints(focal_edge)
        source = edge_source or source
        target = edge_target or target
    if source is None:
        source = next(
            (
                node_id
                for node_id, metadata in node_metadata.items()
                if metadata.get("role") == "focal_sender"
            ),
            None,
        )
    if target is None:
        target = next(
            (
                node_id
                for node_id, metadata in node_metadata.items()
                if metadata.get("role") == "focal_receiver"
            ),
            None,
        )
    return (
        str(source) if source is not None else None,
        str(target) if target is not None else None,
    )


def _deterministic_positions(
    ordered: list[str], focal_source: str | None, focal_target: str | None
) -> dict[str, tuple[float, float]]:
    if focal_source in ordered and focal_target in ordered and focal_source != focal_target:
        positions = {focal_source: (-0.72, 0.0), focal_target: (0.72, 0.0)}
        context = [node_id for node_id in ordered if node_id not in positions]
        if len(context) == 1:
            positions[context[0]] = (0.0, 0.92)
        else:
            for index, node_id in enumerate(context):
                angle = (2 * math.pi * index / max(1, len(context))) + (math.pi / 2)
                positions[node_id] = (1.20 * math.cos(angle), 0.92 * math.sin(angle))
        return positions

    count = len(ordered)
    return {
        node_id: (
            1.05 * math.cos((2 * math.pi * index / count) - math.pi / 2),
            0.90 * math.sin((2 * math.pi * index / count) - math.pi / 2),
        )
        for index, node_id in enumerate(ordered)
    }


def _curved_edge_points(
    start: tuple[float, float],
    end: tuple[float, float],
    offset: float,
    *,
    steps: int = 25,
) -> tuple[list[float], list[float]]:
    x0, y0 = start
    x1, y1 = end
    if start == end:
        radius = 0.22 + abs(offset)
        angles = [2 * math.pi * index / (steps - 1) for index in range(steps)]
        return (
            [x0 + radius * math.cos(angle) for angle in angles],
            [y0 + radius * math.sin(angle) + radius for angle in angles],
        )

    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    control_x = (x0 + x1) / 2 - (dy / length) * offset
    control_y = (y0 + y1) / 2 + (dx / length) * offset
    values = [index / (steps - 1) for index in range(steps)]
    x_values = [
        ((1 - value) ** 2 * x0) + (2 * (1 - value) * value * control_x) + (value**2 * x1)
        for value in values
    ]
    y_values = [
        ((1 - value) ** 2 * y0) + (2 * (1 - value) * value * control_y) + (value**2 * y1)
        for value in values
    ]
    return x_values, y_values


def _node_role(
    node_id: str,
    metadata: dict[str, Any],
    focal_source: str | None,
    focal_target: str | None,
) -> str:
    stored_role = str(metadata.get("role", "")).strip().lower()
    is_sender = node_id == focal_source or stored_role == "focal_sender"
    is_receiver = node_id == focal_target or stored_role == "focal_receiver"
    if is_sender and is_receiver:
        return "focal_account"
    if is_sender:
        return "focal_sender"
    if is_receiver:
        return "focal_receiver"
    if bool(metadata.get("is_seed", False)):
        return "focal_account"
    return "context"


def _display_amount(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A" if value is None else str(value)


def _series_from_columns(
    frame: pd.DataFrame, candidates: tuple[str, ...], *, default: Any = "N/A"
) -> pd.Series:
    column = next((name for name in candidates if name in frame), None)
    if column is None:
        return pd.Series(default, index=frame.index)
    return frame[column].fillna(default)


def build_network_figure(case: dict[str, Any]) -> go.Figure | None:
    """Create a deterministic, investigation-focused directed account network."""

    nodes, raw_edges = _network_records(case)
    edges = [edge for edge in raw_edges[:150] if isinstance(edge, dict)]
    node_ids = {value for edge in edges for value in _edge_endpoints(edge) if value is not None}
    node_ids.update(
        str(node.get("id", node.get("node_id")))
        for node in nodes
        if isinstance(node, dict) and node.get("id", node.get("node_id")) is not None
    )
    if not node_ids or not edges:
        return None

    ordered = sorted(node_ids)
    node_metadata = {
        str(node.get("id", node.get("node_id"))): node
        for node in nodes
        if isinstance(node, dict) and node.get("id", node.get("node_id")) is not None
    }
    focal_source, focal_target = _focal_accounts(case, edges, node_metadata)
    positions = _deterministic_positions(ordered, focal_source, focal_target)

    valid_edges = []
    pair_totals: dict[tuple[str, str], int] = {}
    for edge in edges:
        source_id, target_id = _edge_endpoints(edge)
        if source_id not in positions or target_id not in positions:
            continue
        valid_edges.append((edge, source_id, target_id))
        key = (source_id, target_id)
        pair_totals[key] = pair_totals.get(key, 0) + 1

    figure = go.Figure()
    annotations: list[dict[str, Any]] = []
    pair_seen: dict[tuple[str, str], int] = {}
    legend_roles: set[str] = set()
    for edge, source_id, target_id in valid_edges:
        key = (source_id, target_id)
        occurrence = pair_seen.get(key, 0)
        pair_seen[key] = occurrence + 1
        centered_index = occurrence - ((pair_totals[key] - 1) / 2)
        offset = centered_index * 0.14
        x_values, y_values = _curved_edge_points(positions[source_id], positions[target_id], offset)

        focal = bool(edge.get("is_focal", False))
        prioritized = bool(edge.get("suspicious", edge.get("is_high_risk", False)))
        style_key = "focal" if focal else "prioritized" if prioritized else "context"
        color, width, edge_role = _EDGE_STYLES[style_key]
        amount = _display_amount(edge.get("amount", edge.get("amount_paid")))
        currency = str(edge.get("currency") or edge.get("payment_currency") or "N/A")
        timestamp = str(edge.get("timestamp") or "N/A")
        transaction_id = str(edge.get("transaction_id") or "N/A")
        custom_row = [transaction_id, amount, currency, timestamp, edge_role]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=edge_role,
                legendgroup=f"edge-{style_key}",
                line={"color": color, "width": width},
                customdata=[custom_row for _ in x_values],
                hovertemplate=(
                    f"<b>{source_id} → {target_id}</b>"
                    "<br>Transaction: %{customdata[0]}"
                    "<br>Amount: %{customdata[1]}"
                    "<br>Currency: %{customdata[2]}"
                    "<br>Timestamp: %{customdata[3]}"
                    "<br>Role: %{customdata[4]}<extra></extra>"
                ),
                showlegend=style_key not in legend_roles,
            )
        )
        legend_roles.add(style_key)
        arrow_tip_index = -3 if len(x_values) >= 4 else -1
        arrow_tail_index = -5 if len(x_values) >= 6 else 0
        annotations.append(
            {
                "ax": x_values[arrow_tail_index],
                "ay": y_values[arrow_tail_index],
                "x": x_values[arrow_tip_index],
                "y": y_values[arrow_tip_index],
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 3,
                "arrowsize": 1.15,
                "arrowwidth": max(1.4, width * 0.55),
                "arrowcolor": color,
                "opacity": 0.95,
            }
        )

    node_roles = {
        node_id: _node_role(node_id, node_metadata.get(node_id, {}), focal_source, focal_target)
        for node_id in ordered
    }
    readable_labels: dict[str, str] = {}
    linked_index = 0
    for node_id in ordered:
        role = node_roles[node_id]
        if role == "focal_sender":
            readable_labels[node_id] = "Primary sender"
        elif role == "focal_receiver":
            readable_labels[node_id] = "Selected receiver"
        elif role == "focal_account":
            readable_labels[node_id] = "Primary account"
        else:
            linked_index += 1
            readable_labels[node_id] = f"Linked account {linked_index}"
    for role in _NODE_STYLES:
        if role not in node_roles.values():
            continue
        color, symbol, size, label = _NODE_STYLES[role]
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=label,
                legendgroup=f"node-{role}",
                marker={
                    "size": min(size, 17),
                    "color": color,
                    "symbol": symbol,
                    "line": {"width": 1.5, "color": _COLORS["white"]},
                },
                hoverinfo="skip",
                showlegend=True,
            )
        )

    node_colors: list[str] = []
    node_symbols: list[str] = []
    node_sizes: list[int] = []
    node_customdata: list[list[Any]] = []
    for node_id in ordered:
        metadata = node_metadata.get(node_id, {})
        color, symbol, size, role_label = _NODE_STYLES[node_roles[node_id]]
        node_colors.append(color)
        node_symbols.append(symbol)
        node_sizes.append(size)
        node_customdata.append(
            [
                node_id,
                role_label,
                metadata.get("case_in_degree", "N/A"),
                metadata.get("case_out_degree", "N/A"),
            ]
        )
    figure.add_trace(
        go.Scatter(
            x=[positions[node_id][0] for node_id in ordered],
            y=[positions[node_id][1] for node_id in ordered],
            mode="markers+text",
            text=[readable_labels[node_id] for node_id in ordered],
            textposition="bottom center",
            textfont={"color": _COLORS["navy"], "size": 11},
            customdata=node_customdata,
            hovertemplate=(
                "<b>Account: %{customdata[0]}</b>"
                "<br>Role: %{customdata[1]}"
                "<br>Case in-degree: %{customdata[2]}"
                "<br>Case out-degree: %{customdata[3]}<extra></extra>"
            ),
            marker={
                "size": node_sizes,
                "color": node_colors,
                "symbol": node_symbols,
                "line": {"width": 2.5, "color": _COLORS["white"]},
            },
            name="Accounts",
            showlegend=False,
        )
    )
    figure.update_layout(
        annotations=annotations,
        title={"text": "Who sent money to whom", "x": 0.01, "xanchor": "left"},
        height=500,
        margin={"l": 20, "r": 20, "t": 64, "b": 82},
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        font={"color": _COLORS["navy"]},
        hovermode="closest",
        dragmode="pan",
        clickmode="event+select",
        uirevision=f"argus-network-{case.get('case_id', 'case')}",
        legend={
            "orientation": "h",
            "x": 0.0,
            "y": -0.10,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 11},
            "title": {"text": "Legend"},
        },
        modebar={
            "add": ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
            "bgcolor": "rgba(251,252,250,0.9)",
            "color": _COLORS["slate"],
            "activecolor": _COLORS["teal"],
        },
        xaxis={"visible": False, "fixedrange": False, "zeroline": False},
        yaxis={
            "visible": False,
            "fixedrange": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
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
        _COLORS["amber"]
        if bool(is_focal)
        else _COLORS["red"]
        if bool(is_prioritized)
        else _COLORS["teal"]
        for is_focal, is_prioritized in zip(focal, prioritized, strict=False)
    ]
    symbols = [
        "diamond" if bool(is_focal) else "x" if bool(is_prioritized) else "circle"
        for is_focal, is_prioritized in zip(focal, prioritized, strict=False)
    ]
    transaction_ids = frame.get("transaction_id", pd.Series("N/A", index=frame.index)).astype(str)
    sources = _series_from_columns(frame, ("source", "from_node_id", "sender")).astype(str)
    targets = _series_from_columns(frame, ("target", "to_node_id", "receiver")).astype(str)
    currencies = _series_from_columns(
        frame, ("currency", "payment_currency", "receiving_currency")
    ).astype(str)
    roles = [
        "Focal transfer"
        if bool(is_focal)
        else "Saved prioritized transfer"
        if bool(is_prioritized)
        else "Earlier case context"
        for is_focal, is_prioritized in zip(focal, prioritized, strict=False)
    ]
    customdata = [
        [transaction_id, source, target, currency, role]
        for transaction_id, source, target, currency, role in zip(
            transaction_ids, sources, targets, currencies, roles, strict=False
        )
    ]
    amount_labels = [
        f"{value:,.0f} {currency}"
        for value, currency in zip(frame["_amount"], currencies, strict=False)
    ]
    y_axis: dict[str, Any] = {
        "gridcolor": _COLORS["grid"],
        "fixedrange": False,
        "rangemode": "normal",
    }
    if amount_column is not None and not frame["_amount"].empty:
        minimum = float(frame["_amount"].min())
        maximum = float(frame["_amount"].max())
        spread = maximum - minimum
        padding = max(spread * 0.22, max(abs(maximum), 1.0) * 0.025)
        y_axis["range"] = [minimum - padding, maximum + (padding * 1.8)]
    figure = go.Figure(
        go.Scatter(
            x=frame["_timestamp"],
            y=frame["_amount"],
            mode="lines+markers+text",
            marker={
                "size": 11,
                "color": colors,
                "symbol": symbols,
                "line": {"width": 1.5, "color": _COLORS["white"]},
            },
            line={"color": _COLORS["muted"], "width": 1.6},
            text=amount_labels,
            textposition="top center",
            textfont={"size": 11, "color": _COLORS["navy"]},
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[4]}</b><br>%{x}"
                "<br>Transaction: %{customdata[0]}"
                "<br>%{customdata[1]} → %{customdata[2]}"
                "<br>Amount: %{y:,.2f} %{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={"text": "Case transaction timeline", "x": 0.01, "xanchor": "left"},
        height=320,
        margin={"l": 20, "r": 20, "t": 60, "b": 35},
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        font={"color": _COLORS["navy"]},
        hovermode="closest",
        dragmode="pan",
        xaxis_title="Timestamp (UTC)",
        yaxis_title=y_title,
        xaxis={"gridcolor": _COLORS["grid"], "fixedrange": False},
        yaxis=y_axis,
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
    labels = comparison["model"].map(_display_model_name)
    metric_colors = {
        "pr_auc": _COLORS["teal"],
        "recall_at_k": _COLORS["navy"],
        "precision_at_k": _COLORS["teal_soft"],
        "fpr": _COLORS["amber"],
    }
    for metric in metric_columns:
        figure.add_trace(
            go.Bar(
                name=metric.replace("_", " ").upper(),
                x=labels,
                y=comparison[metric],
                marker_color=metric_colors[metric],
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{metric.replace('_', ' ').upper()}: "
                    "%{y:.4f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title={"text": f"{partition_label} model metrics", "x": 0.01, "xanchor": "left"},
        barmode="group",
        yaxis_title="Metric value",
        height=455,
        margin={"l": 15, "r": 15, "t": 60, "b": 125},
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        font={"color": _COLORS["navy"]},
        xaxis={"gridcolor": _COLORS["grid"]},
        yaxis={"gridcolor": _COLORS["grid"], "rangemode": "tozero"},
        legend={
            "orientation": "h",
            "y": -0.26,
            "yanchor": "top",
            "x": 0.0,
            "xanchor": "left",
        },
    )
    return figure


def build_pr_curve_figure(
    curves: pd.DataFrame, *, partition_label: str = "Validation"
) -> go.Figure:
    """Render saved PR-curve points; never reconstruct curves from aggregate metrics."""

    figure = go.Figure()
    for model, frame in curves.groupby("model", sort=False):
        ordered = frame.sort_values("recall", kind="mergesort")
        canonical = "_".join(str(model).strip().lower().replace("-", " ").split())
        figure.add_trace(
            go.Scatter(
                x=ordered["recall"],
                y=ordered["precision"],
                name=_display_model_name(model),
                mode="lines",
                line={
                    "color": _MODEL_COLORS.get(canonical, _COLORS["teal_soft"]),
                    "width": 3 if canonical == "graph_enhanced_lightgbm" else 2,
                    "dash": "dot"
                    if canonical in {"graphsage", "graphsage_edge_classifier"}
                    else "solid",
                },
                hovertemplate="<br>".join(
                    (
                        "Recall: %{x:.4f}",
                        "Precision: %{y:.4f}<extra>%{fullData.name}</extra>",
                    )
                ),
            )
        )
    figure.update_layout(
        title={
            "text": f"Saved {partition_label.lower()} precision–recall curves",
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=455,
        margin={"l": 15, "r": 15, "t": 60, "b": 110},
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        font={"color": _COLORS["navy"]},
        xaxis={"gridcolor": _COLORS["grid"], "range": [0, 1]},
        yaxis={"gridcolor": _COLORS["grid"], "range": [0, 1]},
        legend={
            "orientation": "h",
            "y": -0.24,
            "yanchor": "top",
            "x": 0.0,
            "xanchor": "left",
        },
    )
    return figure
