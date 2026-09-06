"""Executable Sprint 1 EDA that writes only observations computed from input data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

from argus.graph.build import (
    build_transaction_graph,
    compute_graph_summary,
    extract_suspicious_local_subgraph,
    sample_transaction_frame,
)

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402


class EDAError(ValueError):
    """Raised when real EDA cannot be produced from the supplied canonical data."""


REQUIRED_COLUMNS = {
    "timestamp",
    "from_bank",
    "to_bank",
    "from_node_id",
    "to_node_id",
    "amount_paid",
    "amount_received",
    "payment_format",
    "payment_currency",
    "receiving_currency",
    "is_laundering",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _horizontal_bar(
    table: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    plot_table = table.iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, max(4, len(plot_table) * 0.28)))
    axis.barh(plot_table[label_column].astype(str), plot_table[value_column], color="#3565A8")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, path)


def _time_frequency(timestamps: pd.Series) -> str:
    duration = timestamps.max() - timestamps.min()
    if duration < pd.Timedelta(1, unit="d"):
        return "10min"
    if duration < pd.Timedelta(7, unit="d"):
        return "1h"
    return "1d"


def _artifact_inventory(root: Path, exclude: set[Path] | None = None) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in (exclude or set())}
    inventory: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.resolve() in excluded:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return inventory


def _write_graph_tables(
    graph: nx.MultiDiGraph,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    degree_rows = []
    for node in sorted(graph.nodes, key=str):
        degree_rows.append(
            {
                "node_id": node,
                "in_edge_count": graph.in_degree(node),
                "out_edge_count": graph.out_degree(node),
                "unique_predecessors": len(set(graph.predecessors(node))),
                "unique_successors": len(set(graph.successors(node))),
            }
        )
    degree_table = pd.DataFrame(
        degree_rows,
        columns=[
            "node_id",
            "in_edge_count",
            "out_edge_count",
            "unique_predecessors",
            "unique_successors",
        ],
    )
    degree_table.to_csv(tables_dir / "degree_distribution.csv", index=False)

    components = sorted(
        (len(component) for component in nx.weakly_connected_components(graph)), reverse=True
    )
    component_table = pd.DataFrame(
        {
            "component_rank": range(1, len(components) + 1),
            "node_count": components,
        }
    )
    component_table.to_csv(tables_dir / "connected_components.csv", index=False)
    return degree_table, component_table


def _plot_suspicious_subgraph(
    graph: nx.MultiDiGraph,
    evidence: dict[str, Any],
    path: Path,
    *,
    random_seed: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    if not evidence["available"]:
        axis.text(
            0.5,
            0.5,
            "No labeled-positive transfer exists in the analyzed graph sample.",
            ha="center",
            va="center",
            wrap=True,
        )
        axis.axis("off")
        axis.set_title("Suspicious local subgraph availability")
        _save_figure(figure, path)
        return

    layout = nx.spring_layout(graph, seed=random_seed)
    positive_nodes: set[str] = set()
    edge_colors = []
    for source, target, data in graph.edges(data=True):
        positive = int(data.get("is_laundering") or 0) == 1
        edge_colors.append("#C62828" if positive else "#90A4AE")
        if positive:
            positive_nodes.update((source, target))
    node_colors = ["#EF5350" if node in positive_nodes else "#42A5F5" for node in graph]
    nx.draw_networkx(
        graph,
        pos=layout,
        ax=axis,
        with_labels=False,
        node_size=90,
        width=1.0,
        arrows=True,
        arrowsize=10,
        node_color=node_colors,
        edge_color=edge_colors,
        alpha=0.82,
    )
    axis.set_title("Observed local subgraph around first labeled-positive transfer")
    axis.axis("off")
    _save_figure(figure, path)


def generate_eda_artifacts(
    transactions: pd.DataFrame,
    output_dir: str | Path,
    *,
    sample_size: int | None = None,
    random_seed: int = 42,
    graph_max_edges: int | None = 50_000,
    top_n: int = 20,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate real CSV/JSON/PNG artifacts from a canonical transaction table."""

    missing = sorted(REQUIRED_COLUMNS.difference(transactions.columns))
    if missing:
        raise EDAError(f"Cannot run EDA; missing canonical columns: {missing}")
    if transactions.empty:
        raise EDAError("Cannot run EDA on an empty transaction table")
    if top_n <= 0:
        raise EDAError("top_n must be positive")

    root = Path(output_dir)
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame = sample_transaction_frame(transactions, max_edges=sample_size, random_seed=random_seed)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["amount_paid"] = pd.to_numeric(frame["amount_paid"], errors="raise")
    frame["amount_received"] = pd.to_numeric(frame["amount_received"], errors="raise")
    frame["is_laundering"] = pd.to_numeric(frame["is_laundering"], errors="raise").astype("int8")

    class_table = (
        frame["is_laundering"]
        .value_counts(dropna=False)
        .reindex([0, 1], fill_value=0)
        .rename_axis("is_laundering")
        .rename("count")
        .reset_index()
    )
    class_table["rate"] = class_table["count"] / len(frame)
    class_table.to_csv(tables_dir / "class_distribution.csv", index=False)
    class_table[["is_laundering", "rate"]].to_csv(tables_dir / "laundering_rate.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(
        class_table["is_laundering"].astype(str),
        class_table["count"],
        color=["#3565A8", "#C62828"],
    )
    axis.bar_label(bars, fmt="%d")
    axis.set_title("Observed class distribution")
    axis.set_xlabel("Is laundering label")
    axis.set_ylabel("Transactions (log scale)")
    axis.set_yscale("log")
    _save_figure(figure, figures_dir / "class_distribution.png")

    amount_summary = frame[["amount_paid", "amount_received"]].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    )
    amount_summary.rename_axis("statistic").reset_index().to_csv(
        tables_dir / "amount_summary.csv", index=False
    )
    upper = float(frame["amount_paid"].quantile(0.99))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(frame.loc[frame["amount_paid"].le(upper), "amount_paid"], bins=50, color="#3565A8")
    axis.set_title("Amount paid distribution (observed values through 99th percentile)")
    axis.set_xlabel("Amount paid; source currency not converted")
    axis.set_ylabel("Transactions")
    _save_figure(figure, figures_dir / "amount_distribution.png")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(np.log1p(frame["amount_paid"]), bins=50, color="#5C6BC0")
    axis.set_title("Observed log1p(amount paid) distribution")
    axis.set_xlabel("log1p(amount paid); source currency not converted")
    axis.set_ylabel("Transactions")
    _save_figure(figure, figures_dir / "log_amount_distribution.png")

    frequency = _time_frequency(frame["timestamp"])
    activity = (
        frame.assign(timestamp_bin=frame["timestamp"].dt.floor(frequency))
        .groupby("timestamp_bin", observed=True)
        .agg(
            transactions=("transaction_id", "size"),
            laundering_transactions=("is_laundering", "sum"),
        )
        .reset_index()
    )
    activity.to_csv(tables_dir / "activity_over_time.csv", index=False)
    for column, title, filename, color in (
        (
            "transactions",
            "Observed transaction activity over time",
            "transaction_activity_over_time.png",
            "#3565A8",
        ),
        (
            "laundering_transactions",
            "Observed labeled-laundering activity over time",
            "laundering_activity_over_time.png",
            "#C62828",
        ),
    ):
        figure, axis = plt.subplots(figsize=(10, 4.5))
        axis.plot(
            activity["timestamp_bin"],
            activity[column],
            color=color,
            marker="o",
            markersize=3,
        )
        axis.set_title(title)
        axis.set_xlabel(f"Time bin ({frequency})")
        axis.set_ylabel("Transactions")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.2)
        _save_figure(figure, figures_dir / filename)

    payment_format = (
        frame["payment_format"]
        .value_counts()
        .rename_axis("payment_format")
        .rename("count")
        .reset_index()
    )
    payment_format.to_csv(tables_dir / "payment_format_distribution.csv", index=False)
    _horizontal_bar(
        payment_format,
        label_column="payment_format",
        value_column="count",
        title="Observed payment format distribution",
        xlabel="Transactions",
        path=figures_dir / "payment_format_distribution.png",
    )

    currency = pd.concat(
        [
            frame["payment_currency"]
            .value_counts()
            .rename("count")
            .rename_axis("currency")
            .reset_index()
            .assign(role="payment"),
            frame["receiving_currency"]
            .value_counts()
            .rename("count")
            .rename_axis("currency")
            .reset_index()
            .assign(role="receiving"),
        ],
        ignore_index=True,
    )
    currency.to_csv(tables_dir / "currency_distribution.csv", index=False)
    pivot_currency = currency.pivot(index="currency", columns="role", values="count").fillna(0)
    figure, axis = plt.subplots(figsize=(10, 6))
    pivot_currency.plot.barh(ax=axis, color=["#3565A8", "#80CBC4"])
    axis.set_title("Observed currency distribution by transfer side")
    axis.set_xlabel("Transactions")
    _save_figure(figure, figures_dir / "currency_distribution.png")

    bank_activity = pd.concat(
        [
            frame["from_bank"]
            .value_counts()
            .rename("count")
            .rename_axis("bank_id")
            .reset_index()
            .assign(role="sender"),
            frame["to_bank"]
            .value_counts()
            .rename("count")
            .rename_axis("bank_id")
            .reset_index()
            .assign(role="receiver"),
        ],
        ignore_index=True,
    )
    bank_activity.to_csv(tables_dir / "bank_activity.csv", index=False)
    bank_top = (
        bank_activity.sort_values(["role", "count"], ascending=[True, False])
        .groupby("role", observed=True)
        .head(top_n)
    )
    bank_top = bank_top.assign(label=bank_top["role"] + ":" + bank_top["bank_id"].astype(str))
    _horizontal_bar(
        bank_top.sort_values("count", ascending=False).head(top_n * 2),
        label_column="label",
        value_column="count",
        title=f"Top {top_n} sender and receiver banks in analyzed data",
        xlabel="Transactions",
        path=figures_dir / "bank_activity.png",
    )

    pair_table = (
        frame.groupby(["from_node_id", "to_node_id"], observed=True)
        .agg(
            transfer_count=("transaction_id", "size"),
            total_amount_paid=("amount_paid", "sum"),
            laundering_labels=("is_laundering", "sum"),
        )
        .reset_index()
        .sort_values(["transfer_count", "total_amount_paid"], ascending=False)
    )
    repeated_pairs = pair_table.loc[pair_table["transfer_count"].gt(1)].head(top_n)
    repeated_pairs.to_csv(tables_dir / "repeated_sender_receiver_pairs.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    if repeated_pairs.empty:
        axis.text(0.5, 0.5, "No repeated directed pair in analyzed data", ha="center", va="center")
        axis.axis("off")
    else:
        axis.bar(range(len(repeated_pairs)), repeated_pairs["transfer_count"], color="#7E57C2")
        axis.set_xlabel("Repeated directed pair rank")
        axis.set_ylabel("Transfers")
    axis.set_title("Observed repeated sender-receiver activity")
    _save_figure(figure, figures_dir / "repeated_pair_activity.png")

    outgoing = (
        frame.groupby("from_node_id", observed=True)["to_node_id"]
        .nunique()
        .rename("unique_outgoing_counterparties")
    )
    incoming = (
        frame.groupby("to_node_id", observed=True)["from_node_id"]
        .nunique()
        .rename("unique_incoming_counterparties")
    )
    counterparties = (
        pd.concat([outgoing, incoming], axis=1)
        .fillna(0)
        .astype("int64")
        .rename_axis("node_id")
        .reset_index()
    )
    counterparties.to_csv(tables_dir / "unique_counterparties.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(
        counterparties["unique_outgoing_counterparties"],
        bins=30,
        alpha=0.65,
        label="outgoing",
    )
    axis.hist(
        counterparties["unique_incoming_counterparties"],
        bins=30,
        alpha=0.65,
        label="incoming",
    )
    axis.set_title("Observed unique counterparty distributions")
    axis.set_xlabel("Unique counterparties")
    axis.set_ylabel("Nodes")
    axis.legend()
    _save_figure(figure, figures_dir / "unique_counterparties.png")

    graph = build_transaction_graph(frame, max_edges=graph_max_edges, random_seed=random_seed)
    graph_summary = compute_graph_summary(graph)
    _write_json(graph_summary, root / "graph_summary.json")
    degree_table, component_table = _write_graph_tables(graph, tables_dir)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(degree_table["in_edge_count"], bins=30, alpha=0.65, label="in-degree")
    axis.hist(degree_table["out_edge_count"], bins=30, alpha=0.65, label="out-degree")
    axis.set_title("Observed directed multigraph degree distributions")
    axis.set_xlabel("Incident transfer edges")
    axis.set_ylabel("Nodes")
    axis.legend()
    _save_figure(figure, figures_dir / "in_out_degree_distribution.png")

    figure, axis = plt.subplots(figsize=(9, 5))
    displayed_components = component_table.head(50)
    axis.bar(
        displayed_components["component_rank"],
        displayed_components["node_count"],
        color="#26A69A",
    )
    axis.set_title("Largest observed weakly connected components (up to 50)")
    axis.set_xlabel("Component rank")
    axis.set_ylabel("Nodes")
    _save_figure(figure, figures_dir / "connected_components.png")

    suspicious_graph, suspicious_evidence = extract_suspicious_local_subgraph(graph)
    _write_json(suspicious_evidence, root / "suspicious_subgraph_evidence.json")
    suspicious_rows = []
    for source, target, key, data in suspicious_graph.edges(keys=True, data=True):
        suspicious_rows.append(
            {
                "from_node_id": source,
                "to_node_id": target,
                "edge_key": key,
                "transaction_id": data.get("transaction_id"),
                "timestamp": data.get("timestamp"),
                "amount_paid": data.get("amount_paid"),
                "is_laundering": data.get("is_laundering"),
            }
        )
    pd.DataFrame(
        suspicious_rows,
        columns=[
            "from_node_id",
            "to_node_id",
            "edge_key",
            "transaction_id",
            "timestamp",
            "amount_paid",
            "is_laundering",
        ],
    ).to_csv(tables_dir / "suspicious_local_subgraph_edges.csv", index=False)
    _plot_suspicious_subgraph(
        suspicious_graph,
        suspicious_evidence,
        figures_dir / "suspicious_local_subgraph.png",
        random_seed=random_seed,
    )

    profile = {
        "provenance": provenance or {},
        "analysis_scope": {
            "input_rows": len(transactions),
            "analyzed_rows": len(frame),
            "sampled_inside_eda": len(frame) != len(transactions),
            "random_seed": random_seed,
            "graph_max_edges": graph_max_edges,
            "graph_edges_analyzed": graph.number_of_edges(),
        },
        "dataset": {
            "rows": len(frame),
            "timestamp_minimum": frame["timestamp"].min().isoformat(),
            "timestamp_maximum": frame["timestamp"].max().isoformat(),
            "positive_labels": int(frame["is_laundering"].sum()),
            "laundering_rate": float(frame["is_laundering"].mean()),
            "unique_sender_nodes": int(frame["from_node_id"].nunique()),
            "unique_receiver_nodes": int(frame["to_node_id"].nunique()),
            "unique_nodes": len(set(frame["from_node_id"]).union(frame["to_node_id"])),
            "unique_directed_pairs": len(pair_table),
            "repeated_directed_pairs": int(pair_table["transfer_count"].gt(1).sum()),
            "payment_formats": int(frame["payment_format"].nunique()),
            "payment_currencies": int(frame["payment_currency"].nunique()),
            "receiving_currencies": int(frame["receiving_currency"].nunique()),
            "time_bin_frequency": frequency,
        },
        "graph": graph_summary,
        "suspicious_subgraph": suspicious_evidence,
        "limitations": [
            "Counts describe only the declared analysis scope.",
            (
                "Amounts in different currencies are not converted or directly "
                "aggregated for comparison."
            ),
            "A positive source label is evidence for human review, not a determination of guilt.",
        ],
    }
    _write_json(profile, root / "dataset_profile.json")
    manifest_path = root / "eda_manifest.json"
    manifest = {
        "status": "PASS",
        "generated_by": "argus.eda.generate_eda_artifacts",
        "provenance": provenance or {},
        "artifact_count_excluding_manifest": 0,
        "artifacts": [],
    }
    manifest["artifacts"] = _artifact_inventory(root, exclude={manifest_path})
    manifest["artifact_count_excluding_manifest"] = len(manifest["artifacts"])
    _write_json(manifest, manifest_path)
    return manifest
