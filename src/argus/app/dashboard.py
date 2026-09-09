"""ARGUS Streamlit Network Investigator.

This module renders previously generated product artifacts. It intentionally imports
neither training code nor serialized estimators, so a page refresh cannot fit a model.
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from argus.app.artifacts import ArtifactLoadError, DashboardArtifacts, load_dashboard_artifacts
from argus.app.figures import (
    build_model_metric_figure,
    build_network_figure,
    build_pr_curve_figure,
    build_timeline_figure,
)

_PAGES = (
    "Overview",
    "Investigations",
    "Case Investigator",
    "Model Evidence",
)

_PRIMARY_MODEL = "graph_enhanced_lightgbm"
_MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "graph_enhanced_lightgbm": "Graph-enhanced LightGBM",
    "refined_transaction_lightgbm": "Refined transaction LightGBM",
    "graphsage_edge_classifier": "GraphSAGE",
    "graphsage": "GraphSAGE",
}


def _canonical_model_name(value: Any) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _display_model_name(value: Any) -> str:
    canonical = _canonical_model_name(value)
    if canonical in _MODEL_LABELS:
        return _MODEL_LABELS[canonical]
    return str(value).replace("_", " ").strip().title() or "N/A"


def _model_role(value: Any) -> str:
    canonical = _canonical_model_name(value)
    if canonical == _PRIMARY_MODEL:
        return "Primary operational model"
    if canonical in {"graphsage", "graphsage_edge_classifier"}:
        return "Research comparator"
    return "Comparator"


def _ordered_models(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy with the frozen operational model listed first."""

    order = {
        _PRIMARY_MODEL: 0,
        "refined_transaction_lightgbm": 1,
        "graphsage_edge_classifier": 2,
        "graphsage": 2,
    }
    displayed = comparison.copy()
    displayed["_display_order"] = displayed["model"].map(
        lambda value: order.get(_canonical_model_name(value), 3)
    )
    return displayed.sort_values("_display_order", kind="mergesort").drop(columns="_display_order")


def _humanize(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return str(value).replace("_", " ").strip().title()


def _display_pattern(value: Any) -> str:
    canonical = _canonical_model_name(value)
    if canonical == "high_graphsage_transaction_score":
        return "Elevated transaction ranking (research comparator)"
    return _humanize(value)


def _display_feature_name(value: Any) -> str:
    label = str(value).split("::")[-1]
    for prefix in ("numeric__", "categorical__"):
        if label.startswith(prefix):
            label = label.removeprefix(prefix)
    return _humanize(label)


def _display_evidence_statement(value: Any) -> str:
    return (
        str(value)
        .replace("the supplied neighborhood", "the saved case context")
        .replace("The supplied neighborhood", "The saved case context")
    )


def _display_case_note(case: dict[str, Any], value: Any) -> str:
    text = str(value)
    model = case.get("model_evidence", {})
    if not isinstance(model, dict):
        return text
    model_name = model.get("model_name")
    if model_name:
        text = text.replace(str(model_name), _display_model_name(model_name))
    score_name = model.get("score_name")
    if score_name:
        score_label = (
            "saved research ranking score"
            if _canonical_model_name(model_name) in {"graphsage", "graphsage_edge_classifier"}
            else "saved priority score"
        )
        text = text.replace(str(score_name), score_label)
    return text.replace("The supplied context", "The saved case context")


def _case_model_name(case: dict[str, Any]) -> str | None:
    evidence = case.get("model_evidence", {})
    if not isinstance(evidence, dict):
        return None
    value = evidence.get("model_name")
    return str(value) if value is not None else None


def _queue_model_names(artifacts: DashboardArtifacts) -> tuple[str, ...]:
    names = {
        name for case in artifacts.cases.values() if (name := _case_model_name(case)) is not None
    }
    return tuple(sorted(names))


def _queue_uses_graphsage(artifacts: DashboardArtifacts) -> bool:
    names = _queue_model_names(artifacts)
    return bool(names) and all(
        _canonical_model_name(name) in {"graphsage", "graphsage_edge_classifier"} for name in names
    )


def _display_model_table(comparison: pd.DataFrame, *, final: bool = False) -> pd.DataFrame:
    ordered = _ordered_models(comparison)
    displayed = pd.DataFrame(index=ordered.index)
    displayed["Role"] = ordered["model"].map(_model_role)
    displayed["Model"] = ordered["model"].map(_display_model_name)
    if not final and "feature_family" in ordered:
        displayed["Feature Set"] = ordered["feature_family"].map(_humanize)
    metric_labels = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "fpr": "FPR",
        "recall_at_k": "Recall@K",
        "precision_at_k": "Precision@K",
        "alerts": "Alerts",
    }
    for source, label in metric_labels.items():
        if source in ordered:
            displayed[label] = ordered[source]
    return displayed.reset_index(drop=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configured_artifact_root() -> Path:
    """Return an explicit artifact override or the repository-local default."""

    configured = os.environ.get("ARGUS_ARTIFACT_DIR") or os.environ.get(
        "ARGUS_SPRINT4_ARTIFACT_DIR"
    )
    if configured:
        return Path(configured)
    sprint5_root = _project_root() / "artifacts" / "sprint5"
    final_summary = sprint5_root / "product" / "final_test_summary.json"
    return sprint5_root if final_summary.is_file() else _project_root() / "artifacts" / "sprint4"


@st.cache_data(show_spinner=False)
def _cached_load(path: str) -> DashboardArtifacts:
    return load_dashboard_artifacts(path)


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_money(value: Any, currency: str | None = None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        formatted = f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)
    return f"{formatted} {currency}" if currency else formatted


def _comparison_focus(artifacts: DashboardArtifacts) -> pd.Series:
    """Select only from the validation comparison; final-test rows are held separately."""

    comparison = artifacts.model_comparison
    if "is_champion" in comparison:
        selected = comparison[
            comparison["is_champion"].map(
                lambda value: value is True or str(value).strip().lower() == "true"
            )
        ]
        if not selected.empty:
            return selected.iloc[0]
    if "selected" in comparison:
        selected = comparison[
            comparison["selected"].map(
                lambda value: value is True or str(value).strip().lower() == "true"
            )
        ]
        if not selected.empty:
            return selected.iloc[0]
    return comparison.loc[comparison["pr_auc"].idxmax()]


def _metric_value(summary: dict[str, Any], focus: pd.Series, key: str) -> Any:
    focus_value = focus.get(key)
    if focus_value is not None and not pd.isna(focus_value):
        return focus_value
    aliases = {
        "pr_auc": ("validation_pr_auc", "pr_auc"),
        "recall_at_k": ("validation_recall_at_k", "recall_at_k"),
        "fpr": ("validation_fpr", "fpr"),
        "alerts": ("alert_volume", "alerts", "flagged_transactions"),
    }
    for alias in aliases[key]:
        if alias in summary:
            return summary[alias]
    return focus.get(key)


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(145deg, #f8fafc 0%, #eef6f5 100%); }
        [data-testid="stSidebar"] { background: #102a2a; }
        [data-testid="stSidebar"] * { color: #f8fafc; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.92); border: 1px solid #dbe7e5;
            border-radius: 10px; padding: 14px;
        }
        .argus-eyebrow { color:#0f766e; font-weight:700; letter-spacing:.12em; font-size:.78rem; }
        .evidence-card {
            background:#fff; border-left:4px solid #0f766e; border-radius:8px;
            padding:12px 14px; margin:8px 0; box-shadow:0 1px 4px rgba(15,23,42,.07);
        }
        .evidence-kind {
            color:#0f766e; font-size:.76rem; font-weight:700; text-transform:uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="argus-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def render_executive_dashboard(artifacts: DashboardArtifacts) -> None:
    _page_heading(
        "ARGUS FINANCIAL CRIME INVESTIGATION",
        "Overview",
        "Model results and saved investigation cases from the completed ARGUS study.",
    )
    focus = _comparison_focus(artifacts)
    focus_is_primary = _canonical_model_name(focus.get("model")) == _PRIMARY_MODEL
    summary = artifacts.summary
    first = st.columns(3)
    first[0].metric("Transactions evaluated", _format_count(summary.get("transactions_analyzed")))
    first[1].metric("Saved investigation cases", _format_count(len(artifacts.queue)))
    first[2].metric("High-priority cases", _format_count(summary.get("high_priority_cases")))
    second = st.columns(4)
    second[0].metric(
        "Primary model validation PR-AUC" if focus_is_primary else "Validation PR-AUC",
        _format_metric(_metric_value(summary, focus, "pr_auc")),
    )
    second[1].metric(
        "Validation Recall@K", _format_metric(_metric_value(summary, focus, "recall_at_k"))
    )
    second[2].metric("Validation FPR", _format_metric(_metric_value(summary, focus, "fpr")))
    second[3].metric("Validation alerts", _format_count(_metric_value(summary, focus, "alerts")))
    if focus_is_primary:
        st.caption(
            f"Primary operational ranking: {_display_model_name(focus.get('model'))}. "
            "It was selected using validation evidence before the final test."
        )
    else:
        st.caption(
            f"Loaded validation metrics refer to {_display_model_name(focus.get('model'))}. "
            "The primary Graph-enhanced LightGBM result is not present in this artifact set."
        )
    if _queue_uses_graphsage(artifacts):
        st.info(
            "The saved case examples were produced by the GraphSAGE research comparator. "
            "They are presented as evidence-led case examples and are not relabeled or rescored "
            "as outputs of the primary LightGBM model."
        )

    st.subheader("Model evidence at a glance")
    ordered_comparison = _ordered_models(artifacts.model_comparison)
    card_columns = st.columns(min(4, len(ordered_comparison)))
    for column, (_, model) in zip(card_columns, ordered_comparison.iterrows(), strict=False):
        label = _display_model_name(model["model"])
        column.metric(label, _format_metric(model["pr_auc"]), help="Validation PR-AUC")
        recall = _format_metric(model.get("recall_at_k"))
        precision = _format_metric(model.get("precision_at_k"))
        column.caption(
            f"{_model_role(model['model'])} · Recall@K {recall} · Precision@K {precision}"
        )

    st.plotly_chart(
        build_model_metric_figure(ordered_comparison),
        width="stretch",
        key="executive-model-comparison",
    )
    st.info(
        "This dashboard supports investigation prioritization. It does not establish wrongdoing, "
        "authorize an automatic action, or replace human review."
    )
    if artifacts.final_evaluation is None:
        st.caption("Final-test results are not included in the loaded artifact set.")
    else:
        final = artifacts.final_evaluation
        champion = final.model_comparison.loc[final.model_comparison["is_frozen_champion"]].iloc[0]
        st.subheader("Frozen final evaluation")
        st.caption(
            f"{_display_model_name(champion['model'])} was frozen from validation before the "
            "single final-test evaluation. Test metrics are reporting-only."
        )
        final_metrics = st.columns(4)
        final_metrics[0].metric("Final-test PR-AUC", _format_metric(champion["pr_auc"]))
        final_metrics[1].metric("Final-test ROC-AUC", _format_metric(champion["roc_auc"]))
        final_metrics[2].metric("Final-test F1", _format_metric(champion["f1"]))
        final_metrics[3].metric("Final-test alerts", _format_count(champion["alerts"]))


def _filter_queue(queue: pd.DataFrame, *, score_label: str) -> pd.DataFrame:
    filtered = queue.copy()
    controls = st.columns(3)
    if "priority" in filtered:
        values = sorted(filtered["priority"].dropna().astype(str).unique())
        priority_options = {_humanize(value): value for value in values}
        selected = controls[0].multiselect(
            "Priority", list(priority_options), default=list(priority_options)
        )
        selected_values = [priority_options[label] for label in selected]
        filtered = filtered[filtered["priority"].astype(str).isin(selected_values)]
    else:
        controls[0].caption("Priority filter unavailable")
    if "risk_score" in filtered and not filtered.empty:
        lower = float(filtered["risk_score"].min())
        upper = float(filtered["risk_score"].max())
        if lower < upper:
            minimum = controls[1].slider(
                f"Minimum {score_label.lower()}",
                min_value=lower,
                max_value=upper,
                value=lower,
                help="A saved ranking score, not a calibrated probability of wrongdoing.",
            )
            filtered = filtered[filtered["risk_score"] >= minimum]
        else:
            controls[1].caption(f"All saved cases have a score of {lower:.4f}")
    else:
        controls[1].caption("Score filter unavailable")
    query = controls[2].text_input("Search cases", placeholder="Case ID or pattern")
    if query:
        text = filtered.astype(str).agg(" ".join, axis=1)
        filtered = filtered[text.str.contains(query, case=False, regex=False)]
    return filtered


def render_investigation_queue(artifacts: DashboardArtifacts) -> None:
    research_queue = _queue_uses_graphsage(artifacts)
    score_label = "research score" if research_queue else "priority score"
    _page_heading(
        "HUMAN REVIEW WORKLIST",
        "Investigations",
        "Review and sort saved candidate cases. Queue order supports triage and is not "
        "an accusation.",
    )
    if research_queue:
        st.info(
            "These saved case examples use the GraphSAGE research-comparator ranking. "
            "Graph-enhanced LightGBM remains the primary operational model."
        )
    filtered = _filter_queue(artifacts.queue, score_label=score_label)
    if filtered.empty:
        st.warning("No saved cases match the current filters.")
        return
    sort_labels = {
        "risk_score": score_label.title(),
        "priority": "Priority",
        "total_flow": "Total Flow",
        "transaction_count": "Transfers",
        "case_id": "Case ID",
    }
    sortable = {label: column for column, label in sort_labels.items() if column in filtered}
    selected_sort = st.selectbox("Sort by", list(sortable), index=0)
    sort_column = sortable[selected_sort]
    descending = st.toggle("Descending", value=sort_column != "case_id")
    filtered = filtered.sort_values(sort_column, ascending=not descending, kind="mergesort")
    labels = {
        "priority": "Priority",
        "case_id": "Case ID",
        "risk_score": score_label.title(),
        "account_count": "Accounts",
        "transaction_count": "Transfers",
        "total_flow": "Total Flow",
        "major_pattern": "Primary Pattern",
        "focal_timestamp": "Last Activity",
        "last_activity": "Last Activity",
        "status": "Status",
        "evidence_count": "Evidence Items",
    }
    columns = [column for column in labels if column in filtered]
    display = filtered[columns].copy()
    for column in ("priority", "status"):
        if column in display:
            display[column] = display[column].map(_humanize)
    if "major_pattern" in display:
        display["major_pattern"] = display["major_pattern"].map(_display_pattern)
    for column in ("focal_timestamp", "last_activity"):
        if column in display:
            timestamps = pd.to_datetime(display[column], errors="coerce", utc=True)
            display[column] = timestamps.dt.strftime("%Y-%m-%d %H:%M UTC").fillna("N/A")
    display.rename(columns=labels, inplace=True)
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(f"Showing {len(filtered):,} of {len(artifacts.queue):,} saved cases.")


def _case_value(case: dict[str, Any], queue_row: pd.Series, key: str) -> Any:
    if key in case and not isinstance(case[key], (dict, list)):
        return case[key]
    return queue_row.get(key)


def _case_priority(case: dict[str, Any], queue_row: pd.Series) -> str:
    priority = case.get("priority")
    if isinstance(priority, dict):
        return str(priority.get("band", "N/A"))
    return str(priority if priority is not None else queue_row.get("priority", "N/A"))


def _case_risk(case: dict[str, Any], queue_row: pd.Series) -> Any:
    model = case.get("model_evidence", {})
    if isinstance(model, dict) and model.get("score") is not None:
        return model["score"]
    return _case_value(case, queue_row, "risk_score")


def _case_counts(case: dict[str, Any], queue_row: pd.Series) -> tuple[Any, Any]:
    network = case.get("network", {})
    accounts = _case_value(case, queue_row, "account_count")
    transactions = _case_value(case, queue_row, "transaction_count")
    if accounts is None and isinstance(network, dict) and isinstance(network.get("nodes"), list):
        accounts = len(network["nodes"])
    if transactions is None and isinstance(case.get("transactions"), list):
        transactions = len(case["transactions"])
    return accounts, transactions


def _render_observed_evidence(case: dict[str, Any]) -> None:
    st.subheader("Observed evidence")
    evidence = case["observed_evidence"]
    if len(evidence) < 3:
        st.warning(f"Only {len(evidence)} observed evidence item(s) were saved for this case.")
    for item in evidence:
        kind = html.escape(_humanize(item.get("kind", "Observed fact")))
        statement = html.escape(_display_evidence_statement(item["statement"]))
        scope = item.get("scope")
        scope_labels = {
            "focal_transaction": "Focal transaction",
            "strictly_prior_supplied_neighborhood": "Earlier case context",
            "strictly_prior_neighborhood": "Earlier case context",
        }
        scope_label = scope_labels.get(str(scope), _humanize(scope)) if scope else None
        scope_text = (
            f"<br><small>Context: {html.escape(scope_label)}</small>" if scope_label else ""
        )
        st.markdown(
            f'<div class="evidence-card"><div class="evidence-kind">{kind}</div>'
            f"{statement}{scope_text}</div>",
            unsafe_allow_html=True,
        )


def _render_model_evidence(case: dict[str, Any]) -> None:
    st.subheader("Model explanation")
    model = case.get("model_evidence", {})
    if not model:
        st.warning("No saved model explanation is available for this case.")
        return
    model_name = model.get("model_name", "N/A")
    header = st.columns(4)
    header[0].metric("Model", _display_model_name(model_name))
    header[1].metric("Ranking score", _format_metric(model.get("score")))
    header[2].metric("Threshold", _format_metric(model.get("threshold")))
    header[3].metric("Model role", _model_role(model_name))
    contributions = model.get("feature_contributions", case.get("feature_contributions", []))
    if not isinstance(contributions, list) or not contributions:
        st.caption("No local feature-contribution artifact was saved.")
        return
    frame = pd.DataFrame(contributions)
    feature_column = next(
        (name for name in ("feature", "feature_name", "name") if name in frame), None
    )
    value_column = next(
        (name for name in ("contribution", "shap_value", "attribution") if name in frame), None
    )
    if feature_column is None or value_column is None:
        st.dataframe(frame, width="stretch", hide_index=True)
        return
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame[feature_column] = frame[feature_column].map(_display_feature_name)
    chart = frame.dropna(subset=[value_column]).copy()
    chart = chart.reindex(chart[value_column].abs().sort_values(ascending=True).index).tail(15)
    if not chart.empty:
        figure = go.Figure(
            go.Bar(
                x=chart[value_column],
                y=chart[feature_column],
                orientation="h",
                marker_color=[
                    "#dc2626" if value >= 0 else "#2563eb" for value in chart[value_column]
                ],
            )
        )
        figure.update_layout(
            title="Factors influencing the saved score",
            xaxis_title="Contribution to saved score",
            height=max(320, len(chart) * 28),
            margin={"l": 20, "r": 15, "t": 55, "b": 40},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, width="stretch", key="case-local-explanation")
    st.caption(
        "Model contribution is shown separately from directly observed transaction and network "
        "facts. A model score does not establish wrongdoing."
    )
    if _canonical_model_name(model_name) in {"graphsage", "graphsage_edge_classifier"}:
        st.caption(
            "GraphSAGE is a research comparator. Its saved sigmoid score is uncalibrated and is "
            "not the primary operational ranking."
        )


def _display_transactions(case: dict[str, Any]) -> pd.DataFrame:
    transactions = case.get("transactions")
    if not isinstance(transactions, list) or not transactions:
        return pd.DataFrame()
    frame = pd.DataFrame(transactions)
    aliases = {
        "transaction_id": "Transaction ID",
        "timestamp": "Timestamp",
        "from_node_id": "Sender",
        "source": "Sender",
        "to_node_id": "Receiver",
        "target": "Receiver",
        "amount": "Amount",
        "amount_paid": "Amount Paid",
        "amount_received": "Amount Received",
        "payment_currency": "Payment Currency",
        "receiving_currency": "Receiving Currency",
        "payment_format": "Payment Format",
        "is_focal": "Focal Transaction",
    }
    columns: list[str] = []
    used_labels: set[str] = set()
    for source, label in aliases.items():
        if source in frame and label not in used_labels:
            columns.append(source)
            used_labels.add(label)
    displayed = frame[columns].copy()
    if "timestamp" in displayed:
        timestamps = pd.to_datetime(displayed["timestamp"], errors="coerce", utc=True)
        displayed["timestamp"] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S UTC").fillna("N/A")
    displayed.rename(columns=aliases, inplace=True)
    return displayed


def render_case_investigator(artifacts: DashboardArtifacts) -> None:
    _page_heading(
        "EVIDENCE-LED REVIEW",
        "Case Investigator",
        "Inspect direction, timing, observed facts, and saved model explanations for one case.",
    )
    queue = artifacts.queue
    case_ids = queue["case_id"].astype(str).tolist()
    selected_case_id = st.selectbox("Case", case_ids)
    case = artifacts.cases[selected_case_id]
    queue_row = queue.loc[queue["case_id"].astype(str).eq(selected_case_id)].iloc[0]
    case_model = _case_model_name(case)
    research_case = _canonical_model_name(case_model) in {
        "graphsage",
        "graphsage_edge_classifier",
    }

    accounts, transaction_count = _case_counts(case, queue_row)
    metrics = st.columns(5)
    metrics[0].metric("Priority", _humanize(_case_priority(case, queue_row)))
    score_label = "Research score" if research_case else "Priority score"
    metrics[1].metric(score_label, _format_metric(_case_risk(case, queue_row)))
    metrics[2].metric("Accounts", _format_count(accounts))
    metrics[3].metric("Transactions", _format_count(transaction_count))
    metrics[4].metric(
        "Total flow",
        _format_money(_case_value(case, queue_row, "total_flow"), case.get("currency")),
    )
    st.caption(
        f"Case {selected_case_id} · Status: "
        f"{_humanize(case.get('status', 'pending_human_review'))} · Focal transaction: "
        f"{case.get('transaction_id', case.get('seed_transaction_id', 'N/A'))}"
    )
    if research_case:
        st.info(
            "This saved case was ranked by the GraphSAGE research comparator. "
            "Graph-enhanced LightGBM is the primary operational model."
        )

    network_figure = build_network_figure(case)
    if network_figure is None:
        st.warning("This case has no saved directed-network records to visualize.")
    else:
        st.plotly_chart(network_figure, width="stretch", key="case-network")
        st.caption(
            "Arrowheads preserve sender → receiver direction. Amber marks the selected focal "
            "transaction; red is reserved for an explicitly saved prioritization flag."
        )

    timeline_figure = build_timeline_figure(case)
    if timeline_figure is None:
        st.warning("This case has no timestamped transaction records for a timeline.")
    else:
        st.plotly_chart(timeline_figure, width="stretch", key="case-timeline")

    left, right = st.columns(2)
    with left:
        _render_observed_evidence(case)
    with right:
        pattern = _case_value(case, queue_row, "major_pattern")
        st.subheader("Case context")
        st.write(f"**Primary pattern:** {_display_pattern(pattern)}")
        note = case.get("analyst_note", case.get("case_note"))
        if isinstance(note, dict):
            st.write(_display_case_note(case, note.get("text", "No saved analyst note.")))
        elif note:
            st.write(_display_case_note(case, note))
        else:
            st.caption("No saved case note is available.")
        st.caption("Case notes summarize saved evidence and require analyst verification.")
        st.info("Human review is required. No automatic adverse action is authorized.")

    _render_model_evidence(case)
    transactions = _display_transactions(case)
    if not transactions.empty:
        st.subheader("Transactions in this saved case")
        st.dataframe(transactions, width="stretch", hide_index=True)


def _render_top_k(top_k: pd.DataFrame, *, partition_label: str = "Validation") -> None:
    if top_k.empty:
        st.info(
            "No saved Top-K curve artifact is available; aggregate values above are not "
            "interpolated."
        )
        return
    figure = go.Figure()
    for model, frame in top_k.groupby("model", sort=False):
        ordered = frame.sort_values("k", kind="mergesort")
        label = _display_model_name(model)
        figure.add_trace(
            go.Scatter(
                x=ordered["k"],
                y=ordered["recall_at_k"],
                name=f"{label} recall",
                mode="lines+markers",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=ordered["k"],
                y=ordered["precision_at_k"],
                name=f"{label} precision",
                mode="lines+markers",
                line={"dash": "dot"},
            )
        )
    figure.update_layout(
        title=f"Saved {partition_label.lower()} Recall@K and Precision@K",
        xaxis_title="Alert budget K",
        yaxis_title="Metric value",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    chart_key = f"model-top-k-{partition_label.casefold().replace(' ', '-')}"
    st.plotly_chart(figure, width="stretch", key=chart_key)


def _render_ablation(ablation: pd.DataFrame) -> None:
    if ablation.empty:
        st.info("No saved feature-family ablation is available.")
        return
    displayed = ablation.copy()
    displayed["feature_family"] = displayed["feature_family"].map(_humanize)
    figure = go.Figure(
        go.Bar(
            x=displayed["feature_family"].astype(str),
            y=displayed["pr_auc"],
            marker_color="#0f766e",
        )
    )
    figure.update_layout(
        title="Saved feature-family ablation",
        xaxis_title="Feature family",
        yaxis_title="Validation PR-AUC",
        height=390,
        margin={"l": 15, "r": 15, "t": 55, "b": 110},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", key="model-ablation")
    st.dataframe(
        displayed.rename(columns={"feature_family": "Feature Set", "pr_auc": "PR-AUC"}),
        width="stretch",
        hide_index=True,
    )


def _final_summary_value(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    for container_name in ("protocol", "final_test", "test", "freeze_contract"):
        container = summary.get(container_name)
        if isinstance(container, dict) and key in container:
            return container[key]
    return None


def _render_final_evaluation(artifacts: DashboardArtifacts) -> None:
    final = artifacts.final_evaluation
    if final is None:
        st.divider()
        st.info(
            "Final-test results are not included in the loaded artifact set. Validation model "
            "evidence and saved cases remain available."
        )
        return

    st.divider()
    st.markdown(
        '<div class="argus-eyebrow">FINAL MODEL EVIDENCE</div>',
        unsafe_allow_html=True,
    )
    st.header("Final evaluation")
    st.success(
        "Primary operational model: Graph-enhanced LightGBM. It was selected on validation "
        "before the single final-test evaluation; test metrics did not select or tune the model."
    )

    summary = final.summary
    validation_rate = float(_final_summary_value(summary, "validation_positive_rate"))
    test_rate_value = _final_summary_value(summary, "test_positive_rate")
    if test_rate_value is None:
        test_rate_value = _final_summary_value(summary, "positive_rate")
    test_rate = float(test_rate_value)
    prevalence = st.columns(3)
    prevalence[0].metric("Validation positive rate", f"{validation_rate:.6%}")
    prevalence[1].metric("Final-test positive rate", f"{test_rate:.6%}")
    ratio = f"{test_rate / validation_rate:.3f}×" if validation_rate > 0.0 else "N/A"
    prevalence[2].metric("Prevalence ratio", ratio)
    direction = "higher" if test_rate > validation_rate else "lower"
    st.caption(
        f"Final-test prevalence is {direction} than validation prevalence. Precision and the "
        "number/composition of alerts at the frozen threshold must be interpreted under this "
        "base-rate shift; the shift was not corrected by changing the model or threshold."
    )

    comparison = _ordered_models(final.model_comparison)
    st.dataframe(_display_model_table(comparison, final=True), width="stretch", hide_index=True)
    st.plotly_chart(
        build_model_metric_figure(comparison, partition_label="Final test"),
        width="stretch",
        key="final-model-comparison-bars",
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            build_pr_curve_figure(final.pr_curves, partition_label="Final test"),
            width="stretch",
            key="final-model-pr-curves",
        )
    with right:
        _render_top_k(final.top_k, partition_label="Final test")
    st.warning(
        "The final test was evaluated once after the model, feature set, and threshold were "
        "frozen. No post-test tuning or model reselection was performed."
    )


def render_model_comparison(artifacts: DashboardArtifacts) -> None:
    _page_heading(
        "VALIDATION AND FINAL RESULTS",
        "Model Evidence",
        "Compare the primary operational model with transaction-only and GraphSAGE research "
        "baselines.",
    )
    _render_final_evaluation(artifacts)
    st.divider()
    st.subheader("Validation evidence")
    comparison = _ordered_models(artifacts.model_comparison)
    st.dataframe(_display_model_table(comparison), width="stretch", hide_index=True)
    st.caption(
        "Graph-enhanced LightGBM is the operational ranking model. GraphSAGE is retained as a "
        "research comparator and did not outperform the graph-enhanced tabular model."
    )
    st.plotly_chart(
        build_model_metric_figure(comparison), width="stretch", key="model-comparison-bars"
    )

    left, right = st.columns(2)
    with left:
        if artifacts.pr_curves.empty:
            st.info("No saved PR-curve points are available; the app does not invent a curve.")
        else:
            st.plotly_chart(
                build_pr_curve_figure(artifacts.pr_curves),
                width="stretch",
                key="model-pr-curves",
            )
    with right:
        _render_top_k(artifacts.top_k)
    _render_ablation(artifacts.ablation)

    with st.expander("What these metrics mean for an investigation team"):
        st.markdown(
            """
            - **PR-AUC** measures rare-positive ranking quality across thresholds and is the
              primary metric.
            - **Recall@K** is the share of known positive validation transactions found within K
              alerts.
            - **Precision@K** is the positive share of that fixed analyst workload.
            - **FPR** is the share of negative validation transactions incorrectly flagged.
            - **Alert volume** is the number of transactions a team would need to review at the
              saved threshold.

            These validation metrics support model comparison; they are not final-test performance
            and do not establish that any person or account committed wrongdoing.
            """
        )


def main() -> None:
    """Run the saved-artifact-only Streamlit application."""

    st.set_page_config(page_title="ARGUS Network Investigator", page_icon="◈", layout="wide")
    _inject_style()
    st.sidebar.markdown("## ARGUS")
    st.sidebar.caption("Financial Crime Network Investigator")
    requested_page = st.query_params.get("page")
    page_index = _PAGES.index(requested_page) if requested_page in _PAGES else 0
    page = st.sidebar.radio("Workspace", _PAGES, index=page_index)
    artifact_root = configured_artifact_root()
    st.sidebar.divider()
    st.sidebar.caption("Saved results · No live model execution")

    try:
        artifacts = _cached_load(str(artifact_root))
    except ArtifactLoadError as exc:
        st.error("Saved analysis artifacts are unavailable.")
        st.info(
            "Create the required saved queue, case, and model-comparison artifacts before "
            "starting the application. The application does not train or score models on page load."
        )
        with st.expander("Technical details"):
            st.code(str(exc), language=None)
        st.stop()
        return

    if artifacts.final_evaluation is None:
        st.sidebar.caption("Validation evidence loaded")
    else:
        st.sidebar.caption("Final evaluation loaded")
        st.sidebar.caption("Primary model: Graph-enhanced LightGBM")

    renderers = {
        "Overview": render_executive_dashboard,
        "Investigations": render_investigation_queue,
        "Case Investigator": render_case_investigator,
        "Model Evidence": render_model_comparison,
    }
    renderers[page](artifacts)
    st.divider()
    st.caption(
        "ARGUS is a research decision-support prototype. Scores prioritize human review; "
        "they are not legal conclusions or automatic enforcement decisions."
    )


if __name__ == "__main__":
    main()
