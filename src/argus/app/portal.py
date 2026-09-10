"""Authenticated analyst workspace backed only by validated saved artifacts."""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from argus.app.artifacts import DashboardArtifacts
from argus.app.auth import ANALYST_EMAIL_KEY
from argus.app.figures import (
    build_model_metric_figure,
    build_network_figure,
    build_pr_curve_figure,
    build_timeline_figure,
)
from argus.app.view_models import (
    build_queue_view,
    canonical_token,
    case_currency_context,
    display_pattern,
    display_priority,
    display_queue,
    display_status,
    filter_queue_view,
    humanize,
    sort_queue_view,
)
from argus.app.workflow import (
    ADD_NOTE,
    CLOSE_CASE,
    ESCALATE,
    MARK_FALSE_POSITIVE,
    START_REVIEW,
    WORKFLOW_SESSION_KEY,
    apply_case_action,
    case_with_workflow,
    case_workflow,
)

PAGES = ("Overview", "Investigations", "Case Investigator", "Model Evidence")
PRIMARY_MODEL = "graph_enhanced_lightgbm"
SAVED_RESEARCH_CASE_NOTICE = (
    "Graph-enhanced LightGBM is the primary ranking model. The current saved case examples "
    "are drawn from the GraphSAGE research-comparator artifact set."
)

_MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "graph_enhanced_lightgbm": "Graph-enhanced LightGBM",
    "refined_transaction_lightgbm": "Refined transaction LightGBM",
    "graphsage_edge_classifier": "GraphSAGE",
    "graphsage": "GraphSAGE",
}
_FEATURE_LABELS = {
    "amount_paid": "Payment amount",
    "amount_received": "Received amount",
    "log_amount_paid": "Payment amount (log scale)",
    "log_amount_received": "Received amount (log scale)",
    "pair_previous_transfer_count": "Previous transfers on this route",
    "sender_seconds_since_previous": "Time since sender's previous transfer",
    "receiver_seconds_since_previous": "Time since receiver's previous transfer",
    "sender_previous_outgoing_amount": "Sender's previous outgoing amount",
    "receiver_previous_incoming_amount": "Receiver's previous incoming amount",
    "receiver_previous_transaction_count": "Receiver's earlier transfer count",
    "sender_previous_transaction_count": "Sender's earlier transfer count",
    "sender_prior_fan_in_degree": "Sender's earlier incoming connections",
    "sender_prior_fan_out_degree": "Sender's earlier outgoing connections",
    "receiver_prior_fan_in_degree": "Receiver's earlier incoming connections",
    "receiver_prior_fan_out_degree": "Receiver's earlier outgoing connections",
    "sender_burst_count_1h": "Sender activity in the previous hour",
    "receiver_burst_count_1h": "Receiver activity in the previous hour",
    "day_of_week": "Day of week",
    "hour": "Hour of day",
    "same_bank": "Same-bank route",
    "currency_match": "Matching send and receive currency",
    "from_bank": "Sending bank frequency",
    "payment_currency": "Payment currency",
    "receiving_currency": "Receiving currency",
    "payment_format": "Payment format",
}
_EVIDENCE_KIND_LABELS = {
    "transaction_value": "Transaction value",
    "directed_transfer_route": "Directed transfer route",
    "transaction_time_and_channel": "Transaction time and channel",
    "strictly_prior_directed_pair_activity": "Earlier route activity",
    "strictly_prior_sender_neighborhood": "Earlier sender activity",
    "strictly_prior_receiver_neighborhood": "Earlier receiver activity",
}
_EVIDENCE_SCOPE_LABELS = {
    "focal_transaction": "Focal transaction",
    "strictly_prior_supplied_neighborhood": "Earlier case context",
    "focal_transaction_strictly_prior_feature": "Earlier transaction history",
}


def _canonical_model_name(value: Any) -> str:
    return canonical_token(value)


def _display_model_name(value: Any) -> str:
    canonical = _canonical_model_name(value)
    return _MODEL_LABELS.get(canonical, humanize(value))


def _model_role(value: Any) -> str:
    canonical = _canonical_model_name(value)
    if canonical == PRIMARY_MODEL:
        return "Primary model"
    if canonical in {"graphsage", "graphsage_edge_classifier"}:
        return "Research comparator"
    return "Comparator"


def _ordered_models(comparison: pd.DataFrame) -> pd.DataFrame:
    order = {
        PRIMARY_MODEL: 0,
        "refined_transaction_lightgbm": 1,
        "lightgbm": 1,
        "graphsage_edge_classifier": 2,
        "graphsage": 2,
    }
    displayed = comparison.copy()
    displayed["_display_order"] = displayed["model"].map(
        lambda value: order.get(_canonical_model_name(value), 3)
    )
    return displayed.sort_values("_display_order", kind="mergesort").drop(columns="_display_order")


def _comparison_focus(artifacts: DashboardArtifacts) -> pd.Series:
    """Select only from validation evidence, never final-test rows."""

    comparison = artifacts.model_comparison
    for column in ("is_champion", "selected"):
        if column not in comparison:
            continue
        selected = comparison[
            comparison[column].map(
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


def _format_metric(value: Any, *, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_compact_amount(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(number) >= divisor:
            return f"{number / divisor:,.2f}{suffix}"
    return f"{number:,.2f}"


def _case_summary_card(column: Any, label: str, value: str, *, detail: str = "") -> None:
    title = html.escape(detail or value)
    with column:
        st.markdown(
            '<div class="case-summary-card">'
            f"<span>{html.escape(label)}</span>"
            f'<strong title="{title}">{html.escape(value)}</strong>'
            "</div>",
            unsafe_allow_html=True,
        )


def _page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<p class="argus-eyebrow">{html.escape(eyebrow)}</p>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def _render_empty_state(title: str, copy: str) -> None:
    st.markdown(
        '<div class="argus-empty-state">'
        '<svg viewBox="0 0 64 64" aria-hidden="true">'
        '<circle cx="28" cy="28" r="17"></circle>'
        '<path d="M40 40l12 12M20 28h16M28 20v16"></path>'
        "</svg>"
        f"<div><strong>{html.escape(title)}</strong><span>{html.escape(copy)}</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_research_case_indicator(*, selection: bool = False) -> None:
    label = "Research case selection" if selection else "Research case set"
    detail = (
        "These case examples were selected using the GraphSAGE research comparator. "
        "Graph-enhanced LightGBM remains the primary model."
    )
    st.markdown(
        '<div class="argus-provenance-line">'
        f'<span class="argus-provenance-badge" tabindex="0" title="{html.escape(detail)}">'
        f'{html.escape(label)} <span aria-hidden="true">ⓘ</span></span>'
        f'<span class="argus-provenance-detail">{html.escape(detail)}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def _selected_dataframe_rows(event: Any) -> tuple[int, ...]:
    """Read Streamlit's supported dataframe selection event defensively."""

    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, Mapping):
        selection = event.get("selection")
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, Mapping):
        rows = selection.get("rows", [])
    if not isinstance(rows, (list, tuple)):
        return ()
    selected: list[int] = []
    for row in rows:
        try:
            index = int(row)
        except (TypeError, ValueError):
            continue
        if index >= 0:
            selected.append(index)
    return tuple(selected)


def _workflow_store() -> Mapping[str, Mapping[str, Any]]:
    value = st.session_state.get(WORKFLOW_SESSION_KEY, {})
    return value if isinstance(value, Mapping) else {}


def _queue_view(artifacts: DashboardArtifacts) -> pd.DataFrame:
    return build_queue_view(artifacts.queue, artifacts.cases, _workflow_store())


def _activity_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case_id, overlay in _workflow_store().items():
        activity = overlay.get("activity", []) if isinstance(overlay, Mapping) else []
        if not isinstance(activity, list):
            continue
        for item in activity:
            if isinstance(item, dict):
                records.append({"case_id": case_id, **item})
    return sorted(records, key=lambda item: str(item.get("timestamp", "")), reverse=True)


def render_overview(artifacts: DashboardArtifacts, *, open_case: Callable[[str], None]) -> None:
    _page_heading(
        "ANALYST WORKSPACE",
        "Overview",
        "Open investigations, case context and current review activity in one operational view.",
    )
    worklist = _queue_view(artifacts)
    active_statuses = {"pending_human_review", "pending_review", "in_review", "escalated"}
    active = int(worklist["status"].map(canonical_token).isin(active_statuses).sum())
    in_progress = int(
        worklist["status"].map(canonical_token).isin({"in_review", "escalated"}).sum()
    )
    cases_with_context = int(
        pd.to_numeric(worklist["transaction_count"], errors="coerce").fillna(0).gt(1).sum()
    )
    activity = _activity_records()

    metrics = st.columns(4)
    metrics[0].metric("Open cases", _format_count(active))
    metrics[1].metric("In review", _format_count(in_progress))
    metrics[2].metric(
        "Prior context",
        _format_count(cases_with_context),
        help="Cases containing more than one transaction in the available case context.",
    )
    metrics[3].metric("Analyst actions", _format_count(len(activity)))

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.subheader("Cases requiring attention")
        attention = worklist[
            worklist["status"].map(canonical_token).isin(active_statuses)
        ].sort_values(["priority_rank", "last_activity"], ascending=[True, False])
        if attention.empty:
            _render_empty_state(
                "No cases require attention",
                "Cases that enter review or escalation will appear here.",
            )
        else:
            for row in attention.head(5).to_dict(orient="records"):
                case_id = str(row["case_id"])
                with st.container(border=True):
                    detail, action = st.columns([3.2, 1])
                    detail.markdown(f"**{html.escape(case_id)}**")
                    detail.caption(
                        f"{row['priority_label']} priority · {row['pattern_label']} · "
                        f"{row['status_label']}"
                    )
                    if action.button(
                        "Review",
                        key=f"overview-review-{case_id}",
                        width="stretch",
                    ):
                        open_case(case_id)
    with right:
        priority_bands = worklist["priority_label"].dropna().nunique()
        priority_counts = (
            worklist["priority_label"]
            .value_counts()
            .rename_axis("Priority")
            .reset_index(name="Cases")
        )
        if priority_bands > 1:
            st.subheader("Priority distribution")
            chart_x = priority_counts["Cases"]
            chart_y = priority_counts["Priority"]
            x_title = "Cases"
            y_title = None
            orientation = "h"
        else:
            st.subheader("Case composition")
            transfer_counts = (
                pd.to_numeric(worklist["transaction_count"], errors="coerce")
                .fillna(0)
                .astype(int)
                .value_counts()
                .sort_index()
                .rename_axis("Transfers")
                .reset_index(name="Cases")
            )
            chart_x = transfer_counts["Transfers"]
            chart_y = transfer_counts["Cases"]
            x_title = "Transfers in case context"
            y_title = "Cases"
            orientation = "v"
        figure = go.Figure(
            go.Bar(
                x=chart_x,
                y=chart_y,
                orientation=orientation,
                marker_color="#0F6B66",
                text=priority_counts["Cases"] if priority_bands > 1 else chart_y,
                textposition="auto",
            )
        )
        figure.update_layout(
            height=250,
            margin={"l": 8, "r": 8, "t": 12, "b": 22},
            xaxis_title=x_title,
            yaxis_title=y_title,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        st.caption("Composition is derived from the transaction context stored with each case.")

    st.subheader("Recent activity")
    if not activity:
        _render_empty_state(
            "No recent activity",
            "Start a review or add a note to build the activity trail.",
        )
    else:
        activity_frame = pd.DataFrame(activity[:8])
        displayed = activity_frame[["timestamp", "case_id", "label", "analyst"]].rename(
            columns={
                "timestamp": "Time",
                "case_id": "Case ID",
                "label": "Activity",
                "analyst": "Analyst",
            }
        )
        st.dataframe(displayed, width="stretch", hide_index=True)
    st.caption("Demo actions reset after sign-out.")


def _queue_model_names(artifacts: DashboardArtifacts) -> tuple[str, ...]:
    names: set[str] = set()
    for case in artifacts.cases.values():
        model = case.get("model_evidence", {})
        if isinstance(model, dict) and model.get("model_name"):
            names.add(str(model["model_name"]))
    return tuple(sorted(names))


def _reset_queue_filters() -> None:
    for key in (
        "queue_search",
        "queue_priorities",
        "queue_statuses",
        "queue_patterns",
        "queue_sort",
        "investigation_worklist_table",
        "investigation_selected_case_id",
    ):
        st.session_state.pop(key, None)


def render_investigations(
    artifacts: DashboardArtifacts, *, open_case: Callable[[str], None]
) -> None:
    _page_heading(
        "INVESTIGATION WORKLIST",
        "Investigations",
        "Search, filter and open investigation cases for human review.",
    )
    if all(
        _canonical_model_name(name) in {"graphsage", "graphsage_edge_classifier"}
        for name in _queue_model_names(artifacts)
    ):
        _render_research_case_indicator()
    worklist = _queue_view(artifacts)
    if worklist.empty:
        _render_empty_state(
            "No investigation cases",
            "The current artifact package does not contain a case worklist.",
        )
        return

    search_col, sort_col, reset_col = st.columns([2, 1, 0.65])
    query = search_col.text_input(
        "Search",
        key="queue_search",
        placeholder="Case ID, account or review signal",
        help="Searches identifiers and the context available with these cases.",
    )
    sort_label = sort_col.selectbox(
        "Sort by",
        ("Priority", "Latest activity", "Total flow", "Transfers"),
        key="queue_sort",
    )
    reset_col.markdown('<div class="filter-button-spacer"></div>', unsafe_allow_html=True)
    if reset_col.button("Reset filters", key="queue_reset", width="stretch"):
        _reset_queue_filters()
        st.rerun()

    filter_columns = st.columns(3)
    priority_values = sorted(worklist["priority_label"].unique())
    status_values = sorted(worklist["status_label"].unique())
    pattern_values = sorted(worklist["pattern_label"].unique())
    priorities = filter_columns[0].multiselect(
        "Priority", priority_values, default=priority_values, key="queue_priorities"
    )
    statuses = filter_columns[1].multiselect(
        "Status", status_values, default=status_values, key="queue_statuses"
    )
    patterns = filter_columns[2].multiselect(
        "Review signal", pattern_values, default=pattern_values, key="queue_patterns"
    )
    filtered = filter_queue_view(
        worklist,
        query=query,
        priorities=priorities,
        statuses=statuses,
        patterns=patterns,
    )
    filtered = sort_queue_view(filtered, sort_label)
    if filtered.empty:
        _render_empty_state(
            "No matching cases",
            "Adjust the filters or reset the worklist to continue.",
        )
        if st.button("Reset filters", key="queue_empty_reset"):
            _reset_queue_filters()
            st.rerun()
        return

    table_event = st.dataframe(
        display_queue(filtered),
        width="stretch",
        hide_index=True,
        height=min(650, 86 + 35 * len(filtered)),
        key="investigation_worklist_table",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Priority": st.column_config.TextColumn(width="small"),
            "Case ID": st.column_config.TextColumn(
                width="small", help="The full case ID is shown in the selected-case action panel."
            ),
            "Review Signal": st.column_config.TextColumn(width="medium"),
            "Accounts": st.column_config.NumberColumn(width="small", format="%d"),
            "Transfers": st.column_config.NumberColumn(width="small", format="%d"),
            "Total Flow": st.column_config.TextColumn(width="small"),
            "Last Activity": st.column_config.TextColumn(width="small", help="Displayed in UTC."),
            "Status": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(
        f"Showing {len(filtered):,} of {len(worklist):,} investigation cases. Total-flow values "
        "preserve their recorded currency context; mixed-currency rows are not converted."
    )
    selected_rows = _selected_dataframe_rows(table_event)
    available_ids = filtered["case_id"].astype(str).tolist()
    if selected_rows and selected_rows[0] < len(available_ids):
        st.session_state["investigation_selected_case_id"] = available_ids[selected_rows[0]]
    selected = str(st.session_state.get("investigation_selected_case_id", available_ids[0]))
    if selected not in available_ids:
        selected = available_ids[0]
        st.session_state["investigation_selected_case_id"] = selected
    selected_row = filtered.loc[filtered["case_id"].astype(str).eq(selected)].iloc[0]
    with st.container(key="investigation_action_panel"):
        detail, action = st.columns([4, 1], vertical_alignment="center")
        detail.markdown(f"**{html.escape(selected)}**")
        detail.caption(
            f"{selected_row['priority_label']} priority · {selected_row['pattern_label']} · "
            f"{selected_row['status_label']}"
        )
        if action.button(
            "Open case",
            type="primary",
            key="investigation_open_case",
            width="stretch",
        ):
            open_case(selected)


def _case_counts(case: Mapping[str, Any], queue_row: Mapping[str, Any]) -> tuple[int, int]:
    network = case.get("network", {})
    nodes = network.get("nodes", []) if isinstance(network, dict) else []
    transactions = case.get("transactions", [])
    accounts = queue_row.get("account_count", len(nodes) if isinstance(nodes, list) else 0)
    transfer_count = queue_row.get(
        "transaction_count", len(transactions) if isinstance(transactions, list) else 0
    )
    return int(accounts or 0), int(transfer_count or 0)


def _display_evidence_statement(value: Any) -> str:
    return (
        str(value)
        .replace("the supplied neighborhood", "the earlier case context")
        .replace("The supplied neighborhood", "The earlier case context")
        .replace("transfer(s)", "transfers")
        .replace("transaction(s)", "transactions")
    )


def _render_observed_evidence(case: Mapping[str, Any]) -> None:
    st.markdown('<div class="section-kicker">OBSERVED EVIDENCE</div>', unsafe_allow_html=True)
    st.subheader("What the records show")
    evidence = case.get("observed_evidence", [])
    if not isinstance(evidence, list) or not evidence:
        st.info("No observed evidence is available for this case.")
        return
    visible_evidence = [
        item
        for item in evidence
        if not (
            isinstance(item, dict)
            and canonical_token(item.get("kind")) == "precomputed_strictly_prior_features"
        )
    ]
    if not visible_evidence:
        st.info("No analyst-facing observed evidence is available for this case.")
        return
    if len(visible_evidence) < 3:
        st.warning(
            f"Only {len(visible_evidence)} observed evidence item(s) are available for this case."
        )
    for item in visible_evidence:
        if not isinstance(item, dict) or not item.get("statement"):
            continue
        kind_token = canonical_token(item.get("kind"))
        kind = html.escape(
            _EVIDENCE_KIND_LABELS.get(kind_token, humanize(item.get("kind", "Observed fact")))
        )
        statement = html.escape(_display_evidence_statement(item["statement"]))
        scope_token = canonical_token(item.get("scope"))
        scope = _EVIDENCE_SCOPE_LABELS.get(scope_token, humanize(item.get("scope")))
        st.markdown(
            '<div class="evidence-card observed">'
            f'<div class="evidence-kind">{kind}</div><p>{statement}</p>'
            f'<span class="evidence-scope">Context: {html.escape(scope)}</span></div>',
            unsafe_allow_html=True,
        )


def _display_feature_name(value: Any) -> str:
    raw = str(value).split("::")[-1]
    for prefix in ("numeric__", "categorical__", "frequency__", "onehot__"):
        if raw.startswith(prefix):
            raw = raw.removeprefix(prefix)
    if "=" in raw:
        feature, category = raw.split("=", 1)
        base = _FEATURE_LABELS.get(feature, humanize(feature))
        return f"{base}: {humanize(category)}"
    embedding = re.fullmatch(r"(sender|receiver)_embedding_(\d+)", raw)
    if embedding:
        party, component = embedding.groups()
        return f"{party.title()} network context component {int(component) + 1}"
    burst = re.fullmatch(r"(sender|receiver)_burst_count_(\d+)([hd])", raw)
    if burst:
        party, duration, unit = burst.groups()
        unit_label = "hour" if unit == "h" else "day"
        suffix = "" if duration == "1" else "s"
        return f"{party.title()} activity in the previous {duration} {unit_label}{suffix}"
    rolling = re.fullmatch(r"(sender|receiver)_rolling_(incoming|outgoing)_amount_(\d+)([hd])", raw)
    if rolling:
        party, direction, duration, unit = rolling.groups()
        unit_label = "hour" if unit == "h" else "day"
        suffix = "" if duration == "1" else "s"
        return f"{party.title()} {direction} amount in the previous {duration} {unit_label}{suffix}"
    return _FEATURE_LABELS.get(raw, humanize(raw))


def _render_contributions(
    contributions: Any,
    *,
    title: str,
    key: str,
) -> None:
    if not isinstance(contributions, list) or not contributions:
        st.caption("No local feature-contribution record is available.")
        return
    frame = pd.DataFrame(contributions)
    feature_column = next(
        (name for name in ("feature", "feature_name", "name") if name in frame), None
    )
    value_column = next(
        (name for name in ("contribution", "shap_value", "attribution") if name in frame), None
    )
    if feature_column is None or value_column is None:
        st.caption("The saved explanation cannot be displayed in chart form.")
        return
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame["Readable factor"] = frame[feature_column].map(_display_feature_name)
    chart = frame.dropna(subset=[value_column]).copy()
    chart = chart.reindex(chart[value_column].abs().sort_values().index).tail(10)
    if chart.empty:
        st.caption("The saved explanation contains no finite contribution values.")
        return
    figure = go.Figure(
        go.Bar(
            x=chart[value_column],
            y=chart["Readable factor"],
            orientation="h",
            marker_color=["#B42318" if value >= 0 else "#31688E" for value in chart[value_column]],
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Contribution to the saved model score",
        height=max(320, len(chart) * 34),
        margin={"l": 12, "r": 12, "t": 52, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch", key=key, config={"displaylogo": False})
    with st.expander("Technical details"):
        technical = frame[[feature_column, value_column]].rename(
            columns={feature_column: "Saved feature", value_column: "Contribution"}
        )
        st.dataframe(technical, width="stretch", hide_index=True)


def _render_model_evidence(
    case_id: str, case: Mapping[str, Any], artifacts: DashboardArtifacts
) -> None:
    st.markdown('<div class="section-kicker model">MODEL EVIDENCE</div>', unsafe_allow_html=True)
    st.subheader("How models inform this review")
    _render_research_case_indicator(selection=True)
    saved = case.get("model_evidence", {})
    if not isinstance(saved, dict) or not saved:
        st.info("No model evidence is available for this case.")
    else:
        model_name = saved.get("model_name", "GraphSAGE")
        with st.container(border=True):
            st.markdown("#### Research comparator perspective")
            st.caption("Selected by GraphSAGE research comparator.")
            metrics = st.columns(3)
            metrics[0].metric("Model", _display_model_name(model_name))
            metrics[1].metric(
                "Research ranking score", _format_metric(saved.get("score"), digits=6)
            )
            metrics[2].metric("Case-set rank", _format_count(saved.get("rank")))
            st.write(
                "The GraphSAGE score selected this research case example. It is not a "
                "probability of wrongdoing. Graph-enhanced LightGBM remains the primary model."
            )
            _render_contributions(
                saved.get("feature_contributions", case.get("feature_contributions")),
                title="Factors influencing the research-comparator score",
                key=f"graphsage-contributions-{case_id}",
            )

    primary = artifacts.primary_explanations.get(case_id)
    if primary:
        with st.container(border=True):
            st.markdown("#### Primary model perspective")
            st.caption("Graph-enhanced LightGBM · Native TreeSHAP · Validation case")
            st.write(
                "TreeSHAP shows how the primary model evaluated the same transaction. It did not "
                "generate or rank this research case set."
            )
            _render_contributions(
                primary.get("top_contributions"),
                title="Network and transaction-history contributions",
                key=f"lightgbm-contributions-{case_id}",
            )
            with st.expander("Primary model score details"):
                st.write(f"Raw score: {_format_metric(primary.get('raw_score'), digits=6)}")
                st.write(f"Baseline value: {_format_metric(primary.get('base_value'), digits=6)}")
                st.caption("Raw model scores are technical values, not calibrated probabilities.")
    else:
        st.caption("No compatible primary-model explanation exists for this case.")
    st.warning(
        "Model evidence supports prioritization only. It does not establish wrongdoing or "
        "authorize an automatic account action."
    )


def _transactions_frame(case: Mapping[str, Any]) -> pd.DataFrame:
    records = case.get("transactions", [])
    if not isinstance(records, list) or not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    aliases = {
        "transaction_id": "Transaction ID",
        "timestamp": "Timestamp",
        "from_node_id": "Sender",
        "to_node_id": "Receiver",
        "amount": "Amount",
        "currency": "Currency",
        "payment_format": "Payment Format",
        "is_focal": "Focal Transaction",
    }
    columns = [column for column in aliases if column in frame]
    displayed = frame[columns].copy()
    if "timestamp" in displayed:
        timestamp = pd.to_datetime(displayed["timestamp"], errors="coerce", utc=True)
        displayed["timestamp"] = timestamp.dt.strftime("%d %b %Y · %H:%M:%S UTC").fillna(
            "Not available"
        )
    return displayed.rename(columns=aliases)


def _network_options(case: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    options: dict[str, tuple[str, Mapping[str, Any]]] = {}
    network = case.get("network", {})
    nodes = network.get("nodes", []) if isinstance(network, dict) else []
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id", node.get("id", "Unknown")))
            options[f"Account · {node_id}"] = ("account", node)
    transactions = case.get("transactions", [])
    if isinstance(transactions, list):
        for transaction in transactions:
            if not isinstance(transaction, dict):
                continue
            transaction_id = str(transaction.get("transaction_id", "Unknown"))
            prefix = "Focal transfer" if transaction.get("is_focal") else "Transfer"
            options[f"{prefix} · {transaction_id}"] = ("transaction", transaction)
    return options


def _render_network_inspector(case: Mapping[str, Any]) -> None:
    options = _network_options(case)
    st.markdown("#### Inspect network context")
    if not options:
        st.info("No account or transfer details are available for selection.")
        return
    selected = st.selectbox(
        "Account or transfer",
        list(options),
        key=f"network-selection-{case.get('case_id', 'case')}",
    )
    kind, record = options[selected]
    if kind == "account":
        account_id = record.get("node_id", record.get("id", "Not available"))
        st.write(f"**Account:** `{account_id}`")
        st.write(f"**Network role:** {humanize(record.get('role'))}")
        st.write(f"**Incoming links in case:** {_format_count(record.get('case_in_degree'))}")
        st.write(f"**Outgoing links in case:** {_format_count(record.get('case_out_degree'))}")
        st.caption("Only attributes saved in the case context are shown.")
        return
    st.write(f"**Sender:** `{record.get('from_node_id', 'Not available')}`")
    st.write(f"**Receiver:** `{record.get('to_node_id', 'Not available')}`")
    st.write(
        f"**Amount:** {_format_money(record.get('amount'))} {record.get('currency', '')}".rstrip()
    )
    st.write(f"**Timestamp:** {record.get('timestamp', 'Not available')}")
    st.write(f"**Payment format:** {record.get('payment_format') or 'Not available'}")


def _flash_message() -> None:
    message = st.session_state.pop("argus_workflow_flash", None)
    if message:
        st.success(str(message))


def _actor() -> str:
    return str(st.session_state.get(ANALYST_EMAIL_KEY) or "Demo analyst")


def _apply_action(case_id: str, case: Mapping[str, Any], action: str, **kwargs: Any) -> None:
    result = apply_case_action(
        st.session_state,
        case_id,
        action,
        source_case=case,
        actor=_actor(),
        **kwargs,
    )
    if result.accepted:
        st.session_state["argus_workflow_flash"] = result.message
    else:
        st.session_state["argus_workflow_error"] = result.message


def _render_action_bar(case_id: str, case: Mapping[str, Any], status: str) -> None:
    terminal = canonical_token(status) in {"false_positive", "closed"}
    with st.container(key="case_action_bar"):
        st.markdown("#### Case actions")
        st.caption("Demo actions reset after sign-out.")
        actions = st.columns([1.15, 1, 1.35, 1, 2.2])
        if actions[0].button(
            "Start Review",
            type="primary",
            key=f"start-review-{case_id}",
            disabled=canonical_token(status) not in {"pending_review", "pending_human_review"},
            width="stretch",
        ):
            _apply_action(case_id, case, START_REVIEW)
            st.rerun()
        if actions[1].button(
            "Escalate",
            key=f"escalate-{case_id}",
            disabled=terminal or canonical_token(status) == "escalated",
            width="stretch",
        ):
            st.session_state["argus_pending_case_action"] = {
                "case_id": case_id,
                "action": ESCALATE,
            }
        if actions[2].button(
            "Mark False Positive",
            key=f"false-positive-{case_id}",
            disabled=terminal,
            width="stretch",
        ):
            st.session_state["argus_pending_case_action"] = {
                "case_id": case_id,
                "action": MARK_FALSE_POSITIVE,
            }
        if actions[3].button(
            "Close Case", key=f"close-{case_id}", disabled=terminal, width="stretch"
        ):
            st.session_state["argus_pending_case_action"] = {
                "case_id": case_id,
                "action": CLOSE_CASE,
            }
        actions[4].empty()

    pending_state = st.session_state.get("argus_pending_case_action")
    if not isinstance(pending_state, Mapping) or pending_state.get("case_id") != case_id:
        return
    pending = pending_state.get("action")
    if pending not in {ESCALATE, MARK_FALSE_POSITIVE, CLOSE_CASE}:
        return
    labels = {
        ESCALATE: "Escalate this case for further review?",
        MARK_FALSE_POSITIVE: "Mark this case as a false positive?",
        CLOSE_CASE: "Close this case?",
    }
    st.warning(labels[str(pending)])
    confirm, cancel, _ = st.columns([1, 1, 3])
    if confirm.button("Confirm", type="primary", key=f"confirm-{case_id}-{pending}"):
        _apply_action(case_id, case, str(pending), confirmed=True)
        st.session_state.pop("argus_pending_case_action", None)
        st.rerun()
    if cancel.button("Cancel", key=f"cancel-{case_id}-{pending}"):
        st.session_state.pop("argus_pending_case_action", None)
        st.rerun()


def _render_notes_and_activity(case_id: str, case: Mapping[str, Any]) -> None:
    overlay = case_workflow(st.session_state, case_id, source_case=case)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Analyst notes")
        with st.form(f"case-note-form-{case_id}", clear_on_submit=True):
            note = st.text_area(
                "Add a note",
                placeholder="Record the next verification step or investigation context.",
                max_chars=2000,
            )
            submitted = st.form_submit_button("Save note", type="primary")
        if submitted:
            result = apply_case_action(
                st.session_state,
                case_id,
                ADD_NOTE,
                source_case=case,
                note=note,
                actor=_actor(),
            )
            if result.accepted:
                st.success(result.message)
                overlay = case_workflow(st.session_state, case_id, source_case=case)
            else:
                st.error(result.message)
        notes = overlay.get("notes", [])
        if not notes:
            st.caption("No analyst notes yet.")
        for item in reversed(notes):
            st.markdown(f"**{item['analyst']}** · {item['timestamp']}  \n{item['text']}")
    with right:
        st.subheader("Activity log")
        activity = overlay.get("activity", [])
        if not activity:
            st.caption("No case activity yet.")
        for item in reversed(activity):
            st.markdown(f"**{item['label']}**  \n{item['timestamp']} · {item['analyst']}")
    st.caption("Demo notes and actions reset after sign-out.")


def render_case_investigator(artifacts: DashboardArtifacts) -> None:
    _page_heading(
        "CASE REVIEW",
        "Case Investigator",
        "Understand the account network, observed facts and model contribution before deciding.",
    )
    queue = artifacts.queue
    case_ids = queue["case_id"].astype(str).tolist()
    if not case_ids:
        _render_empty_state(
            "No cases to investigate",
            "Open a case from the investigation worklist when one becomes available.",
        )
        return
    requested = str(st.session_state.get("argus_selected_case_id", case_ids[0]))
    if requested not in artifacts.cases:
        st.warning(
            "The selected case is unavailable. The first investigation case is shown instead."
        )
        requested = case_ids[0]
        st.session_state["argus_selected_case_id"] = requested
    index = case_ids.index(requested) if requested in case_ids else 0
    widget_value = st.session_state.get("case_investigator_select")
    selected_case_id = st.selectbox(
        "Case",
        case_ids,
        index=None if widget_value in case_ids else index,
        key="case_investigator_select",
    )
    if selected_case_id is None:
        selected_case_id = requested
    if selected_case_id != requested:
        st.session_state.pop("argus_pending_case_action", None)
    st.session_state["argus_selected_case_id"] = selected_case_id
    source_case = artifacts.cases.get(selected_case_id)
    if not isinstance(source_case, dict):
        st.error("This case cannot be displayed.")
        return
    queue_rows = queue.loc[queue["case_id"].astype(str).eq(selected_case_id)]
    if queue_rows.empty:
        st.error("This case is not present in the saved worklist.")
        return
    queue_row = queue_rows.iloc[0].to_dict()
    overlay = case_workflow(st.session_state, selected_case_id, source_case=source_case)
    case = case_with_workflow(source_case, overlay)
    status = str(case.get("status", "pending_review"))
    priority_value = case.get("priority", queue_row.get("priority"))
    if isinstance(priority_value, dict):
        priority_value = priority_value.get("band")
    pattern = queue_row.get("major_pattern")
    accounts, transfer_count = _case_counts(case, queue_row)

    title_left, title_right = st.columns([2, 1])
    with title_left:
        st.markdown(f"### {selected_case_id}")
        st.write(
            f"Review **{accounts:,} account(s)** across **{transfer_count:,} transfer(s)**, "
            "then document the appropriate investigation decision."
        )
    with title_right:
        st.markdown(
            f'<div class="case-status-card"><span>{html.escape(display_priority(priority_value))} '
            f"priority</span><strong>{html.escape(display_status(status))}</strong></div>",
            unsafe_allow_html=True,
        )
    summary = st.columns(4)
    _case_summary_card(summary[0], "Review signal", display_pattern(pattern))
    _case_summary_card(summary[1], "Accounts", _format_count(accounts))
    _case_summary_card(summary[2], "Transfers", _format_count(transfer_count))
    total_flow = queue_row.get("total_flow")
    currency_context = case_currency_context(case)
    displayed_flow = (
        "Mixed currencies"
        if currency_context == "Mixed currencies"
        else _format_compact_amount(total_flow)
    )
    _case_summary_card(
        summary[3],
        "Recorded total flow",
        displayed_flow,
        detail="" if currency_context == "Mixed currencies" else _format_money(total_flow),
    )
    st.caption(
        f"Currency context: {currency_context}. Mixed-currency case values are not summed or "
        "converted in this view."
    )
    _flash_message()
    error_message = st.session_state.pop("argus_workflow_error", None)
    if error_message:
        st.error(str(error_message))
    _render_action_bar(selected_case_id, source_case, status)

    st.divider()
    canvas, details = st.columns([2.15, 1], gap="large")
    with canvas:
        st.subheader("Account network")
        network = build_network_figure(case)
        if network is None:
            st.warning("This case has no saved directed-network records to visualize.")
        else:
            st.plotly_chart(
                network,
                width="stretch",
                key=f"case-network-{selected_case_id}",
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "modeBarButtonsToAdd": ["resetScale2d"],
                },
            )
            st.caption(
                "Arrows show sender to receiver. Amber identifies the focal transfer; use hover, "
                "zoom, pan or the reset control to inspect the case context."
            )
    with details:
        _render_network_inspector(case)

    timeline = build_timeline_figure(case)
    if timeline is not None:
        st.plotly_chart(
            timeline,
            width="stretch",
            key=f"case-timeline-{selected_case_id}",
            config={"displaylogo": False},
        )
        st.caption(
            "Timeline amounts use each transaction's recorded currency. Values are not "
            "converted or compared across currencies."
        )

    st.divider()
    _render_observed_evidence(case)
    st.divider()
    _render_model_evidence(selected_case_id, case, artifacts)

    transactions = _transactions_frame(case)
    if not transactions.empty:
        st.subheader("Transactions in this case")
        st.dataframe(transactions, width="stretch", hide_index=True)
    st.divider()
    _render_notes_and_activity(selected_case_id, source_case)


def _display_model_table(comparison: pd.DataFrame, *, final: bool = False) -> pd.DataFrame:
    ordered = _ordered_models(comparison)
    displayed = pd.DataFrame(index=ordered.index)
    displayed["Role"] = ordered["model"].map(_model_role)
    displayed["Model"] = ordered["model"].map(_display_model_name)
    if not final and "feature_family" in ordered:
        displayed["Feature set"] = ordered["feature_family"].map(humanize)
    metric_labels = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "fpr": "FPR",
        "alerts": "Alerts",
    }
    for source, label in metric_labels.items():
        if source in ordered:
            displayed[label] = ordered[source]
    return displayed.reset_index(drop=True)


def _render_top_k(top_k: pd.DataFrame, *, partition_label: str = "Final test") -> None:
    if top_k.empty:
        st.info("No saved Top-K evidence is available.")
        return
    displayed = top_k.copy()
    displayed["model"] = displayed["model"].map(_display_model_name)
    columns = [
        column for column in ("model", "k", "recall_at_k", "precision_at_k") if column in displayed
    ]
    st.dataframe(
        displayed[columns].rename(
            columns={
                "model": "Model",
                "k": "K",
                "recall_at_k": "Recall@K",
                "precision_at_k": "Precision@K",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(f"Saved {partition_label.lower()} values at fixed analyst workload sizes.")


def _render_ablation(ablation: pd.DataFrame) -> None:
    if ablation.empty:
        st.info("No saved feature-family ablation is available.")
        return
    displayed = ablation.copy()
    displayed["_feature_family"] = displayed["feature_family"].map(canonical_token)
    displayed["feature_family"] = displayed["feature_family"].map(humanize)
    figure = go.Figure(
        go.Bar(
            x=displayed["pr_auc"],
            y=displayed["feature_family"],
            orientation="h",
            marker_color="#0F6B66",
            text=displayed["pr_auc"].map(lambda value: f"{float(value):.3f}"),
            textposition="auto",
        )
    )
    figure.update_layout(
        title="How additional context changed validation ranking quality",
        xaxis_title="Validation PR-AUC",
        yaxis_title=None,
        height=360,
        margin={"l": 12, "r": 12, "t": 52, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    if len(displayed) >= 2:
        graph_rows = displayed[
            displayed["_feature_family"].eq("transaction_temporal_history_graph")
        ]
        history_rows = displayed[displayed["_feature_family"].eq("transaction_temporal_history")]
        if not graph_rows.empty and not history_rows.empty:
            improvement = float(graph_rows.iloc[-1]["pr_auc"]) - float(
                history_rows.iloc[-1]["pr_auc"]
            )
            st.write(
                "Adding directed account-network context improved validation PR-AUC by "
                f"**{improvement:.6f}** over transaction and history context alone."
            )


def _final_summary_value(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    for container_name in ("protocol", "final_test", "test", "freeze_contract"):
        container = summary.get(container_name)
        if isinstance(container, dict) and key in container:
            return container[key]
    return None


def render_model_evidence(artifacts: DashboardArtifacts) -> None:
    _page_heading(
        "TECHNICAL EVIDENCE",
        "Model Evidence",
        "Review the frozen primary model and its research comparators away from daily case triage.",
    )
    final = artifacts.final_evaluation
    if final is None:
        st.info("Final-test evidence is not included in this saved artifact set.")
    else:
        comparison = _ordered_models(final.model_comparison)
        champion_rows = comparison[comparison["is_frozen_champion"]]
        if not champion_rows.empty:
            champion = champion_rows.iloc[0]
            st.markdown("### Frozen primary model")
            st.write(
                "**Graph-enhanced LightGBM** was selected using validation evidence before the "
                "single final-test evaluation."
            )
            metrics = st.columns(2)
            metrics[0].metric("Final PR-AUC", _format_metric(champion.get("pr_auc"), digits=6))
            metrics[1].metric("Final ROC-AUC", _format_metric(champion.get("roc_auc"), digits=6))
            threshold_values = (
                ("Precision", _format_metric(champion.get("precision"), digits=6)),
                ("Recall", _format_metric(champion.get("recall"), digits=6)),
                ("F1", _format_metric(champion.get("f1"), digits=6)),
                ("FPR", _format_metric(champion.get("fpr"), digits=6)),
                ("Alert volume", _format_count(champion.get("alerts"))),
            )
            items = "".join(
                f"<div><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
                for label, value in threshold_values
            )
            st.markdown(
                f'<div class="model-threshold-strip">{items}</div>',
                unsafe_allow_html=True,
            )
        st.subheader("Final model comparison")
        st.dataframe(_display_model_table(comparison, final=True), width="stretch", hide_index=True)
        st.plotly_chart(
            build_model_metric_figure(comparison, partition_label="Final test"),
            width="stretch",
            key="final-model-comparison",
            config={"displayModeBar": False},
        )
        st.subheader("Top-K review evidence")
        _render_top_k(final.top_k)
        with st.expander("Precision–recall curves"):
            st.plotly_chart(
                build_pr_curve_figure(final.pr_curves, partition_label="Final test"),
                width="stretch",
                key="final-pr-curves",
                config={"displaylogo": False},
            )

        validation_rate = _final_summary_value(final.summary, "validation_positive_rate")
        test_rate = _final_summary_value(final.summary, "test_positive_rate")
        if validation_rate is not None and test_rate is not None:
            st.caption(
                f"Validation positive rate: {float(validation_rate):.6%} · Final-test positive "
                f"rate: {float(test_rate):.6%}. Precision and alert volume are interpreted under "
                "this observed prevalence shift."
            )

    st.subheader("What network context added")
    _render_ablation(artifacts.ablation)
    with st.expander("Validation comparison and metric guide"):
        validation = _ordered_models(artifacts.model_comparison)
        st.dataframe(_display_model_table(validation), width="stretch", hide_index=True)
        st.markdown(
            """
            - **PR-AUC** is the primary rare-positive ranking metric.
            - **Precision and recall** describe the trade-off at the saved threshold.
            - **FPR** is the share of negative transactions flagged.
            - **Alert volume** is the review workload produced by the saved threshold.
            - **Precision@K and Recall@K** describe performance at fixed review budgets.
            """
        )
    st.warning(
        "IBM AML HI-Small is synthetic. These results are not live-bank validation, and model "
        "output does not establish wrongdoing. GraphSAGE remains a bounded research comparator. "
        "The final test was used once after model and threshold freeze."
    )


def render_page(
    page: str,
    artifacts: DashboardArtifacts,
    *,
    open_case: Callable[[str], None],
) -> None:
    """Render one authenticated portal page."""

    if page == "Overview":
        render_overview(artifacts, open_case=open_case)
    elif page == "Investigations":
        render_investigations(artifacts, open_case=open_case)
    elif page == "Case Investigator":
        render_case_investigator(artifacts)
    elif page == "Model Evidence":
        render_model_evidence(artifacts)
    else:
        st.error("This workspace page is unavailable.")
