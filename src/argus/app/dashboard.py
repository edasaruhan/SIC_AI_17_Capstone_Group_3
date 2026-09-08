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
    "Executive Dashboard",
    "Investigation Queue",
    "Case Investigator",
    "Model Comparison",
)


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
        "ARGUS NETWORK INVESTIGATOR",
        "Executive Dashboard",
        "Validation-only prioritization evidence loaded from the frozen Sprint 4 artifact set.",
    )
    focus = _comparison_focus(artifacts)
    summary = artifacts.summary
    first = st.columns(3)
    first[0].metric("Transactions analyzed", _format_count(summary.get("transactions_analyzed")))
    first[1].metric("GraphSAGE queue alerts", _format_count(summary.get("flagged_transactions")))
    first[2].metric("High-priority cases", _format_count(summary.get("high_priority_cases")))
    second = st.columns(4)
    second[0].metric("Validation PR-AUC", _format_metric(_metric_value(summary, focus, "pr_auc")))
    second[1].metric("Recall@K", _format_metric(_metric_value(summary, focus, "recall_at_k")))
    second[2].metric("FPR", _format_metric(_metric_value(summary, focus, "fpr")))
    second[3].metric("Leader alert volume", _format_count(_metric_value(summary, focus, "alerts")))
    st.caption(
        f"Validation-leader metrics use {focus.get('model', 'the selected model')}; "
        "the queue-alert count refers specifically to the GraphSAGE case-source model."
    )

    st.subheader("Model evidence at a glance")
    card_columns = st.columns(min(4, len(artifacts.model_comparison)))
    for column, (_, model) in zip(
        card_columns, artifacts.model_comparison.iterrows(), strict=False
    ):
        label = str(model.get("version", model["model"])).replace("_", " ").title()
        column.metric(label, _format_metric(model["pr_auc"]), help="Validation PR-AUC")
        recall = _format_metric(model.get("recall_at_k"))
        precision = _format_metric(model.get("precision_at_k"))
        column.caption(f"Recall@K {recall} · Precision@K {precision}")

    st.plotly_chart(
        build_model_metric_figure(artifacts.model_comparison),
        width="stretch",
        key="executive-model-comparison",
    )
    st.info(
        "This dashboard supports investigation prioritization. It does not establish wrongdoing, "
        "authorize an automatic action, or replace human review."
    )
    if artifacts.final_evaluation is None:
        st.caption("Final test remains unopened in this Sprint 4 artifact set.")
    else:
        final = artifacts.final_evaluation
        champion = final.model_comparison.loc[final.model_comparison["is_frozen_champion"]].iloc[0]
        st.subheader("Frozen one-shot final test")
        st.caption(
            "The graph_enhanced_lightgbm champion was frozen from validation before test access. "
            "These test metrics are reporting-only and never drive model selection."
        )
        final_metrics = st.columns(4)
        final_metrics[0].metric("Final-test PR-AUC", _format_metric(champion["pr_auc"]))
        final_metrics[1].metric("Final-test ROC-AUC", _format_metric(champion["roc_auc"]))
        final_metrics[2].metric("Final-test F1", _format_metric(champion["f1"]))
        final_metrics[3].metric("Final-test alerts", _format_count(champion["alerts"]))


def _filter_queue(queue: pd.DataFrame) -> pd.DataFrame:
    filtered = queue.copy()
    controls = st.columns(3)
    if "priority" in filtered:
        values = sorted(filtered["priority"].dropna().astype(str).unique())
        selected = controls[0].multiselect("Priority", values, default=values)
        filtered = filtered[filtered["priority"].astype(str).isin(selected)]
    else:
        controls[0].caption("Priority filter unavailable")
    if "risk_score" in filtered and not filtered.empty:
        lower = float(filtered["risk_score"].min())
        upper = float(filtered["risk_score"].max())
        if lower < upper:
            minimum = controls[1].slider(
                "Minimum uncalibrated ranking score",
                min_value=lower,
                max_value=upper,
                value=lower,
            )
            filtered = filtered[filtered["risk_score"] >= minimum]
        else:
            controls[1].caption(f"All saved cases have ranking score {lower:.4f}")
    else:
        controls[1].caption("Risk-score filter unavailable")
    query = controls[2].text_input("Find case or pattern", placeholder="ARG-0001")
    if query:
        text = filtered.astype(str).agg(" ".join, axis=1)
        filtered = filtered[text.str.contains(query, case=False, regex=False)]
    return filtered


def render_investigation_queue(artifacts: DashboardArtifacts) -> None:
    _page_heading(
        "HUMAN REVIEW WORKLIST",
        "Investigation Queue",
        "Filter and sort saved validation cases. Queue order is evidence, not an accusation.",
    )
    filtered = _filter_queue(artifacts.queue)
    if filtered.empty:
        st.warning("No saved cases match the current filters.")
        return
    sortable = [
        column
        for column in ("risk_score", "priority", "total_flow", "transaction_count", "case_id")
        if column in filtered
    ]
    sort_column = st.selectbox("Sort queue by", sortable, index=0)
    descending = st.toggle("Highest first", value=sort_column != "case_id")
    filtered = filtered.sort_values(sort_column, ascending=not descending, kind="mergesort")
    preferred = [
        "priority",
        "case_id",
        "risk_score",
        "account_count",
        "transaction_count",
        "total_flow",
        "major_pattern",
        "evidence_count",
    ]
    columns = [column for column in preferred if column in filtered]
    columns.extend(column for column in filtered.columns if column not in columns)
    display = filtered[columns].rename(columns={"risk_score": "uncalibrated_ranking_score"})
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
        kind = html.escape(str(item.get("kind", "Observed fact")).replace("_", " "))
        statement = html.escape(str(item["statement"]))
        scope = item.get("scope")
        scope_text = f"<br><small>Scope: {html.escape(str(scope))}</small>" if scope else ""
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
    header = st.columns(4)
    header[0].metric("Model", str(model.get("model_name", "N/A")))
    header[1].metric("Ranking score", _format_metric(model.get("score")))
    header[2].metric("Threshold", _format_metric(model.get("threshold")))
    header[3].metric("Validation rank", _format_count(model.get("rank")))
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
            title="Saved local feature contributions",
            xaxis_title="Contribution to model score",
            height=max(320, len(chart) * 28),
            margin={"l": 20, "r": 15, "t": 55, "b": 40},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, width="stretch", key="case-local-explanation")
    st.caption(
        "Model evidence explains the saved prioritization score. It is separate from directly "
        "observed transaction and graph facts. GraphSAGE sigmoid scores are uncalibrated."
    )


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

    accounts, transaction_count = _case_counts(case, queue_row)
    metrics = st.columns(5)
    metrics[0].metric("Priority", _case_priority(case, queue_row).title())
    metrics[1].metric("Uncalibrated score", _format_metric(_case_risk(case, queue_row)))
    metrics[2].metric("Accounts", _format_count(accounts))
    metrics[3].metric("Transactions", _format_count(transaction_count))
    metrics[4].metric(
        "Total flow",
        _format_money(_case_value(case, queue_row, "total_flow"), case.get("currency")),
    )
    st.caption(
        f"Case {selected_case_id} · Status: {case.get('status', 'pending_human_review')} · "
        f"Seed transaction: {case.get('transaction_id', case.get('seed_transaction_id', 'N/A'))}"
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
        st.write(f"**Major pattern:** {pattern or 'N/A'}")
        note = case.get("analyst_note", case.get("case_note"))
        if isinstance(note, dict):
            st.write(note.get("text", "No saved analyst note."))
            st.caption(f"Note mode: {note.get('mode', 'unspecified')}")
        elif note:
            st.write(str(note))
        else:
            st.caption("No saved case note. The application does not call an LLM on page load.")
        st.info("Human review is required. No automatic adverse action is authorized.")

    _render_model_evidence(case)
    transactions = case.get("transactions")
    if isinstance(transactions, list) and transactions:
        st.subheader("Transactions in this saved case")
        st.dataframe(pd.DataFrame(transactions), width="stretch", hide_index=True)


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
        figure.add_trace(
            go.Scatter(
                x=ordered["k"],
                y=ordered["recall_at_k"],
                name=f"{model} recall",
                mode="lines+markers",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=ordered["k"],
                y=ordered["precision_at_k"],
                name=f"{model} precision",
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
    st.plotly_chart(figure, width="stretch", key="model-top-k")


def _render_ablation(ablation: pd.DataFrame) -> None:
    if ablation.empty:
        st.info("No saved feature-family ablation artifact is available in Sprint 4.")
        return
    figure = go.Figure(
        go.Bar(
            x=ablation["feature_family"].astype(str),
            y=ablation["pr_auc"],
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
    st.dataframe(ablation, width="stretch", hide_index=True)


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
            "No Sprint 5 final-test payload is attached. The validation comparison, queue, and "
            "cases above remain fully usable from the Sprint 4 artifact set."
        )
        return

    st.divider()
    st.markdown(
        '<div class="argus-eyebrow">FROZEN ONE-SHOT FINAL TEST</div>',
        unsafe_allow_html=True,
    )
    st.header("Final evaluation — reporting only")
    st.success(
        "Pre-frozen champion: graph_enhanced_lightgbm. It was selected on validation before "
        "the single test access; test metrics did not select or tune the model."
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

    comparison = final.model_comparison.copy()
    comparison.insert(
        0,
        "frozen_status",
        comparison["is_frozen_champion"].map(
            lambda selected: "PRE-FROZEN CHAMPION" if selected else "comparator only"
        ),
    )
    preferred = [
        "frozen_status",
        "model",
        "version",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "recall_at_k",
        "precision_at_k",
        "alerts",
        "row_count",
        "positive_count",
        "evaluation_partition",
    ]
    columns = [column for column in preferred if column in comparison]
    st.dataframe(comparison[columns], width="stretch", hide_index=True)
    st.plotly_chart(
        build_model_metric_figure(final.model_comparison, partition_label="Final test"),
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
        "The final test has been consumed exactly once. These artifacts are immutable evaluation "
        "evidence; no post-test tuning or champion reselection is permitted."
    )


def render_model_comparison(artifacts: DashboardArtifacts) -> None:
    _page_heading(
        "VALIDATION-ONLY SCIENCE",
        "Model Comparison",
        "Frozen transaction, graph-enhanced, and GraphSAGE evidence under the recorded protocol.",
    )
    comparison = artifacts.model_comparison
    preferred = [
        "version",
        "model",
        "feature_family",
        "scope",
        "pr_auc",
        "recall_at_k",
        "precision_at_k",
        "f1",
        "fpr",
        "alerts",
        "roc_auc",
        "evaluation_partition",
    ]
    columns = [column for column in preferred if column in comparison]
    st.dataframe(comparison[columns], width="stretch", hide_index=True)
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
    _render_final_evaluation(artifacts)


def main() -> None:
    """Run the saved-artifact-only Streamlit application."""

    st.set_page_config(page_title="ARGUS Network Investigator", page_icon="◈", layout="wide")
    _inject_style()
    st.sidebar.markdown("## ARGUS")
    st.sidebar.caption("Network Investigator · Saved artifacts only")
    requested_page = st.query_params.get("page")
    page_index = _PAGES.index(requested_page) if requested_page in _PAGES else 0
    page = st.sidebar.radio("Workspace", _PAGES, index=page_index)
    artifact_root = configured_artifact_root()
    st.sidebar.divider()
    st.sidebar.caption("Artifact source")
    st.sidebar.code(str(artifact_root), language=None)
    st.sidebar.caption("No model training or inference on page load")

    try:
        artifacts = _cached_load(str(artifact_root))
    except ArtifactLoadError as exc:
        st.error("Saved Sprint 4 artifacts are not ready.")
        st.code(str(exc), language=None)
        st.info(
            "The application never trains a model at page load. Run the offline Sprint 4 pipeline "
            "to create the required queue, cases, and validation model-comparison artifacts."
        )
        st.stop()
        return

    if artifacts.final_evaluation is None:
        st.sidebar.caption("Sprint 4 validation · Final test unopened")
    else:
        st.sidebar.caption("Sprint 5 · Frozen one-shot final test")
        st.sidebar.caption("Pre-frozen champion: graph_enhanced_lightgbm")

    renderers = {
        "Executive Dashboard": render_executive_dashboard,
        "Investigation Queue": render_investigation_queue,
        "Case Investigator": render_case_investigator,
        "Model Comparison": render_model_comparison,
    }
    renderers[page](artifacts)
    st.divider()
    st.caption(
        "ARGUS is a research decision-support prototype. Scores prioritize human review; "
        "they are not legal conclusions or automatic enforcement decisions."
    )


if __name__ == "__main__":
    main()
