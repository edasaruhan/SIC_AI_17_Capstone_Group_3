"""Public ARGUS website and session-local demo request experience."""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Mapping
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from argus.app.styles import COLORS, anchor, render_main_content_anchor, section_heading

Navigate = Callable[[str], None]

_GITHUB_ROOT = "https://github.com/edasaruhan/SIC_AI_17_Capstone_Group_3"
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_ROLE_PLACEHOLDER = "Select your role"
_HOW_IT_WORKS_LINK = '<a class="argus-secondary-cta" href="#how-it-works">See How ARGUS Works</a>'


def validate_demo_request(values: Mapping[str, Any]) -> dict[str, str]:
    """Return field-level errors for a demo request without causing side effects."""

    errors: dict[str, str] = {}
    required = {
        "first_name": "Enter your first name.",
        "last_name": "Enter your last name.",
        "work_email": "Enter your work email.",
        "company": "Enter your company or bank.",
        "role": "Select your job role.",
        "message": "Briefly describe your use case.",
    }
    for field, message in required.items():
        if not str(values.get(field, "")).strip():
            errors[field] = message

    email = str(values.get("work_email", "")).strip()
    if email and not _EMAIL_PATTERN.fullmatch(email):
        errors["work_email"] = "Enter a valid email address, such as name@company.com."

    role = str(values.get("role", "")).strip()
    if role == _ROLE_PLACEHOLDER:
        errors["role"] = "Select your job role."

    if not values.get("consent"):
        errors["consent"] = "Confirm that ARGUS may use these details for this demo request."
    return errors


def _public_navigation(navigate: Navigate) -> None:
    with st.container(key="argus_public_nav"):
        brand, links, demo, login = st.columns([1.0, 4.5, 1.2, 1.2], vertical_alignment="center")
        with brand:
            st.markdown('<div class="argus-wordmark">ARGUS</div>', unsafe_allow_html=True)
        with links:
            st.markdown(
                """
                <nav class="argus-nav-links" aria-label="Primary navigation">
                  <div class="argus-desktop-nav">
                    <a href="#product">Product</a>
                    <a href="#how-it-works">How It Works</a>
                    <a href="#analyst-experience">Analyst Experience</a>
                    <a href="#responsible-ai">Security &amp; Responsible AI</a>
                    <a href="#resources">Resources</a>
                    <a href="#contact">Contact</a>
                  </div>
                  <details class="argus-mobile-nav">
                    <summary>Explore</summary>
                    <div class="argus-mobile-nav-panel">
                      <a href="#product">Product</a>
                      <a href="#how-it-works">How It Works</a>
                      <a href="#analyst-experience">Analyst Experience</a>
                      <a href="#responsible-ai">Security &amp; Responsible AI</a>
                      <a href="#resources">Resources</a>
                      <a href="#contact">Contact</a>
                    </div>
                  </details>
                </nav>
                """,
                unsafe_allow_html=True,
            )
        with demo:
            st.button(
                "Request a Demo",
                key="public_nav_demo",
                type="primary",
                width="stretch",
                on_click=navigate,
                args=("demo",),
            )
        with login:
            st.button(
                "Corporate Login",
                key="public_nav_login",
                width="stretch",
                on_click=navigate,
                args=("login",),
            )
    render_main_content_anchor()


def _page_navigation(navigate: Navigate) -> None:
    with st.container(key="argus_public_nav"):
        brand, spacer, home, login = st.columns([1.0, 5.0, 1.0, 1.2], vertical_alignment="center")
        with brand:
            st.markdown('<div class="argus-wordmark">ARGUS</div>', unsafe_allow_html=True)
        spacer.empty()
        with home:
            st.button(
                "Public Site",
                key="public_page_home",
                width="stretch",
                on_click=navigate,
                args=("home",),
            )
        with login:
            st.button(
                "Corporate Login",
                key="public_page_login",
                type="primary",
                width="stretch",
                on_click=navigate,
                args=("login",),
            )
    render_main_content_anchor()


def _render_hero(navigate: Navigate) -> None:
    copy, visual = st.columns([1.08, 0.92], gap="large", vertical_alignment="center")
    with copy:
        st.markdown(
            """
            <section class="argus-hero">
              <div class="argus-eyebrow">Financial crime investigation intelligence</div>
              <h1>See the network behind the transaction.</h1>
              <p class="argus-hero-copy">
                ARGUS helps financial-crime teams prioritize investigations using transaction
                history and account-network context—while keeping every decision in human hands.
              </p>
              <div class="argus-trust-row" aria-label="ARGUS principles">
                <span class="argus-pill">Evidence-led review</span>
                <span class="argus-pill">Human decision-making</span>
                <span class="argus-pill">Network-aware context</span>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        primary, secondary = st.columns(2)
        with primary:
            st.button(
                "Request a Demo",
                key="hero_demo",
                type="primary",
                width="stretch",
                on_click=navigate,
                args=("demo",),
            )
        with secondary:
            st.markdown(_HOW_IT_WORKS_LINK, unsafe_allow_html=True)
    with visual:
        st.markdown(
            """
            <div class="argus-hero-visual">
              <div class="argus-visual-header">
                <span>What ARGUS adds</span>
                <span class="argus-visual-status">Human review</span>
              </div>
              <div class="argus-hero-story">
                <div class="argus-story-step">
                  <span>01</span>
                  <div><strong>A transfer needs attention</strong>
                  <p>The recorded transaction is the starting point, not the conclusion.</p></div>
                </div>
                <div class="argus-story-connector" aria-hidden="true"></div>
                <div class="argus-story-step">
                  <span>02</span>
                  <div><strong>Context reveals the pattern</strong>
                  <p>Recent activity and linked accounts are brought into the same case.</p></div>
                </div>
                <div class="argus-story-connector" aria-hidden="true"></div>
                <div class="argus-story-step active">
                  <span>03</span>
                  <div><strong>The analyst decides what happens next</strong>
                  <p>Observed facts and model guidance stay visibly separate.</p></div>
                </div>
              </div>
              <div class="argus-story-result">
                <span>Outcome</span>
                <strong>A focused case brief with a clear next action</strong>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _preview_figure(expanded: bool) -> go.Figure:
    if expanded:
        positions = {
            "Account 1047": (-1.35, 0.0),
            "Account 3382": (0.0, 0.0),
            "Account 5821": (1.25, 0.78),
            "Account 7714": (1.35, -0.72),
            "Account 9206": (-0.2, -0.95),
            "Account 6630": (-0.72, 0.92),
        }
        edges = (
            ("Account 1047", "Account 3382", True),
            ("Account 3382", "Account 5821", False),
            ("Account 3382", "Account 7714", False),
            ("Account 9206", "Account 3382", False),
            ("Account 5821", "Account 7714", False),
            ("Account 6630", "Account 3382", False),
            ("Account 1047", "Account 6630", False),
        )
        x_range = [-1.9, 1.78]
        y_range = [-1.45, 1.32]
    else:
        positions = {"Account 1047": (-0.78, 0.0), "Account 3382": (0.78, 0.0)}
        edges = (("Account 1047", "Account 3382", True),)
        x_range = [-1.05, 1.05]
        y_range = [-0.48, 0.55]

    figure = go.Figure()
    annotations: list[dict[str, Any]] = []
    for source, target, focal in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        color = COLORS["amber"] if focal else COLORS["slate_500"]
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"color": color, "width": 3 if focal else 1.6},
                hovertemplate=(
                    f"{source} → {target}<br>"
                    f"{'Selected transaction' if focal else 'Network context'}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        annotations.append(
            {
                "x": x1,
                "y": y1,
                "ax": x0,
                "ay": y0,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "arrowhead": 3,
                "arrowsize": 1.05,
                "arrowwidth": 1.5,
                "arrowcolor": color,
                "showarrow": True,
            }
        )

    ordered_nodes = list(positions)
    text_positions = (
        [
            "bottom center",
            "bottom center",
            "middle left",
            "top center",
            "bottom center",
            "bottom center",
        ]
        if expanded
        else "bottom center"
    )
    node_colors = [
        COLORS["navy_950"] if node == "Account 3382" else COLORS["teal_600"]
        for node in ordered_nodes
    ]
    figure.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in ordered_nodes],
            y=[positions[node][1] for node in ordered_nodes],
            mode="markers+text",
            text=ordered_nodes,
            textposition=text_positions,
            hovertext=[f"Illustrative account: {node}" for node in ordered_nodes],
            hoverinfo="text",
            marker={
                "size": 38 if expanded else 44,
                "color": node_colors,
                "line": {"width": 2.5, "color": COLORS["white"]},
            },
            showlegend=False,
        )
    )
    focal_source, focal_target, _ = edges[0]
    focal_x = (positions[focal_source][0] + positions[focal_target][0]) / 2
    focal_y = (positions[focal_source][1] + positions[focal_target][1]) / 2
    annotations.append(
        {
            "x": focal_x,
            "y": focal_y + 0.14,
            "text": "Selected transfer",
            "showarrow": False,
            "font": {"size": 11, "color": "#7A4D10"},
            "bgcolor": "#FFF3DB",
            "bordercolor": "#E8C98F",
            "borderpad": 4,
        }
    )
    figure.update_layout(
        annotations=annotations,
        height=338 if expanded else 270,
        margin={"l": 18, "r": 18, "t": 22, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FBFA",
        xaxis={"visible": False, "range": x_range, "fixedrange": True},
        yaxis={"visible": False, "range": y_range, "fixedrange": True},
        hoverlabel={"bgcolor": COLORS["navy_950"], "font_color": COLORS["white"]},
        uirevision="argus-public-preview",
    )
    return figure


def _render_product_preview() -> None:
    anchor("product")
    section_heading(
        "Product",
        "Move from an isolated transfer to an investigation-ready view.",
        "ARGUS brings transaction history, directed account relationships, and review evidence "
        "into one focused analyst workflow.",
    )
    left, right = st.columns([0.82, 1.18], gap="large", vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="argus-preview-copy">
              <span class="argus-card-label">A clearer starting point</span>
              <h3>One transaction rarely tells the whole story.</h3>
              <p>
                Switch the preview from a single transfer to its surrounding network context.
                The example is illustrative and is not a scored scientific case.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        requested_preview = st.query_params.get("preview")
        if isinstance(requested_preview, list):
            requested_preview = requested_preview[-1] if requested_preview else None
        preview_key = (
            "public_network_preview_expanded"
            if requested_preview == "network"
            else "public_network_preview"
        )
        preview = st.radio(
            "Preview perspective",
            ("Single transaction", "Explore network"),
            index=1 if requested_preview == "network" else 0,
            horizontal=True,
            key=preview_key,
        )
        st.caption("Illustrative network · Demo accounts · Not a scored case")
    with right:
        with st.container(key="public_preview_canvas"):
            state_copy = (
                "6 accounts · 7 directed relationships"
                if preview == "Explore network"
                else "2 accounts · 1 directed relationship"
            )
            st.markdown(
                '<div class="argus-preview-header"><span>Investigation view</span>'
                f"<strong>{state_copy}</strong></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _preview_figure(preview == "Explore network"),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
                key="public_network_preview_chart",
            )


def _render_problem_and_value() -> None:
    st.markdown(
        """
        <section class="argus-value-section">
          <div class="argus-value-intro">
            <div>
              <div class="argus-eyebrow">The investigation challenge</div>
              <h2>Alert volume grows faster than analyst attention.</h2>
              <p>A transfer can look ordinary on its own while timing, history, and surrounding
              relationships reveal where a closer review should begin.</p>
            </div>
            <div class="argus-context-stack" aria-label="From fragmented signals to context">
              <div><span>01</span><strong>Isolated alert</strong>
              <small>Limited context</small></div>
              <svg viewBox="0 0 36 22" aria-hidden="true"><path d="M2 11h28m-7-7 7 7-7 7"/></svg>
              <div class="active"><span>02</span><strong>ARGUS review</strong>
              <small>Network and evidence together</small></div>
            </div>
          </div>
          <div class="argus-pillars" aria-label="ARGUS product pillars">
            <article>
              <svg viewBox="0 0 32 32" aria-hidden="true"><path d="M6 8h20M6 16h13M6 24h8"/>
              <circle cx="24" cy="20" r="4"/><path d="m27 23 3 3"/></svg>
              <span>Prioritize</span><strong class="argus-pillar-title">Focus attention</strong>
              <p>Turn ranking evidence into a focused investigation worklist.</p>
            </article>
            <article>
              <svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="7" cy="16" r="4"/>
              <circle cx="25" cy="8" r="4"/><circle cx="25" cy="24" r="4"/>
              <path d="m11 15 10-5m-10 7 10 5"/></svg>
              <span>Connect</span><strong class="argus-pillar-title">See relationships</strong>
              <p>Follow direction, timing, and the account network around a case.</p>
            </article>
            <article>
              <svg viewBox="0 0 32 32" aria-hidden="true">
              <path d="M6 5h20v22H6zM11 11h10M11 16h10M11 21h6"/>
              <circle cx="24" cy="23" r="5"/></svg>
              <span>Explain</span><strong class="argus-pillar-title">Review with clarity</strong>
              <p>Keep observed facts visibly separate from model-based evidence.</p>
            </article>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_how_it_works() -> None:
    anchor("how-it-works")
    section_heading(
        "How it works",
        "A clear path from alert to human decision.",
        "ARGUS organizes complex evidence into three steps an investigator can follow.",
    )
    st.markdown(
        """
        <ol class="argus-process" aria-label="ARGUS investigation process">
          <li><span class="argus-process-node">01</span><div><strong>Prioritize</strong>
          <p>Bring the cases that need attention to the top of the worklist.</p></div></li>
          <li><span class="argus-process-node">02</span><div><strong>Understand</strong>
          <p>Review the transfer, recent activity and connected accounts together.</p></div></li>
          <li><span class="argus-process-node">03</span><div><strong>Decide</strong>
          <p>Record a documented human decision with the evidence kept in context.</p></div></li>
        </ol>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Technical model roles"):
        st.markdown(
            "**Graph-enhanced LightGBM** is the primary scientific and operational ranking "
            "model. **GraphSAGE** is retained as a bounded research comparator. The two roles "
            "are not interchangeable."
        )


def _render_outcomes() -> None:
    st.markdown(
        """
        <section class="argus-outcome-band">
          <div class="argus-outcome-visual" aria-hidden="true">
            <svg viewBox="0 0 220 150">
              <path class="route" d="M26 34h72c24 0 24 38 48 38h48"/>
              <path class="route" d="M26 76h52c30 0 31 42 61 42h55"/>
              <path class="focal" d="M26 118h36c35 0 45-18 76-18h56"/>
              <circle cx="26" cy="34" r="7"/><circle cx="26" cy="76" r="7"/>
              <circle cx="26" cy="118" r="9" class="hot"/>
              <rect x="152" y="22" width="42" height="24" rx="6"/>
              <rect x="152" y="60" width="42" height="24" rx="6"/>
              <rect x="152" y="106" width="42" height="24" rx="6" class="selected"/>
            </svg>
            <span>Signals become a review path</span>
          </div>
          <div class="argus-outcome-copy">
            <div class="argus-eyebrow light">Operational value</div>
            <h2>Give investigators context they can act on.</h2>
            <div class="argus-outcome-list">
              <span>Prioritize limited analyst attention</span>
              <span>Expose directed account-network context</span>
              <span>Move directly from worklist to investigation</span>
              <span>Separate observed and model evidence</span>
              <span>Preserve human review</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_analyst_experience() -> None:
    anchor("analyst-experience")
    section_heading(
        "Analyst experience",
        "A direct path from workload to decision.",
        "The workspace is organized around the questions an investigator needs to answer—not "
        "around a machine-learning pipeline.",
    )
    st.markdown(
        """
        <section class="argus-analyst-journey">
          <ol class="argus-journey-rail">
            <li><span>01</span><div><strong>Overview</strong>
            <small>See active work</small></div></li>
            <li><span>02</span><div><strong>Investigations</strong>
            <small>Find and open a case</small></div></li>
            <li><span>03</span><div><strong>Case Investigator</strong>
            <small>Review network and evidence</small></div></li>
            <li><span>04</span><div><strong>Analyst Decision</strong>
            <small>Document the next action</small></div></li>
          </ol>
          <div class="argus-workspace-preview" aria-label="Illustrative analyst workspace">
            <div class="argus-preview-sidebar"><b>ARGUS</b><i></i><i></i>
            <i class="active"></i><i></i></div>
            <div class="argus-preview-main">
              <div class="argus-preview-top"><span>ILLUSTRATIVE · A-2048</span>
              <em>IN REVIEW</em></div>
              <div class="argus-preview-grid">
                <div class="argus-mini-network">
                  <svg viewBox="0 0 280 185" aria-hidden="true">
                    <path d="M45 42L137 91M45 142L137 91M151 91L235 46M151 91L235 139"/>
                    <path class="focal" d="M50 91h76"/>
                    <circle cx="40" cy="42" r="12"/><circle cx="40" cy="91" r="17"/>
                    <circle cx="40" cy="142" r="12"/><circle cx="140" cy="91" r="20" class="core"/>
                    <circle cx="240" cy="42" r="13"/><circle cx="240" cy="142" r="13"/>
                  </svg>
                </div>
                <div class="argus-mini-evidence"><strong>Observed evidence</strong>
                  <i></i><i></i><i class="short"></i><strong>Model evidence</strong>
                  <i></i><i class="short"></i>
                </div>
              </div>
              <div class="argus-preview-actions"><span>Start review</span>
              <span class="secondary">Escalate</span>
              <span class="muted">Close case</span></div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_responsible_ai() -> None:
    anchor("responsible-ai")
    section_heading(
        "Security & Responsible AI",
        "Decision support with clear boundaries.",
        "ARGUS keeps model evidence in context and leaves investigation decisions with trained "
        "professionals.",
    )
    st.markdown(
        """
        <div class="argus-trust-grid">
          <article><span class="argus-trust-icon">
            <svg viewBox="0 0 32 32" aria-hidden="true">
            <path d="M16 3 27 8v8c0 7-4.7 11.2-11 13-6.3-1.8-11-6-11-13V8z"/>
            <path d="m10.5 16 3.5 3.5 7.5-8"/></svg></span>
            <div><strong>Human decision remains final</strong>
            <p>A model score supports review; it does not establish wrongdoing or authorize
            action.</p></div>
          </article>
          <article><span class="argus-trust-icon">
            <svg viewBox="0 0 32 32" aria-hidden="true">
            <rect x="3" y="6" width="11" height="20" rx="3"/>
            <rect x="18" y="6" width="11" height="20" rx="3"/>
            <path d="M8 12h2M8 17h2M23 12h2M23 17h2"/></svg></span>
            <div><strong>Evidence types stay separate</strong>
            <p>Observed transaction facts remain distinct from model reasoning throughout
            review.</p></div>
          </article>
          <article><span class="argus-trust-icon">
            <svg viewBox="0 0 32 32" aria-hidden="true">
            <path d="M5 28V12h22v16M9 12V7h14v5M11 18h3M18 18h3M11 23h3M18 23h3"/>
            </svg></span>
            <div><strong>Deployment requires institution controls</strong>
            <p>Production use requires identity, security, governance, validation, and
            monitoring.</p></div>
          </article>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _truthy_series(frame: Any, column: str) -> Any:
    return frame[column].map(lambda value: value is True or str(value).strip().lower() == "true")


def _scientific_evidence(artifacts: Any | None) -> list[tuple[str, str]]:
    if artifacts is None or getattr(artifacts, "final_evaluation", None) is None:
        return []
    comparison = artifacts.final_evaluation.model_comparison
    if comparison.empty or "is_frozen_champion" not in comparison:
        return []
    champion_rows = comparison[_truthy_series(comparison, "is_frozen_champion")]
    if champion_rows.empty:
        return []
    champion = champion_rows.iloc[0]
    evidence: list[tuple[str, str]] = []
    value = champion.get("pr_auc")
    if value is not None:
        try:
            evidence.append(("Final PR-AUC", f"{float(value):.3f}"))
        except (TypeError, ValueError):
            pass

    top_k = artifacts.final_evaluation.top_k
    if not top_k.empty and {"model", "k", "precision_at_k"}.issubset(top_k.columns):
        canonical = top_k["model"].astype(str).str.lower().str.replace("-", "_", regex=False)
        at_100 = top_k[(canonical == "graph_enhanced_lightgbm") & top_k["k"].eq(100)]
        if not at_100.empty:
            try:
                evidence.append(("Precision@100", f"{float(at_100.iloc[0]['precision_at_k']):.3f}"))
            except (TypeError, ValueError):
                pass
    return evidence


def _render_scientific_credibility(artifacts: Any | None, navigate: Navigate) -> None:
    anchor("resources")
    evidence = _scientific_evidence(artifacts)
    if evidence:
        values = dict(evidence)
        st.markdown(
            f"""
            <section class="argus-credibility-strip">
              <div class="argus-credibility-label"><span>Project evaluation</span>
              <strong>IBM AML HI-Small</strong></div>
              <div><strong>5.08M</strong><span>transaction project dataset</span></div>
              <div><strong>Graph-enhanced LightGBM</strong><span>primary model</span></div>
              <div><strong>{html.escape(values.get("Final PR-AUC", "—"))}</strong>
              <span>Final PR-AUC</span></div>
              <div><strong>{html.escape(values.get("Precision@100", "—"))}</strong>
              <span>Precision@100</span></div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="argus-resource-note">
              <span>TRANSPARENT BY DESIGN</span>
              <strong>Model evidence is kept separate from daily case review.</strong>
              <p>The demo workspace explains what each saved metric means and clearly labels
              the limits of synthetic evidence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    link, note = st.columns([1.2, 4.8], vertical_alignment="center")
    link.button(
        "View Model Evidence",
        key="public_resources",
        on_click=navigate,
        args=("resources",),
        width="stretch",
    )
    note.caption("Technical details remain available without interrupting the analyst workflow.")


def _render_contact(navigate: Navigate) -> None:
    anchor("contact")
    with st.container(key="contact_callout"):
        copy, action = st.columns([4.4, 1.2], gap="large", vertical_alignment="center")
        with copy:
            st.markdown(
                """
                <div class="argus-eyebrow light">Request a demonstration</div>
                <h2>Bring network context into your investigation workflow.</h2>
                <p>Tell us about your investigation workflow and explore how ARGUS could fit
                your review process.</p>
                """,
                unsafe_allow_html=True,
            )
        with action:
            st.button(
                "Request a Demo",
                key="contact_demo",
                type="primary",
                width="stretch",
                on_click=navigate,
                args=("demo",),
            )


def _render_footer(navigate: Navigate) -> None:
    with st.container(key="public_footer"):
        st.markdown(
            """
            <footer class="argus-footer">
              <strong>ARGUS</strong>
              <span>Financial Crime Investigation &amp; Account Network Intelligence</span>
            </footer>
            """,
            unsafe_allow_html=True,
        )
        product, resources, contact, privacy, terms, login = st.columns(6)
        product.button("Product", key="footer_product", on_click=navigate, args=("home",))
        resources.button(
            "Resources", key="footer_resources", on_click=navigate, args=("resources",)
        )
        contact.button("Contact", key="footer_contact", on_click=navigate, args=("demo",))
        privacy.button("Privacy", key="footer_privacy", on_click=navigate, args=("privacy",))
        terms.button("Terms", key="footer_terms", on_click=navigate, args=("terms",))
        login.button("Login", key="footer_login", on_click=navigate, args=("login",))
        st.caption("Project experience · Human review required · MIT-licensed source code")


def render_public_home(navigate: Navigate, artifacts: Any | None = None) -> None:
    """Render the complete public-facing ARGUS experience."""

    _public_navigation(navigate)
    _render_hero(navigate)
    _render_product_preview()
    _render_how_it_works()
    _render_analyst_experience()
    _render_responsible_ai()
    _render_scientific_credibility(artifacts, navigate)
    _render_contact(navigate)
    _render_footer(navigate)


def _reset_demo_request() -> None:
    for key in (
        "demo_first_name",
        "demo_last_name",
        "demo_work_email",
        "demo_company",
        "demo_role",
        "demo_message",
        "demo_consent",
        "argus_demo_request_submitted",
        "argus_demo_request_name",
        "argus_demo_request_errors",
    ):
        st.session_state.pop(key, None)


def _field_error(errors: Mapping[str, str], field: str, target: Any = st) -> None:
    message = errors.get(field)
    if message:
        target.markdown(
            f'<div class="argus-field-error" role="alert">{message}</div>',
            unsafe_allow_html=True,
        )


def render_demo_request(navigate: Navigate) -> None:
    """Render a validated, session-local request-a-demo flow."""

    _page_navigation(navigate)
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    section_heading(
        "Request a demo",
        "Start a conversation about your investigation workflow.",
        "Tell us where network context and evidence-led review could support your team.",
    )

    if st.session_state.get("argus_demo_request_submitted"):
        name = st.session_state.get("argus_demo_request_name", "there")
        st.success(f"Thank you, {name}. Your request is ready for this demo session.")
        st.markdown(
            '<div class="argus-prototype-note"><strong>Demo environment</strong> · No external '
            "CRM submission is connected. No message was sent.</div>",
            unsafe_allow_html=True,
        )
        first, second, remainder = st.columns([1.2, 1.2, 3.6])
        first.button(
            "Submit Another",
            key="demo_again",
            on_click=_reset_demo_request,
            width="stretch",
        )
        second.button(
            "Return to Site",
            key="demo_success_home",
            type="primary",
            on_click=navigate,
            args=("home",),
            width="stretch",
        )
        remainder.empty()
        return

    stored_errors = st.session_state.get("argus_demo_request_errors", {})
    errors = dict(stored_errors) if isinstance(stored_errors, Mapping) else {}
    if errors:
        st.error("Please correct the highlighted request details.")
    with st.form("argus_demo_request_form", clear_on_submit=False):
        first, last = st.columns(2)
        first_name = first.text_input("First name", key="demo_first_name")
        _field_error(errors, "first_name", first)
        last_name = last.text_input("Last name", key="demo_last_name")
        _field_error(errors, "last_name", last)
        work_email = st.text_input(
            "Work email", key="demo_work_email", placeholder="name@company.com"
        )
        _field_error(errors, "work_email")
        company = st.text_input("Company / Bank", key="demo_company")
        _field_error(errors, "company")
        role = st.selectbox(
            "Job role",
            (
                _ROLE_PLACEHOLDER,
                "Financial Crime / AML",
                "Fraud Operations",
                "Compliance",
                "Risk Management",
                "Data / Technology",
                "Executive Leadership",
                "Other",
            ),
            key="demo_role",
        )
        _field_error(errors, "role")
        message = st.text_area(
            "Message or use case",
            key="demo_message",
            placeholder="Tell us what your investigation team needs to understand or prioritize.",
        )
        _field_error(errors, "message")
        consent = st.checkbox(
            "I agree that these details may be used to respond to this demo request.",
            key="demo_consent",
        )
        _field_error(errors, "consent")
        submitted = st.form_submit_button("Request a Demo", type="primary", width="stretch")

    if submitted:
        values = {
            "first_name": first_name,
            "last_name": last_name,
            "work_email": work_email,
            "company": company,
            "role": role,
            "message": message,
            "consent": consent,
        }
        errors = validate_demo_request(values)
        if errors:
            st.session_state["argus_demo_request_errors"] = errors
            st.rerun()
        else:
            st.session_state.pop("argus_demo_request_errors", None)
            st.session_state["argus_demo_request_submitted"] = True
            st.session_state["argus_demo_request_name"] = first_name.strip()
            st.rerun()
    st.caption("Required fields must be complete. No form data is written to project artifacts.")
    st.button(
        "Return to Public Site",
        key="demo_home",
        on_click=navigate,
        args=("home",),
    )


def render_public_information(route: str, navigate: Navigate) -> None:
    """Render small public resource, privacy, and terms views with working navigation."""

    if route not in {"resources", "privacy", "terms"}:
        raise ValueError(f"Unsupported public information route: {route}")
    _page_navigation(navigate)
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    if route == "resources":
        section_heading(
            "Resources",
            "Review the evidence behind the ARGUS project.",
            "Project documentation separates scientific evaluation, model limitations, and "
            "the analyst-facing product experience.",
        )
        columns = st.columns(3, gap="medium")
        resources = (
            (
                "Project repository",
                "Source code, tests, and human-authored project documentation.",
                _GITHUB_ROOT,
            ),
            (
                "Model card",
                "Primary model role, evaluation context, limitations, and intended use.",
                f"{_GITHUB_ROOT}/blob/main/docs/MODEL_CARD.md",
            ),
            (
                "Experiment protocol",
                "Chronological evaluation and one-time final-test safeguards.",
                f"{_GITHUB_ROOT}/blob/main/docs/EXPERIMENT_PROTOCOL.md",
            ),
        )
        for index, (title, copy, url) in enumerate(resources):
            with columns[index]:
                st.markdown(
                    f'<div class="argus-card"><h3>{title}</h3><p>{copy}</p></div>',
                    unsafe_allow_html=True,
                )
                st.link_button("Open Resource", url, width="stretch")
        st.link_button(
            "IBM AML-Data Collection",
            "https://github.com/IBM/AML-Data",
            help="The source collection for the synthetic HI-Small project dataset.",
        )
    elif route == "privacy":
        section_heading(
            "Privacy notice",
            "Demo data stays within the running session.",
            "This project notice describes the demonstration environment and is not a corporate "
            "privacy policy.",
        )
        st.markdown(
            """
            - The demo-request experience is not connected to email, CRM, or a contact database.
            - Login and analyst workflow state may be held temporarily in the active Streamlit
              session and is cleared when that session resets.
            - Session-only notes and actions are not written into scientific artifact files.
            - A production deployment would require the deploying organization's privacy,
              retention, access-control, and data-governance policies.
            """
        )
    else:
        section_heading(
            "Terms of use",
            "Use ARGUS as decision-support research, not as an enforcement system.",
            "The public demonstration communicates the intended analyst workflow and the limits "
            "of the completed project evaluation.",
        )
        st.markdown(
            """
            - Model output prioritizes human review and does not establish wrongdoing.
            - No automatic account blocking, accusation, or adverse action is authorized.
            - Results use the synthetic IBM AML HI-Small dataset and do not represent live-bank
              production validation.
            - Production use would require independent validation, governance, security, and
              integration work by the deploying organization.
            """
        )
        st.link_button("View MIT License", f"{_GITHUB_ROOT}/blob/main/LICENSE")

    st.button(
        "Return to Public Site",
        key=f"{route}_home",
        type="primary",
        on_click=navigate,
        args=("home",),
    )
