"""Public ARGUS website and session-local demo request experience."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from argus.app.styles import COLORS, anchor, section_heading

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
                  <a href="#product">Product</a>
                  <a href="#how-it-works">How It Works</a>
                  <a href="#analyst-experience">Analyst Experience</a>
                  <a href="#responsible-ai">Security &amp; Responsible AI</a>
                  <a href="#resources">Resources</a>
                  <a href="#contact">Contact</a>
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
                <span>Investigation context</span>
                <span class="argus-visual-status">Human review</span>
              </div>
              <div class="argus-visual-path">
                <svg viewBox="0 0 520 285" role="img"
                     aria-label="An illustrative transfer expanding into an account network">
                  <defs>
                    <marker id="argus-arrow" viewBox="0 0 10 10" refX="8" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#b7791f"></path>
                    </marker>
                    <marker id="argus-context-arrow" viewBox="0 0 10 10" refX="8" refY="5"
                            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#668390"></path>
                    </marker>
                  </defs>
                  <g stroke="#668390" stroke-width="2" fill="none"
                     marker-end="url(#argus-context-arrow)">
                    <path d="M88 72 L231 129"></path>
                    <path d="M270 137 L411 78"></path>
                    <path d="M270 151 L416 216"></path>
                    <path d="M99 224 L230 157"></path>
                  </g>
                  <path d="M93 145 L222 145" stroke="#b7791f" stroke-width="5"
                        marker-end="url(#argus-arrow)"></path>
                  <g fill="#147d76" stroke="#edf7f5" stroke-width="4">
                    <circle cx="70" cy="63" r="20"></circle>
                    <circle cx="70" cy="145" r="27"></circle>
                    <circle cx="76" cy="235" r="20"></circle>
                    <circle cx="250" cy="145" r="31" fill="#123149"></circle>
                    <circle cx="435" cy="67" r="22"></circle>
                    <circle cx="440" cy="228" r="22"></circle>
                  </g>
                  <g fill="#d8e8e8" font-family="Segoe UI, sans-serif" font-size="13">
                    <text x="42" y="34">History</text>
                    <text x="24" y="188">Transfer</text>
                    <text x="224" y="197">Account</text>
                    <text x="396" y="34">Network</text>
                    <text x="399" y="271">Context</text>
                  </g>
                </svg>
              </div>
              <div class="argus-visual-caption">Transaction → network → investigation</div>
              <div class="argus-evidence-lanes">
                <div class="argus-evidence-lane">
                  <strong>Observed evidence</strong>
                  <span>Direction, timing, transfer history</span>
                </div>
                <div class="argus-evidence-lane">
                  <strong>Model evidence</strong>
                  <span>Review priority, explained separately</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _preview_figure(expanded: bool) -> go.Figure:
    if expanded:
        positions = {
            "Account 1047": (-1.25, 0.05),
            "Account 3382": (0.0, 0.05),
            "Account 5821": (1.2, 0.85),
            "Account 7714": (1.35, -0.75),
            "Account 9206": (0.0, -1.0),
        }
        edges = (
            ("Account 1047", "Account 3382", True),
            ("Account 3382", "Account 5821", False),
            ("Account 3382", "Account 7714", False),
            ("Account 9206", "Account 3382", False),
            ("Account 5821", "Account 7714", False),
        )
    else:
        positions = {"Account 1047": (-0.9, 0.0), "Account 3382": (0.9, 0.0)}
        edges = (("Account 1047", "Account 3382", True),)

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
            textposition="bottom center",
            hovertext=[f"Illustrative account: {node}" for node in ordered_nodes],
            hoverinfo="text",
            marker={
                "size": 25,
                "color": node_colors,
                "line": {"width": 2, "color": COLORS["white"]},
            },
            showlegend=False,
        )
    )
    figure.update_layout(
        annotations=annotations,
        height=390,
        margin={"l": 12, "r": 12, "t": 20, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "range": [-1.7, 1.75]},
        yaxis={"visible": False, "range": [-1.35, 1.25], "scaleanchor": "x"},
        hoverlabel={"bgcolor": COLORS["navy_950"], "font_color": COLORS["white"]},
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
            <div class="argus-card">
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
        preview = st.radio(
            "Preview perspective",
            ("Single transaction", "Explore network"),
            horizontal=True,
            key="public_network_preview",
        )
        preview_note = " · ".join(
            ("Illustrative product preview", "Generic accounts", "No model performance claim")
        )
        st.caption(preview_note)
    with right:
        st.plotly_chart(
            _preview_figure(preview == "Explore network"),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key=f"public_preview_{preview.casefold().replace(' ', '_')}",
        )


def _render_problem_and_value() -> None:
    st.divider()
    section_heading(
        "The investigation challenge",
        "Alert volume grows faster than analyst attention.",
        "A transfer can look ordinary on its own while its timing, history, or surrounding "
        "account relationships deserve closer review.",
    )
    problem, response = st.columns(2, gap="large")
    with problem:
        st.markdown(
            """
            <div class="argus-card">
              <span class="argus-card-label">Operational pressure</span>
              <h3>Too many signals, too little context</h3>
              <p>
                Investigation teams need to decide where to look first without losing the
                transaction trail or the network surrounding it.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with response:
        st.markdown(
            """
            <div class="argus-card">
              <span class="argus-card-label">ARGUS response</span>
              <h3>Priority with evidence attached</h3>
              <p>
                ARGUS links a review priority to directed transfers, historical behavior,
                observed evidence, and a separate model explanation.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    columns = st.columns(3, gap="medium")
    concepts = (
        (
            "Prioritize",
            "Which cases deserve attention first?",
            "Turn a saved ranking into a focused investigation worklist.",
        ),
        (
            "Connect",
            "How are the accounts and transfers related?",
            "Follow direction, timing, and the account network around a case.",
        ),
        (
            "Explain",
            "What supports the review?",
            "Keep directly observed facts separate from model-based evidence.",
        ),
    )
    for column, (label, title, copy) in zip(columns, concepts, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="argus-card">
                  <span class="argus-card-label">{label}</span>
                  <h3>{title}</h3>
                  <p>{copy}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_how_it_works() -> None:
    anchor("how-it-works")
    st.divider()
    section_heading(
        "How it works",
        "A disciplined path from transaction data to analyst review.",
        "The operational journey stays simple even when the underlying evidence includes time "
        "and network context.",
    )
    steps = (
        ("01", "Transaction data", "Read the recorded transfer and its attributes."),
        ("02", "Historical context", "Look only at activity available before the transfer."),
        ("03", "Account network", "Trace directed relationships around the sender and receiver."),
        ("04", "Risk prioritization", "Rank what deserves limited analyst attention first."),
        ("05", "Investigation", "Review evidence and make a human decision."),
    )
    columns = st.columns(5, gap="small")
    for column, (number, title, copy) in zip(columns, steps, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="argus-flow-step">
                  <span class="argus-flow-number">{number}</span>
                  <h3>{title}</h3>
                  <p>{copy}</p>
                </div>
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
    st.divider()
    section_heading(
        "Operational value",
        "Give investigators context they can act on.",
        "ARGUS is designed to support prioritization and review—not to replace professional "
        "judgment or claim an investigation outcome.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            """
            - Prioritize a constrained investigation workload
            - Surface directed account-network context
            - Move from a queue entry into the relevant case directly
            """
        )
    with right:
        st.markdown(
            """
            - Separate observed facts from model reasoning
            - Support evidence-led analyst decisions
            - Preserve human review before any adverse action
            """
        )


def _render_analyst_experience() -> None:
    anchor("analyst-experience")
    st.divider()
    section_heading(
        "Analyst experience",
        "A direct path from workload to decision.",
        "The secure-style portal is organized around the questions an investigator needs to "
        "answer—not around a machine-learning pipeline.",
    )
    journey = (
        ("Overview", "See what requires attention and return to active work."),
        ("Investigations", "Search, filter, sort, and open the saved case worklist."),
        ("Case Investigator", "Inspect the network, timeline, and observed evidence."),
        ("Analyst Decision", "Record session-only notes, escalation, or case status."),
    )
    columns = st.columns(4, gap="small")
    for index, (title, copy) in enumerate(journey, start=1):
        with columns[index - 1]:
            st.markdown(
                f"""
                <div class="argus-card">
                  <span class="argus-card-label">Step {index}</span>
                  <h3>{title}</h3>
                  <p>{copy}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_responsible_ai() -> None:
    anchor("responsible-ai")
    st.divider()
    section_heading(
        "Security & Responsible AI",
        "Decision support with clear boundaries.",
        "ARGUS keeps model evidence in context and leaves investigation decisions with trained "
        "professionals.",
    )
    principles, deployment = st.columns(2, gap="large")
    with principles:
        st.markdown(
            """
            <div class="argus-card">
              <span class="argus-card-label">Review safeguards</span>
              <h3>Human review remains mandatory</h3>
              <p>
                A model score is not proof of wrongdoing. ARGUS does not autonomously accuse a
                person, block an account, or authorize enforcement action. Observed evidence and
                model evidence remain visibly separate.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with deployment:
        st.markdown(
            """
            <div class="argus-card">
              <span class="argus-card-label">Deployment boundary</span>
              <h3>Prototype evidence, not a production certification</h3>
              <p>
                A live-bank deployment would require the bank's own identity, access, security,
                data-governance, validation, monitoring, and integration controls. No external
                certification or regulatory approval is claimed.
              </p>
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
    for column, label in (("pr_auc", "Final test PR-AUC"), ("roc_auc", "Final test ROC-AUC")):
        value = champion.get(column)
        if value is not None:
            try:
                evidence.append((label, f"{float(value):.6f}"))
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
    st.divider()
    section_heading(
        "Scientific credibility",
        "Frozen evaluation evidence, presented with its limits.",
        "Graph-enhanced LightGBM was selected before the final test. Results come from the "
        "synthetic IBM AML HI-Small project dataset and are not live-bank validation.",
    )
    evidence = _scientific_evidence(artifacts)
    if evidence:
        columns = st.columns(len(evidence), gap="medium")
        for column, (label, value) in zip(columns, evidence, strict=True):
            with column:
                st.markdown(
                    f"""
                    <div class="argus-card">
                      <div class="argus-proof-value">{value}</div>
                      <div class="argus-proof-label">{label} · Project dataset</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info(
            "The public experience is available without local scientific artifacts. "
            "Verified metrics appear when the saved final-evaluation bundle is loaded."
        )
    st.caption(
        "The final test was used once after the model, feature set, and threshold were frozen. "
        "Model output does not establish wrongdoing."
    )
    st.button(
        "Explore Project Resources",
        key="public_resources",
        on_click=navigate,
        args=("resources",),
    )


def _render_contact(navigate: Navigate) -> None:
    anchor("contact")
    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="argus-callout">
          <div class="argus-eyebrow" style="color:#7fd0c8">Request a demonstration</div>
          <h2>See how a network-aware investigation can fit your review workflow.</h2>
          <p>
            Share your use case through the prototype request flow. No email or CRM submission
            occurs without a connected service.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, remainder = st.columns([1.35, 4.65])
    with left:
        st.button(
            "Request a Demo",
            key="contact_demo",
            type="primary",
            width="stretch",
            on_click=navigate,
            args=("demo",),
        )
    remainder.empty()


def _render_footer(navigate: Navigate) -> None:
    st.markdown(
        """
        <footer class="argus-footer">
          <strong style="color:#071c2c;letter-spacing:.12em">ARGUS</strong><br>
          Financial-crime investigation and account-network intelligence prototype.
        </footer>
        """,
        unsafe_allow_html=True,
    )
    product, resources, contact, privacy, terms, login = st.columns(6)
    product.button("Product", key="footer_product", on_click=navigate, args=("home",))
    resources.button("Resources", key="footer_resources", on_click=navigate, args=("resources",))
    contact.button("Contact", key="footer_contact", on_click=navigate, args=("demo",))
    privacy.button("Privacy", key="footer_privacy", on_click=navigate, args=("privacy",))
    terms.button("Terms", key="footer_terms", on_click=navigate, args=("terms",))
    login.button("Login", key="footer_login", on_click=navigate, args=("login",))
    st.caption("ARGUS · Academic decision-support prototype · MIT-licensed source code")


def render_public_home(navigate: Navigate, artifacts: Any | None = None) -> None:
    """Render the complete public-facing ARGUS experience."""

    _public_navigation(navigate)
    _render_hero(navigate)
    _render_product_preview()
    _render_problem_and_value()
    _render_how_it_works()
    _render_outcomes()
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
        "This prototype validates your request locally. It is not connected to email, a CRM, "
        "or another submission service.",
    )

    if st.session_state.get("argus_demo_request_submitted"):
        name = st.session_state.get("argus_demo_request_name", "there")
        st.success(f"Thank you, {name}. Your request is ready for this demo session.")
        st.markdown(
            '<div class="argus-prototype-note">No message was sent. This confirmation exists '
            "only in the current application session.</div>",
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
            "Review the evidence behind the ARGUS prototype.",
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
            "Prototype privacy notice",
            "Session data stays within the running demonstration.",
            "ARGUS is an academic prototype and does not present this page as a corporate "
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
            "Prototype terms",
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
