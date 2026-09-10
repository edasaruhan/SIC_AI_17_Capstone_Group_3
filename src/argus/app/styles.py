"""Shared visual language for the ARGUS public site and analyst portal."""

from __future__ import annotations

import streamlit as st

COLORS = {
    "navy_950": "#071C2C",
    "navy_800": "#123149",
    "teal_700": "#0F6B66",
    "teal_600": "#147D76",
    "teal_100": "#DDF0ED",
    "off_white": "#F7F8F5",
    "white": "#FFFFFF",
    "slate_700": "#425466",
    "slate_500": "#647482",
    "border": "#D8E1E5",
    "amber": "#B7791F",
    "critical": "#B42318",
}


def inject_global_styles() -> None:
    """Apply a restrained, responsive enterprise-fintech design system."""

    st.markdown(
        """
        <style>
        :root {
            --argus-navy-950: #071c2c;
            --argus-navy-800: #123149;
            --argus-teal-700: #0f6b66;
            --argus-teal-600: #147d76;
            --argus-teal-100: #ddf0ed;
            --argus-off-white: #f7f8f5;
            --argus-white: #ffffff;
            --argus-slate-700: #425466;
            --argus-slate-500: #647482;
            --argus-border: #d8e1e5;
            --argus-amber: #b7791f;
            --argus-critical: #b42318;
            --argus-shadow: 0 12px 32px rgba(7, 28, 44, 0.07);
            --argus-radius-sm: 6px;
            --argus-radius-md: 10px;
            --argus-radius-lg: 14px;
        }

        html { scroll-behavior: smooth; }

        .stApp {
            background: var(--argus-off-white);
            color: var(--argus-navy-950);
            font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(247, 248, 245, 0.90);
            backdrop-filter: blur(10px);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1280px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            color: var(--argus-navy-950);
            letter-spacing: -0.025em;
        }

        h1 { line-height: 1.05; }
        h2 { line-height: 1.14; }
        p, li { line-height: 1.65; }

        a {
            color: var(--argus-teal-700);
            text-decoration-thickness: 1px;
            text-underline-offset: 3px;
        }

        a:hover { color: var(--argus-navy-800); }

        button:focus-visible,
        a:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="button"]:focus-visible {
            outline: 3px solid rgba(20, 125, 118, 0.38) !important;
            outline-offset: 2px !important;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stLinkButton"] > a {
            border-radius: var(--argus-radius-sm);
            min-height: 2.75rem;
            font-weight: 650;
            letter-spacing: 0.005em;
            box-shadow: none;
            transition: border-color 120ms ease, background 120ms ease,
                        color 120ms ease, transform 120ms ease;
        }

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover,
        div[data-testid="stLinkButton"] > a:hover {
            transform: translateY(-1px);
        }

        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"],
        div[data-testid="stLinkButton"] > a[kind="primary"] {
            background: var(--argus-teal-700) !important;
            border-color: var(--argus-teal-700) !important;
            color: var(--argus-white) !important;
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
        div[data-testid="stLinkButton"] > a[kind="primary"]:hover {
            background: #0b5d58 !important;
            border-color: #0b5d58 !important;
        }

        div[data-testid="stButton"] > button:disabled,
        div[data-testid="stFormSubmitButton"] > button:disabled {
            background: #e6ebed !important;
            border-color: #d1dade !important;
            color: #7a8993 !important;
            cursor: not-allowed;
            transform: none;
        }

        div[data-testid="stForm"] {
            background: var(--argus-white);
            border: 1px solid var(--argus-border);
            border-radius: var(--argus-radius-lg);
            padding: 1.4rem;
            box-shadow: var(--argus-shadow);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stMultiSelect"] > div > div {
            border-radius: var(--argus-radius-sm);
        }

        div[data-testid="stAlert"] {
            border-radius: var(--argus-radius-md);
            border-width: 1px;
        }

        div[data-testid="stMetric"] {
            background: var(--argus-white);
            border: 1px solid var(--argus-border);
            border-radius: var(--argus-radius-md);
            padding: 1rem 1.1rem;
            box-shadow: none;
        }

        [data-testid="stSidebar"] {
            background: var(--argus-navy-950);
            border-right: 1px solid rgba(255, 255, 255, 0.10);
        }

        [data-testid="stSidebar"] * { color: #f7fbfb; }

        .argus-anchor {
            display: block;
            position: relative;
            top: -5.5rem;
            visibility: hidden;
        }

        .argus-eyebrow {
            color: var(--argus-teal-700);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            line-height: 1.4;
            margin-bottom: 0.65rem;
            text-transform: uppercase;
        }

        .argus-section-heading {
            font-size: clamp(1.75rem, 3vw, 2.55rem);
            margin: 0 0 0.75rem;
            max-width: 760px;
        }

        .argus-section-copy {
            color: var(--argus-slate-700);
            font-size: 1.04rem;
            margin: 0 0 1.75rem;
            max-width: 720px;
        }

        .argus-hero {
            padding: clamp(2.4rem, 7vw, 6.25rem) 0 clamp(2rem, 5vw, 4.5rem);
        }

        .argus-hero h1 {
            font-size: clamp(2.75rem, 6.2vw, 5.45rem);
            line-height: 0.98;
            margin: 0 0 1.35rem;
            max-width: 820px;
        }

        .argus-hero-copy {
            color: var(--argus-slate-700);
            font-size: clamp(1.05rem, 2vw, 1.28rem);
            line-height: 1.62;
            max-width: 700px;
        }

        .argus-hero-visual {
            background: var(--argus-navy-950);
            border: 1px solid #1c4156;
            border-radius: var(--argus-radius-lg);
            box-shadow: 0 20px 48px rgba(7, 28, 44, 0.16);
            color: #e9f3f4;
            min-height: 31rem;
            overflow: hidden;
            padding: 1.35rem;
            position: relative;
        }

        .argus-hero-visual::before {
            background: radial-gradient(circle, rgba(20, 125, 118, 0.26), transparent 68%);
            content: "";
            height: 21rem;
            position: absolute;
            right: -7rem;
            top: -8rem;
            width: 21rem;
        }

        .argus-visual-header {
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
            display: flex;
            font-size: 0.77rem;
            font-weight: 650;
            justify-content: space-between;
            letter-spacing: 0.04em;
            padding-bottom: 0.9rem;
            position: relative;
            text-transform: uppercase;
        }

        .argus-visual-status {
            align-items: center;
            color: #a9d7d2;
            display: inline-flex;
            gap: 0.4rem;
        }

        .argus-visual-status::before {
            background: #5dc2b8;
            border-radius: 50%;
            content: "";
            height: 0.45rem;
            width: 0.45rem;
        }

        .argus-visual-path {
            margin: 1.2rem auto 0.8rem;
            max-width: 28rem;
            position: relative;
            width: 100%;
        }

        .argus-visual-path svg { display: block; height: auto; width: 100%; }

        .argus-visual-caption {
            color: #bad0d6;
            font-size: 0.78rem;
            margin: 0.25rem 0 1rem;
            text-align: center;
        }

        .argus-evidence-lanes {
            display: grid;
            gap: 0.7rem;
            grid-template-columns: 1fr 1fr;
            position: relative;
        }

        .argus-evidence-lane {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: var(--argus-radius-sm);
            padding: 0.8rem;
        }

        .argus-evidence-lane strong {
            color: #ffffff;
            display: block;
            font-size: 0.8rem;
            margin-bottom: 0.2rem;
        }

        .argus-evidence-lane span { color: #bad0d6; font-size: 0.74rem; }

        .argus-trust-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.35rem;
        }

        .argus-pill {
            align-items: center;
            background: var(--argus-teal-100);
            border: 1px solid #c5e4df;
            border-radius: 999px;
            color: #0b5753;
            display: inline-flex;
            font-size: 0.79rem;
            font-weight: 650;
            padding: 0.38rem 0.68rem;
        }

        .argus-card {
            background: var(--argus-white);
            border: 1px solid var(--argus-border);
            border-radius: var(--argus-radius-lg);
            height: 100%;
            padding: 1.45rem;
        }

        .argus-card h3 {
            font-size: 1.08rem;
            margin: 0 0 0.6rem;
        }

        .argus-card p {
            color: var(--argus-slate-700);
            font-size: 0.94rem;
            margin: 0;
        }

        .argus-card-label {
            color: var(--argus-teal-700);
            display: block;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.12em;
            margin-bottom: 0.7rem;
            text-transform: uppercase;
        }

        .argus-flow-step {
            border-left: 3px solid var(--argus-teal-600);
            min-height: 8.5rem;
            padding: 0.25rem 0.75rem 0.25rem 1rem;
        }

        .argus-flow-number {
            color: var(--argus-teal-700);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.1em;
        }

        .argus-flow-step h3 {
            font-size: 1rem;
            margin: 0.45rem 0;
        }

        .argus-flow-step p {
            color: var(--argus-slate-700);
            font-size: 0.88rem;
            line-height: 1.5;
            margin: 0;
        }

        .argus-callout {
            background: var(--argus-navy-950);
            border-radius: var(--argus-radius-lg);
            color: #edf7f5;
            padding: clamp(1.6rem, 4vw, 3rem);
        }

        .argus-callout h2, .argus-callout h3 { color: #ffffff; }
        .argus-callout p { color: #c9dadf; }

        .argus-proof-value {
            color: var(--argus-navy-950);
            font-size: clamp(1.75rem, 3vw, 2.5rem);
            font-weight: 760;
            letter-spacing: -0.04em;
            line-height: 1.1;
        }

        .argus-proof-label {
            color: var(--argus-slate-500);
            font-size: 0.78rem;
            font-weight: 650;
            margin-top: 0.35rem;
        }

        .argus-demo-label {
            color: var(--argus-slate-500);
            font-size: 0.78rem;
            font-weight: 650;
            margin-bottom: 0.4rem;
        }

        .argus-footer {
            border-top: 1px solid var(--argus-border);
            color: var(--argus-slate-500);
            font-size: 0.84rem;
            margin-top: 4rem;
            padding: 2rem 0 0.5rem;
        }

        .argus-wordmark {
            color: var(--argus-navy-950);
            font-size: 1.18rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            line-height: 2.75rem;
        }

        .argus-nav-links {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem 1.15rem;
            min-height: 2.75rem;
        }

        .argus-nav-links a {
            color: var(--argus-slate-700);
            font-size: 0.84rem;
            font-weight: 620;
            text-decoration: none;
            white-space: nowrap;
        }

        .argus-nav-links a:hover { color: var(--argus-teal-700); }

        .argus-secondary-cta {
            align-items: center;
            background: transparent;
            border: 1px solid var(--argus-border);
            border-radius: var(--argus-radius-sm);
            color: var(--argus-navy-950);
            display: inline-flex;
            font-size: 0.9rem;
            font-weight: 650;
            justify-content: center;
            min-height: 2.75rem;
            padding: 0 1rem;
            text-decoration: none;
            width: 100%;
        }

        .argus-secondary-cta:hover {
            background: #eef3f3;
            border-color: #a9bdc1;
            color: var(--argus-navy-950);
        }

        .st-key-argus_public_nav {
            background: rgba(247, 248, 245, 0.96);
            border-bottom: 1px solid var(--argus-border);
            padding: 0.55rem 0;
            position: sticky;
            top: 0;
            z-index: 999;
        }

        .st-key-argus_public_nav [data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        .st-key-argus_public_nav div[data-testid="stButton"] > button {
            min-height: 2.35rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        .argus-prototype-note {
            background: #f0f5f6;
            border: 1px solid var(--argus-border);
            border-radius: var(--argus-radius-sm);
            color: var(--argus-slate-700);
            font-size: 0.82rem;
            padding: 0.75rem 0.9rem;
        }

        .argus-field-error {
            color: var(--argus-critical);
            font-size: 0.78rem;
            font-weight: 620;
            margin: -0.25rem 0 0.55rem;
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1.1rem;
                padding-right: 1.1rem;
                padding-top: 0.75rem;
            }
            .argus-hero { padding-top: 2.5rem; }
            .argus-hero-visual { min-height: 28rem; }
            .argus-flow-step { min-height: 0; margin-bottom: 0.75rem; }
            .argus-nav-links { gap: 0.45rem 0.8rem; }
            .argus-nav-links a { font-size: 0.77rem; }
        }

        @media (max-width: 640px) {
            .argus-hero h1 { font-size: clamp(2.45rem, 13vw, 3.45rem); }
            .argus-hero-copy { font-size: 1rem; }
            .argus-card { padding: 1.1rem; }
            .argus-nav-links { line-height: 1.8; }
            .argus-callout { padding: 1.4rem; }
            .argus-hero-visual { min-height: 0; padding: 1rem; }
            .argus-evidence-lanes { grid-template-columns: 1fr; }
            .st-key-argus_public_nav { position: relative; }
        }

        @media (prefers-reduced-motion: reduce) {
            html { scroll-behavior: auto; }
            * { transition: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_public_shell() -> None:
    """Hide analyst navigation and style public/login-only surfaces."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

        .login-spacer { height: clamp(1rem, 5vw, 4rem); }

        .login-context {
            background: var(--argus-navy-950);
            border-radius: var(--argus-radius-lg);
            color: #dce9ec;
            min-height: 30rem;
            padding: clamp(2rem, 5vw, 4.25rem);
        }

        .login-context h1 {
            color: #ffffff;
            font-size: clamp(2.35rem, 4.5vw, 4rem);
            line-height: 1.04;
            margin: 0.7rem 0 1.25rem;
        }

        .login-context p { color: #c4d4d9; font-size: 1.08rem; max-width: 34rem; }
        .argus-eyebrow.light { color: #80c9c2; }

        .login-boundary {
            border-top: 1px solid rgba(255,255,255,.16);
            display: grid;
            gap: .3rem;
            margin-top: 3rem;
            padding-top: 1.25rem;
        }

        .login-boundary strong { color: #ffffff; }
        .login-boundary span { color: #aebfc5; font-size: .88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_portal_shell() -> None:
    """Apply desktop-first analyst-workspace styling."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: block; min-width: 18rem; }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1rem; }
        [data-testid="stSidebar"] .argus-wordmark { color: #ffffff; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 6px;
            margin: .15rem 0;
            padding: .38rem .5rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: transparent;
            border-color: rgba(255,255,255,.35);
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: rgba(255,255,255,.09);
            border-color: rgba(255,255,255,.55);
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button p {
            color: inherit !important;
        }
        [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2.25rem; }

        .portal-identity {
            background: rgba(255,255,255,.07);
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 8px;
            margin: .8rem 0 1.2rem;
            padding: .8rem;
        }
        .portal-identity strong { display: block; font-size: .84rem; }
        .portal-identity span { color: #b8cbd0 !important; font-size: .76rem; }

        .section-kicker {
            color: var(--argus-teal-700);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .14em;
            margin-top: .35rem;
        }
        .section-kicker.model { color: #315f85; }

        .evidence-card {
            background: #ffffff;
            border: 1px solid var(--argus-border);
            border-left: 4px solid var(--argus-teal-600);
            border-radius: 8px;
            margin: .7rem 0;
            padding: 1rem 1.05rem;
        }
        .evidence-card p { color: var(--argus-navy-800); margin: .25rem 0 .45rem; }
        .evidence-kind {
            color: var(--argus-teal-700);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .09em;
            text-transform: uppercase;
        }
        .evidence-scope { color: var(--argus-slate-500); font-size: .76rem; }

        .case-status-card {
            align-items: flex-end;
            background: #ffffff;
            border: 1px solid var(--argus-border);
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            gap: .25rem;
            padding: 1rem 1.1rem;
        }
        .case-status-card span {
            color: var(--argus-amber);
            font-size: .72rem;
            font-weight: 750;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .case-status-card strong { color: var(--argus-navy-950); font-size: 1.05rem; }

        .case-summary-card {
            background: #ffffff;
            border: 1px solid var(--argus-border);
            border-radius: 10px;
            min-height: 6.4rem;
            padding: 1rem 1.1rem;
        }
        .case-summary-card span {
            color: var(--argus-slate-700);
            display: block;
            font-size: .82rem;
            margin-bottom: .55rem;
        }
        .case-summary-card strong {
            color: var(--argus-navy-950);
            display: block;
            font-size: clamp(1.25rem, 2vw, 1.65rem);
            font-weight: 500;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        .filter-button-spacer { height: 1.72rem; }
        [data-testid="stDataFrame"] { border: 1px solid var(--argus-border); border-radius: 8px; }
        [data-testid="stMetricValue"] { font-size: clamp(1.3rem, 2.2vw, 1.9rem); }

        @media (max-width: 900px) {
            [data-testid="stSidebar"] { min-width: 15rem; }
            [data-testid="stMainBlockContainer"] { padding-top: 1.2rem; }
            .case-status-card { align-items: flex-start; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def anchor(name: str) -> None:
    """Render a stable in-page destination for public navigation."""

    st.markdown(f'<span id="{name}" class="argus-anchor"></span>', unsafe_allow_html=True)


def section_heading(eyebrow: str, title: str, description: str) -> None:
    """Render a consistent public-site section heading."""

    st.markdown(
        f"""
        <div class="argus-eyebrow">{eyebrow}</div>
        <h2 class="argus-section-heading">{title}</h2>
        <p class="argus-section-copy">{description}</p>
        """,
        unsafe_allow_html=True,
    )
