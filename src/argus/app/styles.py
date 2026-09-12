# ruff: noqa: E501 -- embedded CSS is kept readable by selector and declaration group
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

        .argus-skip-link {
            background: var(--argus-navy-950);
            border-radius: var(--argus-radius-sm);
            color: #ffffff;
            font-size: .86rem;
            font-weight: 700;
            left: 1rem;
            padding: .65rem .9rem;
            position: fixed;
            top: -5rem;
            transition: top 120ms ease;
            z-index: 2000;
        }

        .argus-skip-link:focus {
            color: #ffffff;
            top: .75rem;
        }

        .argus-main-anchor { display: block; scroll-margin-top: 5.5rem; }

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
            padding: clamp(2.25rem, 5vw, 4.5rem) 0 clamp(1rem, 2vw, 1.75rem);
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
            min-height: 30rem;
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
            margin: .55rem auto .7rem;
            max-width: 31rem;
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

        .argus-desktop-nav {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem 1.15rem;
        }

        .argus-mobile-nav { display: none; position: relative; }
        .argus-mobile-nav summary {
            border: 1px solid var(--argus-border);
            border-radius: var(--argus-radius-sm);
            color: var(--argus-navy-950);
            cursor: pointer;
            font-size: .82rem;
            font-weight: 700;
            list-style: none;
            padding: .55rem .75rem;
            width: fit-content;
        }
        .argus-mobile-nav summary::-webkit-details-marker { display: none; }
        .argus-mobile-nav summary::after { content: " +"; }
        .argus-mobile-nav[open] summary::after { content: " −"; }
        .argus-mobile-nav-panel {
            background: #ffffff;
            border: 1px solid var(--argus-border);
            border-radius: var(--argus-radius-md);
            box-shadow: var(--argus-shadow);
            display: grid;
            gap: .15rem;
            left: 0;
            min-width: 14rem;
            padding: .5rem;
            position: absolute;
            top: calc(100% + .4rem);
            z-index: 1200;
        }
        .argus-mobile-nav-panel a {
            border-radius: 4px;
            padding: .5rem .55rem;
        }
        .argus-mobile-nav-panel a:hover { background: #eef3f3; }

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

        .argus-visual-toolbar {
            align-items: center;
            color: #91aab3;
            display: flex;
            font-size: .68rem;
            gap: .4rem;
            margin-top: .75rem;
            position: relative;
        }
        .argus-visual-toolbar > span {
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 999px;
            padding: .28rem .52rem;
        }
        .argus-visual-toolbar > span.active {
            background: rgba(93,194,184,.14);
            border-color: rgba(93,194,184,.32);
            color: #aee1dc;
        }
        .argus-visual-toolbar .argus-visual-case {
            border: 0;
            color: #77939d;
            margin-left: auto;
            padding-right: 0;
        }
        .argus-lane-icon {
            align-items: center;
            border: 1px solid rgba(127,208,200,.38);
            border-radius: 50%;
            color: #91d8d1 !important;
            display: inline-flex !important;
            float: left;
            font-size: .61rem !important;
            height: 1.7rem;
            justify-content: center;
            margin-right: .6rem;
            width: 1.7rem;
        }

        .argus-preview-copy { padding: .6rem 1.8rem .6rem 0; }
        .argus-preview-copy h3 { font-size: 1.55rem; margin: 0 0 .7rem; }
        .argus-preview-copy p { color: var(--argus-slate-700); margin: 0 0 1.2rem; }
        .st-key-public_preview_canvas {
            background: #ffffff;
            border: 1px solid var(--argus-border);
            border-radius: var(--argus-radius-lg);
            box-shadow: var(--argus-shadow);
            overflow: hidden;
            padding: .9rem 1rem .2rem;
        }
        .argus-preview-header {
            align-items: center;
            border-bottom: 1px solid var(--argus-border);
            color: var(--argus-slate-500);
            display: flex;
            font-size: .72rem;
            justify-content: space-between;
            letter-spacing: .06em;
            padding: .2rem .2rem .75rem;
            text-transform: uppercase;
        }
        .argus-preview-header strong { color: var(--argus-teal-700); font-size: .7rem; }

        .argus-value-section { padding: 5.5rem 0 4.5rem; }
        .argus-value-intro {
            align-items: center;
            display: grid;
            gap: clamp(2rem, 6vw, 6rem);
            grid-template-columns: 1.15fr .85fr;
        }
        .argus-value-intro h2 { font-size: clamp(2rem, 4vw, 3.25rem); margin: 0 0 1rem; }
        .argus-value-intro p { color: var(--argus-slate-700); max-width: 42rem; }
        .argus-context-stack {
            align-items: center;
            background: #eef4f3;
            border: 1px solid #d3e1e0;
            border-radius: var(--argus-radius-lg);
            display: grid;
            gap: .75rem;
            grid-template-columns: 1fr 2.25rem 1fr;
            padding: 1.25rem;
        }
        .argus-context-stack > div {
            background: rgba(255,255,255,.72);
            border: 1px solid var(--argus-border);
            border-radius: 9px;
            display: grid;
            gap: .2rem;
            min-height: 7rem;
            padding: 1rem;
        }
        .argus-context-stack > div.active { background: #0f6b66; border-color: #0f6b66; }
        .argus-context-stack span { color: var(--argus-amber); font-size: .7rem; font-weight: 800; }
        .argus-context-stack strong { font-size: .9rem; }
        .argus-context-stack small { color: var(--argus-slate-500); }
        .argus-context-stack .active strong { color: #fff; }
        .argus-context-stack .active small { color: #bfe1dd; }
        .argus-context-stack svg { fill: none; stroke: #789099; stroke-width: 1.8; width: 100%; }
        .argus-pillars {
            border-top: 1px solid var(--argus-border);
            display: grid;
            gap: 2.5rem;
            grid-template-columns: repeat(3, 1fr);
            margin-top: 4rem;
            padding-top: 2rem;
        }
        .argus-pillars article { align-items: flex-start; display: flex; flex-direction: column; padding-right: 1.25rem; }
        .argus-pillars svg {
            fill: none;
            stroke: var(--argus-teal-700);
            stroke-linecap: round;
            stroke-linejoin: round;
            stroke-width: 1.7;
            width: 2rem;
        }
        .argus-pillars span { color: var(--argus-teal-700); display: block; font-size: .72rem; font-weight: 800; letter-spacing: .11em; margin-top: 1rem; text-transform: uppercase; }
        .argus-pillar-title { color: var(--argus-navy-950); display: block; font-size: 1.3rem; line-height: 1.2; margin: .35rem 0 .65rem; }
        .argus-pillars p { color: var(--argus-slate-700); font-size: .9rem; margin: 0 !important; }

        .argus-process {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            list-style: none;
            margin: 2.75rem 0 1.5rem;
            padding: 0;
            position: relative;
        }
        .argus-process::before {
            background: linear-gradient(90deg, #0f6b66, #8abcb7);
            content: "";
            height: 2px;
            left: 10%;
            position: absolute;
            right: 10%;
            top: 1.4rem;
        }
        .argus-process li { padding: 0 .65rem; position: relative; text-align: center; }
        .argus-process-node {
            align-items: center;
            background: var(--argus-off-white);
            border: 2px solid var(--argus-teal-600);
            border-radius: 50%;
            color: var(--argus-teal-700);
            display: inline-flex;
            font-size: .7rem;
            font-weight: 800;
            height: 2.8rem;
            justify-content: center;
            margin-bottom: 1rem;
            position: relative;
            width: 2.8rem;
            z-index: 1;
        }
        .argus-process strong { display: block; font-size: .95rem; }
        .argus-process p { color: var(--argus-slate-700); font-size: .82rem; line-height: 1.45; margin: .45rem 0 0; }

        .argus-outcome-band {
            align-items: center;
            background: var(--argus-navy-950);
            border-radius: var(--argus-radius-lg);
            color: #dce9ec;
            display: grid;
            gap: clamp(2rem, 5vw, 5rem);
            grid-template-columns: .75fr 1.25fr;
            margin: 5.5rem 0;
            overflow: hidden;
            padding: clamp(2rem, 5vw, 3.5rem);
        }
        .argus-outcome-copy h2 { color: #fff; font-size: clamp(1.8rem, 3.4vw, 2.8rem); margin: 0 0 1.3rem; }
        .argus-outcome-visual { text-align: center; }
        .argus-outcome-visual svg { max-width: 17rem; width: 100%; }
        .argus-outcome-visual svg * { fill: rgba(255,255,255,.05); stroke: #7899a4; stroke-width: 2; }
        .argus-outcome-visual .route { fill: none; }
        .argus-outcome-visual .focal { fill: none; stroke: #d7a250; stroke-width: 3.5; }
        .argus-outcome-visual .hot, .argus-outcome-visual .selected { fill: #b7791f; stroke: #f1cc8f; }
        .argus-outcome-visual span { color: #91aab3; display: block; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
        .argus-outcome-list { display: grid; gap: .4rem 1.5rem; grid-template-columns: 1fr 1fr; }
        .argus-outcome-list span { border-top: 1px solid rgba(255,255,255,.14); color: #d6e3e6; font-size: .9rem; padding: .75rem 0; }
        .argus-outcome-list span::before { color: #72c5bc; content: "✓"; margin-right: .55rem; }

        .argus-analyst-journey {
            align-items: stretch;
            display: grid;
            gap: clamp(2rem, 5vw, 5rem);
            grid-template-columns: .8fr 1.2fr;
            margin: 2.4rem 0 5.5rem;
        }
        .argus-journey-rail { list-style: none; margin: 0; padding: .4rem 0; position: relative; }
        .argus-journey-rail::before { background: #bfd4d3; bottom: 2rem; content: ""; left: 1.15rem; position: absolute; top: 2rem; width: 2px; }
        .argus-journey-rail li { align-items: center; display: flex; gap: 1rem; padding: .65rem 0; position: relative; }
        .argus-journey-rail li > span { align-items: center; background: #e3f1ef; border: 1px solid #b6d8d4; border-radius: 50%; color: #0f6b66; display: inline-flex; font-size: .66rem; font-weight: 800; height: 2.35rem; justify-content: center; width: 2.35rem; z-index: 1; }
        .argus-journey-rail strong { display: block; font-size: .96rem; }
        .argus-journey-rail small { color: var(--argus-slate-500); }
        .argus-workspace-preview { background: #fff; border: 1px solid var(--argus-border); border-radius: var(--argus-radius-lg); box-shadow: var(--argus-shadow); display: grid; grid-template-columns: 4rem 1fr; min-height: 24rem; overflow: hidden; }
        .argus-preview-sidebar { background: #092236; display: flex; flex-direction: column; gap: .75rem; padding: 1rem .8rem; }
        .argus-preview-sidebar b { color: #fff; font-size: .62rem; letter-spacing: .1em; }
        .argus-preview-sidebar i { background: rgba(255,255,255,.16); border-radius: 4px; height: .45rem; width: 100%; }
        .argus-preview-sidebar i.active { background: #5dc2b8; }
        .argus-preview-main { padding: 1.1rem; }
        .argus-preview-top { align-items: center; border-bottom: 1px solid #e0e7e9; display: flex; font-size: .72rem; font-weight: 700; justify-content: space-between; padding-bottom: .8rem; }
        .argus-preview-top em { background: #e6f3f1; border-radius: 999px; color: #0f6b66; font-size: .59rem; font-style: normal; padding: .3rem .5rem; }
        .argus-preview-grid { display: grid; gap: .8rem; grid-template-columns: 1.25fr .75fr; padding: 1rem 0; }
        .argus-mini-network { align-items: center; background: #f4f7f6; border-radius: 8px; display: flex; justify-content: center; min-height: 13rem; }
        .argus-mini-network svg { width: 95%; }
        .argus-mini-network path { fill: none; stroke: #8ba2aa; stroke-width: 1.8; }
        .argus-mini-network path.focal { stroke: #b7791f; stroke-width: 4; }
        .argus-mini-network circle { fill: #147d76; stroke: #fff; stroke-width: 3; }
        .argus-mini-network circle.core { fill: #123149; }
        .argus-mini-evidence { border: 1px solid #e0e7e9; border-radius: 8px; padding: .85rem; }
        .argus-mini-evidence strong { color: #234153; display: block; font-size: .67rem; margin: .35rem 0 .55rem; }
        .argus-mini-evidence i { background: #dce5e6; border-radius: 5px; display: block; height: .42rem; margin: .42rem 0; width: 100%; }
        .argus-mini-evidence i.short { width: 62%; }
        .argus-preview-actions { display: flex; gap: .5rem; }
        .argus-preview-actions span { background: #0f6b66; border-radius: 5px; color: #fff; font-size: .62rem; padding: .5rem .65rem; }
        .argus-preview-actions span.secondary { background: #fff; border: 1px solid #0f6b66; color: #0f6b66; }
        .argus-preview-actions span.muted { background: #edf1f2; color: #536975; }

        .argus-trust-grid { border-bottom: 1px solid var(--argus-border); border-top: 1px solid var(--argus-border); display: grid; grid-template-columns: repeat(3,1fr); margin: 2rem 0 5rem; }
        .argus-trust-grid article { display: flex; gap: 1rem; padding: 1.6rem 1.5rem 1.6rem 0; }
        .argus-trust-grid article + article { border-left: 1px solid var(--argus-border); padding-left: 1.5rem; }
        .argus-trust-icon { align-items: center; background: #e5f2f0; border-radius: 8px; display: inline-flex; flex: 0 0 2.7rem; height: 2.7rem; justify-content: center; }
        .argus-trust-icon svg { fill: none; stroke: #0f6b66; stroke-linecap: round; stroke-linejoin: round; stroke-width: 1.6; width: 1.55rem; }
        .argus-trust-grid strong { display: block; font-size: .95rem; }
        .argus-trust-grid p { color: var(--argus-slate-700); font-size: .82rem; line-height: 1.5; margin: .4rem 0 0; }

        .argus-credibility-strip { align-items: center; background: #edf3f2; border: 1px solid #d5e2e0; border-radius: 12px; display: grid; gap: 0; grid-template-columns: 1.25fr repeat(3,1fr) .75fr; margin: 1rem 0 1rem; padding: 1rem 0; }
        .argus-credibility-strip > div { border-left: 1px solid #d2dedc; display: flex; flex-direction: column; min-height: 3.4rem; padding: .35rem 1rem; }
        .argus-credibility-strip > div:first-child { border-left: 0; }
        .argus-credibility-strip strong { color: var(--argus-navy-950); font-size: 1rem; }
        .argus-credibility-strip span { color: var(--argus-slate-500); font-size: .7rem; margin-top: .2rem; }
        .argus-credibility-label span { color: var(--argus-teal-700); font-weight: 800; letter-spacing: .09em; margin: 0 0 .25rem; text-transform: uppercase; }

        .st-key-contact_callout { background: var(--argus-navy-950); border-radius: var(--argus-radius-lg); margin-top: 5rem; padding: clamp(1.8rem,4vw,3rem); }
        .st-key-contact_callout h2 { color: #fff; font-size: clamp(1.7rem,3.5vw,2.7rem); margin: 0 0 .6rem; }
        .st-key-contact_callout p { color: #c9dadf; margin: 0; }
        .st-key-contact_callout div[data-testid="stButton"] > button[kind="primary"] { background: #f2c879 !important; border-color: #f2c879 !important; color: #34240c !important; }
        .st-key-contact_callout div[data-testid="stButton"] > button[kind="primary"] p { color: #34240c !important; }
        .st-key-public_footer { margin-top: 3.5rem; }
        .argus-footer { align-items: baseline; display: flex; justify-content: space-between; margin: 0; padding: 2rem 0 1rem; }
        .argus-footer strong { color: #071c2c; font-size: 1rem; letter-spacing: .14em; }
        .argus-footer span { color: var(--argus-slate-500); }
        .st-key-public_footer div[data-testid="stButton"] > button { background: transparent; border: 0; color: var(--argus-slate-700); min-height: 2rem; padding: .2rem; }
        .st-key-public_footer div[data-testid="stButton"] > button:hover { color: var(--argus-teal-700); transform: none; }

        @media (max-width: 1200px) {
            .argus-nav-links { gap: .4rem .58rem; }
            .argus-nav-links a { font-size: .74rem; }
            .st-key-argus_public_nav div[data-testid="stButton"] > button {
                padding-left: .5rem;
                padding-right: .5rem;
            }
            .st-key-argus_public_nav div[data-testid="stButton"] > button p { font-size: .76rem; }
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
            .argus-desktop-nav { display: none; }
            .argus-mobile-nav { display: block; }
            .argus-value-intro, .argus-outcome-band, .argus-analyst-journey { grid-template-columns: 1fr; }
            .argus-pillars { gap: 1.4rem; }
            .argus-process { gap: .5rem; grid-template-columns: 1fr; margin-left: .4rem; }
            .argus-process::before { bottom: 2rem; height: auto; left: 1.4rem; right: auto; top: 1.4rem; width: 2px; }
            .argus-process li { align-items: flex-start; display: grid; gap: 1rem; grid-template-columns: 2.8rem 1fr; padding: .55rem 0; text-align: left; }
            .argus-process-node { margin: 0; }
            .argus-outcome-visual { display: none; }
            .argus-trust-grid { grid-template-columns: 1fr; }
            .argus-trust-grid article + article { border-left: 0; border-top: 1px solid var(--argus-border); padding-left: 0; }
            .argus-credibility-strip { grid-template-columns: repeat(2, 1fr); }
            .argus-credibility-strip > div { border-bottom: 1px solid #d2dedc; }
            .argus-credibility-strip > div:nth-child(odd) { border-left: 0; }
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
            .argus-value-section { padding: 4rem 0 3rem; }
            .argus-context-stack { grid-template-columns: 1fr; }
            .argus-context-stack svg { height: 1.5rem; transform: rotate(90deg); }
            .argus-pillars { grid-template-columns: 1fr; margin-top: 2.5rem; }
            .argus-pillars article { border-top: 1px solid var(--argus-border); padding-top: 1.25rem; }
            .argus-outcome-list, .argus-preview-grid { grid-template-columns: 1fr; }
            .argus-outcome-band { margin: 4rem 0; }
            .argus-workspace-preview { grid-template-columns: 3.3rem 1fr; }
            .argus-preview-sidebar { align-items: center; padding: 1rem .35rem; }
            .argus-preview-sidebar b { font-size: .46rem; letter-spacing: .04em; white-space: nowrap; }
            .argus-mini-network { min-height: 10rem; }
            .argus-preview-actions { flex-wrap: wrap; }
            .argus-credibility-strip { grid-template-columns: 1fr; }
            .argus-credibility-strip > div { border-left: 0; }
            .argus-footer { align-items: flex-start; flex-direction: column; gap: .35rem; }
            .st-key-contact_callout { padding: 1.5rem; }
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

        @media (max-width: 640px) {
            .login-spacer { height: .5rem; }
            .login-context { min-height: 0; padding: 1.5rem; }
            .login-context h1 { font-size: 2.25rem; margin-bottom: .8rem; }
            .login-context p { font-size: .95rem; }
            .login-boundary { margin-top: 1.25rem; padding-top: .85rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_portal_shell() -> None:
    """Apply responsive analyst-workspace styling."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            max-width: 18rem !important;
            min-width: 18rem !important;
            width: 18rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1rem;
            width: 100% !important;
        }
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

        .portal-status-grid {
            display: grid;
            gap: .15rem .55rem;
            grid-template-columns: auto 1fr;
            margin: .2rem 0 .8rem;
        }
        .portal-status-grid span { color: #91a9b1 !important; font-size: .65rem; text-transform: uppercase; }
        .portal-status-grid strong { color: #f4f8f8; font-size: .72rem; font-weight: 600; }

        .section-kicker {
            color: var(--argus-teal-700);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .14em;
            margin-top: .35rem;
        }
        .section-kicker.model { color: #315f85; }

        .argus-metric-context {
            color: var(--argus-slate-500);
            font-size: .76rem;
            line-height: 1.4;
            margin: -.5rem .25rem 1.1rem;
            min-height: 2.1rem;
        }

        .argus-review-path {
            background: #edf3f2;
            border: 1px solid #d5e2e0;
            border-radius: 12px;
            display: grid;
            gap: 0;
            grid-template-columns: repeat(3, 1fr);
            list-style: none;
            margin: .35rem 0 1.5rem;
            padding: 0;
        }
        .argus-review-path li {
            align-items: center;
            display: flex;
            gap: .75rem;
            min-width: 0;
            padding: .85rem 1rem;
        }
        .argus-review-path li + li { border-left: 1px solid #d5e2e0; }
        .argus-review-path li > b {
            align-items: center;
            background: #d9e6e4;
            border-radius: 999px;
            color: var(--argus-teal-700);
            display: inline-flex;
            flex: 0 0 1.8rem;
            font-size: .74rem;
            height: 1.8rem;
            justify-content: center;
        }
        .argus-review-path li.active > b { background: var(--argus-teal-700); color: #ffffff; }
        .argus-review-path strong { display: block; font-size: .8rem; }
        .argus-review-path span {
            color: var(--argus-slate-500);
            display: block;
            font-size: .68rem;
            line-height: 1.35;
            margin-top: .12rem;
        }

        .st-key-overview_next_action {
            background: linear-gradient(120deg, #0a2a3e 0%, #0f5f5b 100%);
            border: 1px solid rgba(15,107,102,.24);
            border-radius: 12px;
            box-shadow: 0 12px 28px rgba(7,28,44,.12);
            margin: .25rem 0 1.5rem;
            padding: 1rem 1.15rem;
        }
        .argus-next-action-copy span,
        .argus-review-start-copy span,
        .argus-snapshot-heading span,
        .argus-insight-banner > span {
            display: block;
            font-size: .66rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .argus-next-action-copy span { color: #8fd8d1; }
        .argus-next-action-copy strong { color: #ffffff; display: block; font-size: 1.12rem; }
        .argus-next-action-copy p { color: #c8dadd; font-size: .82rem; margin: .2rem 0 0; }
        .st-key-overview_next_action div[data-testid="stButton"] > button[kind="primary"] {
            background: #f2c879 !important;
            border-color: #f2c879 !important;
            color: #34240c !important;
        }
        .st-key-overview_next_action div[data-testid="stButton"] > button[kind="primary"] p {
            color: #34240c !important;
        }

        .argus-results-summary {
            align-items: center;
            background: #eef4f3;
            border: 1px solid #d4e1df;
            border-radius: 9px;
            display: grid;
            gap: 1rem;
            grid-template-columns: auto auto 1fr;
            margin: .75rem 0;
            padding: .7rem .9rem;
        }
        .argus-results-summary > div {
            border-right: 1px solid #ccdcd9;
            min-width: 6rem;
            padding-right: 1rem;
        }
        .argus-results-summary strong { display: block; font-size: 1rem; }
        .argus-results-summary span { color: var(--argus-slate-500); font-size: .68rem; }
        .argus-results-summary p {
            color: var(--argus-slate-700);
            font-size: .74rem;
            line-height: 1.4;
            margin: 0;
        }

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
        .case-summary-card small {
            color: var(--argus-slate-500);
            display: block;
            font-size: .7rem;
            line-height: 1.35;
            margin-top: .55rem;
        }

        .argus-empty-state {
            align-items: center;
            background: #f1f6f5;
            border: 1px dashed #b7cbc9;
            border-radius: 10px;
            display: flex;
            gap: 1rem;
            margin: .7rem 0 1.2rem;
            padding: 1.15rem;
        }
        .argus-empty-state svg { fill: none; flex: 0 0 2.7rem; stroke: #3d7b77; stroke-linecap: round; stroke-width: 1.8; width: 2.7rem; }
        .argus-empty-state strong { display: block; font-size: .9rem; }
        .argus-empty-state span { color: var(--argus-slate-500); display: block; font-size: .78rem; margin-top: .2rem; }

        .argus-provenance-line { align-items: center; display: flex; gap: .7rem; margin: -.1rem 0 1rem; }
        .argus-provenance-badge { background: #eef2f6; border: 1px solid #d5dee6; border-radius: 999px; color: #315f85; cursor: help; display: inline-flex; font-size: .7rem; font-weight: 700; gap: .35rem; padding: .3rem .55rem; white-space: nowrap; }
        .argus-provenance-detail { color: var(--argus-slate-500); font-size: .72rem; }

        .st-key-investigation_action_panel {
            background: #ffffff;
            border: 1px solid var(--argus-border);
            border-radius: 9px;
            margin-top: .7rem;
            padding: .75rem .9rem;
        }
        .st-key-investigation_action_panel p { margin-bottom: .15rem; }
        .argus-selected-case-copy > span {
            color: var(--argus-teal-700);
            display: block;
            font-size: .64rem;
            font-weight: 800;
            letter-spacing: .12em;
        }
        .argus-selected-case-copy > strong { display: block; font-size: 1rem; margin-top: .12rem; }
        .argus-selected-case-copy p {
            color: var(--argus-slate-700);
            font-size: .78rem;
            line-height: 1.4;
            margin: .18rem 0;
        }
        .argus-selected-case-copy small { color: var(--argus-slate-500); font-size: .7rem; }

        .st-key-case_review_start {
            background: #eaf3f1;
            border: 1px solid #cbdedb;
            border-left: 4px solid var(--argus-teal-700);
            border-radius: 10px;
            margin: 1rem 0 .5rem;
            padding: .85rem 1rem;
        }
        .argus-review-start-copy span { color: var(--argus-teal-700); }
        .argus-review-start-copy strong { display: block; font-size: .95rem; margin-top: .08rem; }
        .argus-review-start-copy p {
            color: var(--argus-slate-700);
            font-size: .78rem;
            line-height: 1.45;
            margin: .18rem 0 0;
        }
        .st-key-case_action_bar {
            background: #f0f5f4;
            border: 1px solid #d4e1df;
            border-radius: 10px;
            margin: 1rem 0 .35rem;
            padding: .85rem 1rem;
        }
        .st-key-case_action_bar h4 { font-size: .85rem; margin: 0; }
        .st-key-case_action_bar div[data-testid="stCaptionContainer"] { margin-bottom: .35rem; }
        .st-key-case_action_bar div[data-testid="stButton"] > button { min-height: 2.35rem; }
        .st-key-case_action_bar .st-key-false-positive button,
        .st-key-case_action_bar .st-key-close button { color: #7a3440; }

        .model-threshold-strip {
            background: #f0f5f4;
            border: 1px solid #d5e1df;
            border-radius: 9px;
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            margin: .75rem 0 1.4rem;
        }
        .model-threshold-strip > div { border-left: 1px solid #d5e1df; padding: .7rem .85rem; }
        .model-threshold-strip > div:first-child { border-left: 0; }
        .model-threshold-strip span { color: var(--argus-slate-500); display: block; font-size: .68rem; }
        .model-threshold-strip strong { display: block; font-size: .9rem; margin-top: .15rem; }
        .model-threshold-strip small {
            color: var(--argus-slate-500);
            display: block;
            font-size: .64rem;
            line-height: 1.35;
            margin-top: .22rem;
        }

        .argus-snapshot-heading {
            border-left: 3px solid var(--argus-teal-700);
            margin: 1rem 0 .8rem;
            padding: .15rem 0 .15rem .85rem;
        }
        .argus-snapshot-heading span { color: var(--argus-teal-700); }
        .argus-snapshot-heading strong { display: block; font-size: 1rem; margin-top: .08rem; }
        .argus-snapshot-heading p {
            color: var(--argus-slate-700);
            font-size: .78rem;
            line-height: 1.45;
            margin: .2rem 0 0;
            max-width: 56rem;
        }

        .argus-metric-guide {
            display: grid;
            gap: .65rem;
            grid-template-columns: repeat(4, 1fr);
            margin: .65rem 0 1rem;
        }
        .argus-metric-guide article {
            background: #ffffff;
            border: 1px solid var(--argus-border);
            border-radius: 9px;
            padding: .75rem .8rem;
        }
        .argus-metric-guide span {
            color: var(--argus-teal-700);
            display: block;
            font-size: .65rem;
            font-weight: 800;
            letter-spacing: .08em;
        }
        .argus-metric-guide strong { display: block; font-size: .78rem; margin-top: .16rem; }
        .argus-metric-guide p {
            color: var(--argus-slate-500);
            font-size: .69rem;
            line-height: 1.4;
            margin: .18rem 0 0;
        }

        .argus-insight-banner {
            background: #e9f4f1;
            border: 1px solid #c8dfda;
            border-radius: 10px;
            margin: .5rem 0 1.35rem;
            padding: .9rem 1rem;
        }
        .argus-insight-banner > span { color: var(--argus-teal-700); }
        .argus-insight-banner strong { display: block; font-size: .92rem; margin-top: .1rem; }
        .argus-insight-banner p {
            color: var(--argus-slate-700);
            font-size: .78rem;
            line-height: 1.5;
            margin: .22rem 0 0;
        }

        .filter-button-spacer { height: 1.72rem; }
        [data-testid="stDataFrame"] { border: 1px solid var(--argus-border); border-radius: 8px; }
        [data-testid="stMetricValue"] { font-size: clamp(1.3rem, 2.2vw, 1.9rem); }

        @media (max-width: 900px) {
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                max-width: 15rem !important;
                min-width: 15rem !important;
                width: 15rem !important;
            }
            [data-testid="stMainBlockContainer"] { padding-top: 1.2rem; }
            .case-status-card { align-items: flex-start; }
            .argus-provenance-line { align-items: flex-start; flex-direction: column; }
            .argus-review-path { grid-template-columns: 1fr; }
            .argus-review-path li + li { border-left: 0; border-top: 1px solid #d5e2e0; }
            .argus-results-summary { grid-template-columns: repeat(2, auto); }
            .argus-results-summary p { grid-column: 1 / -1; }
            .argus-metric-guide { grid-template-columns: repeat(2, 1fr); }
            .model-threshold-strip { grid-template-columns: repeat(2, 1fr); }
            .model-threshold-strip > div { border-bottom: 1px solid #d5e1df; }
        }

        @media (max-width: 640px) {
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                max-width: 86vw !important;
                min-width: 0 !important;
                width: 86vw !important;
            }
            [data-testid="stMainBlockContainer"] {
                padding-left: .85rem;
                padding-right: .85rem;
                padding-top: .8rem;
            }
            .case-status-card { margin-top: .35rem; }
            .argus-metric-context { min-height: 0; }
            .argus-results-summary { grid-template-columns: 1fr 1fr; }
            .argus-results-summary > div { min-width: 0; }
            .argus-metric-guide { grid-template-columns: 1fr; }
            .model-threshold-strip { grid-template-columns: 1fr; }
            .model-threshold-strip > div {
                border-left: 0;
                padding: .65rem .8rem;
            }
            .st-key-investigation_action_panel,
            .st-key-case_action_bar { padding: .7rem .75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def anchor(name: str) -> None:
    """Render a stable in-page destination for public navigation."""

    st.markdown(f'<span id="{name}" class="argus-anchor"></span>', unsafe_allow_html=True)


def render_skip_link() -> None:
    """Expose a keyboard-only shortcut past repeated navigation."""

    st.markdown(
        '<a class="argus-skip-link" href="#argus-main-content">Skip to content</a>',
        unsafe_allow_html=True,
    )


def render_main_content_anchor() -> None:
    """Mark the beginning of route-specific content for keyboard navigation."""

    st.markdown(
        '<span id="argus-main-content" class="argus-main-anchor" tabindex="-1"></span>',
        unsafe_allow_html=True,
    )


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
