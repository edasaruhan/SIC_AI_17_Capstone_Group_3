"""Visually separate prototype corporate-login experience."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from argus.app.auth import sign_in

Navigate = Callable[[str], None]


def render_login(navigate: Navigate) -> None:
    """Render safe demo authentication without claiming production identity controls."""

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown('<div class="argus-wordmark">ARGUS</div>', unsafe_allow_html=True)
    with top_right:
        st.button(
            "Public Site",
            key="login_public_site",
            width="stretch",
            on_click=navigate,
            args=("home",),
        )

    st.markdown('<div class="login-spacer"></div>', unsafe_allow_html=True)
    context, form_column = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
    with context:
        st.markdown(
            """
            <div class="login-context">
              <div class="argus-eyebrow light">SECURE-STYLE ANALYST ACCESS</div>
              <h1>Enter the investigation workspace.</h1>
              <p>
                Review prioritized cases, follow directed transaction networks, and record
                session-only analyst decisions in a focused workspace.
              </p>
              <div class="login-boundary">
                <strong>Prototype access</strong>
                <span>This demonstration is not connected to a bank identity provider.</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with form_column:
        st.markdown("## Corporate Login")
        st.caption("Use a work email to start a local demonstration session.")
        show_password = st.checkbox("Show password", key="login_show_password")
        with st.form("corporate_login_form", clear_on_submit=False):
            email = st.text_input(
                "Corporate email",
                placeholder="analyst@institution.example",
                autocomplete="email",
            )
            password = st.text_input(
                "Password",
                type="default" if show_password else "password",
                autocomplete="current-password",
            )
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            result = sign_in(st.session_state, email, password)
            if result.accepted:
                st.session_state["argus_login_notice"] = result.message
                st.session_state["argus_portal_page"] = "Overview"
                st.session_state["argus_nav_override"] = "Overview"
                navigate("portal")
                st.rerun()
            else:
                st.error(result.message)

        if st.button("Corporate SSO", key="corporate_sso", width="stretch"):
            st.session_state["argus_sso_notice"] = True
        if st.session_state.get("argus_sso_notice"):
            st.info(
                "Corporate SSO is shown as a product option but is not connected in this "
                "prototype. Use the demo sign-in form above."
            )
        st.caption(
            "Credentials are used only to establish the current demo session and are not "
            "stored by ARGUS."
        )
