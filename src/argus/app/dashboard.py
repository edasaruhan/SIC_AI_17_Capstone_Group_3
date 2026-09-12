"""Route the ARGUS public site, analyst access, and artifact-backed workspace."""

from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st

from argus.app import portal as portal_views
from argus.app.artifacts import ArtifactLoadError, DashboardArtifacts, load_dashboard_artifacts
from argus.app.auth import (
    ANALYST_EMAIL_KEY,
    ROUTE_KEY,
    initialize_auth_session,
    is_authenticated,
    sign_out,
)
from argus.app.login import render_login
from argus.app.portal import (
    PAGES,
    PRIMARY_MODEL,
    SAVED_RESEARCH_CASE_NOTICE,
    render_investigations,
    render_model_evidence,
    render_overview,
    render_page,
)
from argus.app.public_site import (
    render_demo_request,
    render_public_home,
    render_public_information,
)
from argus.app.styles import (
    inject_global_styles,
    inject_portal_shell,
    inject_public_shell,
    render_main_content_anchor,
    render_skip_link,
)

_PAGES = PAGES
_PRIMARY_MODEL = PRIMARY_MODEL
_SAVED_RESEARCH_CASE_NOTICE = SAVED_RESEARCH_CASE_NOTICE
_comparison_focus = portal_views._comparison_focus
_metric_value = portal_views._metric_value
_PUBLIC_ROUTES = {"home", "demo", "login", "resources", "privacy", "terms"}
_VALID_ROUTES = _PUBLIC_ROUTES | {"portal"}
_TRACKED_DEMO_NOTICE = (
    "Illustrative synthetic demo · These walkthrough cases are not IBM HI-Small records and "
    "are not frozen scientific model outputs. No result from this demo should be reported as "
    "project evaluation evidence."
)
_PORTAL_WIDGET_KEYS = {
    "case_investigator_select",
    "case_back_to_investigations",
    "investigation_selected_case_id",
    "investigation_worklist_table",
    "investigation_open_case",
    "overview_case_select",
    "overview_open_case",
    "portal_nav",
    "queue_patterns",
    "queue_priorities",
    "queue_search",
    "queue_sort",
    "queue_statuses",
}
_PORTAL_WIDGET_PREFIXES = (
    "cancel-",
    "close-",
    "confirm-",
    "escalate-",
    "false-positive-",
    "network-selection-",
    "overview-review-",
    "start-review-",
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
    if final_summary.is_file():
        return sprint5_root
    sprint4_root = _project_root() / "artifacts" / "sprint4"
    sprint4_payloads = (
        sprint4_root / "dashboard_bundle.json",
        sprint4_root / "product" / "dashboard_bundle.json",
        sprint4_root / "product" / "investigation_queue.csv",
    )
    if any(path.is_file() for path in sprint4_payloads):
        return sprint4_root
    return _project_root() / "demo" / "artifacts"


@st.cache_data(show_spinner=False)
def _cached_load(path: str) -> DashboardArtifacts:
    return load_dashboard_artifacts(path)


def _query_value(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        value = value[-1] if value else None
    return str(value) if value not in (None, "") else None


def _normalize_route(value: object) -> str:
    route = str(value or "home").strip().casefold()
    if route == "public":
        return "home"
    return route if route in _VALID_ROUTES else "home"


def _requested_route() -> str:
    query_route = _query_value("view")
    if query_route is not None:
        route = _normalize_route(query_route)
        st.session_state[ROUTE_KEY] = route
        return route
    return _normalize_route(st.session_state.get(ROUTE_KEY, "home"))


def _navigate(route: str) -> None:
    """Update the product route without creating a second frontend router."""

    target = _normalize_route(route)
    st.session_state["argus_scroll_top"] = True
    st.session_state[ROUTE_KEY] = target
    st.query_params.clear()
    st.query_params["view"] = target
    if target == "portal":
        page = str(st.session_state.get("argus_portal_page", "Overview"))
        if page not in PAGES:
            page = "Overview"
        st.query_params["page"] = page


def _optional_artifacts() -> DashboardArtifacts | None:
    try:
        return _cached_load(str(configured_artifact_root()))
    except ArtifactLoadError:
        return None


def _render_public_route(route: str) -> None:
    inject_public_shell()
    render_skip_link()
    if route == "home":
        render_public_home(_navigate, artifacts=_optional_artifacts())
    elif route == "demo":
        render_demo_request(_navigate)
    elif route == "login":
        if st.session_state.pop("argus_auth_guard", False):
            st.info("Sign in to access the analyst workspace.")
        render_login(_navigate)
    else:
        render_public_information(route, _navigate)


def _selected_portal_page() -> str:
    requested = _query_value("page")
    if requested in PAGES:
        st.session_state["argus_portal_page"] = requested
        return requested
    stored = str(st.session_state.get("argus_portal_page", "Overview"))
    return stored if stored in PAGES else "Overview"


def _sync_portal_route(page: str) -> None:
    st.session_state["argus_scroll_top"] = True
    st.session_state["argus_portal_page"] = page
    st.session_state[ROUTE_KEY] = "portal"
    st.query_params.clear()
    st.query_params["view"] = "portal"
    st.query_params["page"] = page


def _open_case(case_id: str) -> None:
    st.session_state["argus_selected_case_id"] = case_id
    st.session_state["case_investigator_select"] = case_id
    st.session_state["argus_nav_override"] = "Case Investigator"
    st.session_state.pop("argus_pending_case_action", None)
    _sync_portal_route("Case Investigator")
    st.rerun()


def _back_to_investigations() -> None:
    st.session_state["argus_nav_override"] = "Investigations"
    st.session_state["argus_portal_page"] = "Investigations"
    st.session_state.pop("argus_pending_case_action", None)


def _logout() -> None:
    sign_out(st.session_state)
    st.session_state["argus_clear_portal_widgets"] = True
    st.session_state.pop("argus_portal_page", None)
    st.session_state.pop("argus_pending_case_action", None)
    st.session_state.pop("argus_sso_notice", None)
    _navigate("home")


def _clear_portal_widget_state_if_requested() -> None:
    if not st.session_state.pop("argus_clear_portal_widgets", False):
        return
    for key in list(st.session_state):
        if key in _PORTAL_WIDGET_KEYS or str(key).startswith(_PORTAL_WIDGET_PREFIXES):
            st.session_state.pop(key, None)


def _render_portal_sidebar(page: str) -> str:
    st.sidebar.markdown('<div class="argus-wordmark">ARGUS</div>', unsafe_allow_html=True)
    st.sidebar.caption("Financial Crime Investigation")
    analyst = html.escape(str(st.session_state.get(ANALYST_EMAIL_KEY) or "Demo analyst"))
    st.sidebar.markdown(
        '<div class="portal-identity"><strong>ARGUS Investigation Workspace</strong>'
        f"<span>{analyst}</span></div>",
        unsafe_allow_html=True,
    )
    override = st.session_state.pop("argus_nav_override", None)
    if override in PAGES:
        st.session_state["portal_nav"] = override
        page = str(override)
    elif "portal_nav" not in st.session_state:
        st.session_state["portal_nav"] = page
    selected = st.sidebar.radio("Workspace", PAGES, index=None, key="portal_nav")
    if selected not in PAGES:
        selected = page
    page_guidance = {
        "Overview": "See current workload and continue with the highest-priority open case.",
        "Investigations": "Search, filter and select a case from the review queue.",
        "Case Investigator": "Move from observed records to model context, then decide.",
        "Model Evidence": "Understand model quality, workload and research limitations.",
    }
    st.sidebar.caption(page_guidance[selected])
    if selected != page:
        _sync_portal_route(selected)
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <div class="portal-status-grid">
          <span>Model</span><strong>Graph-enhanced LightGBM</strong>
          <span>Environment</span><strong>Demo</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("Help"):
        st.write(
            "Start in Investigations, select a case, then review observed evidence before model "
            "evidence. Demo actions reset after sign-out."
        )
    if st.sidebar.button("Log out", key="portal_logout", width="stretch"):
        _logout()
        st.rerun()
    return selected


def _scroll_to_top_if_requested() -> None:
    if not st.session_state.pop("argus_scroll_top", False):
        return
    st.html(
        """
        <script>
        const documentRoot = window.parent.document;
        const scrollers = [
          documentRoot.querySelector('[data-testid="stMain"]'),
          documentRoot.querySelector('[data-testid="stAppViewContainer"]'),
          documentRoot.scrollingElement
        ];
        requestAnimationFrame(() => {
          scrollers.forEach((node) => node && node.scrollTo({top: 0, left: 0}));
        });
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _render_artifact_error() -> None:
    st.markdown('<p class="argus-eyebrow">WORKSPACE STATUS</p>', unsafe_allow_html=True)
    st.title("Investigation workspace unavailable")
    st.error("The saved investigation package could not be loaded in this environment.")
    st.write(
        "The public site and demo login remain available. A verified saved-artifact package is "
        "required to open analyst cases; ARGUS will not train or score a model on page load."
    )
    controls = st.columns([1, 1, 3])
    if controls[0].button("Retry", type="primary", key="artifact_retry"):
        _cached_load.clear()
        st.rerun()
    if controls[1].button("Public Site", key="artifact_public"):
        _navigate("home")
        st.rerun()


def _render_portal() -> None:
    if not is_authenticated(st.session_state):
        st.session_state["argus_auth_guard"] = True
        _navigate("login")
        st.rerun()
    inject_portal_shell()
    render_skip_link()
    page = _render_portal_sidebar(_selected_portal_page())
    render_main_content_anchor()
    _scroll_to_top_if_requested()
    login_notice = st.session_state.pop("argus_login_notice", None)
    if login_notice:
        st.success(str(login_notice))
    with st.spinner("Loading the investigation workspace…"):
        try:
            artifacts = _cached_load(str(configured_artifact_root()))
        except ArtifactLoadError:
            _render_artifact_error()
            return
    if artifacts.is_tracked_demo:
        st.warning(_TRACKED_DEMO_NOTICE, icon="⚠️")
    render_page(
        page,
        artifacts,
        open_case=_open_case,
        back_to_investigations=_back_to_investigations,
    )
    st.divider()
    st.caption(
        "Decision support only · Human review remains mandatory · Model output is not an "
        "automatic enforcement decision."
    )


def render_executive_dashboard(artifacts: DashboardArtifacts) -> None:
    """Backward-compatible entry point for the operational Overview renderer."""

    render_overview(artifacts, open_case=_open_case)


def render_investigation_queue(artifacts: DashboardArtifacts) -> None:
    """Backward-compatible entry point for the investigation worklist renderer."""

    render_investigations(artifacts, open_case=_open_case)


def render_model_comparison(artifacts: DashboardArtifacts) -> None:
    """Backward-compatible entry point for technical model evidence."""

    render_model_evidence(artifacts)


def main() -> None:
    """Run the end-to-end ARGUS product experience."""

    st.set_page_config(
        page_title="ARGUS | Financial Crime Investigation",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_auth_session(st.session_state)
    _clear_portal_widget_state_if_requested()
    inject_global_styles()
    route = _requested_route()
    _scroll_to_top_if_requested()
    if route == "portal":
        _render_portal()
    else:
        _render_public_route(route)


if __name__ == "__main__":
    main()
