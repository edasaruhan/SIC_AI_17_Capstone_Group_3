from __future__ import annotations

import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from argus.app.public_site import _preview_figure, validate_demo_request


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _app() -> AppTest:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    return AppTest.from_file(str(app_path)).run(timeout=20)


def test_public_home_and_corporate_login_routes_work(monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_DEMO_EMAIL", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_PASSWORD", raising=False)
    app = _app()

    assert not app.exception
    assert _button(app, "Request a Demo")
    _button(app, "Corporate Login").click()
    app.run(timeout=20)

    assert not app.exception
    assert [item.label for item in app.text_input] == ["Corporate email", "Password"]
    app.text_input[0].set_value("not-an-email")
    app.text_input[1].set_value("prototype-access")
    _button(app, "Sign in").click()
    app.run(timeout=20)
    assert any("valid corporate email" in item.value for item in app.error)


def test_public_navigation_anchors_have_matching_sections() -> None:
    app = _app()
    markup = " ".join(str(item.value) for item in app.markdown)

    assert 'class="argus-skip-link" href="#argus-main-content"' in markup
    assert 'id="argus-main-content"' in markup
    assert '<details class="argus-mobile-nav">' in markup
    assert "<summary>Explore</summary>" in markup
    for anchor in (
        "product",
        "how-it-works",
        "analyst-experience",
        "responsible-ai",
        "resources",
        "contact",
    ):
        assert f'href="#{anchor}"' in markup
        assert f'id="{anchor}"' in markup

    assert "What ARGUS adds" in markup
    assert "The analyst decides what happens next" in markup
    assert "ILLUSTRATIVE CASE · A-2048" not in markup
    assert "Alert volume grows faster" not in markup


def test_public_resource_and_legal_routes_render() -> None:
    routes = {
        "View Model Evidence": "Review the evidence behind the ARGUS project.",
        "Privacy": "Privacy notice",
        "Terms": "Terms of use",
    }

    for button_label, expected_copy in routes.items():
        app = _app()
        _button(app, button_label).click()
        app.run(timeout=20)
        visible = " ".join(str(item.value) for item in app.markdown)
        assert not app.exception
        assert expected_copy in visible


def test_demo_request_validation_rejects_missing_and_invalid_values() -> None:
    errors = validate_demo_request(
        {
            "first_name": "",
            "last_name": "",
            "work_email": "invalid",
            "company": "",
            "role": "Select your role",
            "message": "",
            "consent": False,
        }
    )

    assert set(errors) == {
        "first_name",
        "last_name",
        "work_email",
        "company",
        "role",
        "message",
        "consent",
    }


def test_demo_request_valid_payload_has_no_errors() -> None:
    assert not validate_demo_request(
        {
            "first_name": "Gizem",
            "last_name": "Özcan",
            "work_email": "gizem@institution.example",
            "company": "Example Institution",
            "role": "Financial Crime Analyst",
            "message": "Evaluate the investigation workflow.",
            "consent": True,
        }
    )


def test_demo_request_form_completes_as_session_only_flow() -> None:
    app = _app()
    _button(app, "Request a Demo").click()
    app.run(timeout=20)

    for field, value in zip(
        app.text_input,
        ("Gizem", "Özcan", "gizem@institution.example", "Example Institution"),
        strict=True,
    ):
        field.set_value(value)
    app.selectbox[0].set_value("Financial Crime / AML")
    app.text_area[0].set_value("Evaluate the investigation workflow.")
    app.checkbox[0].check()
    _button(app, "Request a Demo").click()
    app.run(timeout=20)

    assert not app.exception
    assert any("Thank you, Gizem" in item.value for item in app.success)
    confirmation = " ".join(str(item.value) for item in app.markdown)
    assert "Demo environment" in confirmation
    assert "No external CRM submission is connected" in confirmation
    assert "No message was sent" in confirmation


def test_demo_request_form_shows_helpful_inline_errors() -> None:
    app = _app()
    _button(app, "Request a Demo").click()
    app.run(timeout=20)
    _button(app, "Request a Demo").click()
    app.run(timeout=20)

    assert not app.exception
    assert any("Please correct" in item.value for item in app.error)
    visible = " ".join(str(item.value) for item in app.markdown)
    assert "Enter your first name." in visible
    assert "Enter your work email." in visible
    assert "Confirm that ARGUS may use these details" in visible


def test_login_uses_enterprise_copy_and_reveals_sso_limit_only_after_click() -> None:
    app = _app()
    _button(app, "Corporate Login").click()
    app.run(timeout=20)

    visible = " ".join(str(item.value) for group in (app.markdown, app.caption) for item in group)
    assert "ANALYST ACCESS" in visible
    assert "Sign in with your institutional account." in visible
    assert "Demo Environment" in visible
    assert "Demo actions are not persisted after sign-out." in visible
    assert "SECURE-STYLE ANALYST ACCESS" not in visible
    assert "Prototype access" not in visible
    assert not app.info

    _button(app, "Corporate SSO").click()
    app.run(timeout=20)
    assert any("Corporate SSO is not connected" in item.value for item in app.info)


def test_public_network_preview_expands_context_without_scientific_claims() -> None:
    single = _preview_figure(False)
    expanded = _preview_figure(True)

    assert len(single.data[-1].x) == 2
    assert len(expanded.data[-1].x) == 6
    assert len(single.data) == 2
    assert len(expanded.data) == 8
    assert single.data[0].line.color == "#B7791F"
    assert single.data[0].line.width > expanded.data[1].line.width
    assert single.layout.height == 270
    assert expanded.layout.height == 338
    assert "scaleanchor" not in single.layout.yaxis.to_plotly_json()
    assert all("Account" in str(label) for label in expanded.data[-1].text)


def test_public_network_preview_control_switches_to_expanded_context() -> None:
    app = _app()

    app.radio[0].set_value("Explore network")
    app.run(timeout=20)

    assert not app.exception
    assert app.radio[0].value == "Explore network"
    assert any("6 accounts" in str(item.value) for item in app.markdown)


def test_streamlit_product_chrome_uses_supported_minimal_configuration() -> None:
    config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["client"]["toolbarMode"] == "minimal"
    assert config["client"]["showSidebarNavigation"] is False


def test_portal_sidebar_keeps_streamlit_collapse_behavior_and_mobile_width() -> None:
    styles_path = Path(__file__).resolve().parents[1] / "src" / "argus" / "app" / "styles.py"
    styles = styles_path.read_text(encoding="utf-8")

    assert "transform: none !important" not in styles
    assert "max-width: 86vw !important" in styles
    assert "min-width: 0 !important" in styles
    assert ".argus-mobile-nav { display: block; }" in styles
