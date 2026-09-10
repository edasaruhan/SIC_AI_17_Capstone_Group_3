from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from argus.app.public_site import validate_demo_request


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


def test_public_resource_and_legal_routes_render() -> None:
    routes = {
        "Explore Project Resources": "Review the evidence behind the ARGUS prototype.",
        "Privacy": "Prototype privacy notice",
        "Terms": "Prototype terms",
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
    assert any("No message was sent" in item.value for item in app.markdown)


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
