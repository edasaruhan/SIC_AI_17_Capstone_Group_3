"""Exercise the ARGUS product journey from immutable final-evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

from streamlit.testing.v1 import AppTest


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _element(elements, label: str):
    return next(element for element in elements if element.label == label)


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int, str]]:
    result: dict[str, tuple[int, int, str]] = {}
    if not root.is_dir():
        return result
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        metadata = path.stat()
        result[str(path.relative_to(root))] = (
            metadata.st_size,
            metadata.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return result


def _validation_root(artifact_root: Path) -> Path | None:
    reference = artifact_root / "product" / "validation_artifact_reference.json"
    if not reference.is_file():
        return None
    payload = json.loads(reference.read_text(encoding="utf-8"))
    configured = payload.get("validation_artifact_root")
    if not isinstance(configured, str) or not configured.strip():
        return None
    return (artifact_root / configured).resolve()


def _exercise_demo_request(app: AppTest) -> AppTest:
    _button(app, "Request a Demo").click()
    app.run(timeout=30)
    values = {
        "First name": "ARGUS",
        "Last name": "Smoke",
        "Work email": "smoke@institution.example",
        "Company / Bank": "Example Institution",
    }
    for label, value in values.items():
        _element(app.text_input, label).set_value(value)
    _element(app.selectbox, "Job role").set_value("Financial Crime / AML")
    _element(app.text_area, "Message or use case").set_value("Review the investigation workflow.")
    consent_label = "I agree that these details may be used to respond to this demo request."
    _element(app.checkbox, consent_label).check()
    _button(app, "Request a Demo").click()
    app.run(timeout=30)
    if app.exception or not any("Thank you, ARGUS" in item.value for item in app.success):
        raise RuntimeError("Demo request flow did not complete")
    _button(app, "Return to Site").click()
    return app.run(timeout=30)


def _login(app: AppTest) -> AppTest:
    email = f"smoke-{secrets.token_hex(4)}@argus.example"
    password = secrets.token_urlsafe(16)
    os.environ["ARGUS_DEMO_EMAIL"] = email
    os.environ["ARGUS_DEMO_PASSWORD"] = password
    _button(app, "Corporate Login").click()
    app.run(timeout=30)
    app.text_input[0].set_value(email)
    app.text_input[1].set_value(password)
    _button(app, "Sign in").click()
    return app.run(timeout=30)


def _open_case_from_worklist(app: AppTest) -> AppTest:
    app.sidebar.radio[0].set_value("Investigations")
    app.run(timeout=30)
    _element(app.text_input, "Search").set_value("no-such-argus-case")
    app.run(timeout=30)
    visible = " ".join(str(item.value) for item in app.markdown)
    if "No matching cases" not in visible:
        raise RuntimeError("Investigation empty-filter state did not render")
    _button(app, "Reset filters").click()
    app.run(timeout=30)
    if app.exception or not app.dataframe:
        raise RuntimeError("Investigation worklist did not render")
    _button(app, "Open case").click()
    return app.run(timeout=30)


def _select_case(app: AppTest, index: int) -> AppTest:
    selector = _element(app.selectbox, "Case")
    if len(selector.options) <= index:
        raise RuntimeError("Not enough cases to exercise workflow actions")
    selector.set_value(selector.options[index])
    return app.run(timeout=30)


def _confirm_action(app: AppTest, label: str) -> AppTest:
    _button(app, label).click()
    app.run(timeout=30)
    _button(app, "Confirm").click()
    return app.run(timeout=30)


def _exercise_case_workflow(app: AppTest) -> AppTest:
    if app.exception or not app.title or app.title[0].value != "Case Investigator":
        raise RuntimeError("Case Investigator did not open")

    _button(app, "Start Review").click()
    app.run(timeout=30)
    _element(app.text_area, "Add a note").set_value("Smoke-test review note.")
    _button(app, "Save note").click()
    app.run(timeout=30)
    if not any("Smoke-test review note." in str(item.value) for item in app.markdown):
        raise RuntimeError("Analyst note was not reflected in the session")
    app = _confirm_action(app, "Escalate")

    app = _select_case(app, 1)
    app = _confirm_action(app, "Mark False Positive")

    app = _select_case(app, 2)
    app = _confirm_action(app, "Close Case")
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/sprint5"),
    )
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    artifact_root = arguments.artifact_root.expanduser()
    if not artifact_root.is_absolute():
        artifact_root = (project_root / artifact_root).resolve()
    else:
        artifact_root = artifact_root.resolve()
    final_summary = artifact_root / "product" / "final_test_summary.json"
    if not final_summary.is_file():
        print(f"Final Streamlit artifact smoke status: FAIL (missing {final_summary})")
        return 1

    os.environ["ARGUS_ARTIFACT_DIR"] = str(artifact_root)
    os.environ.pop("ARGUS_SPRINT4_ARTIFACT_DIR", None)
    protected_roots = [artifact_root]
    validation_root = _validation_root(artifact_root)
    if validation_root is not None and validation_root.is_dir():
        protected_roots.append(validation_root)
    before_snapshot = {str(root): _snapshot_tree(root) for root in protected_roots}
    app = AppTest.from_file(str(project_root / "app.py")).run(timeout=30)
    if app.exception or not any(button.label == "Corporate Login" for button in app.button):
        print("Final Streamlit artifact smoke status: FAIL (public site)")
        return 1
    try:
        app = _exercise_demo_request(app)
    except RuntimeError as error:
        print(f"Final Streamlit artifact smoke status: FAIL ({error})")
        return 1
    app = _login(app)
    if app.exception or not app.title or app.title[0].value != "Overview":
        print("Final Streamlit artifact smoke status: FAIL (Overview)")
        return 1
    try:
        app = _open_case_from_worklist(app)
        app = _exercise_case_workflow(app)
    except RuntimeError as error:
        print(f"Final Streamlit artifact smoke status: FAIL ({error})")
        return 1

    app.sidebar.radio[0].set_value("Model Evidence")
    app.run(timeout=30)
    if app.exception or not app.title or app.title[0].value != "Model Evidence":
        print("Final Streamlit artifact smoke status: FAIL (Model Evidence)")
        return 1

    final_copy = [str(item.value) for item in app.markdown]
    if not any("Frozen primary model" in value for value in final_copy):
        print("Final Streamlit artifact smoke status: FAIL (frozen champion absent)")
        return 1

    _button(app, "Log out").click()
    app.run(timeout=30)
    if app.exception or not any(button.label == "Corporate Login" for button in app.button):
        print("Final Streamlit artifact smoke status: FAIL (logout)")
        return 1
    after_snapshot = {str(root): _snapshot_tree(root) for root in protected_roots}
    if before_snapshot != after_snapshot:
        print("Final Streamlit artifact smoke status: FAIL (artifact files changed)")
        return 1

    print("Final Streamlit artifact smoke status: PASS")
    print(f"Artifact root: {artifact_root}")
    print("Rendered flow: Public site, Demo request, Login, Overview, Investigations, ")
    print("Case Investigator actions, Model Evidence, Logout")
    print("Artifact read-only check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
