"""Render every Sprint 4 Streamlit screen from the saved real artifacts."""

from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path

from streamlit.testing.v1 import AppTest


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/sprint4/product"),
    )
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    artifact_root = arguments.artifact_root
    if not artifact_root.is_absolute():
        artifact_root = (project_root / artifact_root).resolve()
    os.environ["ARGUS_SPRINT4_ARTIFACT_DIR"] = str(artifact_root)

    pages = (
        "Overview",
        "Investigations",
        "Case Investigator",
        "Model Evidence",
    )
    app = AppTest.from_file(str(project_root / "app.py")).run(timeout=30)
    if app.exception or not any(button.label == "Corporate Login" for button in app.button):
        print("Sprint 4 Streamlit artifact smoke status: FAIL (public site)")
        return 1
    app = _login(app)
    rendered: list[str] = []
    for page in pages:
        if page != pages[0]:
            app.sidebar.radio[0].set_value(page)
            app.run(timeout=30)
        if app.exception:
            print(f"Sprint 4 Streamlit artifact smoke status: FAIL ({page})")
            for exception in app.exception:
                print(exception.value)
            return 1
        if not app.title or app.title[0].value != page:
            print(f"Sprint 4 Streamlit artifact smoke status: FAIL ({page} title)")
            return 1
        rendered.append(page)

    _button(app, "Log out").click()
    app.run(timeout=30)
    if app.exception or not any(button.label == "Corporate Login" for button in app.button):
        print("Sprint 4 Streamlit artifact smoke status: FAIL (logout)")
        return 1

    print("Sprint 4 Streamlit artifact smoke status: PASS")
    print(f"Artifact root: {artifact_root}")
    print(f"Rendered screens: {', '.join(rendered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
