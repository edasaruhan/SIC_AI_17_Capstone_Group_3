"""Render all four Streamlit screens from saved final-evaluation artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
from pathlib import Path

from streamlit.testing.v1 import AppTest


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _hash_tree(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not root.is_dir():
        return result
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


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
    sibling_validation = artifact_root.parent / "sprint4"
    if sibling_validation.is_dir():
        protected_roots.append(sibling_validation)
    before_hashes = {str(root): _hash_tree(root) for root in protected_roots}
    pages = (
        "Overview",
        "Investigations",
        "Case Investigator",
        "Model Evidence",
    )
    app = AppTest.from_file(str(project_root / "app.py")).run(timeout=30)
    if app.exception or not any(button.label == "Corporate Login" for button in app.button):
        print("Final Streamlit artifact smoke status: FAIL (public site)")
        return 1
    app = _login(app)
    rendered: list[str] = []
    for page in pages:
        if page != pages[0]:
            if not app.sidebar.radio:
                print(f"Final Streamlit artifact smoke status: FAIL ({page} navigation)")
                return 1
            app.sidebar.radio[0].set_value(page)
            app.run(timeout=30)
        if app.exception:
            print(f"Final Streamlit artifact smoke status: FAIL ({page})")
            for exception in app.exception:
                print(exception.value)
            return 1
        if not app.title or app.title[0].value != page:
            print(f"Final Streamlit artifact smoke status: FAIL ({page} title)")
            return 1
        rendered.append(page)

    final_copy = [str(item.value) for item in app.markdown]
    if not any("Frozen primary model" in value for value in final_copy):
        print("Final Streamlit artifact smoke status: FAIL (frozen champion absent)")
        return 1

    _button(app, "Log out").click()
    app.run(timeout=30)
    if app.exception or not any(button.label == "Corporate Login" for button in app.button):
        print("Final Streamlit artifact smoke status: FAIL (logout)")
        return 1
    after_hashes = {str(root): _hash_tree(root) for root in protected_roots}
    if before_hashes != after_hashes:
        print("Final Streamlit artifact smoke status: FAIL (artifact files changed)")
        return 1

    print("Final Streamlit artifact smoke status: PASS")
    print(f"Artifact root: {artifact_root}")
    print(f"Rendered screens: {', '.join(rendered)}")
    print("Artifact read-only check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
