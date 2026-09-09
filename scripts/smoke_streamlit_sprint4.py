"""Render every Sprint 4 Streamlit screen from the saved real artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest


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

    print("Sprint 4 Streamlit artifact smoke status: PASS")
    print(f"Artifact root: {artifact_root}")
    print(f"Rendered screens: {', '.join(rendered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
