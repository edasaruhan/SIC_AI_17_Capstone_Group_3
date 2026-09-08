"""Render all four Streamlit screens from saved final-evaluation artifacts only."""

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
    pages = (
        "Executive Dashboard",
        "Investigation Queue",
        "Case Investigator",
        "Model Comparison",
    )
    app = AppTest.from_file(str(project_root / "app.py")).run(timeout=30)
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

    final_headers = [str(item.value) for item in app.header]
    final_successes = [str(item.value) for item in app.success]
    if not any("Final evaluation" in value for value in final_headers):
        print("Final Streamlit artifact smoke status: FAIL (final section absent)")
        return 1
    if not any("Pre-frozen champion" in value for value in final_successes):
        print("Final Streamlit artifact smoke status: FAIL (frozen champion absent)")
        return 1

    print("Final Streamlit artifact smoke status: PASS")
    print(f"Artifact root: {artifact_root}")
    print(f"Rendered screens: {', '.join(rendered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
