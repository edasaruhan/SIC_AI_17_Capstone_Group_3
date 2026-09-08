"""Capture four real headless-browser screenshots from saved Sprint 5 artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path


def _browser() -> Path:
    candidates = (
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    discovered = shutil.which("chrome") or shutil.which("msedge")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("Chrome or Microsoft Edge is required for screenshot capture")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _wait_for_server(url: str, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Streamlit did not become ready within {timeout_seconds} seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/sprint5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sprint5/screenshots"))
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    artifact_root = (project_root / arguments.artifact_root).resolve()
    output_dir = (project_root / arguments.output_dir).resolve()
    if not (artifact_root / "product" / "final_test_summary.json").is_file():
        raise FileNotFoundError("Verified Sprint 5 product artifacts are required")
    output_dir.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    environment = os.environ.copy()
    environment["ARGUS_ARTIFACT_DIR"] = str(artifact_root)
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(project_root / "app.py"),
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=127.0.0.1",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    server = subprocess.Popen(
        command,
        cwd=project_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    browser = _browser()
    pages = {
        "executive_dashboard": "Executive Dashboard",
        "investigation_queue": "Investigation Queue",
        "case_investigator": "Case Investigator",
        "model_comparison": "Model Comparison",
    }
    created: list[Path] = []
    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_server(base_url)
        with tempfile.TemporaryDirectory(prefix="argus_s5_browser_") as browser_profile:
            for slug, page in pages.items():
                destination = (output_dir / f"{slug}.png").resolve()
                url = f"{base_url}/?{urllib.parse.urlencode({'page': page})}"
                completed = subprocess.run(
                    [
                        str(browser),
                        "--headless=new",
                        "--disable-gpu",
                        "--disable-extensions",
                        "--hide-scrollbars",
                        "--no-first-run",
                        f"--user-data-dir={browser_profile}",
                        "--window-size=1440,1400",
                        "--virtual-time-budget=12000",
                        f"--screenshot={destination}",
                        url,
                    ],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=40,
                    check=False,
                )
                if completed.returncode != 0 or not destination.is_file():
                    raise RuntimeError(
                        f"Headless screenshot failed for {page}: "
                        f"{completed.stdout}\n{completed.stderr}"
                    )
                created.append(destination)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
    print("Final Streamlit screenshots: PASS")
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
