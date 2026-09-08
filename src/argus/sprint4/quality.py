"""Post-run independent verification and repository quality checks for Sprint 4."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.config import get_path, load_config
from argus.modeling.artifacts import atomic_write_json, write_run_manifest
from argus.sprint4.reporting import render_sprint4_report
from argus.sprint4.verify import verify_sprint4_run

_QUALITY_PREFIX = ".quality_"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _run_check(command: list[str], project_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    return {
        "command": subprocess.list2cmdline(command),
        "exit_code": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _safe_cleanup(work_dir: Path, run_dir: Path) -> None:
    resolved_work = work_dir.resolve()
    resolved_run = run_dir.resolve()
    if resolved_work.parent != resolved_run or not resolved_work.name.startswith(_QUALITY_PREFIX):
        raise RuntimeError(f"Refusing to remove unexpected quality path: {work_dir}")
    if work_dir.exists():
        shutil.rmtree(work_dir)


def run_sprint4_quality_checks(
    config_path: str | Path = "configs/sprint4.yaml",
) -> dict[str, Any]:
    """Verify artifacts, run the complete suite, and refresh evidence/report."""

    config = load_config(config_path)
    project_root = Path(config["_meta"]["project_root"])
    run_dir = get_path(config, "run_dir")
    report_path = get_path(config, "generated_report")
    print("[quality] Independently verifying Sprint 4 artifacts", flush=True)
    verification = verify_sprint4_run(config_path)
    work_dir = run_dir / f"{_QUALITY_PREFIX}{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=False, exist_ok=False)
    try:
        print("[quality] Running full pytest, Ruff lint/format, and pip check", flush=True)
        checks = {
            "pytest": _run_check(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    str(work_dir / "pytest"),
                ],
                project_root,
            ),
            "ruff_lint": _run_check(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--no-cache",
                    "src",
                    "scripts",
                    "tests",
                    "app.py",
                ],
                project_root,
            ),
            "ruff_format": _run_check(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "format",
                    "--no-cache",
                    "--check",
                    "src",
                    "scripts",
                    "tests",
                    "app.py",
                ],
                project_root,
            ),
            "pip_check": _run_check([sys.executable, "-m", "pip", "check"], project_root),
            "streamlit_artifact_smoke": _run_check(
                [
                    sys.executable,
                    "scripts/smoke_streamlit_sprint4.py",
                    "--artifact-root",
                    str(run_dir / "product"),
                ],
                project_root,
            ),
        }
    finally:
        _safe_cleanup(work_dir, run_dir)

    passed = verification["status"] == "PASS" and all(
        result["exit_code"] == 0 for result in checks.values()
    )
    pytest_output = checks["pytest"]["stdout"] + checks["pytest"]["stderr"]
    match = re.search(r"(?P<count>\d+) passed in (?P<seconds>\d+(?:\.\d+)?)s", pytest_output)
    quality = {
        "status": "PASS" if passed else "FAIL",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "pytest_passed": int(match.group("count")) if match else None,
        "pytest_reported_seconds": float(match.group("seconds")) if match else None,
        "artifact_verification_status": verification["status"],
        "checks": checks,
    }
    atomic_write_json(quality, run_dir / "quality_report.json")
    acceptance_path = run_dir / "acceptance_checklist.json"
    acceptance = _load_object(acceptance_path)
    acceptance["automated_quality_checks"] = passed
    atomic_write_json(acceptance, acceptance_path)

    manifest = _load_object(run_dir / "run_manifest.json")
    manifest["quality"] = quality
    manifest["acceptance"] = acceptance
    manifest["status"] = "PASS" if passed and all(acceptance.values()) else "FAIL_QUALITY_CHECKS"
    for path in (
        "artifacts/sprint4/verification_report.json",
        "artifacts/sprint4/quality_report.json",
    ):
        if path not in manifest["core_artifact_paths"]:
            manifest["core_artifact_paths"].append(path)
    render_sprint4_report(manifest, run_dir / "SPRINT_4_STATUS.md")
    refreshed = write_run_manifest(manifest, run_dir)
    render_sprint4_report(refreshed, report_path)
    final_verification = verify_sprint4_run(config_path)
    print(f"[quality] Sprint 4 quality status: {quality['status']}", flush=True)
    return {
        **quality,
        "final_artifact_verification_status": final_verification["status"],
        "artifact_count_excluding_manifest": final_verification[
            "refreshed_artifact_count_excluding_manifest"
        ],
        "report_path": str(report_path),
    }


__all__ = ["run_sprint4_quality_checks"]
