"""Post-run Sprint 2 test, lint, dependency, and report finalization."""

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
from argus.modeling.derived import refresh_sprint2_derived_artifacts
from argus.modeling.reporting import (
    comparison_context_from_run,
    plot_metric_comparison,
    render_sprint2_markdown,
    write_comparison_table,
)
from argus.modeling.verify import verify_sprint2_run

_QUALITY_WORK_PREFIX = ".q_"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _run_check(command: list[str], project_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
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
        "exit_code": result.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": "PASS" if result.returncode == 0 else "FAIL",
    }


def _safe_cleanup(work_dir: Path, run_dir: Path) -> None:
    resolved_work = work_dir.resolve()
    resolved_run = run_dir.resolve()
    if resolved_work.parent != resolved_run or not resolved_work.name.startswith(
        _QUALITY_WORK_PREFIX
    ):
        raise RuntimeError(f"Refusing to remove unexpected quality work path: {work_dir}")
    if resolved_work.exists():
        shutil.rmtree(resolved_work)


def run_sprint2_quality_checks(
    config_path: str | Path = "configs/baseline.yaml",
) -> dict[str, Any]:
    """Run final checks, persist evidence, and regenerate the Sprint 2 report."""

    config = load_config(config_path)
    project_root = Path(config["_meta"]["project_root"])
    run_dir = get_path(config, "run_dir")
    report_path = get_path(config, "generated_report")
    print("[quality] Refreshing deterministic derived artifacts", flush=True)
    derived_refresh = refresh_sprint2_derived_artifacts(config_path)
    print("[quality] Running independent saved-artifact verification", flush=True)
    artifact_verification = verify_sprint2_run(config_path)

    # Keep the base path deliberately short: several Windows tests add nested
    # directories whose complete path would otherwise exceed legacy MAX_PATH.
    work_dir = run_dir / f"{_QUALITY_WORK_PREFIX}{uuid.uuid4().hex[:6]}"
    work_dir.mkdir(parents=False, exist_ok=False)
    try:
        print("[quality] Running pytest, Ruff, and pip check", flush=True)
        checks = {
            "pytest": _run_check(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
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
                ],
                project_root,
            ),
            "pip_check": _run_check(
                [sys.executable, "-m", "pip", "check"],
                project_root,
            ),
        }
    finally:
        _safe_cleanup(work_dir, run_dir)

    passed = all(check["exit_code"] == 0 for check in checks.values())
    pytest_text = checks["pytest"]["stdout"] + checks["pytest"]["stderr"]
    match = re.search(r"(?P<count>\d+) passed in (?P<seconds>\d+(?:\.\d+)?)s", pytest_text)
    pytest_count = int(match.group("count")) if match else None
    pytest_seconds = float(match.group("seconds")) if match else None
    quality = {
        "status": "PASS" if passed else "FAIL",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "pytest_passed": pytest_count,
        "pytest_reported_seconds": pytest_seconds,
        "artifact_verification_status": artifact_verification["status"],
        "derived_artifact_refresh": derived_refresh,
        "artifact_validation_rows_recomputed": artifact_verification["validation_predictions"][
            "rows"
        ],
        "checks": checks,
    }
    atomic_write_json(quality, run_dir / "quality_report.json")

    manifest = _load_json(run_dir / "run_manifest.json")
    manifest["quality_verification"] = quality
    manifest["acceptance"]["automated_quality_checks"] = passed
    manifest["status"] = "PASS" if passed else "FAIL_QUALITY_CHECKS"

    comparison = _load_json(run_dir / "model_comparison.json")
    upstream = _load_json(get_path(config, "upstream_manifest"))
    manifest["provenance"]["raw_sources"] = {
        "transactions": upstream["provenance"]["transactions"],
        "accounts": upstream["provenance"]["accounts"],
    }
    comparison_context = comparison_context_from_run(
        provenance=manifest["provenance"],
        feature_manifest=manifest["feature_contract"],
        configuration=manifest["configuration"],
        libraries=manifest["libraries"],
        feature_scope=str(config["baseline"]["feature_scope"]),
    )
    comparison["provenance"] = comparison_context
    atomic_write_json(comparison, run_dir / "model_comparison.json")
    comparison_table = write_comparison_table(
        comparison["model_results"],
        run_dir / "model_comparison.csv",
        context=comparison_context,
    )
    plot_metric_comparison(
        comparison_table,
        run_dir / "figures" / "metric_comparison.png",
    )
    summary = (
        f"{pytest_count} tests passed in {pytest_seconds:.2f}s; Ruff lint and format, "
        "pip check, and independent saved-artifact verification passed."
        if passed and pytest_count is not None and pytest_seconds is not None
        else "One or more post-run quality checks failed; inspect quality_report.json."
    )
    artifact_paths = list(manifest["core_artifact_paths"])
    for path in (
        "artifacts/sprint2/verification_report.json",
        "artifacts/sprint2/quality_report.json",
    ):
        if path not in artifact_paths:
            artifact_paths.append(path)
    manifest["core_artifact_paths"] = artifact_paths
    render_sprint2_markdown(
        model_results=comparison["model_results"],
        champion=manifest["champion"],
        prevalence=manifest["split_prevalence"],
        runtime_seconds=manifest["runtime_seconds_by_stage"],
        artifact_paths=artifact_paths,
        destination=report_path,
        verification={"status": quality["status"], "summary": summary},
        total_runtime_seconds=float(manifest["runtime_seconds"]),
    )
    refreshed = write_run_manifest(manifest, run_dir)
    print(f"[quality] Final quality status: {quality['status']}", flush=True)
    return {
        **quality,
        "artifact_count_excluding_manifest": refreshed["artifact_count_excluding_manifest"],
        "report_path": str(report_path),
    }
