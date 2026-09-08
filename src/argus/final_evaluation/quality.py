"""Post-run, saved-artifact-only quality checks for the final evaluation.

This module is deliberately downstream of the one-shot evaluation.  It may
publish verification/quality reports, but it never runs the final pipeline,
opens raw or split data, loads an estimator, or performs inference.  The final
run manifest remains the immutable baseline throughout this process.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.config import get_path, load_config
from argus.final_evaluation.verify import (
    publish_final_verification,
    verify_final_evaluation,
)
from argus.modeling.artifacts import atomic_write_json, sha256_file

_QUALITY_SCHEMA = "argus.final_evaluation.quality.v1"


class FinalEvaluationQualityError(RuntimeError):
    """Raised when the immutable quality-report boundary is unsafe."""


def _run_check(command: list[str], project_root: Path) -> dict[str, Any]:
    """Run one repository check without a shell or Python bytecode writes."""

    started = time.perf_counter()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
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


def _pytest_summary(check: Mapping[str, Any]) -> tuple[int | None, float | None]:
    output = f"{check.get('stdout', '')}\n{check.get('stderr', '')}"
    passed_match = re.search(r"(?P<count>\d+)\s+passed\b", output)
    seconds_matches = list(re.finditer(r"\bin\s+(?P<seconds>\d+(?:\.\d+)?)s\b", output))
    passed = int(passed_match.group("count")) if passed_match else None
    seconds = float(seconds_matches[-1].group("seconds")) if seconds_matches else None
    return passed, seconds


def _quality_report_path(config: Mapping[str, Any], run_dir: Path) -> Path:
    sprint5 = config.get("sprint5")
    if not isinstance(sprint5, Mapping):
        raise FinalEvaluationQualityError("Missing sprint5 configuration")
    outputs = sprint5.get("outputs")
    if not isinstance(outputs, Mapping):
        raise FinalEvaluationQualityError("Missing sprint5.outputs configuration")
    relative = outputs.get("quality_report_json")
    if not isinstance(relative, str) or not relative.strip():
        raise FinalEvaluationQualityError("Missing sprint5.outputs.quality_report_json")
    configured = Path(relative)
    if configured.is_absolute():
        raise FinalEvaluationQualityError("Final quality report path must be run-relative")
    report_path = (run_dir / configured).resolve()
    try:
        report_path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise FinalEvaluationQualityError(
            "Final quality report path escapes the final run directory"
        ) from exc
    if report_path == (run_dir / "run_manifest.json").resolve():
        raise FinalEvaluationQualityError("Quality report may not replace run_manifest.json")
    return report_path


def _inventory_paths(manifest_path: Path) -> set[str]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalEvaluationQualityError(f"Could not read final run manifest: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
        raise FinalEvaluationQualityError("Final run manifest has no artifact inventory")
    paths: set[str] = set()
    for item in payload["artifacts"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise FinalEvaluationQualityError("Final run manifest inventory is malformed")
        paths.add(str(item["path"]).replace("\\", "/"))
    return paths


def _check_commands(
    *, project_root: Path, run_dir: Path, temporary_root: Path
) -> dict[str, list[str]]:
    del project_root  # The caller supplies it to _run_check; commands stay repo-relative.
    return {
        "pytest": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(temporary_root / "pytest"),
        ],
        "ruff_format": [
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
        "ruff_lint": [
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
        "pip_check": [sys.executable, "-m", "pip", "check"],
        "streamlit_artifact_smoke": [
            sys.executable,
            "scripts/smoke_streamlit_final.py",
            "--artifact-root",
            str(run_dir),
        ],
    }


def run_final_evaluation_quality_checks(
    config_path: str | Path = "configs/final_evaluation.yaml",
) -> dict[str, Any]:
    """Publish verification and run post-run checks without re-baselining evidence."""

    config = load_config(config_path)
    project_root = Path(str(config["_meta"]["project_root"])).resolve()
    run_dir = get_path(config, "run_dir").resolve()
    manifest_path = run_dir / "run_manifest.json"
    quality_path = _quality_report_path(config, run_dir)
    quality_relative = quality_path.relative_to(run_dir).as_posix()
    inventory_paths = _inventory_paths(manifest_path)
    if quality_relative in inventory_paths:
        raise FinalEvaluationQualityError(
            "quality_report.json must remain external to the immutable manifest inventory"
        )

    manifest_sha_before = sha256_file(manifest_path)
    print("[quality] Publishing saved-artifact-only final verification", flush=True)
    published_verification = publish_final_verification(config_path)
    manifest_sha_after_publication = sha256_file(manifest_path)

    print(
        "[quality] Running full pytest, Ruff format/lint, pip check, and final Streamlit smoke",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="argus_final_quality_") as temporary:
        commands = _check_commands(
            project_root=project_root,
            run_dir=run_dir,
            temporary_root=Path(temporary),
        )
        checks = {name: _run_check(command, project_root) for name, command in commands.items()}

    manifest_sha_after_checks = sha256_file(manifest_path)
    final_verification_error: dict[str, str] | None = None
    try:
        final_verification = verify_final_evaluation(config_path)
    except Exception as exc:  # Preserve a machine-readable FAIL report for verifier failures.
        final_verification = None
        final_verification_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    manifest_sha_after_pure_verification = sha256_file(manifest_path)

    expected_sha = manifest_sha_before
    publication_sha = published_verification.get("run_manifest_sha256")
    final_sha = (
        final_verification.get("run_manifest_sha256")
        if isinstance(final_verification, Mapping)
        else None
    )
    manifest_unchanged = all(
        value == expected_sha
        for value in (
            manifest_sha_after_publication,
            manifest_sha_after_checks,
            manifest_sha_after_pure_verification,
            publication_sha,
            final_sha,
        )
    )
    published_passed = published_verification.get("status") == "PASS"
    final_verification_passed = (
        isinstance(final_verification, Mapping) and final_verification.get("status") == "PASS"
    )
    checks_passed = all(result.get("exit_code") == 0 for result in checks.values())
    passed = published_passed and final_verification_passed and checks_passed and manifest_unchanged
    pytest_passed, pytest_seconds = _pytest_summary(checks["pytest"])

    quality: dict[str, Any] = {
        "schema": _QUALITY_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "mode": "post_run_saved_artifacts_only_no_pipeline_no_raw_test_no_estimator_no_inference",
        "artifact_verification_status": published_verification.get("status"),
        "final_artifact_verification_status": (
            final_verification.get("status") if isinstance(final_verification, Mapping) else "FAIL"
        ),
        "final_artifact_verification_error": final_verification_error,
        "pytest_passed": pytest_passed,
        "pytest_reported_seconds": pytest_seconds,
        "artifact_count_excluding_manifest": published_verification.get("inventory", {}).get(
            "unique_paths"
        ),
        "quality_report_path": str(quality_path),
        "quality_report_outside_manifest_inventory": True,
        "run_manifest_sha256_before": manifest_sha_before,
        "run_manifest_sha256_after": manifest_sha_after_pure_verification,
        "run_manifest_unchanged": manifest_unchanged,
        "manifest_integrity": {
            "status": "PASS" if manifest_unchanged else "FAIL",
            "sha256_before": manifest_sha_before,
            "sha256_after_verification_publication": manifest_sha_after_publication,
            "sha256_after_repository_checks": manifest_sha_after_checks,
            "sha256_after_pure_verification": manifest_sha_after_pure_verification,
            "published_verification_reported_sha256": publication_sha,
            "pure_verification_reported_sha256": final_sha,
            "unchanged": manifest_unchanged,
            "manifest_rewritten_or_rebaselined": False,
        },
        "quality_layer_operations": {
            "final_pipeline_invoked": False,
            "raw_features_or_split_opened": False,
            "raw_test_opened": False,
            "estimator_loaded": False,
            "inference_performed": False,
            "saved_prediction_artifact_verified": True,
        },
        "checks": checks,
    }

    # This report is intentionally not added to the immutable run-manifest inventory.
    atomic_write_json(quality, quality_path)
    manifest_sha_after_quality_report = sha256_file(manifest_path)
    quality["manifest_integrity"]["sha256_after_external_quality_report"] = (
        manifest_sha_after_quality_report
    )
    final_manifest_unchanged = (
        manifest_unchanged and manifest_sha_after_quality_report == expected_sha
    )
    quality["manifest_integrity"]["unchanged"] = final_manifest_unchanged
    quality["manifest_integrity"]["status"] = "PASS" if final_manifest_unchanged else "FAIL"
    quality["run_manifest_sha256_after"] = manifest_sha_after_quality_report
    quality["run_manifest_unchanged"] = final_manifest_unchanged
    if not final_manifest_unchanged:
        quality["status"] = "FAIL"
    atomic_write_json(quality, quality_path)

    # Hash once more after the finalized external report; never repair/rewrite the manifest.
    manifest_sha_at_return = sha256_file(manifest_path)
    if manifest_sha_at_return != expected_sha:
        quality["status"] = "FAIL"
        quality["run_manifest_unchanged"] = False
        quality["manifest_integrity"]["status"] = "FAIL"
        quality["manifest_integrity"]["unchanged"] = False
        quality["manifest_integrity"]["sha256_at_return"] = manifest_sha_at_return
        atomic_write_json(quality, quality_path)

    print(f"[quality] Final evaluation quality status: {quality['status']}", flush=True)
    return quality


__all__ = [
    "FinalEvaluationQualityError",
    "run_final_evaluation_quality_checks",
]
