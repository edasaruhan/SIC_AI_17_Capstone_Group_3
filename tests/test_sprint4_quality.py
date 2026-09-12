from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from argus.sprint4 import quality


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _seed_run(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    run_dir = tmp_path / "sprint4"
    run_dir.mkdir()
    report_path = tmp_path / "SPRINT_4_STATUS.md"
    _write_json(run_dir / "acceptance_checklist.json", {"artifacts_complete": True})
    _write_json(
        run_dir / "run_manifest.json",
        {
            "status": "PASS",
            "core_artifact_paths": [],
        },
    )
    config = {
        "_meta": {"project_root": str(tmp_path)},
        "paths": {
            "run_dir": str(run_dir),
            "generated_report": str(report_path),
        },
    }
    return config, run_dir, report_path


def _passing_check(command: list[str], project_root: Path) -> dict[str, Any]:
    del project_root
    return {
        "command": " ".join(command),
        "exit_code": 0,
        "runtime_seconds": 0.01,
        "stdout": "420 passed in 4.50s\n" if "pytest" in command else "",
        "stderr": "",
        "status": "PASS",
    }


@pytest.mark.parametrize(
    ("initial_verification", "expected_status"),
    [("PASS", "PASS"), ("FAIL", "FAIL")],
)
def test_sprint4_quality_respects_artifact_verification_and_runs_final_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_verification: str,
    expected_status: str,
) -> None:
    config, run_dir, report_path = _seed_run(tmp_path)
    verification_calls = 0
    commands: list[list[str]] = []
    manifests: list[dict[str, Any]] = []

    monkeypatch.setattr(quality, "load_config", lambda _: config)

    def verify(_: object) -> dict[str, Any]:
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 1:
            return {"status": initial_verification}
        return {
            "status": "PASS",
            "refreshed_artifact_count_excluding_manifest": 11,
        }

    def run_check(command: list[str], project_root: Path) -> dict[str, Any]:
        commands.append(command)
        return _passing_check(command, project_root)

    def write_manifest(payload: dict[str, Any], run_path: Path) -> dict[str, Any]:
        assert run_path == run_dir
        manifests.append(payload)
        return payload

    monkeypatch.setattr(quality, "verify_sprint4_run", verify)
    monkeypatch.setattr(quality, "_run_check", run_check)
    monkeypatch.setattr(quality, "render_sprint4_report", lambda *_args: None)
    monkeypatch.setattr(quality, "write_run_manifest", write_manifest)

    result = quality.run_sprint4_quality_checks("synthetic.yaml")

    assert result["status"] == expected_status
    assert result["pytest_passed"] == 420
    assert result["final_artifact_verification_status"] == "PASS"
    assert result["artifact_count_excluding_manifest"] == 11
    assert result["report_path"] == str(report_path)
    assert verification_calls == 2
    assert len(commands) == 5
    assert all("app.py" in command for command in commands if "ruff" in command)
    assert any(
        any("smoke_streamlit_sprint4.py" in part for part in command) for command in commands
    )
    assert manifests[-1]["status"] == (
        "PASS" if expected_status == "PASS" else "FAIL_QUALITY_CHECKS"
    )
    acceptance = json.loads((run_dir / "acceptance_checklist.json").read_text(encoding="utf-8"))
    assert acceptance["automated_quality_checks"] is (expected_status == "PASS")
    assert not any(path.name.startswith(".quality_") for path in run_dir.iterdir())


def test_sprint4_quality_cleanup_accepts_only_scoped_child(tmp_path: Path) -> None:
    run_dir = tmp_path / "sprint4"
    work_dir = run_dir / ".quality_12345678"
    work_dir.mkdir(parents=True)
    (work_dir / "temporary.txt").write_text("temporary", encoding="utf-8")

    quality._safe_cleanup(work_dir, run_dir)

    assert not work_dir.exists()


def test_sprint4_quality_cleanup_rejects_unexpected_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "sprint4"
    work_dir = run_dir / "model"
    work_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="unexpected quality path"):
        quality._safe_cleanup(work_dir, run_dir)


def test_sprint4_quality_json_loader_requires_an_object(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    _write_json(payload, ["not", "an", "object"])

    with pytest.raises(RuntimeError, match="Expected a JSON object"):
        quality._load_object(payload)
