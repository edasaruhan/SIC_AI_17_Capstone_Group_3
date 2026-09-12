from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from argus.modeling import quality


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _seed_run(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    run_dir = tmp_path / "sprint2"
    run_dir.mkdir()
    report_path = tmp_path / "SPRINT_2_STATUS.md"
    upstream_path = tmp_path / "upstream_manifest.json"
    _write_json(
        upstream_path,
        {
            "provenance": {
                "transactions": {"sha256": "1" * 64},
                "accounts": {"sha256": "2" * 64},
            }
        },
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "status": "PASS",
            "acceptance": {},
            "provenance": {"dataset": "synthetic", "frozen_inputs": {}},
            "feature_contract": {},
            "configuration": {},
            "libraries": {},
            "champion": {"champion_model": "lightgbm"},
            "split_prevalence": {},
            "runtime_seconds_by_stage": {},
            "runtime_seconds": 1.0,
            "core_artifact_paths": [],
        },
    )
    _write_json(run_dir / "model_comparison.json", {"model_results": {}})
    config = {
        "_meta": {"project_root": str(tmp_path)},
        "paths": {
            "run_dir": str(run_dir),
            "generated_report": str(report_path),
            "upstream_manifest": str(upstream_path),
        },
        "baseline": {"feature_scope": "transaction_only"},
    }
    return config, run_dir, report_path


def _check_result(command: list[str], *, exit_code: int = 0) -> dict[str, Any]:
    is_pytest = "pytest" in command
    return {
        "command": " ".join(command),
        "exit_code": exit_code,
        "runtime_seconds": 0.01,
        "stdout": "420 passed in 4.50s\n" if is_pytest else "",
        "stderr": "" if exit_code == 0 else "synthetic failure",
        "status": "PASS" if exit_code == 0 else "FAIL",
    }


def _patch_quality_run(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    *,
    artifact_status: str,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    commands: list[list[str]] = []
    written_manifests: list[dict[str, Any]] = []
    monkeypatch.setattr(quality, "load_config", lambda _: config)
    monkeypatch.setattr(
        quality,
        "refresh_sprint2_derived_artifacts",
        lambda _: {"status": "PASS"},
    )
    monkeypatch.setattr(
        quality,
        "verify_sprint2_run",
        lambda _: {
            "status": artifact_status,
            "validation_predictions": {"rows": 12},
        },
    )

    def run_check(command: list[str], project_root: Path) -> dict[str, Any]:
        assert project_root == Path(config["_meta"]["project_root"])
        commands.append(command)
        return _check_result(command)

    def write_manifest(payload: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        del run_dir
        written_manifests.append(payload)
        return {**payload, "artifact_count_excluding_manifest": 7}

    monkeypatch.setattr(quality, "_run_check", run_check)
    monkeypatch.setattr(quality, "comparison_context_from_run", lambda **_: {})
    monkeypatch.setattr(
        quality,
        "write_comparison_table",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(quality, "plot_metric_comparison", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(quality, "render_sprint2_markdown", lambda **_kwargs: None)
    monkeypatch.setattr(quality, "write_run_manifest", write_manifest)
    return commands, written_manifests


def test_sprint2_quality_passes_and_lints_application_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run_dir, report_path = _seed_run(tmp_path)
    commands, manifests = _patch_quality_run(monkeypatch, config, artifact_status="PASS")

    result = quality.run_sprint2_quality_checks("synthetic.yaml")

    assert result["status"] == "PASS"
    assert result["pytest_passed"] == 420
    assert result["pytest_reported_seconds"] == pytest.approx(4.5)
    assert result["artifact_count_excluding_manifest"] == 7
    assert result["report_path"] == str(report_path)
    ruff_commands = [command for command in commands if "ruff" in command]
    assert len(ruff_commands) == 2
    assert all("app.py" in command for command in ruff_commands)
    assert manifests[-1]["status"] == "PASS"
    assert manifests[-1]["acceptance"]["automated_quality_checks"] is True
    assert not any(path.name.startswith(".q_") for path in run_dir.iterdir())


def test_sprint2_quality_fails_when_artifact_verification_is_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run_dir, _ = _seed_run(tmp_path)
    _, manifests = _patch_quality_run(monkeypatch, config, artifact_status="FAIL")

    result = quality.run_sprint2_quality_checks("synthetic.yaml")

    assert result["status"] == "FAIL"
    assert result["artifact_verification_status"] == "FAIL"
    assert all(check["status"] == "PASS" for check in result["checks"].values())
    assert manifests[-1]["status"] == "FAIL_QUALITY_CHECKS"
    assert manifests[-1]["acceptance"]["automated_quality_checks"] is False
    saved = json.loads((run_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert saved["status"] == "FAIL"


def test_sprint2_quality_cleanup_accepts_only_scoped_child(tmp_path: Path) -> None:
    run_dir = tmp_path / "sprint2"
    work_dir = run_dir / ".q_123456"
    work_dir.mkdir(parents=True)
    (work_dir / "temporary.txt").write_text("temporary", encoding="utf-8")

    quality._safe_cleanup(work_dir, run_dir)

    assert not work_dir.exists()


def test_sprint2_quality_cleanup_rejects_unexpected_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "sprint2"
    work_dir = run_dir / "models"
    work_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="unexpected quality work path"):
        quality._safe_cleanup(work_dir, run_dir)


def test_sprint2_quality_json_loader_requires_an_object(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    _write_json(payload, ["not", "an", "object"])

    with pytest.raises(RuntimeError, match="Expected a JSON object"):
        quality._load_json(payload)
