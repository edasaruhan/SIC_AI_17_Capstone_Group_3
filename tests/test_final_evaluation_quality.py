from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from argus.final_evaluation import quality


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    run_dir = tmp_path / "sprint5"
    run_dir.mkdir()
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "argus.final_evaluation.run_manifest.v1",
                "artifacts": [
                    {
                        "path": "sealed.json",
                        "size_bytes": 3,
                        "sha256": "0" * 64,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "_meta": {"project_root": str(tmp_path)},
        "paths": {"run_dir": str(run_dir)},
        "sprint5": {"outputs": {"quality_report_json": "quality_report.json"}},
    }
    return config, run_dir, manifest_path


def _verification(manifest_path: Path) -> dict[str, Any]:
    return {
        "status": "PASS",
        "run_manifest_sha256": _sha256(manifest_path),
        "inventory": {"unique_paths": 1},
        "predictions": {"rows": 2},
        "metrics": {"frozen_champion": "graph_enhanced_lightgbm"},
    }


def _passing_check(command: list[str], project_root: Path) -> dict[str, Any]:
    del project_root
    is_pytest = "pytest" in command
    return {
        "command": " ".join(command),
        "exit_code": 0,
        "runtime_seconds": 0.01,
        "stdout": "321 passed, 2 warnings in 12.34s\n" if is_pytest else "",
        "stderr": "",
        "status": "PASS",
    }


def test_quality_is_external_and_keeps_manifest_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run_dir, manifest_path = _config(tmp_path)
    original_bytes = manifest_path.read_bytes()
    events: list[str] = []

    monkeypatch.setattr(quality, "load_config", lambda _: config)

    def publish(_: object) -> dict[str, Any]:
        events.append("publish")
        return _verification(manifest_path)

    def run_check(command: list[str], project_root: Path) -> dict[str, Any]:
        events.append(Path(command[1]).name if len(command) == 2 else command[-1])
        return _passing_check(command, project_root)

    def verify(_: object) -> dict[str, Any]:
        events.append("pure_verify")
        return _verification(manifest_path)

    monkeypatch.setattr(quality, "publish_final_verification", publish)
    monkeypatch.setattr(quality, "_run_check", run_check)
    monkeypatch.setattr(quality, "verify_final_evaluation", verify)

    result = quality.run_final_evaluation_quality_checks("synthetic.yaml")

    assert result["status"] == "PASS"
    assert result["pytest_passed"] == 321
    assert result["pytest_reported_seconds"] == pytest.approx(12.34)
    assert result["run_manifest_unchanged"] is True
    assert result["manifest_integrity"]["manifest_rewritten_or_rebaselined"] is False
    assert result["quality_report_outside_manifest_inventory"] is True
    assert manifest_path.read_bytes() == original_bytes
    assert events[0] == "publish"
    assert events[-1] == "pure_verify"
    assert list(result["checks"]) == [
        "pytest",
        "ruff_format",
        "ruff_lint",
        "pip_check",
        "streamlit_artifact_smoke",
    ]

    report_path = run_dir / "quality_report.json"
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "quality_report.json" not in {row["path"] for row in manifest["artifacts"]}


def test_quality_reports_check_failure_but_still_runs_final_pure_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run_dir, manifest_path = _config(tmp_path)
    pure_verify_calls = 0

    monkeypatch.setattr(quality, "load_config", lambda _: config)
    monkeypatch.setattr(
        quality,
        "publish_final_verification",
        lambda _: _verification(manifest_path),
    )

    def one_failed_check(command: list[str], project_root: Path) -> dict[str, Any]:
        result = _passing_check(command, project_root)
        if "ruff" in command and "check" in command:
            result.update(exit_code=1, status="FAIL", stderr="synthetic lint failure")
        return result

    def verify(_: object) -> dict[str, Any]:
        nonlocal pure_verify_calls
        pure_verify_calls += 1
        return _verification(manifest_path)

    monkeypatch.setattr(quality, "_run_check", one_failed_check)
    monkeypatch.setattr(quality, "verify_final_evaluation", verify)

    result = quality.run_final_evaluation_quality_checks("synthetic.yaml")

    assert result["status"] == "FAIL"
    assert result["checks"]["ruff_lint"]["status"] == "FAIL"
    assert result["run_manifest_unchanged"] is True
    assert result["final_artifact_verification_status"] == "PASS"
    assert pure_verify_calls == 1
    assert (run_dir / "quality_report.json").is_file()


def test_quality_detects_manifest_mutation_and_never_rebaselines_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, manifest_path = _config(tmp_path)
    original_sha = _sha256(manifest_path)

    monkeypatch.setattr(quality, "load_config", lambda _: config)

    def mutating_publish(_: object) -> dict[str, Any]:
        result = _verification(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["unexpected_mutation"] = True
        manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(quality, "publish_final_verification", mutating_publish)
    monkeypatch.setattr(quality, "_run_check", _passing_check)
    monkeypatch.setattr(
        quality,
        "verify_final_evaluation",
        lambda _: {
            **_verification(manifest_path),
            "run_manifest_sha256": _sha256(manifest_path),
        },
    )

    result = quality.run_final_evaluation_quality_checks("synthetic.yaml")

    assert result["status"] == "FAIL"
    assert result["run_manifest_unchanged"] is False
    assert result["manifest_integrity"]["status"] == "FAIL"
    assert result["manifest_integrity"]["sha256_before"] == original_sha
    assert _sha256(manifest_path) != original_sha
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["unexpected_mutation"] is True


def test_quality_refuses_to_overwrite_an_inventoried_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, manifest_path = _config(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"].append(
        {"path": "quality_report.json", "size_bytes": 1, "sha256": "1" * 64}
    )
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(quality, "load_config", lambda _: config)

    with pytest.raises(
        quality.FinalEvaluationQualityError,
        match="external to the immutable manifest inventory",
    ):
        quality.run_final_evaluation_quality_checks("synthetic.yaml")


def test_final_quality_entrypoints_do_not_import_execution_or_inference_modules() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sources = [
        project_root / "src" / "argus" / "final_evaluation" / "quality.py",
        project_root / "scripts" / "verify_final_evaluation.py",
        project_root / "scripts" / "validate_final_evaluation.py",
        project_root / "scripts" / "smoke_streamlit_final.py",
    ]
    forbidden = (
        "argus.final_evaluation.pipeline",
        "argus.full_pipeline",
        "argus.pipeline",
        "argus.data",
        "argus.features",
        "argus.gnn",
        "argus.modeling.models",
        "argus.modeling.refinement_models",
    )
    imports: set[str] = set()
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert not any(
        imported == prefix or imported.startswith(f"{prefix}.")
        for imported in imports
        for prefix in forbidden
    )
