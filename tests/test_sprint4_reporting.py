from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from argus.sprint4 import reporting


def _payload() -> dict[str, object]:
    comparison = [
        {
            "model": model,
            "row_count": 761_749,
            "average_precision": average_precision,
            "roc_auc": 0.8,
            "precision_at_k": 0.02,
            "recall_at_k": 0.1,
            "precision": 0.03,
            "recall": 0.2,
            "f1": 0.05,
            "false_positive_rate": 0.01,
            "alert_count": 5_000,
        }
        for model, average_precision in (
            ("graph_enhanced_lightgbm", 0.47),
            ("refined_transaction_lightgbm", 0.35),
            ("graphsage_edge_classifier", 0.009),
        )
    ]
    return {
        "status": "PASS",
        "target_design": {"unsupported_account_label_created": False},
        "experiment_scope": "sampled_train_graph_full_validation_evaluation",
        "model_comparison": comparison,
        "sampling": {
            "training_context": {
                "sampled_rows": 300_000,
                "population_rows": 3_000_000,
                "coverage_fraction": 0.1,
                "maximum_timestamp": "2022-09-07T00:00:00",
            },
            "inference_context": {
                "sampled_rows": 300_000,
                "population_rows": 3_500_000,
                "coverage_fraction": 0.0857142857,
            },
            "supervised_training": {
                "sampled_rows": 50_000,
                "sampled_positives": 1_000,
                "minimum_timestamp": "2022-09-07T00:01:00",
            },
        },
        "product": {
            "cases": {
                "count": 25,
                "minimum_observed_evidence": 3,
                "deterministic_fallback_notes": 25,
            }
        },
        "runtime_seconds_by_stage": {"evaluation": 2.5, "training": 10.0},
        "runtime_seconds": 12.5,
        "acceptance": {"artifacts_complete": True, "test_sealed": True},
        "quality": {
            "status": "PASS",
            "pytest_passed": 449,
            "artifact_verification_status": "PASS",
            "checks": {
                "pytest": {"status": "PASS"},
                "ruff_lint": {"status": "PASS"},
            },
        },
    }


def test_sprint4_report_is_rendered_only_from_manifest_values(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "nested" / "SPRINT_4_STATUS.md"

    result = reporting.render_sprint4_report(payload, destination)

    assert result == destination
    rendered = destination.read_text(encoding="utf-8")
    assert "Sprint 4 status: **PASS**" in rendered
    assert "FULL 761,749 rows" in rendered
    assert "| graph_enhanced_lightgbm | 0.47000000 |" in rendered
    assert "300,000 / 3,000,000 edges (0.10000000)" in rendered
    assert "Saved cases: 25" in rendered
    assert "`evaluation`: 2.500 seconds" in rendered
    assert "`total`: 12.500 seconds" in rendered
    assert "**PASS** - artifacts_complete" in rendered
    assert "Pytest passed: 449" in rendered
    assert "`ruff_lint`: **PASS**" in rendered
    assert "final-test opening" in rendered.lower()
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_sprint4_report_omits_pending_quality_section_when_not_supplied(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    payload.pop("quality")
    destination = tmp_path / "SPRINT_4_STATUS.md"

    reporting.render_sprint4_report(payload, destination)

    rendered = destination.read_text(encoding="utf-8")
    assert "Automated quality evidence" not in rendered
    assert reporting._metric(None) == "N/A"


def test_sprint4_report_preserves_existing_file_when_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "SPRINT_4_STATUS.md"
    destination.write_text("previous report\n", encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(reporting.os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        reporting.render_sprint4_report(_payload(), destination)

    assert destination.read_text(encoding="utf-8") == "previous report\n"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
