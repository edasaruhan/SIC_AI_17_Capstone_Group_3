"""Synthetic orchestration tests for the one-shot final-evaluation boundary."""

from __future__ import annotations

import ast
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from argus.final_evaluation import pipeline
from argus.final_evaluation import verify as final_verify
from argus.modeling.artifacts import atomic_write_json


class _FakeConnection:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("connection_closed")


def _config(run_dir: Path) -> dict[str, Any]:
    return {
        "_meta": {"config_file": "synthetic.yaml", "project_root": str(run_dir.parent)},
        "project": {"random_seed": 42},
        "paths": {"run_dir": str(run_dir)},
        "sprint5": {
            "one_shot_gate": {
                "freeze_contract_path": "freeze_contract.json",
                "receipt_path": "FINAL_TEST_OPENED.json",
            },
            "outputs": {"failure_marker_json": "FINAL_EVALUATION_FAILED.json"},
        },
    }


def _install_orchestration_stubs(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    events: list[str],
) -> None:
    config = _config(run_dir)
    monkeypatch.setattr(pipeline, "load_config", lambda _path: config)
    monkeypatch.setattr(
        pipeline,
        "get_path",
        lambda _config, key: Path(_config["paths"][key]),
    )

    def build_contract(_config: dict[str, Any]) -> dict[str, Any]:
        events.append("freeze_verified_in_memory")
        return {"frozen_champion": "graph_enhanced_lightgbm"}

    monkeypatch.setattr(pipeline, "build_freeze_contract", build_contract)
    monkeypatch.setattr(
        pipeline.duckdb,
        "connect",
        lambda: _FakeConnection(events),
    )
    monkeypatch.setattr(
        pipeline,
        "_configure_duckdb",
        lambda *_args, **_kwargs: events.append("duckdb_configured"),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_frozen_runtime",
        lambda *_args: events.append("runtime_loaded") or object(),
    )

    original_receipt = pipeline.create_exclusive_access_receipt

    def open_receipt(
        path: Path,
        *,
        freeze_contract_path: Path,
        freeze_contract: dict[str, Any],
    ) -> dict[str, Any]:
        assert freeze_contract_path.is_file()
        events.append("receipt_opened")
        return original_receipt(
            path,
            freeze_contract_path=freeze_contract_path,
            freeze_contract=freeze_contract,
        )

    monkeypatch.setattr(pipeline, "create_exclusive_access_receipt", open_receipt)


def test_one_shot_receipt_precedes_first_scoring_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "sprint5"
    events: list[str] = []
    _install_orchestration_stubs(monkeypatch, run_dir, events)

    def score(*_args: Any) -> object:
        assert (run_dir / "FINAL_TEST_OPENED.json").is_file()
        events.append("test_scored")
        return object()

    expected = pipeline.FinalEvaluationRunResult(
        run_dir=run_dir,
        manifest_path=run_dir / "run_manifest.json",
        report_path=tmp_path / "report.md",
        manifest={"status": "PASS", "frozen_champion": "graph_enhanced_lightgbm"},
    )
    monkeypatch.setattr(pipeline, "_score_final_test", score)
    monkeypatch.setattr(
        pipeline,
        "_persist_success",
        lambda *_args, **_kwargs: events.append("artifacts_published") or expected,
    )

    result = pipeline.run_final_evaluation("synthetic.yaml")

    assert result is expected
    assert events.index("freeze_verified_in_memory") < events.index("runtime_loaded")
    assert events.index("runtime_loaded") < events.index("receipt_opened")
    assert events.index("receipt_opened") < events.index("test_scored")
    assert events.index("test_scored") < events.index("artifacts_published")


def test_post_receipt_failure_is_marked_and_cannot_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "sprint5"
    events: list[str] = []
    _install_orchestration_stubs(monkeypatch, run_dir, events)

    def fail_score(*_args: Any) -> object:
        assert (run_dir / "FINAL_TEST_OPENED.json").is_file()
        raise RuntimeError("synthetic inference failure")

    def mark_failure(
        _config: dict[str, Any],
        *,
        receipt_path: Path,
        stage: str,
        error: BaseException,
    ) -> None:
        assert receipt_path.is_file()
        (run_dir / "FINAL_EVALUATION_FAILED.json").write_text(
            json.dumps({"stage": stage, "error": str(error), "retry_permitted": False}),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline, "_score_final_test", fail_score)
    monkeypatch.setattr(pipeline, "_write_failure_marker", mark_failure)

    with pytest.raises(pipeline.FinalEvaluationError, match="synthetic inference failure"):
        pipeline.run_final_evaluation("synthetic.yaml")

    assert (run_dir / "FINAL_TEST_OPENED.json").is_file()
    assert (run_dir / "FINAL_EVALUATION_FAILED.json").is_file()
    with pytest.raises(pipeline.FinalEvaluationError, match="cannot be retried"):
        pipeline.run_final_evaluation("synthetic.yaml")


def test_pipeline_exposes_no_training_or_selection_call() -> None:
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding="utf-8"))
    forbidden_calls = {
        "fit",
        "fit_duckdb",
        "fit_pandas",
        "fit_transform",
        "partial_fit",
        "select_champion",
        "tune",
        "optimize_threshold",
    }
    observed = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    observed.update(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    assert observed.isdisjoint(forbidden_calls)


def test_existing_receipt_blocks_before_contract_or_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "sprint5"
    run_dir.mkdir()
    (run_dir / "FINAL_TEST_OPENED.json").write_text("{}\n", encoding="utf-8")
    config = _config(run_dir)
    monkeypatch.setattr(pipeline, "load_config", lambda _path: config)
    monkeypatch.setattr(
        pipeline,
        "get_path",
        lambda _config, key: Path(_config["paths"][key]),
    )
    monkeypatch.setattr(
        pipeline,
        "build_freeze_contract",
        lambda _config: pytest.fail("contract must not run after receipt consumption"),
    )

    with pytest.raises(pipeline.FinalEvaluationError, match="cannot be retried"):
        pipeline.run_final_evaluation("synthetic.yaml")


def test_persisted_payload_passes_independent_saved_artifact_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "sprint5"
    sprint4_dir = tmp_path / "sprint4"
    sprint4_dir.mkdir()
    outputs = {
        "validation_reference_json": "validation_reference.json",
        "prevalence_comparison_json": "prevalence_comparison.json",
        "prevalence_comparison_csv": "prevalence_comparison.csv",
        "final_metrics_json": "final_metrics.json",
        "final_metrics_csv": "final_metrics.csv",
        "final_model_comparison_json": "final_model_comparison.json",
        "final_model_comparison_csv": "final_model_comparison.csv",
        "final_pr_curves_csv": "final_pr_curves.csv",
        "final_top_k_metrics_json": "final_top_k_metrics.json",
        "final_top_k_metrics_csv": "final_top_k_metrics.csv",
        "final_test_summary_json": "final_test_summary.json",
        "final_test_predictions_parquet": "final_test_predictions.parquet",
        "run_manifest_json": "run_manifest.json",
        "verification_report_json": "verification_report.json",
        "quality_report_json": "quality_report.json",
        "failure_marker_json": "FINAL_EVALUATION_FAILED.json",
    }
    config = {
        "_meta": {"config_file": "synthetic.yaml", "project_root": str(tmp_path)},
        "project": {"random_seed": 42},
        "paths": {
            "run_dir": str(run_dir),
            "generated_report": str(tmp_path / "generated" / "final.md"),
            "sprint4_run_dir": str(sprint4_dir),
        },
        "sprint5": {
            "outputs": outputs,
            "parquet_compression": "zstd",
            "ranking_tie_break": "source_row_number_ascending",
            "top_k": [1, 3, 5],
        },
    }
    names = (
        "graph_enhanced_lightgbm",
        "refined_transaction_lightgbm",
        "graphsage_edge_classifier",
    )
    roles = {
        "graph_enhanced_lightgbm": "frozen_champion",
        "refined_transaction_lightgbm": "comparator",
        "graphsage_edge_classifier": "comparator",
    }
    thresholds = {
        "graph_enhanced_lightgbm": 0.1,
        "refined_transaction_lightgbm": 0.2,
        "graphsage_edge_classifier": 0.3,
    }
    timestamps = pd.date_range("2022-01-01", periods=12, freq="min")
    labels = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)
    metadata = pd.DataFrame(
        {
            "transaction_id": [f"tx-{index:02d}" for index in range(12)],
            "source_row_number": np.arange(1, 13, dtype=np.int64),
            "timestamp": timestamps,
            "is_laundering": labels,
        }
    )
    freeze = {
        "schema": "argus.final_evaluation.freeze_contract.v1",
        "created_before_final_test_access": True,
        "frozen_champion": "graph_enhanced_lightgbm",
        "selection_partition": "validation",
        "evaluation_partition": "test",
        "model_selection_locked_before_test": True,
        "thresholds_locked_before_test": True,
        "final_test_rows_read": False,
        "final_test_labels_read": False,
        "frozen_references": {"sprint4_manifest": {"path": "run_manifest.json"}},
        "validation_partition": {
            "rows": 10,
            "positives": 2,
            "negatives": 8,
            "positive_rate": 0.2,
        },
        "test_partition_metadata_only": {
            "rows": 12,
            "positives": 4,
            "negatives": 8,
            "positive_rate": 1 / 3,
            "minimum_timestamp": timestamps.min().isoformat(),
            "maximum_timestamp": timestamps.max().isoformat(),
        },
        "models": {
            name: {
                "role": roles[name],
                "feature_family": f"synthetic_{name}",
                "model": {"path": f"{name}.model", "sha256": "a" * 64},
                "preprocessor_state": {
                    "path": f"{name}.state",
                    "sha256": "b" * 64,
                },
                "threshold_source": {
                    "path": f"{name}.thresholds",
                    "sha256": "c" * 64,
                },
                "frozen_raw_threshold": thresholds[name],
                "validation_average_precision": 0.4,
            }
            for name in names
        },
        "top_k": [1, 3, 5],
    }
    run_dir.mkdir()
    freeze_path = run_dir / "freeze_contract.json"
    atomic_write_json(freeze, freeze_path)
    opening = pipeline.create_exclusive_access_receipt(
        run_dir / "FINAL_TEST_OPENED.json",
        freeze_contract_path=freeze_path,
        freeze_contract=freeze,
    )
    raw_scores = {
        "graph_enhanced_lightgbm": np.linspace(-2.0, 2.0, 12),
        "refined_transaction_lightgbm": np.linspace(2.0, -2.0, 12),
        "graphsage_edge_classifier": np.array(
            [-1.0, 0.8, -0.7, -0.6, 0.7, -0.5, -0.4, 0.6, -0.3, -0.2, -0.1, 0.5]
        ),
    }
    identity_audit = {
        "schema": "argus.final_evaluation.test_identity_audit.v1",
        "status": "PASS",
        "exact_split_membership_join_performed_inside_authorized_one_shot_run": True,
        "raw_test_reopened_by_post_run_verifier": False,
    }
    graph_audit = {
        "schema": "argus.final_evaluation.graph_inference.v1",
        "message_context_partition": "train",
        "message_context_rows": 300_000,
        "validation_edges_used_for_message_passing": False,
        "test_edges_used_for_message_passing": False,
        "test_labels_used_for_graph_construction": False,
    }
    scored = pipeline._ScoredFinalTest(
        metadata=metadata,
        raw_scores=raw_scores,
        identity_audit=identity_audit,
        graph_audit=graph_audit,
        scoring_audit={"same_authorized_one_shot_run": True},
    )
    recorder = pipeline._StageRecorder()
    recorder.seconds["synthetic"] = 0.01

    result = pipeline._persist_success(
        config,
        freeze,
        opening,
        scored,
        recorder,
        started_at=datetime.now(UTC),
        started_monotonic=time.perf_counter(),
    )

    monkeypatch.setattr(final_verify, "load_config", lambda _path: config)
    monkeypatch.setattr(final_verify, "get_path", lambda _config, _key: run_dir)
    verification = final_verify.verify_final_evaluation("synthetic.yaml")
    inventoried = {row["path"] for row in result.manifest["artifacts"]}
    assert verification["status"] == "PASS"
    assert "quality_report.json" not in inventoried
    assert "verification_report.json" not in inventoried
