from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import argus.modeling.refinement as refinement_module
from argus.modeling.preprocessing import ALWAYS_FORBIDDEN_PREDICTORS
from argus.modeling.refinement import (
    RefinementPipelineError,
    _compact_score_diagnostics,
    _contract_for_experiment,
    _novel_graph_contract,
    _prepare_run_directory,
    _selected_threshold_value,
    _stable_digest,
    _trial_eligible,
    _write_ablation_artifacts,
    _write_comparison_artifacts,
    run_sprint3_refinement,
)


def test_novel_graph_contract_adds_only_three_nonredundant_fields() -> None:
    contract = _novel_graph_contract()
    baseline = _contract_for_experiment("transaction_temporal_history")
    assert set(contract.numeric_features).difference(baseline.numeric_features) == {
        "sender_prior_fan_in_degree",
        "receiver_prior_fan_out_degree",
        "pair_previous_transfer_count",
    }
    assert set(ALWAYS_FORBIDDEN_PREDICTORS).issubset(contract.forbidden_predictors)


def test_stable_digest_is_key_order_independent() -> None:
    assert _stable_digest({"b": 2, "a": 1}) == _stable_digest({"a": 1, "b": 2})


def test_compact_score_diagnostics_records_probability_collapse() -> None:
    raw = np.array([-1000.0, -999.0, 999.0, 1000.0])
    probability = np.array([0.0, 0.0, 1.0, 1.0])
    result = _compact_score_diagnostics(raw, probability)
    assert result["probability_exact_zero_count"] == 2
    assert result["probability_exact_one_count"] == 2
    assert result["probability_collapsed_raw_ranking"] is True


def test_trial_eligibility_rejects_nonconverged_logistic() -> None:
    eligible, reasons = _trial_eligible(
        "logistic_regression",
        {"converged": False},
        {"applicable": False},
        {"probability_boundary_fraction": 0.0},
    )
    assert eligible is False
    assert reasons == ["logistic_solver_did_not_converge"]


def test_trial_eligibility_rejects_pathological_lightgbm_leaf() -> None:
    eligible, reasons = _trial_eligible(
        "lightgbm",
        {"applicable": False},
        {"trees_with_absolute_leaf_over_1m": 1},
        {"probability_boundary_fraction": 0.0},
    )
    assert eligible is False
    assert "pathological_leaf_values_over_one_million" in reasons


def test_selected_threshold_reads_predeclared_joint_point() -> None:
    threshold = _selected_threshold_value(
        {
            "summaries": {
                "predeclared_joint_primary_constraint": {"operating_point": {"threshold": 0.75}}
            }
        },
        np.array([0.1, 0.9]),
    )
    assert threshold == 0.75


def test_prepare_run_directory_preserves_only_marker(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "sprint3"
    (run_dir / "nested").mkdir(parents=True)
    (run_dir / "nested" / "old.json").write_text("{}", encoding="utf-8")
    (run_dir / ".gitkeep").write_text("", encoding="utf-8")
    _prepare_run_directory(run_dir)
    assert sorted(item.name for item in run_dir.iterdir()) == [".gitkeep"]


def test_prepare_run_directory_rejects_unexpected_path(tmp_path: Path) -> None:
    with pytest.raises(RefinementPipelineError, match="unexpected run directory"):
        _prepare_run_directory(tmp_path / "other")


def test_frozen_input_failure_preserves_previous_successful_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "artifacts" / "sprint3"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "run_manifest.json"
    sentinel.write_text('{"status":"PASS"}', encoding="utf-8")
    paths = {
        "feature_table": tmp_path / "features.parquet",
        "split_table": tmp_path / "splits.parquet",
        "split_metadata": tmp_path / "split.json",
        "upstream_manifest": tmp_path / "upstream.json",
        "run_dir": run_dir,
        "generated_report": tmp_path / "report.md",
        "sprint2_run_dir": tmp_path / "sprint2",
    }
    config = {
        "_meta": {"project_root": str(tmp_path)},
        "sprint3": {"final_test_access": False, "full_data": True, "sampled": False},
        "baseline": {},
    }
    monkeypatch.setattr(refinement_module, "load_config", lambda _path: config)
    monkeypatch.setattr(refinement_module, "get_path", lambda _config, key: paths[key])

    def fail_frozen_verification(**_kwargs: object) -> object:
        raise RefinementPipelineError("frozen input mismatch")

    monkeypatch.setattr(
        refinement_module,
        "_verify_frozen_inputs",
        fail_frozen_verification,
    )

    with pytest.raises(RefinementPipelineError, match="frozen input mismatch"):
        run_sprint3_refinement(tmp_path / "refinement.yaml")

    assert sentinel.read_text(encoding="utf-8") == '{"status":"PASS"}'


def _metrics(average_precision: float) -> dict[str, object]:
    return {
        "average_precision": average_precision,
        "roc_auc": 0.7,
        "threshold_metrics": {
            "threshold": 0.5,
            "precision": 0.1,
            "recall": 0.2,
            "f1": 0.1333333333,
            "false_positive_rate": 0.01,
            "alert_count": 25,
        },
        "top_k_metrics": [
            {
                "requested_k": 100,
                "precision_at_k": 0.05,
                "recall_at_k": 0.25,
                "tie_break_material_at_cutoff": False,
            }
        ],
    }


def _outer_result(family: str, average_precision: float) -> dict[str, object]:
    metrics = _metrics(average_precision)
    return {
        "candidate_id": "candidate",
        "feature_family": family,
        "ranking_score_type": "lightgbm_raw_margin",
        "ranking_metrics": metrics,
        "threshold_optimization": {
            "summaries": {
                "predeclared_joint_primary_constraint": {
                    "operating_point": metrics["threshold_metrics"]
                }
            }
        },
        "saturation": {
            "raw_score": {
                "top_k": [
                    {
                        "requested_k": 100,
                        "tie_break_material_at_cutoff": False,
                        "expected_precision_at_k": 0.05,
                        "minimum_precision_at_k": 0.05,
                        "maximum_precision_at_k": 0.05,
                        "expected_true_positives_at_k": 5.0,
                        "minimum_true_positives_at_k": 5,
                        "maximum_true_positives_at_k": 5,
                    }
                ]
            }
        },
        "fit_seconds": 1.0,
        "predict_seconds": 0.1,
        "model_family": "lightgbm",
        "transformed_feature_count": 10,
    }


def test_unified_comparison_contains_baseline_refined_graph_and_operational_metrics(
    tmp_path: Path,
) -> None:
    refined = {"lightgbm": _outer_result("transaction_temporal_history", 0.2)}
    sprint2_metrics = _metrics(0.1)
    sprint2_manifest = {
        "models": {
            "lightgbm": {
                "validation_metrics": sprint2_metrics,
                "runtime_seconds": {"fit": 2.0, "validation_predict": 0.2},
            }
        }
    }
    sprint2_saturation = {
        "models": {
            "lightgbm": {
                "ranking_score_type": "lightgbm_raw_margin",
                "raw_ranking_metrics": _metrics(0.12),
                "audit": {"raw_score": {"top_k": []}},
            }
        }
    }
    graph = _outer_result("transaction_temporal_history_graph", 0.22)

    frame, rows = _write_comparison_artifacts(
        tmp_path,
        sprint2_manifest=sprint2_manifest,
        sprint2_saturation=sprint2_saturation,
        refined=refined,
        graph_result=graph,
        champion={"champion_model": "lightgbm"},
    )

    assert len(frame) == 1
    assert {row["stage"] for row in rows} == {
        "sprint2_baseline_recomputed_raw_ranking",
        "sprint3_refined_transaction_baseline",
        "sprint3_graph_enhanced",
    }
    saved = pd.read_csv(tmp_path / "baseline_vs_refined.csv")
    assert {
        "precision_at_100",
        "recall_at_100",
        "f1",
        "false_positive_rate",
        "alert_count",
    }.issubset(saved.columns)
    assert saved.loc[
        saved["stage"] == "sprint2_baseline_recomputed_raw_ranking",
        "average_precision",
    ].iloc[0] == pytest.approx(0.12)
    refined_row = next(
        row for row in rows if row["stage"] == "sprint3_refined_transaction_baseline"
    )
    assert refined_row["top_k_metrics"][0]["expected_true_positives_at_k"] == 5.0
    assert refined_row["top_k_metrics"][0]["deterministic_precision_at_k"] == 0.05


def test_ablation_writer_labels_primary_and_sensitivity_and_recomputes_delta(
    tmp_path: Path,
) -> None:
    values = {
        "transaction_only": 0.10,
        "transaction_temporal_history": 0.20,
        "transaction_temporal_history_graph": 0.23,
        "transaction_temporal_history_graph_novel3": 0.21,
    }
    results = {family: _outer_result(family, score) for family, score in values.items()}
    frame, conclusion = _write_ablation_artifacts(
        tmp_path,
        ablation_results=results,
        parameter_digest="params",
        split_digest="split",
        random_seed=42,
        duplicate_evidence={"mismatch_rows_full_train_plus_validation": 0},
    )

    assert frame.loc[frame["feature_family"] == "transaction_only", "scope"].iloc[0] == "primary"
    assert (
        frame.loc[
            frame["feature_family"] == "transaction_temporal_history_graph_novel3",
            "scope",
        ].iloc[0]
        == "sensitivity_duplicate_removed"
    )
    assert conclusion["absolute_average_precision_delta"] == pytest.approx(0.03)
    assert conclusion["same_evaluation_protocol"] is True
