from __future__ import annotations

import pytest

from argus.modeling.refinement_reporting import (
    Sprint3ReportError,
    render_sprint3_markdown,
)


def _payload() -> dict[str, object]:
    top_k = [
        {
            "requested_k": 100,
            "deterministic_precision_at_k": 0.04,
            "deterministic_recall_at_k": 0.02,
            "deterministic_true_positives_at_k": 4,
            "expected_true_positives_at_k": 3.5,
            "minimum_true_positives_at_k": 2,
            "maximum_true_positives_at_k": 5,
            "tie_break_material_at_cutoff": True,
        }
    ]
    return {
        "status": "PASS",
        "selection": {
            "champion_model": "lightgbm_refined",
            "selection_partition": "validation",
            "primary_metric": "average_precision",
            "final_test_used": False,
        },
        "scope": {"sprint4_started": False},
        "graph_value_conclusion": {
            "comparison": "C vs B",
            "average_precision_delta": 0.001,
            "same_protocol_verified": True,
        },
        "baseline_vs_refined": [
            {
                "stage": "Sprint 2 baseline",
                "model": "random_forest",
                "validation_metrics": {
                    "average_precision": 0.08,
                    "roc_auc": 0.77,
                    "top_k_metrics": top_k,
                    "threshold_metrics": {
                        "precision": 0.02,
                        "recall": 0.30,
                        "f1": 0.0375,
                        "false_positive_rate": 0.01,
                        "alert_count": 5000,
                    },
                },
            },
            {
                "stage": "Sprint 3 refined",
                "model": "lightgbm_refined",
                "metrics": {
                    "pr_auc_average_precision": 0.10,
                    "roc_auc_secondary": 0.81,
                    "precision": 0.03,
                    "recall": 0.35,
                    "f1": 0.055,
                    "fpr": 0.009,
                    "alert_volume": 4800,
                },
                "top_k_metrics": top_k,
            },
        ],
        "temporal_cv": [
            {
                "candidate_id": "lgb_stable",
                "model_family": "lightgbm",
                "fold_id": 1,
                "train_row_count": 1000,
                "validation_row_count": 500,
                "metrics": {"average_precision": 0.07, "roc_auc": 0.75},
                "eligible": True,
                "runtime_seconds": {"fit": 1.2},
            }
        ],
        "ablation": [
            {
                "feature_family": "A_transaction_only",
                "scope": "primary",
                "model_family": "lightgbm",
                "parameter_digest": "abc123",
                "feature_count": 49,
                "average_precision": 0.09,
                "roc_auc": 0.79,
                "delta_vs_b": -0.01,
            },
            {
                "feature_family": "C_graph",
                "scope": "primary",
                "model_family": "lightgbm",
                "parameter_digest": "abc123",
                "feature_count": 79,
                "average_precision": 0.101,
                "roc_auc": 0.82,
                "delta_vs_b": 0.001,
            },
        ],
        "threshold_analysis": [
            {
                "model": "lightgbm_refined",
                "selection_rule": "joint_budget",
                "status": "selected",
                "operating_point": {
                    "threshold": 1.25,
                    "precision": 0.03,
                    "recall": 0.35,
                    "f1": 0.055,
                    "false_positive_rate": 0.009,
                    "alert_count": 4800,
                },
            }
        ],
        "saturation": [
            {
                "stage": "Sprint 2 baseline",
                "model": "logistic_regression",
                "raw": {"unique_score_count": 400000},
                "probability": {
                    "unique_score_count": 60000,
                    "exact_zero_count": 35,
                    "exact_one_count": 60119,
                },
                "raw_average_precision": 0.008,
                "probability_average_precision": 0.006,
                "material_top_k_ties": 3,
                "finding": "Probability saturation collapsed distinct raw margins.",
            }
        ],
        "prevalence": {
            partition: {
                "rows": 1000,
                "positive_labels": 2,
                "positive_rate": 0.002,
                "minimum_timestamp": "2022-01-01T00:00:00",
                "maximum_timestamp": "2022-01-02T00:00:00",
            }
            for partition in ("train", "validation", "test")
        },
        "runtime_seconds": {"full_pipeline": 12.5, "temporal_cv": 8.2},
        "artifact_paths": [
            {
                "path": "artifacts/sprint3/model_comparison.csv",
                "sha256": "deadbeef",
            },
            "artifacts/sprint3/ablation.csv",
        ],
        "acceptance": [
            {
                "criterion": "Temporal CV completed",
                "status": "PASS",
                "evidence": "3/3 folds",
            },
            {
                "criterion": "Final test remained unopened",
                "status": "PASS",
                "evidence": "metadata only",
            },
        ],
        "quality": {
            "status": "PASS",
            "pytest_summary": "101 passed",
            "ruff_summary": "All checks passed",
        },
    }


def test_render_sprint3_markdown_covers_evidence_sections(tmp_path) -> None:
    output = render_sprint3_markdown(
        payload=_payload(),
        destination=tmp_path / "SPRINT_3_STATUS.md",
    )

    text = output.read_text(encoding="utf-8")
    assert "lightgbm_refined" in text
    assert "0.10000000" in text
    assert "Expanding-window temporal cross-validation" in text
    assert "A_transaction_only" in text
    assert "C_graph" in text
    assert "joint_budget" in text
    assert "60,119" in text
    assert "Probability saturation collapsed distinct raw margins" in text
    assert "Precision@K" in text
    assert "2-5" in text
    assert "final test remains unopened" in text.lower()
    assert "Sprint 4 / GraphSAGE: **NOT STARTED**" in text
    assert "101 passed" in text
    assert "deadbeef" in text


@pytest.mark.parametrize(
    ("selection_value", "scope_value", "message"),
    [
        ({"final_test_used": True}, {"sprint4_started": False}, "final-test use"),
        ({"final_test_used": False}, {"sprint4_started": True}, "Sprint 4 started"),
    ],
)
def test_render_rejects_scope_violations_or_missing_evidence(
    tmp_path, selection_value, scope_value, message
) -> None:
    payload = _payload()
    payload["selection"] = selection_value
    payload["scope"] = scope_value

    with pytest.raises(Sprint3ReportError, match=message):
        render_sprint3_markdown(payload=payload, destination=tmp_path / "report.md")


def test_renderer_tolerates_missing_optional_scope_and_sections(tmp_path) -> None:
    output = render_sprint3_markdown({}, tmp_path / "report.md")

    text = output.read_text(encoding="utf-8")
    assert "NOT RECORDED" in text
    assert "No temporal-CV records were supplied" in text
    assert "No artifact paths were supplied" in text


def test_renderer_consumes_full_manifest_section_shapes(tmp_path) -> None:
    compact = _payload()
    baseline_rows = compact.pop("baseline_vs_refined")
    graph_conclusion = compact.pop("graph_value_conclusion")
    cv_rows = compact.pop("temporal_cv")
    ablation_rows = compact.pop("ablation")
    threshold_rows = compact.pop("threshold_analysis")
    compact["champion"] = compact.pop("selection")
    compact["models"] = {row["model"]: row for row in baseline_rows}
    compact["temporal_cv"] = {
        "folds": [
            {
                "fold": 1,
                "train_minimum_timestamp": "2022-01-01",
                "train_maximum_timestamp": "2022-01-02",
                "validation_minimum_timestamp": "2022-01-03",
                "validation_maximum_timestamp": "2022-01-04",
                "train_rows": 1000,
                "train_positive_labels": 2,
                "validation_rows": 500,
                "validation_positive_labels": 1,
            }
        ],
        "candidate_summary": cv_rows,
        "selected_candidates": {
            "lightgbm": {
                "candidate_id": "lgb_stable",
                "selection_metric_value": 0.07,
                "selection_partition": "inner_temporal_validation_folds",
            }
        },
    }
    compact["ablation"] = {
        "rows": ablation_rows,
        "graph_value_conclusion": graph_conclusion,
    }
    compact["threshold_analysis"] = {"models": {"lightgbm": threshold_rows[0]}}
    compact["split_prevalence"] = compact.pop("prevalence")
    compact["runtime_seconds_by_stage"] = compact.pop("runtime_seconds")
    compact["core_artifact_paths"] = compact.pop("artifact_paths")

    text = render_sprint3_markdown(compact, tmp_path / "manifest-report.md").read_text(
        encoding="utf-8"
    )

    assert "Fold boundaries" in text
    assert "Selected candidates" in text
    assert "lgb_stable" in text
    assert "C vs B" in text
    assert "lightgbm_refined" in text


def test_renderer_supports_mapping_rows_and_acceptance_flags(tmp_path) -> None:
    payload = _payload()
    payload["baseline_vs_refined"] = {
        "refined_lr": {
            "stage": "Sprint 3 refined",
            "metrics": {"average_precision": 0.05},
        }
    }
    payload["acceptance"] = {
        "convergence achieved": True,
        "graph protocol verified": {"status": "PASS", "evidence": "same digest"},
    }

    text = render_sprint3_markdown(
        payload=payload,
        destination=tmp_path / "report.md",
    ).read_text(encoding="utf-8")

    assert "refined_lr" in text
    assert "convergence achieved" in text
    assert "same digest" in text
