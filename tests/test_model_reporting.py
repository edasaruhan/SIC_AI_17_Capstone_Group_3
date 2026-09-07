from __future__ import annotations

import numpy as np

from argus.modeling.reporting import (
    _deterministic_plot_indices,
    comparison_frame,
    plot_metric_comparison,
    plot_precision_recall_curves,
    render_sprint2_markdown,
    write_comparison_table,
)


def test_plot_decimation_is_deterministic_bounded_and_keeps_endpoints() -> None:
    indices = _deterministic_plot_indices(100_001, 10_000)

    assert len(indices) == 10_000
    assert indices[0] == 0
    assert indices[-1] == 100_000
    assert np.all(np.diff(indices) > 0)
    assert np.array_equal(indices, _deterministic_plot_indices(100_001, 10_000))


def _result(name: str, average_precision: float) -> dict[str, object]:
    return {
        "model": name,
        "threshold_basis": "fixed_from_config_not_optimized",
        "runtime_seconds": {"fit": 1.0, "validation_predict": 0.2},
        "validation_metrics": {
            "row_count": 4,
            "positive_count": 1,
            "positive_rate": 0.25,
            "average_precision": average_precision,
            "roc_auc": 0.75,
            "threshold_metrics": {
                "threshold": 0.5,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 2 / 3,
                "false_positive_rate": 1 / 3,
                "alert_count": 2,
                "alert_rate": 0.5,
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 0,
                "true_negative": 2,
            },
            "top_k_metrics": [
                {
                    "requested_k": 2,
                    "precision_at_k": 0.5,
                    "recall_at_k": 1.0,
                    "true_positives": 1,
                    "cutoff_score": 0.5,
                    "cutoff_tie_group_size": 3,
                    "selected_from_cutoff_tie_group": 1,
                    "tie_break_material_at_cutoff": True,
                }
            ],
        },
    }


def test_comparison_and_report_are_generated_from_result_objects(tmp_path) -> None:
    results = {"model_b": _result("model_b", 0.4), "model_a": _result("model_a", 0.6)}
    frame = comparison_frame(results)
    assert frame["model"].tolist() == ["model_a", "model_b"]
    assert frame["cutoff_tie_group_size_at_2"].tolist() == [3, 3]
    assert frame["tie_break_material_at_2"].tolist() == [True, True]

    csv_path = tmp_path / "comparison.csv"
    write_comparison_table(results, csv_path)
    assert csv_path.is_file()
    assert "pr_auc_average_precision" in csv_path.read_text(encoding="utf-8")

    metric_figure = plot_metric_comparison(frame, tmp_path / "metrics.png")
    curve_figure = plot_precision_recall_curves(
        np.array([0, 1, 0, 0]),
        {
            "model_a": np.array([0.1, 0.9, 0.4, 0.2]),
            "model_b": np.array([0.2, 0.8, 0.3, 0.1]),
        },
        tmp_path / "curves.png",
    )
    assert metric_figure.stat().st_size > 0
    assert curve_figure.stat().st_size > 0

    prevalence = {
        name: {
            "rows": 4,
            "positive_labels": 1,
            "positive_rate": 0.25,
            "minimum_timestamp": "2022-01-01T00:00:00",
            "maximum_timestamp": "2022-01-01T00:03:00",
        }
        for name in ("train", "validation", "test")
    }
    report = render_sprint2_markdown(
        model_results=results,
        champion={"champion_model": "model_a"},
        prevalence=prevalence,
        runtime_seconds={"modeling": 2.0},
        artifact_paths=["artifacts/sprint2/model_comparison.csv"],
        destination=tmp_path / "SPRINT_2_STATUS.md",
    )
    text = report.read_text(encoding="utf-8")
    assert "model_a" in text
    assert "0.60000000" in text
    assert "Final test model inference/evaluation: **NOT USED**" in text
    assert "Top-K cutoff tie diagnostics" in text
    assert "source_row_number" in text
