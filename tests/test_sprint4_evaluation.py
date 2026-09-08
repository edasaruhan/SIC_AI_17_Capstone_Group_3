from __future__ import annotations

import numpy as np
import pytest

from argus.sprint4.evaluation import (
    Sprint4EvaluationError,
    build_pr_curve_table,
    compare_validation_models,
    graphsage_score_diagnostics,
)


def test_comparison_uses_same_validation_rows_and_joint_constraint() -> None:
    labels = np.array([0, 1, 0, 1, 0, 0], dtype=np.int8)
    rows = np.arange(1, 7)
    scores = {
        "graphsage": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.0]),
        "reference": np.array([0.2, 0.6, 0.1, 0.7, 0.4, 0.3]),
    }
    comparison, thresholds, top_k = compare_validation_models(
        labels,
        rows,
        scores,
        top_k=(1, 3),
        alert_budget=3,
        fpr_ceiling=0.5,
    )
    assert {row["model"] for row in comparison} == set(scores)
    assert all(row["row_count"] == 6 for row in comparison)
    assert all(row["test_metrics_used"] is False for row in comparison)
    assert all(
        item["summaries"]["predeclared_joint_primary_constraint"]["status"] == "selected"
        for item in thresholds.values()
    )
    assert len(top_k) == 4


def test_comparison_rejects_misaligned_score_vector() -> None:
    with pytest.raises(Sprint4EvaluationError, match="misaligned"):
        compare_validation_models(
            [0, 1],
            [1, 2],
            {"bad": [0.1]},
            top_k=(1,),
            alert_budget=1,
            fpr_ceiling=0.5,
        )


def test_pr_curve_thinning_is_explicit_and_bounded() -> None:
    labels = np.tile([0, 1], 50)
    scores = {"model": np.linspace(0.0, 1.0, 100)}
    frame = build_pr_curve_table(labels, scores, maximum_points_per_model=12)
    assert len(frame) <= 12
    assert frame["downsampled_for_display_only"].all()
    assert frame["source_curve_points"].iloc[0] > len(frame)


def test_graphsage_diagnostics_keeps_raw_and_probability_separate() -> None:
    audit = graphsage_score_diagnostics(
        [0, 1, 0, 1],
        [-1000.0, 1000.0, -2.0, 2.0],
        [0.0, 1.0, 0.1, 0.9],
        [1, 2, 3, 4],
        top_k=(1, 2),
    )
    assert audit["partition"] == "validation"
    assert audit["raw_score"]["available"] is True
    assert audit["probability"]["exact_one_count"] == 1
