from __future__ import annotations

import json

import numpy as np
import pytest

from argus.modeling.refinement_metrics import (
    RefinementMetricError,
    analyze_validation_score_saturation,
    build_validation_threshold_frontier,
    optimize_validation_thresholds,
)


def test_saturation_diagnostics_separate_raw_and_probability_collapse() -> None:
    result = analyze_validation_score_saturation(
        [1, 0, 1, 0, 1],
        probabilities=np.array([1.0, 1.0, 0.7, 0.0, 0.0], dtype=np.float32),
        raw_scores=np.array([10.0, 9.0, 1.0, -2.0, -10.0], dtype=np.float64),
        source_row_number=[5, 1, 3, 2, 4],
        top_k_values=[1, 3],
        partition="validation",
        near_tolerance=1e-4,
    )

    probability = result["probability"]
    assert probability["input_dtype"] == "float32"
    assert probability["exact_one_count"] == 2
    assert probability["exact_one_positive_count"] == 1
    assert probability["exact_one_negative_count"] == 1
    assert probability["near_one_count"] == 2
    assert probability["exact_zero_count"] == 2
    assert probability["exact_zero_positive_count"] == 1
    assert probability["exact_zero_negative_count"] == 1
    assert probability["near_zero_count"] == 2
    assert probability["unique_score_count"] == 3
    assert probability["unique_score_fraction"] == pytest.approx(3 / 5)
    assert probability["largest_tie_group_size"] == 2
    assert probability["largest_tie_group_score"] == 1.0

    raw = result["raw_score"]
    assert raw["input_dtype"] == "float64"
    assert raw["unique_score_count"] == 5
    assert set(raw["absolute_decision_margin_quantiles"]) == {
        "q00",
        "q01",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "q99",
        "q100",
    }
    cross = result["cross_representation"]
    assert cross["probability_has_fewer_unique_scores_than_raw"] is True
    assert cross["probability_tie_groups_containing_multiple_raw_scores"] == 2
    assert cross["rows_in_probability_ties_containing_multiple_raw_scores"] == 4
    assert "probability_representation_collapses_distinguishable_raw_scores" in result["warnings"]


def test_top_k_tie_bounds_expose_source_row_number_dependence() -> None:
    result = analyze_validation_score_saturation(
        [1, 0, 1, 0],
        probabilities=[1.0, 1.0, 0.7, 0.2],
        raw_scores=[3.0, 2.0, 1.0, -1.0],
        source_row_number=[20, 10, 30, 40],
        top_k_values=[1],
        partition="validation",
    )
    top_k = result["probability"]["top_k"][0]

    assert top_k["tie_break_material_at_cutoff"] is True
    assert top_k["cutoff_tie_group_size"] == 2
    assert top_k["cutoff_tie_positive_count"] == 1
    assert top_k["selected_from_cutoff_tie_group"] == 1
    assert top_k["deterministic_true_positives_at_k"] == 0
    assert top_k["expected_true_positives_at_k"] == pytest.approx(0.5)
    assert top_k["minimum_true_positives_at_k"] == 0
    assert top_k["maximum_true_positives_at_k"] == 1
    assert top_k["expected_precision_at_k"] == pytest.approx(0.5)


def test_missing_raw_scores_are_explicit_and_json_safe() -> None:
    result = analyze_validation_score_saturation(
        [0, 1],
        probabilities=[0.2, 0.8],
        source_row_number=[1, 2],
        top_k_values=[1],
        partition="validation",
    )

    assert result["raw_score"] == {
        "available": False,
        "reason": "raw_scores_not_supplied",
    }
    assert result["cross_representation"] is None
    assert "raw_scores_unavailable" in result["warnings"][-1]
    json.dumps(result, allow_nan=False)


def test_exact_frontier_admits_complete_score_groups() -> None:
    frontier = build_validation_threshold_frontier(
        [1, 0, 1, 0],
        [0.9, 0.9, 0.5, 0.1],
        partition="validation",
    )

    assert frontier["observed_unique_score_count"] == 3
    assert frontier["frontier_point_count_including_no_alert_sentinel"] == 4
    sentinel, first, second, third = frontier["points"]
    assert sentinel["threshold"] is None
    assert sentinel["alert_count"] == 0
    assert first["threshold"] == pytest.approx(0.9)
    assert first["admitted_score_group_size"] == 2
    assert first["admitted_score_group_true_positives"] == 1
    assert first["alert_count"] == 2
    assert first["true_positive"] == 1
    assert first["false_positive"] == 1
    assert second["threshold"] == pytest.approx(0.5)
    assert second["alert_count"] == 3
    assert second["recall"] == 1.0
    assert third["alert_count"] == 4
    json.dumps(frontier, allow_nan=False)


def test_threshold_optimizer_produces_all_four_validation_summaries() -> None:
    result = optimize_validation_thresholds(
        [1, 0, 1, 0, 0],
        [0.9, 0.8, 0.7, 0.6, 0.1],
        partition="validation",
        fpr_ceiling=0.0,
        alert_budget=2,
        primary_fpr_ceiling=1 / 3,
        primary_alert_budget=3,
    )
    summaries = result["summaries"]

    assert result["test_partition_used"] is False
    assert summaries["maximum_f1"]["operating_point"]["threshold"] == pytest.approx(0.7)
    assert summaries["maximum_f1"]["objective_value"] == pytest.approx(0.8)
    assert summaries["maximum_recall_under_fpr_ceiling"]["operating_point"][
        "threshold"
    ] == pytest.approx(0.9)
    # Both 0.9 and 0.8 recall one positive under budget=2; fewer alerts wins.
    assert summaries["maximum_recall_under_alert_budget"]["operating_point"][
        "threshold"
    ] == pytest.approx(0.9)
    primary = summaries["predeclared_joint_primary_constraint"]
    assert primary["operating_point"]["threshold"] == pytest.approx(0.7)
    assert primary["operating_point"]["recall"] == 1.0
    assert "minimize_alert_count" in result["selection_tie_rule"]
    assert "frontier" not in result
    json.dumps(result, allow_nan=False)


def test_zero_alert_budget_selects_explicit_sentinel() -> None:
    result = optimize_validation_thresholds(
        [0, 1],
        [0.2, 0.8],
        partition="validation",
        fpr_ceiling=0.1,
        alert_budget=0,
        primary_fpr_ceiling=0.1,
        primary_alert_budget=0,
        include_frontier=True,
    )

    budget = result["summaries"]["maximum_recall_under_alert_budget"]
    assert budget["status"] == "selected"
    assert budget["operating_point"]["threshold"] is None
    assert budget["operating_point"]["alert_count"] == 0
    assert result["frontier"]["points"][0]["threshold_kind"].startswith("no_alert")


def test_no_positive_labels_make_recall_objectives_explicitly_unavailable() -> None:
    result = optimize_validation_thresholds(
        [0, 0, 0],
        [0.9, 0.5, 0.1],
        partition="validation",
        fpr_ceiling=0.1,
        alert_budget=1,
        primary_fpr_ceiling=0.1,
        primary_alert_budget=1,
    )

    for summary in result["summaries"].values():
        assert summary["status"] == "unavailable"
        assert summary["operating_point"] is None
        assert "No positive" in summary["reason"]
    assert result["warnings"]
    json.dumps(result, allow_nan=False)


def test_no_negative_labels_only_disable_fpr_constrained_summaries() -> None:
    result = optimize_validation_thresholds(
        [1, 1],
        [0.8, 0.2],
        partition="validation",
        fpr_ceiling=0.0,
        alert_budget=1,
        primary_fpr_ceiling=0.0,
        primary_alert_budget=1,
    )

    assert result["summaries"]["maximum_f1"]["status"] == "selected"
    assert result["summaries"]["maximum_recall_under_alert_budget"]["status"] == "selected"
    assert result["summaries"]["maximum_recall_under_fpr_ceiling"]["status"] == "unavailable"
    assert result["summaries"]["predeclared_joint_primary_constraint"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "call",
    [
        lambda: analyze_validation_score_saturation(
            [0, 1],
            probabilities=[0.1, 0.9],
            source_row_number=[1, 2],
            top_k_values=[1],
            partition="test",
        ),
        lambda: build_validation_threshold_frontier(
            [0, 1],
            [0.1, 0.9],
            partition="test",
        ),
        lambda: optimize_validation_thresholds(
            [0, 1],
            [0.1, 0.9],
            partition="train",
            fpr_ceiling=0.1,
            alert_budget=1,
            primary_fpr_ceiling=0.1,
            primary_alert_budget=1,
        ),
    ],
)
def test_every_public_api_rejects_non_validation_partition(call) -> None:
    with pytest.raises(RefinementMetricError, match="validation-only"):
        call()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"probabilities": [-0.1, 0.9]}, "inclusive interval"),
        ({"probabilities": [0.1, float("nan")]}, "finite"),
        ({"probabilities": [0.1, 0.9], "source_row_number": [1, 1]}, "unique"),
        ({"probabilities": [0.1, 0.9], "near_tolerance": 0.5}, "strictly between"),
    ],
)
def test_saturation_diagnostics_reject_invalid_inputs(kwargs, message: str) -> None:
    inputs = {
        "probabilities": [0.1, 0.9],
        "source_row_number": [1, 2],
        "top_k_values": [1],
        "partition": "validation",
    }
    inputs.update(kwargs)
    with pytest.raises(RefinementMetricError, match=message):
        analyze_validation_score_saturation([0, 1], **inputs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fpr_ceiling": 1.1}, "fpr_ceiling"),
        ({"alert_budget": -1}, "alert_budget"),
        ({"primary_fpr_ceiling": float("nan")}, "primary_fpr_ceiling"),
        ({"primary_alert_budget": True}, "primary_alert_budget"),
        ({"include_frontier": 1}, "include_frontier"),
    ],
)
def test_threshold_optimizer_rejects_invalid_constraints(kwargs, message: str) -> None:
    inputs = {
        "partition": "validation",
        "fpr_ceiling": 0.1,
        "alert_budget": 1,
        "primary_fpr_ceiling": 0.1,
        "primary_alert_budget": 1,
    }
    inputs.update(kwargs)
    with pytest.raises(RefinementMetricError, match=message):
        optimize_validation_thresholds([0, 1], [0.1, 0.9], **inputs)
