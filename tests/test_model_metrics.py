from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from argus.modeling.metrics import (
    MetricInputError,
    compute_average_precision,
    compute_roc_auc,
    compute_top_k_metrics,
    evaluate_binary_predictions,
)


def test_metric_bundle_matches_known_binary_ranking_and_threshold_values() -> None:
    result = evaluate_binary_predictions(
        y_true=np.array([0, 1, 1, 0]),
        y_score=np.array([0.1, 0.8, 0.4, 0.7]),
        source_row_number=np.array([10, 11, 12, 13]),
        threshold=0.5,
        top_k_values=[1, 2, 10],
    )

    assert result["pr_auc_method"] == "non_interpolated_average_precision"
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["pr_auc"] == result["average_precision"]
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["roc_auc_role"] == "secondary"
    assert result["positive_rate"] == pytest.approx(0.5)

    threshold = result["threshold_metrics"]
    assert threshold == {
        "threshold": 0.5,
        "threshold_comparison": "score_greater_than_or_equal",
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_positive_rate": 0.5,
        "alert_count": 2,
        "alert_rate": 0.5,
    }

    top_k = result["top_k_metrics"]
    assert top_k[0]["precision_at_k"] == 1.0
    assert top_k[0]["recall_at_k"] == 0.5
    assert top_k[1]["precision_at_k"] == 0.5
    assert top_k[1]["recall_at_k"] == 0.5
    assert top_k[2]["requested_k"] == 10
    assert top_k[2]["effective_k"] == 4
    assert top_k[2]["was_clipped_to_partition_size"] is True
    assert top_k[2]["precision_at_k"] == 0.5
    assert top_k[2]["recall_at_k"] == 1.0


def test_top_k_ties_use_source_row_number_but_average_precision_groups_ties() -> None:
    labels = [1, 0]
    scores = [0.5, 0.5]

    top_k = compute_top_k_metrics(labels, scores, [20, 10], top_k_values=[1])

    assert top_k[0]["true_positives"] == 0
    assert top_k[0]["precision_at_k"] == 0.0
    assert top_k[0]["recall_at_k"] == 0.0
    assert top_k[0]["cutoff_tie_group_size"] == 2
    assert top_k[0]["selected_from_cutoff_tie_group"] == 1
    assert top_k[0]["cutoff_tie_group_fully_included"] is False
    assert top_k[0]["tie_break_material_at_cutoff"] is True
    assert compute_average_precision(labels, scores) == pytest.approx(0.5)
    assert compute_roc_auc(labels, scores) == pytest.approx(0.5)


def test_vectorized_roc_auc_matches_reference_with_multiple_tie_groups() -> None:
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.7, 0.1, 0.7, 0.1, 0.5, 0.9])

    assert compute_roc_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_undefined_class_denominators_are_null_not_fabricated_zeroes() -> None:
    no_positives = evaluate_binary_predictions(
        [0, 0],
        [0.9, 0.1],
        [1, 2],
        threshold=0.5,
        top_k_values=[1],
    )
    assert no_positives["average_precision"] is None
    assert no_positives["roc_auc"] is None
    assert no_positives["threshold_metrics"]["recall"] is None
    assert no_positives["threshold_metrics"]["f1"] is None
    assert no_positives["top_k_metrics"][0]["recall_at_k"] is None
    assert no_positives["warnings"]

    no_negatives = evaluate_binary_predictions(
        [1, 1],
        [0.9, 0.1],
        [1, 2],
        threshold=0.5,
        top_k_values=[1],
    )
    assert no_negatives["average_precision"] == 1.0
    assert no_negatives["roc_auc"] is None
    assert no_negatives["threshold_metrics"]["false_positive_rate"] is None
    assert no_negatives["warnings"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"y_true": [0, 1], "y_score": [0.1], "source_row_number": [1, 2]},
            "same number",
        ),
        (
            {"y_true": [0, 2], "y_score": [0.1, 0.2], "source_row_number": [1, 2]},
            "binary labels",
        ),
        (
            {
                "y_true": [0, 1],
                "y_score": [0.1, math.nan],
                "source_row_number": [1, 2],
            },
            "finite values",
        ),
        (
            {"y_true": [0, 1], "y_score": [0.1, 0.2], "source_row_number": [1, 1]},
            "must be unique",
        ),
    ],
)
def test_metric_bundle_rejects_invalid_or_nondeterministic_inputs(
    kwargs: dict[str, list[float]],
    message: str,
) -> None:
    with pytest.raises(MetricInputError, match=message):
        evaluate_binary_predictions(**kwargs, threshold=0.5, top_k_values=[1])


@pytest.mark.parametrize("top_k_values", [[], [0], [True], [1, 1]])
def test_top_k_configuration_must_be_nonempty_positive_and_unique(
    top_k_values: list[int],
) -> None:
    with pytest.raises(MetricInputError):
        compute_top_k_metrics([0, 1], [0.1, 0.9], [1, 2], top_k_values=top_k_values)
