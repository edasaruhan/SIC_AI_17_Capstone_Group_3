"""Validation-only score diagnostics and exact threshold optimization.

Sprint 3 needs to distinguish ranking quality from artifacts introduced by a
probability transform.  This module therefore reports raw-margin and probability
score saturation separately, quantifies uncertainty when a configured Top-K
cuts through a tied score group, and optimizes operating thresholds only on the
validation partition.

Threshold candidates are *exact score groups*: an inclusive threshold admits a
whole tied group.  No arbitrary threshold grid and no row-level tie breaking is
used for threshold selection.  Top-K remains deterministic by sorting score
descending and ``source_row_number`` ascending, but its expected/minimum/maximum
true-positive counts expose how material that arbitrary tie break is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from argus.modeling.metrics import MetricInputError


class RefinementMetricError(MetricInputError):
    """Raised when Sprint 3 diagnostic or threshold inputs are invalid."""


_SELECTION_TIE_RULE = (
    "maximize_objective_then_minimize_alert_count_then_minimize_false_positives_"
    "then_choose_highest_threshold"
)


def _require_validation_partition(partition: str) -> None:
    if partition != "validation":
        raise RefinementMetricError(
            "Sprint 3 score diagnostics and threshold optimization are validation-only; "
            f"received partition={partition!r}"
        )


def _validated_labels(y_true: Sequence[int] | np.ndarray) -> np.ndarray:
    labels = np.asarray(y_true)
    if labels.ndim != 1:
        raise RefinementMetricError("y_true must be one-dimensional")
    if labels.size == 0:
        raise RefinementMetricError("At least one validation row is required")
    if labels.dtype.kind not in {"b", "i", "u", "f"}:
        raise RefinementMetricError("y_true must contain numeric binary labels")
    numeric = labels.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise RefinementMetricError("y_true must contain only finite binary labels 0 and 1")
    return numeric.astype(np.int8, copy=False)


def _validated_scores(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    expected_size: int,
    probability: bool,
) -> tuple[np.ndarray, str]:
    original = np.asarray(values)
    if original.ndim != 1 or original.size != expected_size:
        raise RefinementMetricError(
            f"{name} must be one-dimensional and align one-to-one with y_true"
        )
    try:
        numeric = original.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise RefinementMetricError(f"{name} must contain numeric values") from exc
    if not np.isfinite(numeric).all():
        raise RefinementMetricError(f"{name} must contain only finite values")
    if probability and (np.any(numeric < 0.0) or np.any(numeric > 1.0)):
        raise RefinementMetricError("probabilities must lie in the inclusive interval [0, 1]")
    return numeric, str(original.dtype)


def _validated_source_rows(
    source_row_number: Sequence[int] | np.ndarray,
    *,
    expected_size: int,
) -> np.ndarray:
    rows = np.asarray(source_row_number)
    if rows.ndim != 1 or rows.size != expected_size:
        raise RefinementMetricError(
            "source_row_number must be one-dimensional and align one-to-one with y_true"
        )
    try:
        numeric = rows.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise RefinementMetricError("source_row_number must contain integer values") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise RefinementMetricError("source_row_number must contain finite integer values")
    integer_rows = numeric.astype(np.int64)
    if np.unique(integer_rows).size != integer_rows.size:
        raise RefinementMetricError(
            "source_row_number values must be unique for deterministic Top-K tie breaking"
        )
    return integer_rows


def _validated_top_k_values(top_k_values: Iterable[int]) -> tuple[int, ...]:
    values = tuple(top_k_values)
    if not values:
        raise RefinementMetricError("top_k_values must contain at least one positive integer")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise RefinementMetricError("Every configured K must be a positive integer")
        normalized.append(int(value))
    if len(set(normalized)) != len(normalized):
        raise RefinementMetricError("top_k_values must not contain duplicates")
    return tuple(normalized)


def _validated_near_tolerance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RefinementMetricError("near_tolerance must be numeric")
    tolerance = float(value)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 0.5:
        raise RefinementMetricError("near_tolerance must be finite and strictly between 0 and 0.5")
    return tolerance


def _validated_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RefinementMetricError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RefinementMetricError(f"{name} must be finite and in [0, 1]")
    return result


def _validated_alert_budget(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise RefinementMetricError(f"{name} must be a non-negative integer")
    return int(value)


def _margin_quantiles(scores: np.ndarray, *, center: float) -> dict[str, float]:
    margins = np.abs(scores - center)
    probabilities = np.array([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    values = np.quantile(margins, probabilities)
    return {
        f"q{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, values, strict=True)
    }


def _tie_aware_top_k(
    labels: np.ndarray,
    scores: np.ndarray,
    source_rows: np.ndarray,
    *,
    top_k_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    ranking = np.lexsort((source_rows, -scores))
    ranked_labels = labels[ranking]
    ranked_scores = scores[ranking]
    cumulative_true_positives = np.cumsum(ranked_labels, dtype=np.int64)
    positive_count = int(labels.sum())

    results: list[dict[str, Any]] = []
    for requested_k in top_k_values:
        effective_k = min(requested_k, labels.size)
        cutoff_score = float(ranked_scores[effective_k - 1])
        strictly_above = scores > cutoff_score
        cutoff_tie = scores == cutoff_score
        strictly_above_count = int(np.count_nonzero(strictly_above))
        fixed_true_positives = int(labels[strictly_above].sum())
        tie_group_size = int(np.count_nonzero(cutoff_tie))
        tie_positive_count = int(labels[cutoff_tie].sum())
        tie_negative_count = tie_group_size - tie_positive_count
        selected_from_tie = effective_k - strictly_above_count

        selected_mask_in_ranking = ranked_scores[:effective_k] == cutoff_score
        deterministic_tie_true_positives = int(
            ranked_labels[:effective_k][selected_mask_in_ranking].sum()
        )
        deterministic_total = int(cumulative_true_positives[effective_k - 1])
        expected_tie_true_positives = selected_from_tie * tie_positive_count / tie_group_size
        minimum_tie_true_positives = max(0, selected_from_tie - tie_negative_count)
        maximum_tie_true_positives = min(selected_from_tie, tie_positive_count)

        expected_total = fixed_true_positives + expected_tie_true_positives
        minimum_total = fixed_true_positives + minimum_tie_true_positives
        maximum_total = fixed_true_positives + maximum_tie_true_positives
        results.append(
            {
                "requested_k": requested_k,
                "effective_k": effective_k,
                "was_clipped_to_partition_size": effective_k != requested_k,
                "cutoff_score": cutoff_score,
                "strictly_above_cutoff_count": strictly_above_count,
                "cutoff_tie_group_size": tie_group_size,
                "cutoff_tie_positive_count": tie_positive_count,
                "cutoff_tie_negative_count": tie_negative_count,
                "selected_from_cutoff_tie_group": selected_from_tie,
                "cutoff_tie_group_fully_included": selected_from_tie == tie_group_size,
                "tie_break_material_at_cutoff": selected_from_tie < tie_group_size,
                "deterministic_tie_break": "source_row_number_ascending",
                "deterministic_true_positives_from_cutoff_tie": (deterministic_tie_true_positives),
                "expected_true_positives_from_cutoff_tie": float(expected_tie_true_positives),
                "minimum_true_positives_from_cutoff_tie": minimum_tie_true_positives,
                "maximum_true_positives_from_cutoff_tie": maximum_tie_true_positives,
                "deterministic_true_positives_at_k": deterministic_total,
                "expected_true_positives_at_k": float(expected_total),
                "minimum_true_positives_at_k": minimum_total,
                "maximum_true_positives_at_k": maximum_total,
                "deterministic_precision_at_k": float(deterministic_total / effective_k),
                "expected_precision_at_k": float(expected_total / effective_k),
                "minimum_precision_at_k": float(minimum_total / effective_k),
                "maximum_precision_at_k": float(maximum_total / effective_k),
                "deterministic_recall_at_k": (
                    None if positive_count == 0 else float(deterministic_total / positive_count)
                ),
                "expected_recall_at_k": (
                    None if positive_count == 0 else float(expected_total / positive_count)
                ),
                "minimum_recall_at_k": (
                    None if positive_count == 0 else float(minimum_total / positive_count)
                ),
                "maximum_recall_at_k": (
                    None if positive_count == 0 else float(maximum_total / positive_count)
                ),
            }
        )
    return results


def _score_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    source_rows: np.ndarray,
    *,
    dtype_name: str,
    representation: str,
    near_tolerance: float,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    unique_scores, counts = np.unique(scores, return_counts=True)
    largest_count = int(counts.max())
    # If several groups are equally large, the highest score is the most relevant
    # to analyst-priority saturation and is selected deterministically.
    largest_score = float(unique_scores[counts == largest_count][-1])
    exact_zero = scores == 0.0
    exact_one = scores == 1.0
    near_zero = np.abs(scores) <= near_tolerance
    near_one = np.abs(scores - 1.0) <= near_tolerance
    largest_tie = scores == largest_score
    positive = labels == 1
    margin_center = 0.5 if representation == "probability" else 0.0
    summary: dict[str, Any] = {
        "available": True,
        "representation": representation,
        "input_dtype": dtype_name,
        "row_count": int(scores.size),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "exact_zero_count": int(np.count_nonzero(exact_zero)),
        "exact_zero_positive_count": int(np.count_nonzero(exact_zero & positive)),
        "exact_zero_negative_count": int(np.count_nonzero(exact_zero & ~positive)),
        "exact_zero_fraction": float(np.mean(exact_zero)),
        "exact_one_count": int(np.count_nonzero(exact_one)),
        "exact_one_positive_count": int(np.count_nonzero(exact_one & positive)),
        "exact_one_negative_count": int(np.count_nonzero(exact_one & ~positive)),
        "exact_one_fraction": float(np.mean(exact_one)),
        "near_tolerance": near_tolerance,
        "near_zero_definition": "absolute_score_less_than_or_equal_to_tolerance",
        "near_zero_count": int(np.count_nonzero(near_zero)),
        "near_zero_positive_count": int(np.count_nonzero(near_zero & positive)),
        "near_zero_negative_count": int(np.count_nonzero(near_zero & ~positive)),
        "near_zero_fraction": float(np.mean(near_zero)),
        "near_one_definition": "absolute_score_minus_one_less_than_or_equal_to_tolerance",
        "near_one_count": int(np.count_nonzero(near_one)),
        "near_one_positive_count": int(np.count_nonzero(near_one & positive)),
        "near_one_negative_count": int(np.count_nonzero(near_one & ~positive)),
        "near_one_fraction": float(np.mean(near_one)),
        "unique_score_count": int(unique_scores.size),
        "unique_score_fraction": float(unique_scores.size / scores.size),
        "largest_tie_group_size": largest_count,
        "largest_tie_group_fraction": float(largest_count / scores.size),
        "largest_tie_group_score": largest_score,
        "largest_tie_group_positive_count": int(np.count_nonzero(largest_tie & positive)),
        "largest_tie_group_negative_count": int(np.count_nonzero(largest_tie & ~positive)),
        "decision_margin_definition": (
            "absolute_probability_minus_0.5"
            if representation == "probability"
            else "absolute_raw_score_minus_0.0"
        ),
        "absolute_decision_margin_quantiles": _margin_quantiles(
            scores,
            center=margin_center,
        ),
        "top_k": _tie_aware_top_k(
            labels,
            scores,
            source_rows,
            top_k_values=top_k_values,
        ),
    }
    if representation == "probability":
        summary.update(
            {
                "near_zero_definition": "score_less_than_or_equal_to_tolerance",
                "near_zero_count": int(np.count_nonzero(scores <= near_tolerance)),
                "near_zero_positive_count": int(
                    np.count_nonzero((scores <= near_tolerance) & positive)
                ),
                "near_zero_negative_count": int(
                    np.count_nonzero((scores <= near_tolerance) & ~positive)
                ),
                "near_zero_fraction": float(np.mean(scores <= near_tolerance)),
                "near_one_definition": "score_greater_than_or_equal_to_one_minus_tolerance",
                "near_one_count": int(np.count_nonzero(scores >= 1.0 - near_tolerance)),
                "near_one_positive_count": int(
                    np.count_nonzero((scores >= 1.0 - near_tolerance) & positive)
                ),
                "near_one_negative_count": int(
                    np.count_nonzero((scores >= 1.0 - near_tolerance) & ~positive)
                ),
                "near_one_fraction": float(np.mean(scores >= 1.0 - near_tolerance)),
            }
        )
    return summary


def _cross_representation_diagnostics(
    raw_scores: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    raw_unique_count = int(np.unique(raw_scores).size)
    probability_unique_count = int(np.unique(probabilities).size)

    order = np.lexsort((raw_scores, probabilities))
    sorted_probability = probabilities[order]
    sorted_raw = raw_scores[order]
    group_starts = np.concatenate(
        ([0], np.flatnonzero(sorted_probability[:-1] != sorted_probability[1:]) + 1)
    )
    group_ends = np.concatenate((group_starts[1:], [probabilities.size]))
    group_sizes = group_ends - group_starts

    raw_value_start = np.ones(probabilities.size, dtype=np.int64)
    raw_value_start[1:] = (sorted_probability[1:] != sorted_probability[:-1]) | (
        sorted_raw[1:] != sorted_raw[:-1]
    )
    distinct_raw_per_probability = np.add.reduceat(raw_value_start, group_starts)
    merged = distinct_raw_per_probability > 1

    return {
        "raw_unique_score_count": raw_unique_count,
        "probability_unique_score_count": probability_unique_count,
        "probability_unique_count_loss_from_raw": max(
            0,
            raw_unique_count - probability_unique_count,
        ),
        "probability_has_fewer_unique_scores_than_raw": (
            probability_unique_count < raw_unique_count
        ),
        "probability_tie_groups_containing_multiple_raw_scores": int(np.count_nonzero(merged)),
        "rows_in_probability_ties_containing_multiple_raw_scores": int(group_sizes[merged].sum()),
        "largest_probability_tie_containing_multiple_raw_scores": (
            int(group_sizes[merged].max()) if np.any(merged) else 0
        ),
        "interpretation": (
            "A non-zero merged group count is direct evidence that the probability "
            "representation collapses distinguishable raw scores; it does not by itself "
            "identify whether clipping, floating-point rounding, or another transform caused it."
        ),
    }


def analyze_validation_score_saturation(
    y_true: Sequence[int] | np.ndarray,
    *,
    probabilities: Sequence[float] | np.ndarray,
    source_row_number: Sequence[int] | np.ndarray,
    top_k_values: Iterable[int],
    partition: str,
    raw_scores: Sequence[float] | np.ndarray | None = None,
    near_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Diagnose validation score saturation without interpreting tied Top-K as superiority.

    ``raw_scores`` should be the model decision margin or raw logit before conversion
    to probability.  When unavailable, that limitation is explicitly represented.
    """

    _require_validation_partition(partition)
    labels = _validated_labels(y_true)
    probability_values, probability_dtype = _validated_scores(
        probabilities,
        name="probabilities",
        expected_size=labels.size,
        probability=True,
    )
    source_rows = _validated_source_rows(source_row_number, expected_size=labels.size)
    k_values = _validated_top_k_values(top_k_values)
    tolerance = _validated_near_tolerance(near_tolerance)

    probability_summary = _score_summary(
        labels,
        probability_values,
        source_rows,
        dtype_name=probability_dtype,
        representation="probability",
        near_tolerance=tolerance,
        top_k_values=k_values,
    )

    warnings: list[str] = []
    if int(labels.sum()) == 0:
        warnings.append("no_positive_labels: recall and true-positive opportunity are undefined")
    if int(labels.sum()) == labels.size:
        warnings.append("no_negative_labels: false-positive rate is undefined")
    if probability_summary["exact_one_count"]:
        warnings.append("exact_probability_one_saturation_observed")
    if probability_summary["exact_zero_count"]:
        warnings.append("exact_probability_zero_saturation_observed")
    if any(item["tie_break_material_at_cutoff"] for item in probability_summary["top_k"]):
        warnings.append(
            "probability_top_k_intersects_tied_score_group: deterministic Top-K depends on "
            "source_row_number; use expected/minimum/maximum bounds"
        )

    raw_summary: dict[str, Any]
    cross_representation: dict[str, Any] | None
    if raw_scores is None:
        raw_summary = {
            "available": False,
            "reason": "raw_scores_not_supplied",
        }
        cross_representation = None
        warnings.append(
            "raw_scores_unavailable: probability-output saturation cannot be separated from "
            "raw model-margin saturation"
        )
    else:
        raw_values, raw_dtype = _validated_scores(
            raw_scores,
            name="raw_scores",
            expected_size=labels.size,
            probability=False,
        )
        raw_summary = _score_summary(
            labels,
            raw_values,
            source_rows,
            dtype_name=raw_dtype,
            representation="raw_score",
            near_tolerance=tolerance,
            top_k_values=k_values,
        )
        cross_representation = _cross_representation_diagnostics(
            raw_values,
            probability_values,
        )
        if cross_representation["probability_has_fewer_unique_scores_than_raw"]:
            warnings.append("probability_representation_collapses_distinguishable_raw_scores")

    return {
        "schema_version": 1,
        "partition": "validation",
        "validation_only": True,
        "row_count": int(labels.size),
        "positive_count": int(labels.sum()),
        "negative_count": int(labels.size - labels.sum()),
        "probability": probability_summary,
        "raw_score": raw_summary,
        "cross_representation": cross_representation,
        "top_k_interpretation": (
            "Deterministic values use score descending then source_row_number ascending. "
            "Expected/minimum/maximum values quantify selection uncertainty when K cuts a tie."
        ),
        "warnings": warnings,
    }


def _frontier_arrays(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    group_ends = np.concatenate(
        (np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), [scores.size - 1])
    )
    thresholds = ranked_scores[group_ends]
    alerts = group_ends.astype(np.int64) + 1
    true_positives = np.cumsum(ranked_labels, dtype=np.int64)[group_ends]
    false_positives = alerts - true_positives
    group_sizes = np.diff(np.concatenate(([-1], group_ends))).astype(np.int64)
    group_true_positives = np.diff(np.concatenate(([0], true_positives))).astype(np.int64)

    # Index zero is an explicit no-alert sentinel.  Actual threshold candidates
    # follow in descending observed-score order and admit whole equal-score groups.
    return {
        "thresholds": np.concatenate(([np.nan], thresholds)),
        "alerts": np.concatenate(([0], alerts)),
        "true_positives": np.concatenate(([0], true_positives)),
        "false_positives": np.concatenate(([0], false_positives)),
        "group_sizes": np.concatenate(([0], group_sizes)),
        "group_true_positives": np.concatenate(([0], group_true_positives)),
    }


def _frontier_metric_vectors(
    labels: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    alerts = arrays["alerts"]
    true_positives = arrays["true_positives"]
    false_positives = arrays["false_positives"]
    precision = np.divide(
        true_positives,
        alerts,
        out=np.zeros(alerts.size, dtype=np.float64),
        where=alerts > 0,
    )
    if positive_count:
        recall = true_positives.astype(np.float64) / positive_count
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(alerts.size, dtype=np.float64),
            where=(precision + recall) > 0.0,
        )
    else:
        recall = np.full(alerts.size, np.nan, dtype=np.float64)
        f1 = np.full(alerts.size, np.nan, dtype=np.float64)
    false_positive_rate = (
        false_positives.astype(np.float64) / negative_count
        if negative_count
        else np.full(alerts.size, np.nan, dtype=np.float64)
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def _frontier_point(
    labels: np.ndarray,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    *,
    index: int,
) -> dict[str, Any]:
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    alerts = int(arrays["alerts"][index])
    true_positive = int(arrays["true_positives"][index])
    false_positive = int(arrays["false_positives"][index])
    sentinel = index == 0
    return {
        "threshold": None if sentinel else float(arrays["thresholds"][index]),
        "threshold_kind": (
            "no_alert_sentinel_above_maximum_score" if sentinel else "observed_score_group"
        ),
        "threshold_comparison": (
            "score_greater_than_maximum_observed_score"
            if sentinel
            else "score_greater_than_or_equal"
        ),
        "admitted_score_group_size": int(arrays["group_sizes"][index]),
        "admitted_score_group_true_positives": int(arrays["group_true_positives"][index]),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": positive_count - true_positive,
        "true_negative": negative_count - false_positive,
        "precision": float(metrics["precision"][index]),
        "recall": (None if not positive_count else float(metrics["recall"][index])),
        "f1": None if not positive_count else float(metrics["f1"][index]),
        "false_positive_rate": (
            None if not negative_count else float(metrics["false_positive_rate"][index])
        ),
        "alert_count": alerts,
        "alert_rate": float(alerts / labels.size),
    }


def _frontier_points(
    labels: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    metrics = _frontier_metric_vectors(labels, arrays)
    return [
        _frontier_point(labels, arrays, metrics, index=index)
        for index in range(arrays["alerts"].size)
    ]


def build_validation_threshold_frontier(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    partition: str,
) -> dict[str, Any]:
    """Return the complete validation frontier over observed equal-score groups."""

    _require_validation_partition(partition)
    labels = _validated_labels(y_true)
    scores, dtype_name = _validated_scores(
        y_score,
        name="y_score",
        expected_size=labels.size,
        probability=False,
    )
    arrays = _frontier_arrays(labels, scores)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    warnings: list[str] = []
    if positive_count == 0:
        warnings.append("no_positive_labels: recall and F1 are null; optimization is unavailable")
    if negative_count == 0:
        warnings.append("no_negative_labels: false_positive_rate is null")
    return {
        "schema_version": 1,
        "partition": "validation",
        "validation_only": True,
        "score_dtype": dtype_name,
        "row_count": int(labels.size),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "observed_unique_score_count": int(arrays["alerts"].size - 1),
        "frontier_point_count_including_no_alert_sentinel": int(arrays["alerts"].size),
        "candidate_definition": (
            "one inclusive threshold per observed equal-score group plus an explicit "
            "no-alert sentinel"
        ),
        "points": _frontier_points(labels, arrays),
        "warnings": warnings,
    }


def _unavailable_selection(*, objective: str, reason: str, constraints: dict[str, Any]) -> dict:
    return {
        "status": "unavailable",
        "objective": objective,
        "constraints": constraints,
        "selection_tie_rule": _SELECTION_TIE_RULE,
        "reason": reason,
        "operating_point": None,
    }


def _select_operating_point(
    labels: np.ndarray,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    *,
    objective: str,
    constraints: dict[str, Any],
    eligible: np.ndarray,
) -> dict[str, Any]:
    objective_key = "f1" if objective == "maximize_f1" else "recall"
    objective_values = metrics[objective_key]
    eligible_indices = np.flatnonzero(eligible & np.isfinite(objective_values))
    if eligible_indices.size == 0:
        return {
            "status": "no_feasible_threshold",
            "objective": objective,
            "constraints": constraints,
            "selection_tie_rule": _SELECTION_TIE_RULE,
            "reason": "No exact score-group threshold satisfies all constraints",
            "operating_point": None,
        }

    best_objective = float(np.max(objective_values[eligible_indices]))
    tied = eligible_indices[objective_values[eligible_indices] == best_objective]
    minimum_alerts = np.min(arrays["alerts"][tied])
    tied = tied[arrays["alerts"][tied] == minimum_alerts]
    minimum_false_positives = np.min(arrays["false_positives"][tied])
    tied = tied[arrays["false_positives"][tied] == minimum_false_positives]
    # Frontier thresholds are descending.  Index zero is the above-maximum
    # sentinel; therefore the first remaining point also has the highest threshold.
    selected_index = int(tied[0])
    selected = _frontier_point(
        labels,
        arrays,
        metrics,
        index=selected_index,
    )
    return {
        "status": "selected",
        "objective": objective,
        "objective_value": best_objective,
        "constraints": constraints,
        "selection_tie_rule": _SELECTION_TIE_RULE,
        "operating_point": selected.copy(),
    }


def optimize_validation_thresholds(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    partition: str,
    fpr_ceiling: float,
    alert_budget: int,
    primary_fpr_ceiling: float,
    primary_alert_budget: int,
    include_frontier: bool = False,
) -> dict[str, Any]:
    """Select validation operating points from the exact score-group frontier.

    The four summaries are descriptive validation-only choices.  The predeclared
    primary operating point jointly enforces its FPR ceiling and alert budget,
    then maximizes recall.  ``include_frontier=False`` avoids embedding a
    potentially large frontier in summary JSON; callers can persist the complete
    frontier separately with :func:`build_validation_threshold_frontier`.
    """

    _require_validation_partition(partition)
    validated_fpr = _validated_probability(fpr_ceiling, name="fpr_ceiling")
    validated_alert_budget = _validated_alert_budget(alert_budget, name="alert_budget")
    validated_primary_fpr = _validated_probability(
        primary_fpr_ceiling,
        name="primary_fpr_ceiling",
    )
    validated_primary_alert_budget = _validated_alert_budget(
        primary_alert_budget,
        name="primary_alert_budget",
    )
    if not isinstance(include_frontier, bool):
        raise RefinementMetricError("include_frontier must be boolean")

    labels = _validated_labels(y_true)
    scores, dtype_name = _validated_scores(
        y_score,
        name="y_score",
        expected_size=labels.size,
        probability=False,
    )
    arrays = _frontier_arrays(labels, scores)
    metrics = _frontier_metric_vectors(labels, arrays)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    warnings: list[str] = []
    if positive_count == 0:
        warnings.append("no_positive_labels: recall and F1 are null; optimization is unavailable")
    if negative_count == 0:
        warnings.append("no_negative_labels: false_positive_rate is null")

    if positive_count == 0:
        no_positive = "No positive validation labels; recall and F1 are undefined"
        max_f1 = _unavailable_selection(
            objective="maximize_f1",
            reason=no_positive,
            constraints={},
        )
        recall_under_fpr = _unavailable_selection(
            objective="maximize_recall",
            reason=no_positive,
            constraints={"false_positive_rate_at_most": validated_fpr},
        )
        recall_under_budget = _unavailable_selection(
            objective="maximize_recall",
            reason=no_positive,
            constraints={"alert_count_at_most": validated_alert_budget},
        )
        primary = _unavailable_selection(
            objective="maximize_recall",
            reason=no_positive,
            constraints={
                "false_positive_rate_at_most": validated_primary_fpr,
                "alert_count_at_most": validated_primary_alert_budget,
            },
        )
    else:
        max_f1 = _select_operating_point(
            labels,
            arrays,
            metrics,
            objective="maximize_f1",
            constraints={},
            eligible=np.ones(arrays["alerts"].size, dtype=bool),
        )
        if negative_count == 0:
            recall_under_fpr = _unavailable_selection(
                objective="maximize_recall",
                reason="No negative validation labels; false-positive rate is undefined",
                constraints={"false_positive_rate_at_most": validated_fpr},
            )
            primary = _unavailable_selection(
                objective="maximize_recall",
                reason="No negative validation labels; false-positive rate is undefined",
                constraints={
                    "false_positive_rate_at_most": validated_primary_fpr,
                    "alert_count_at_most": validated_primary_alert_budget,
                },
            )
        else:
            recall_under_fpr = _select_operating_point(
                labels,
                arrays,
                metrics,
                objective="maximize_recall",
                constraints={"false_positive_rate_at_most": validated_fpr},
                eligible=metrics["false_positive_rate"] <= validated_fpr,
            )
            primary = _select_operating_point(
                labels,
                arrays,
                metrics,
                objective="maximize_recall",
                constraints={
                    "false_positive_rate_at_most": validated_primary_fpr,
                    "alert_count_at_most": validated_primary_alert_budget,
                },
                eligible=(metrics["false_positive_rate"] <= validated_primary_fpr)
                & (arrays["alerts"] <= validated_primary_alert_budget),
            )
        recall_under_budget = _select_operating_point(
            labels,
            arrays,
            metrics,
            objective="maximize_recall",
            constraints={"alert_count_at_most": validated_alert_budget},
            eligible=arrays["alerts"] <= validated_alert_budget,
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "partition": "validation",
        "validation_only": True,
        "test_partition_used": False,
        "candidate_definition": (
            "one inclusive threshold per observed equal-score group plus an explicit "
            "no-alert sentinel"
        ),
        "frontier_point_count_including_no_alert_sentinel": int(arrays["alerts"].size),
        "selection_tie_rule": _SELECTION_TIE_RULE,
        "summaries": {
            "maximum_f1": max_f1,
            "maximum_recall_under_fpr_ceiling": recall_under_fpr,
            "maximum_recall_under_alert_budget": recall_under_budget,
            "predeclared_joint_primary_constraint": primary,
        },
        "warnings": warnings,
    }
    if include_frontier:
        result["frontier"] = {
            "schema_version": 1,
            "partition": "validation",
            "validation_only": True,
            "score_dtype": dtype_name,
            "row_count": int(labels.size),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "observed_unique_score_count": int(arrays["alerts"].size - 1),
            "frontier_point_count_including_no_alert_sentinel": int(arrays["alerts"].size),
            "candidate_definition": result["candidate_definition"],
            "points": _frontier_points(labels, arrays),
            "warnings": warnings,
        }
    return result


__all__ = [
    "RefinementMetricError",
    "analyze_validation_score_saturation",
    "build_validation_threshold_frontier",
    "optimize_validation_thresholds",
]
