"""Deterministic binary-ranking metrics for severely imbalanced baselines.

``PR-AUC`` is deliberately reported as average precision rather than as a
trapezoidal integral over the precision-recall curve.  Ranking ties are grouped
for average precision and ROC-AUC.  Operational top-K metrics use an explicit,
stable order: score descending, then ``source_row_number`` ascending.

Undefined denominators are represented by ``None`` so they remain distinguishable
from an observed zero when serialized to JSON.  The one exception is precision
with zero predicted alerts, which follows the common zero-division convention and
is reported as ``0.0``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


class MetricInputError(ValueError):
    """Raised when prediction inputs cannot produce defensible metrics."""


def _validated_labels_and_scores(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score)
    if labels.ndim != 1 or scores.ndim != 1:
        raise MetricInputError("y_true and y_score must both be one-dimensional")
    if labels.size == 0:
        raise MetricInputError("Prediction metrics require at least one row")
    if labels.size != scores.size:
        raise MetricInputError("y_true and y_score must contain the same number of rows")
    if labels.dtype.kind not in {"b", "i", "u", "f"}:
        raise MetricInputError("y_true must contain numeric binary labels")

    numeric_labels = labels.astype(np.float64, copy=False)
    if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, (0.0, 1.0)).all():
        raise MetricInputError("y_true must contain only finite binary labels 0 and 1")
    try:
        numeric_scores = scores.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise MetricInputError("y_score must contain numeric values") from exc
    if not np.isfinite(numeric_scores).all():
        raise MetricInputError("y_score must contain only finite values")

    return numeric_labels.astype(np.int8, copy=False), numeric_scores


def _validated_source_rows(
    source_row_number: Sequence[int] | np.ndarray,
    *,
    expected_size: int,
) -> np.ndarray:
    rows = np.asarray(source_row_number)
    if rows.ndim != 1 or rows.size != expected_size:
        raise MetricInputError(
            "source_row_number must be one-dimensional and align one-to-one with predictions"
        )
    if rows.dtype.kind not in {"i", "u", "f"}:
        raise MetricInputError("source_row_number must contain numeric integer values")
    numeric_rows = rows.astype(np.float64, copy=False)
    if (
        not np.isfinite(numeric_rows).all()
        or not np.equal(numeric_rows, np.floor(numeric_rows)).all()
    ):
        raise MetricInputError("source_row_number must contain finite integer values")
    integer_rows = numeric_rows.astype(np.int64)
    if np.unique(integer_rows).size != integer_rows.size:
        raise MetricInputError(
            "source_row_number values must be unique to guarantee deterministic top-K ranking"
        )
    return integer_rows


def _validated_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise MetricInputError("threshold must be a finite numeric value")
    value = float(threshold)
    if not np.isfinite(value):
        raise MetricInputError("threshold must be a finite numeric value")
    return value


def _validated_top_k_values(top_k_values: Iterable[int]) -> tuple[int, ...]:
    values = tuple(top_k_values)
    if not values:
        raise MetricInputError("top_k_values must contain at least one positive integer")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise MetricInputError("Every configured K must be a positive integer")
        normalized.append(int(value))
    if len(set(normalized)) != len(normalized):
        raise MetricInputError("top_k_values must not contain duplicates")
    return tuple(normalized)


def _average_precision_from_validated(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None

    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    cumulative_true_positives = np.cumsum(ranked_labels, dtype=np.int64)

    # A precision-recall operating point exists only after an entire equal-score
    # group is admitted.  Grouping ties keeps AP independent of arbitrary row order.
    group_ends = np.concatenate(
        (np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), [ranked_scores.size - 1])
    )
    true_positives_at_threshold = cumulative_true_positives[group_ends]
    predicted_positives_at_threshold = group_ends + 1
    precision = true_positives_at_threshold / predicted_positives_at_threshold
    positive_increments = np.diff(np.concatenate(([0], true_positives_at_threshold)))
    return float(np.sum((positive_increments / positive_count) * precision))


def compute_average_precision(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> float | None:
    """Compute non-interpolated average precision; return ``None`` with no positives."""

    labels, scores = _validated_labels_and_scores(y_true, y_score)
    return _average_precision_from_validated(labels, scores)


def _roc_auc_from_validated(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(scores, kind="stable")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    group_starts = np.concatenate(
        ([0], np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]) + 1)
    )
    group_ends = np.concatenate((group_starts[1:], [ranked_scores.size]))
    group_sizes = group_ends - group_starts
    positives_in_group = np.add.reduceat(ranked_labels.astype(np.int64, copy=False), group_starts)
    negatives_in_group = group_sizes - positives_in_group
    negatives_below = np.cumsum(negatives_in_group, dtype=np.int64) - negatives_in_group

    # Twice the Mann-Whitney concordant-pair count stays integral: a positive
    # scores 2 for every negative below it and 1 for every tied negative.  This
    # vectorization is mathematically identical to the group loop but avoids a
    # Python iteration per distinct score on full validation vectors.
    twice_concordant_pairs = np.sum(
        positives_in_group * (2 * negatives_below + negatives_in_group),
        dtype=np.int64,
    )
    return float(twice_concordant_pairs / (2 * positive_count * negative_count))


def compute_roc_auc(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> float | None:
    """Compute tie-aware ROC-AUC; return ``None`` when either class is absent."""

    labels, scores = _validated_labels_and_scores(y_true, y_score)
    return _roc_auc_from_validated(labels, scores)


def _threshold_metrics_from_validated(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    predicted_positive = scores >= threshold
    positive = labels == 1
    true_positive = int(np.count_nonzero(predicted_positive & positive))
    false_positive = int(np.count_nonzero(predicted_positive & ~positive))
    false_negative = int(np.count_nonzero(~predicted_positive & positive))
    true_negative = int(np.count_nonzero(~predicted_positive & ~positive))
    alert_count = true_positive + false_positive
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive

    precision = true_positive / alert_count if alert_count else 0.0
    recall = true_positive / positive_count if positive_count else None
    false_positive_rate = false_positive / negative_count if negative_count else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0.0
        else (0.0 if recall is not None else None)
    )
    return {
        "threshold": threshold,
        "threshold_comparison": "score_greater_than_or_equal",
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": float(precision),
        "recall": None if recall is None else float(recall),
        "f1": None if f1 is None else float(f1),
        "false_positive_rate": (
            None if false_positive_rate is None else float(false_positive_rate)
        ),
        "alert_count": alert_count,
        "alert_rate": float(alert_count / labels.size),
    }


def compute_threshold_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compute confusion-derived metrics at a fixed, inclusive score threshold."""

    labels, scores = _validated_labels_and_scores(y_true, y_score)
    return _threshold_metrics_from_validated(
        labels,
        scores,
        threshold=_validated_threshold(threshold),
    )


def _top_k_metrics_from_validated(
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

    metrics: list[dict[str, Any]] = []
    for requested_k in top_k_values:
        effective_k = min(requested_k, labels.size)
        true_positives = int(cumulative_true_positives[effective_k - 1])
        cutoff_score = float(ranked_scores[effective_k - 1])
        cutoff_group = ranked_scores == cutoff_score
        cutoff_tie_group_size = int(np.count_nonzero(cutoff_group))
        selected_from_cutoff_group = int(np.count_nonzero(cutoff_group[:effective_k]))
        metrics.append(
            {
                "requested_k": requested_k,
                "effective_k": effective_k,
                "was_clipped_to_partition_size": effective_k != requested_k,
                "true_positives": true_positives,
                "false_positives": effective_k - true_positives,
                "precision_at_k": float(true_positives / effective_k),
                "recall_at_k": (
                    None if positive_count == 0 else float(true_positives / positive_count)
                ),
                "cutoff_score": cutoff_score,
                "cutoff_tie_group_size": cutoff_tie_group_size,
                "selected_from_cutoff_tie_group": selected_from_cutoff_group,
                "cutoff_tie_group_fully_included": (
                    selected_from_cutoff_group == cutoff_tie_group_size
                ),
                "tie_break_material_at_cutoff": (
                    cutoff_tie_group_size > selected_from_cutoff_group
                ),
            }
        )
    return metrics


def compute_top_k_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    source_row_number: Sequence[int] | np.ndarray,
    *,
    top_k_values: Iterable[int],
) -> list[dict[str, Any]]:
    """Compute deterministic Precision@K and Recall@K for configured K values."""

    labels, scores = _validated_labels_and_scores(y_true, y_score)
    source_rows = _validated_source_rows(source_row_number, expected_size=labels.size)
    values = _validated_top_k_values(top_k_values)
    return _top_k_metrics_from_validated(
        labels,
        scores,
        source_rows,
        top_k_values=values,
    )


def evaluate_binary_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    source_row_number: Sequence[int] | np.ndarray,
    *,
    threshold: float,
    top_k_values: Iterable[int],
) -> dict[str, Any]:
    """Return one JSON-safe metric bundle from one frozen prediction vector."""

    labels, scores = _validated_labels_and_scores(y_true, y_score)
    source_rows = _validated_source_rows(source_row_number, expected_size=labels.size)
    threshold_value = _validated_threshold(threshold)
    k_values = _validated_top_k_values(top_k_values)
    average_precision = _average_precision_from_validated(labels, scores)
    positive_count = int(labels.sum())

    warnings: list[str] = []
    if positive_count == 0:
        warnings.append("no_positive_labels: average_precision and recall metrics are null")
    if positive_count == labels.size:
        warnings.append("no_negative_labels: roc_auc and false_positive_rate are null")

    return {
        "row_count": int(labels.size),
        "positive_count": positive_count,
        "negative_count": int(labels.size - positive_count),
        "positive_rate": float(positive_count / labels.size),
        "pr_auc": average_precision,
        "average_precision": average_precision,
        "pr_auc_method": "non_interpolated_average_precision",
        "roc_auc": _roc_auc_from_validated(labels, scores),
        "roc_auc_role": "secondary",
        "threshold_metrics": _threshold_metrics_from_validated(
            labels,
            scores,
            threshold=threshold_value,
        ),
        "top_k_metrics": _top_k_metrics_from_validated(
            labels,
            scores,
            source_rows,
            top_k_values=k_values,
        ),
        "top_k_order": "score_descending_then_source_row_number_ascending",
        "undefined_metric_representation": None,
        "warnings": warnings,
    }
