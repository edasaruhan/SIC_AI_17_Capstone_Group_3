"""Pure metric construction for the frozen one-shot final evaluation.

This module deliberately exposes no threshold optimizer or model selector. Every
operating threshold is supplied by the pre-test freeze contract, and model roles
are carried through rather than inferred from test performance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from argus.modeling.metrics import MetricInputError, evaluate_binary_predictions
from argus.sprint4.evaluation import build_pr_curve_table


class FinalEvaluationMetricError(ValueError):
    """Raised when frozen final-evaluation vectors or metadata are invalid."""


def stable_sigmoid(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a numerically stable sigmoid without clipping or ranking changes."""

    try:
        raw = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FinalEvaluationMetricError("Raw model scores must be numeric") from exc
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
        raise FinalEvaluationMetricError("Raw model scores must be a finite rank-one vector")
    result = np.empty_like(raw)
    nonnegative = raw >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-raw[nonnegative]))
    exponential = np.exp(raw[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise FinalEvaluationMetricError("Sigmoid scores must be finite and lie in [0, 1]")
    return result


def evaluate_frozen_models(
    labels: Sequence[int] | np.ndarray,
    source_rows: Sequence[int] | np.ndarray,
    raw_scores: Mapping[str, Sequence[float] | np.ndarray],
    *,
    frozen_thresholds: Mapping[str, float],
    model_roles: Mapping[str, str],
    validation_average_precision: Mapping[str, float],
    top_k: Sequence[int],
    frozen_champion: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """Evaluate already-frozen models without selecting or optimizing on test."""

    names = set(raw_scores)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise FinalEvaluationMetricError("Raw scores require non-empty string model names")
    for label, values in (
        ("frozen_thresholds", frozen_thresholds),
        ("model_roles", model_roles),
        ("validation_average_precision", validation_average_precision),
    ):
        if set(values) != names:
            raise FinalEvaluationMetricError(
                f"{label} model names differ: expected={sorted(names)}, actual={sorted(values)}"
            )
    if frozen_champion not in names:
        raise FinalEvaluationMetricError("Frozen champion is absent from final score vectors")
    if any(role not in {"frozen_champion", "comparator"} for role in model_roles.values()):
        raise FinalEvaluationMetricError("Model roles must be frozen_champion or comparator")
    if model_roles[frozen_champion] != "frozen_champion":
        raise FinalEvaluationMetricError("Frozen champion must have the frozen_champion role")
    if sum(role == "frozen_champion" for role in model_roles.values()) != 1:
        raise FinalEvaluationMetricError("Exactly one model must be marked frozen_champion")

    # Preserve the caller's values until the shared metric validator has checked
    # them.  Casting first would silently turn values such as label 257 into 1
    # (int8 wraparound) or truncate non-integral source row identifiers.
    label_values = np.asarray(labels)
    row_values = np.asarray(source_rows)
    comparison: list[dict[str, Any]] = []
    top_k_rows: list[dict[str, Any]] = []
    score_arrays: dict[str, np.ndarray] = {}
    for model_name in sorted(names):
        try:
            scores = np.asarray(raw_scores[model_name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise FinalEvaluationMetricError(
                f"Raw score vector is not numeric for {model_name}"
            ) from exc
        score_arrays[model_name] = scores
        raw_threshold = frozen_thresholds[model_name]
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, Real):
            raise FinalEvaluationMetricError(f"Frozen threshold is not numeric for {model_name}")
        threshold = float(raw_threshold)
        if not math.isfinite(threshold):
            raise FinalEvaluationMetricError(f"Frozen threshold is non-finite for {model_name}")
        raw_validation_ap = validation_average_precision[model_name]
        if isinstance(raw_validation_ap, bool) or not isinstance(raw_validation_ap, Real):
            raise FinalEvaluationMetricError(
                f"Validation average precision is not numeric for {model_name}"
            )
        validation_ap = float(raw_validation_ap)
        if not math.isfinite(validation_ap) or not 0.0 <= validation_ap <= 1.0:
            raise FinalEvaluationMetricError(
                f"Validation average precision is outside [0, 1] for {model_name}"
            )
        try:
            metrics = evaluate_binary_predictions(
                label_values,
                scores,
                row_values,
                threshold=threshold,
                top_k_values=top_k,
            )
        except MetricInputError as exc:
            raise FinalEvaluationMetricError(
                f"Invalid frozen evaluation vectors for {model_name}: {exc}"
            ) from exc
        threshold_metrics = metrics["threshold_metrics"]
        row = {
            "model": model_name,
            "role": model_roles[model_name],
            "partition": "test",
            "evaluation_partition": "test",
            "evaluation_role": "one_shot_confirmatory",
            "row_count": int(metrics["row_count"]),
            "positive_count": int(metrics["positive_count"]),
            "negative_count": int(metrics["negative_count"]),
            "positive_rate": float(metrics["positive_count"] / metrics["row_count"]),
            "average_precision": metrics["average_precision"],
            "pr_auc": metrics["average_precision"],
            "roc_auc": metrics["roc_auc"],
            "threshold": threshold,
            "threshold_source_partition": "validation",
            "threshold_comparison": "score_greater_than_or_equal",
            "true_positive": int(threshold_metrics["true_positive"]),
            "false_positive": int(threshold_metrics["false_positive"]),
            "false_negative": int(threshold_metrics["false_negative"]),
            "true_negative": int(threshold_metrics["true_negative"]),
            "precision": threshold_metrics["precision"],
            "recall": threshold_metrics["recall"],
            "f1": threshold_metrics["f1"],
            "false_positive_rate": threshold_metrics["false_positive_rate"],
            "alert_count": int(threshold_metrics["alert_count"]),
            "alert_volume": int(threshold_metrics["alert_count"]),
            "alert_rate": threshold_metrics["alert_rate"],
            "validation_average_precision": validation_ap,
            "champion_frozen_before_test": model_name == frozen_champion,
            "model_selection_used_test": False,
            "threshold_tuned_on_test": False,
            "feature_selection_used_test": False,
            "retrained_after_test": False,
        }
        comparison.append(row)
        for top_k_row in metrics["top_k_metrics"]:
            top_k_rows.append(
                {
                    "model": model_name,
                    "role": model_roles[model_name],
                    "partition": "test",
                    "evaluation_role": "one_shot_confirmatory",
                    **top_k_row,
                    "model_selection_used_test": False,
                }
            )

    # Presentation order is fixed by the pre-test role, never by a test metric.
    role_order = {"frozen_champion": 0, "comparator": 1}
    comparison.sort(key=lambda row: (role_order.get(str(row["role"]), 99), str(row["model"])))
    curves = build_pr_curve_table(label_values, score_arrays)
    curves.insert(1, "partition", "test")
    curves.insert(2, "evaluation_role", "one_shot_confirmatory")
    return comparison, top_k_rows, curves


__all__ = [
    "FinalEvaluationMetricError",
    "evaluate_frozen_models",
    "stable_sigmoid",
]
