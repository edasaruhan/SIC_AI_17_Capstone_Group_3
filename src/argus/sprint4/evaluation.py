"""Fair validation-only comparison helpers for Sprint 4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from argus.modeling.metrics import evaluate_binary_predictions
from argus.modeling.refinement_metrics import (
    analyze_validation_score_saturation,
    optimize_validation_thresholds,
)


class Sprint4EvaluationError(ValueError):
    """Raised when model-comparison vectors are not strictly aligned."""


def _vectors(
    labels: Sequence[int] | np.ndarray,
    source_rows: Sequence[int] | np.ndarray,
    scores: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    y = np.asarray(labels, dtype=np.int8)
    rows = np.asarray(source_rows, dtype=np.int64)
    if y.ndim != 1 or rows.ndim != 1 or y.size == 0 or y.size != rows.size:
        raise Sprint4EvaluationError("Labels and source rows must be aligned one-dimensional data")
    if not np.isin(y, (0, 1)).all() or np.unique(rows).size != rows.size:
        raise Sprint4EvaluationError("Validation labels must be binary and source rows unique")
    if not scores:
        raise Sprint4EvaluationError("At least one model score vector is required")
    score_arrays: dict[str, np.ndarray] = {}
    for name, values in scores.items():
        if not isinstance(name, str) or not name:
            raise Sprint4EvaluationError("Model names must be non-empty strings")
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or vector.size != y.size or not np.isfinite(vector).all():
            raise Sprint4EvaluationError(f"Score vector {name!r} is invalid or misaligned")
        score_arrays[name] = vector
    return y, rows, score_arrays


def compare_validation_models(
    labels: Sequence[int] | np.ndarray,
    source_rows: Sequence[int] | np.ndarray,
    scores: Mapping[str, Sequence[float] | np.ndarray],
    *,
    top_k: Sequence[int],
    alert_budget: int,
    fpr_ceiling: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all models with the identical full validation vectors and constraints."""

    y, rows, score_arrays = _vectors(labels, source_rows, scores)
    comparison: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {}
    top_k_rows: list[dict[str, Any]] = []
    for model_name in sorted(score_arrays):
        model_scores = score_arrays[model_name]
        threshold = optimize_validation_thresholds(
            y,
            model_scores,
            partition="validation",
            fpr_ceiling=fpr_ceiling,
            alert_budget=alert_budget,
            primary_fpr_ceiling=fpr_ceiling,
            primary_alert_budget=alert_budget,
        )
        primary = threshold["summaries"]["predeclared_joint_primary_constraint"]
        if primary["status"] != "selected":
            raise Sprint4EvaluationError(f"No valid primary operating point for {model_name}")
        selected_threshold = float(primary["operating_point"]["threshold"])
        metrics = evaluate_binary_predictions(
            y,
            model_scores,
            rows,
            threshold=selected_threshold,
            top_k_values=top_k,
        )
        threshold["model"] = model_name
        thresholds[model_name] = threshold
        comparison.append(
            {
                "model": model_name,
                "partition": "validation",
                "evaluation_partition": "validation",
                "evaluation_scope": "full_frozen_outer_validation",
                "row_count": metrics["row_count"],
                "validation_row_count": metrics["row_count"],
                "positive_count": metrics["positive_count"],
                "average_precision": metrics["average_precision"],
                "pr_auc": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "threshold": selected_threshold,
                **{
                    key: metrics["threshold_metrics"][source]
                    for key, source in {
                        "precision": "precision",
                        "recall": "recall",
                        "f1": "f1",
                        "false_positive_rate": "false_positive_rate",
                        "alert_count": "alert_count",
                    }.items()
                },
                "test_metrics_used": False,
            }
        )
        for item in metrics["top_k_metrics"]:
            top_k_rows.append({"model": model_name, **item})
    return comparison, thresholds, top_k_rows


def build_pr_curve_table(
    labels: Sequence[int] | np.ndarray,
    scores: Mapping[str, Sequence[float] | np.ndarray],
    *,
    maximum_points_per_model: int = 2000,
) -> pd.DataFrame:
    """Return a deterministically thinned, display-sized PR-curve table."""

    if maximum_points_per_model < 2:
        raise Sprint4EvaluationError("maximum_points_per_model must be at least two")
    y = np.asarray(labels, dtype=np.int8)
    records: list[pd.DataFrame] = []
    for model_name in sorted(scores):
        precision, recall, thresholds = precision_recall_curve(y, np.asarray(scores[model_name]))
        raw_points = len(precision)
        if raw_points > maximum_points_per_model:
            indices = np.unique(
                np.linspace(0, raw_points - 1, maximum_points_per_model, dtype=np.int64)
            )
        else:
            indices = np.arange(raw_points, dtype=np.int64)
        threshold_values = np.full(raw_points, np.nan, dtype=np.float64)
        threshold_values[: len(thresholds)] = thresholds
        records.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "precision": precision[indices],
                    "recall": recall[indices],
                    "threshold": threshold_values[indices],
                    "source_curve_points": raw_points,
                    "display_curve_points": len(indices),
                    "downsampled_for_display_only": raw_points > len(indices),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def graphsage_score_diagnostics(
    labels: Sequence[int] | np.ndarray,
    raw_logits: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    source_rows: Sequence[int] | np.ndarray,
    *,
    top_k: Sequence[int],
) -> dict[str, Any]:
    """Record raw/probability ties without converting Top-K into an unsupported claim."""

    return analyze_validation_score_saturation(
        labels,
        raw_scores=raw_logits,
        probabilities=probabilities,
        source_row_number=source_rows,
        top_k_values=top_k,
        partition="validation",
    )


__all__ = [
    "Sprint4EvaluationError",
    "build_pr_curve_table",
    "compare_validation_models",
    "graphsage_score_diagnostics",
]
