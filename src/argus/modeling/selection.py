"""Validation-only selection of the transaction baseline champion."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import Any


class ChampionSelectionError(ValueError):
    """Raised when champion selection could use invalid or non-validation evidence."""


def select_transaction_baseline_champion(
    validation_average_precision: Mapping[str, float],
    *,
    partition: str = "validation",
    test_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the maximum validation AP, breaking exact ties by model name.

    The deliberately narrow input is a model-name to *validation* average-
    precision mapping.  Supplying any test result, including an empty mapping, is
    rejected so a caller cannot accidentally make test-aware selection logic.
    """

    if partition != "validation":
        raise ChampionSelectionError(
            "Transaction Baseline Champion selection accepts validation results only"
        )
    if test_metrics is not None:
        raise ChampionSelectionError(
            "Final-test metrics are forbidden during baseline champion selection"
        )
    if not isinstance(validation_average_precision, Mapping) or not validation_average_precision:
        raise ChampionSelectionError("At least one validation average-precision result is required")

    candidates: list[tuple[str, float]] = []
    for model_name, metric_value in validation_average_precision.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ChampionSelectionError("Every candidate must have a non-empty model name")
        if model_name != model_name.strip():
            raise ChampionSelectionError("Model names must not contain surrounding whitespace")
        if isinstance(metric_value, bool) or not isinstance(metric_value, Real):
            raise ChampionSelectionError(
                f"Validation average precision for {model_name!r} must be numeric"
            )
        score = float(metric_value)
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ChampionSelectionError(
                f"Validation average precision for {model_name!r} must be finite and in [0, 1]"
            )
        candidates.append((model_name, score))

    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
    champion_name, champion_score = ranked[0]
    return {
        "title": "Transaction Baseline Champion",
        "champion_model": champion_name,
        "selection_partition": "validation",
        "selection_metric": "average_precision",
        "selection_metric_role": "primary_pr_auc",
        "selection_metric_value": champion_score,
        "selection_rule": "maximum_validation_average_precision",
        "tie_break_rule": "model_name_ascending",
        "test_metrics_used": False,
        "ranked_candidates": [
            {"rank": rank, "model_name": name, "average_precision": score}
            for rank, (name, score) in enumerate(ranked, start=1)
        ],
    }
