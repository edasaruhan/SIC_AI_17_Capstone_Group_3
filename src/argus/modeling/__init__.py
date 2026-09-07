"""Leakage-safe transaction-baseline modeling utilities."""

from argus.modeling.metrics import (
    MetricInputError,
    compute_average_precision,
    compute_roc_auc,
    compute_threshold_metrics,
    compute_top_k_metrics,
    evaluate_binary_predictions,
)
from argus.modeling.selection import (
    ChampionSelectionError,
    select_transaction_baseline_champion,
)

__all__ = [
    "ChampionSelectionError",
    "MetricInputError",
    "compute_average_precision",
    "compute_roc_auc",
    "compute_threshold_metrics",
    "compute_top_k_metrics",
    "evaluate_binary_predictions",
    "select_transaction_baseline_champion",
]
