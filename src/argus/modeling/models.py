"""Fixed Sprint 2 baseline model factory and bounded prediction helpers."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


class ModelConfigurationError(ValueError):
    """Raised when a baseline model configuration is unsafe or ambiguous."""


@dataclass(frozen=True)
class FittedBaseline:
    """A fitted estimator and its measured training runtime."""

    name: str
    estimator: Any
    fit_seconds: float
    implementation: str
    effective_parameters: dict[str, Any]


_LOGISTIC_KEYS = {
    "loss",
    "penalty",
    "alpha",
    "max_iter",
    "tol",
    "average",
    "class_weight",
    "early_stopping",
    "n_jobs",
}
_FOREST_KEYS = {
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "bootstrap",
    "max_samples",
    "n_jobs",
}
_LIGHTGBM_KEYS = {
    "boosting_type",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "max_bin",
    "deterministic",
    "force_col_wise",
    "n_jobs",
    "verbosity",
}


def _model_parameters(
    model_config: Mapping[str, Any], allowed: set[str], name: str
) -> tuple[str, dict[str, Any]]:
    implementation = model_config.get("implementation")
    if not isinstance(implementation, str) or not implementation:
        raise ModelConfigurationError(f"baseline.models.{name}.implementation is required")
    parameters = {key: value for key, value in model_config.items() if key != "implementation"}
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        raise ModelConfigurationError(
            f"Unknown baseline.models.{name} parameters: {', '.join(unexpected)}"
        )
    return implementation, parameters


def _training_class_counts(labels: np.ndarray) -> tuple[int, int]:
    values = np.asarray(labels)
    positives = int(np.count_nonzero(values == 1))
    negatives = int(np.count_nonzero(values == 0))
    if positives == 0 or negatives == 0 or positives + negatives != values.size:
        raise ModelConfigurationError(
            "Training labels must be binary and contain both classes before fitting baselines"
        )
    return negatives, positives


def build_baseline_models(
    models_config: Mapping[str, Mapping[str, Any]],
    *,
    random_seed: int,
    training_labels: np.ndarray,
) -> dict[str, tuple[Any, str, dict[str, Any]]]:
    """Build the three fixed baselines using training-label imbalance only."""

    expected = {"logistic_regression", "random_forest", "lightgbm"}
    if set(models_config) != expected:
        missing = sorted(expected - set(models_config))
        extra = sorted(set(models_config) - expected)
        raise ModelConfigurationError(
            f"Exactly three Sprint 2 models are required; missing={missing}, extra={extra}"
        )

    negative_count, positive_count = _training_class_counts(training_labels)
    scale_pos_weight = negative_count / positive_count

    logistic_impl, logistic_params = _model_parameters(
        models_config["logistic_regression"], _LOGISTIC_KEYS, "logistic_regression"
    )
    if logistic_params.get("loss") != "log_loss":
        raise ModelConfigurationError(
            "Logistic Regression must use SGDClassifier(loss='log_loss') in the full-data path"
        )
    logistic = SGDClassifier(random_state=random_seed, **logistic_params)

    forest_impl, forest_params = _model_parameters(
        models_config["random_forest"], _FOREST_KEYS, "random_forest"
    )
    forest = RandomForestClassifier(random_state=random_seed, **forest_params)

    lightgbm_impl, lightgbm_params = _model_parameters(
        models_config["lightgbm"], _LIGHTGBM_KEYS, "lightgbm"
    )
    lightgbm_params = {**lightgbm_params, "scale_pos_weight": scale_pos_weight}
    lightgbm = LGBMClassifier(random_state=random_seed, **lightgbm_params)

    return {
        "logistic_regression": (logistic, logistic_impl, dict(logistic.get_params())),
        "random_forest": (forest, forest_impl, dict(forest.get_params())),
        "lightgbm": (lightgbm, lightgbm_impl, dict(lightgbm.get_params())),
    }


def fit_baseline(
    name: str,
    estimator: Any,
    implementation: str,
    effective_parameters: Mapping[str, Any],
    matrix: np.ndarray,
    labels: np.ndarray,
) -> FittedBaseline:
    """Fit one estimator and return a runtime-bearing immutable wrapper."""

    started = time.perf_counter()
    estimator.fit(matrix, labels)
    elapsed = time.perf_counter() - started
    return FittedBaseline(
        name=name,
        estimator=estimator,
        fit_seconds=elapsed,
        implementation=implementation,
        effective_parameters=dict(effective_parameters),
    )


def predict_positive_probability(
    estimator: Any,
    matrix: np.ndarray,
    *,
    batch_rows: int,
) -> tuple[np.ndarray, float]:
    """Predict class-1 scores in bounded batches and return measured runtime."""

    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    row_count = int(matrix.shape[0])
    scores = np.empty(row_count, dtype=np.float64)
    started = time.perf_counter()
    for start in range(0, row_count, batch_rows):
        stop = min(start + batch_rows, row_count)
        probabilities = np.asarray(estimator.predict_proba(matrix[start:stop]))
        if probabilities.ndim != 2 or probabilities.shape != (stop - start, 2):
            raise RuntimeError(
                "Baseline predict_proba must return an (n_rows, 2) probability matrix"
            )
        scores[start:stop] = probabilities[:, 1]
    elapsed = time.perf_counter() - started
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise RuntimeError("Baseline validation scores must be finite values in [0, 1]")
    return scores, elapsed


def json_safe_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize estimator parameters for deterministic JSON metadata."""

    normalized: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, float) and math.isnan(value):
            normalized[key] = None
        elif value is None or isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        elif isinstance(value, np.generic):
            normalized[key] = value.item()
        elif isinstance(value, (list, tuple)):
            normalized[key] = list(value)
        elif isinstance(value, dict):
            normalized[key] = dict(value)
        else:
            normalized[key] = repr(value)
    return normalized
