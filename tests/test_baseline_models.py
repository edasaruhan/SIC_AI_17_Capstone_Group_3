from __future__ import annotations

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from argus.modeling.models import (
    ModelConfigurationError,
    build_baseline_models,
    predict_positive_probability,
)


def _config() -> dict[str, dict[str, object]]:
    return {
        "logistic_regression": {
            "implementation": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss",
            "max_iter": 5,
            "tol": 0.01,
        },
        "random_forest": {
            "implementation": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 2,
            "n_jobs": 1,
        },
        "lightgbm": {
            "implementation": "lightgbm.LGBMClassifier",
            "n_estimators": 2,
            "verbosity": -1,
            "n_jobs": 1,
        },
    }


def test_factory_builds_exact_three_families_and_train_only_weight() -> None:
    labels = np.array([0, 0, 0, 1], dtype=np.uint8)
    built = build_baseline_models(_config(), random_seed=42, training_labels=labels)

    assert set(built) == {"logistic_regression", "random_forest", "lightgbm"}
    assert isinstance(built["logistic_regression"][0], SGDClassifier)
    assert isinstance(built["random_forest"][0], RandomForestClassifier)
    assert isinstance(built["lightgbm"][0], LGBMClassifier)
    assert built["lightgbm"][0].get_params()["scale_pos_weight"] == 3.0


def test_factory_rejects_non_logistic_sgd_loss() -> None:
    config = _config()
    config["logistic_regression"]["loss"] = "hinge"
    with pytest.raises(ModelConfigurationError, match="loss='log_loss'"):
        build_baseline_models(
            config,
            random_seed=42,
            training_labels=np.array([0, 1], dtype=np.uint8),
        )


def test_factory_rejects_one_class_training_labels() -> None:
    with pytest.raises(ModelConfigurationError, match="both classes"):
        build_baseline_models(
            _config(),
            random_seed=42,
            training_labels=np.zeros(3, dtype=np.uint8),
        )


def test_batched_probability_prediction_preserves_order() -> None:
    class Estimator:
        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            positive = matrix[:, 0]
            return np.column_stack((1.0 - positive, positive))

    matrix = np.array([[0.1], [0.4], [0.9]], dtype=np.float32)
    scores, seconds = predict_positive_probability(Estimator(), matrix, batch_rows=2)
    np.testing.assert_allclose(scores, [0.1, 0.4, 0.9])
    assert seconds >= 0.0
