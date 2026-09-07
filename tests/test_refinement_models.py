from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from argus.modeling.refinement_models import (
    RefinementModelError,
    build_refinement_candidate,
    convergence_evidence,
    fit_candidate,
    predict_dual_scores,
    resolve_positive_weight,
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    matrix, labels = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    return matrix.astype(np.float32), labels.astype(np.int8)


def test_train_only_weight_policies_are_exact() -> None:
    labels = np.array([0] * 99 + [1])
    value, evidence = resolve_positive_weight("sqrt_train_ratio", labels)
    assert value == {0: 1.0, 1: pytest.approx(np.sqrt(99))}
    assert evidence["train_negative_count"] == 99
    assert evidence["train_positive_count"] == 1
    assert evidence["fit_scope"] == "fold_train_only_or_outer_train_only"


def test_unknown_weight_policy_and_single_class_are_rejected() -> None:
    with pytest.raises(RefinementModelError, match="Unknown"):
        resolve_positive_weight("future_labels", np.array([0, 1]))
    with pytest.raises(RefinementModelError, match="both binary classes"):
        resolve_positive_weight("none", np.zeros(4, dtype=np.int8))


@pytest.mark.parametrize(
    ("family", "config", "expected_type"),
    [
        (
            "logistic_regression",
            {
                "candidate_id": "lr",
                "solver": "saga",
                "penalty": "l2",
                "C": 0.1,
                "class_weight_policy": "none",
                "max_iter": 200,
                "tol": 1e-3,
            },
            "decision_function_margin",
        ),
        (
            "random_forest",
            {
                "candidate_id": "rf",
                "n_estimators": 10,
                "max_depth": 4,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight_policy": "balanced_subsample",
                "bootstrap": True,
                "max_samples": 0.8,
            },
            "positive_class_probability",
        ),
        (
            "lightgbm",
            {
                "candidate_id": "lgb",
                "scale_pos_weight_policy": "sqrt_train_ratio",
                "n_estimators": 10,
                "learning_rate": 0.05,
                "num_leaves": 7,
                "max_depth": 4,
                "min_child_samples": 5,
                "min_child_weight": 1.0,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
                "max_delta_step": 1.0,
            },
            "lightgbm_raw_margin",
        ),
    ],
)
def test_refinement_candidates_fit_and_emit_dual_scores(
    family: str,
    config: dict[str, object],
    expected_type: str,
) -> None:
    matrix, labels = _data()
    candidate = build_refinement_candidate(
        family,
        config,
        training_labels=labels,
        random_seed=42,
    )
    assert fit_candidate(candidate, matrix, labels) >= 0.0
    scores = predict_dual_scores(candidate.estimator, matrix, batch_rows=37)
    assert scores.ranking_score_type == expected_type
    assert scores.ranking_score.shape == (300,)
    assert np.isfinite(scores.ranking_score).all()
    assert np.all((scores.probability >= 0) & (scores.probability <= 1))


def test_logistic_convergence_metadata_is_explicit() -> None:
    matrix, labels = _data()
    candidate = build_refinement_candidate(
        "logistic_regression",
        {
            "candidate_id": "lr",
            "solver": "saga",
            "penalty": "l2",
            "C": 0.01,
            "class_weight_policy": "none",
            "max_iter": 500,
            "tol": 1e-2,
        },
        training_labels=labels,
        random_seed=42,
    )
    fit_candidate(candidate, matrix, labels)
    evidence = convergence_evidence(candidate.estimator, [])
    assert evidence["applicable"] is True
    assert evidence["converged"] is True
    assert evidence["observed_n_iter"] < evidence["configured_max_iter"]
