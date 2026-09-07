"""Bounded, deterministic estimator helpers for Sprint 3 refinement.

Ranking uses decision margins where an estimator exposes them.  Probabilities
are retained separately for saturation diagnosis; they are not allowed to hide
distinct large margins behind a numerically rounded 0.0 or 1.0 value.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class RefinementModelError(ValueError):
    """Raised when a Sprint 3 model candidate violates its contract."""


@dataclass(frozen=True)
class CandidateModel:
    """Unfitted model with fully resolved train-only imbalance parameters."""

    model_family: str
    candidate_id: str
    estimator: Any
    implementation: str
    effective_parameters: dict[str, Any]
    imbalance_policy: dict[str, Any]


@dataclass(frozen=True)
class DualScores:
    """Raw ranking scores and the corresponding probability output."""

    ranking_score: np.ndarray
    probability: np.ndarray
    ranking_score_type: str
    runtime_seconds: float


def _class_counts(labels: np.ndarray) -> tuple[int, int]:
    values = np.asarray(labels)
    positives = int(np.count_nonzero(values == 1))
    negatives = int(np.count_nonzero(values == 0))
    if positives <= 0 or negatives <= 0 or positives + negatives != values.size:
        raise RefinementModelError("Training labels must contain both binary classes")
    return negatives, positives


def resolve_positive_weight(policy: str, labels: np.ndarray) -> tuple[Any, dict[str, Any]]:
    """Resolve a declared imbalance policy from training labels only."""

    negatives, positives = _class_counts(labels)
    ratio = negatives / positives
    if policy == "none":
        value: Any = None
        positive_weight = 1.0
    elif policy == "balanced":
        value = "balanced"
        positive_weight = (negatives + positives) / (2.0 * positives)
    elif policy == "balanced_subsample":
        value = "balanced_subsample"
        positive_weight = None
    elif policy == "sqrt_train_ratio":
        positive_weight = math.sqrt(ratio)
        value = {0: 1.0, 1: positive_weight}
    elif policy == "capped_100":
        positive_weight = min(100.0, ratio)
        value = {0: 1.0, 1: positive_weight}
    else:
        raise RefinementModelError(f"Unknown class-weight policy: {policy!r}")
    return value, {
        "policy": policy,
        "train_negative_count": negatives,
        "train_positive_count": positives,
        "train_negative_to_positive_ratio": ratio,
        "resolved_positive_weight": positive_weight,
        "fit_scope": "fold_train_only_or_outer_train_only",
    }


def build_refinement_candidate(
    model_family: str,
    candidate_config: Mapping[str, Any],
    *,
    training_labels: np.ndarray,
    random_seed: int,
    n_jobs: int = 1,
) -> CandidateModel:
    """Build one explicitly bounded Sprint 3 candidate."""

    config = dict(candidate_config)
    candidate_id = config.pop("candidate_id", None)
    if not isinstance(candidate_id, str) or not candidate_id:
        raise RefinementModelError("Every refinement candidate needs a candidate_id")

    if model_family == "logistic_regression":
        policy = str(config.pop("class_weight_policy"))
        requested_penalty = str(config.pop("penalty"))
        if requested_penalty != "l2":
            raise RefinementModelError("Sprint 3 Logistic Regression currently requires L2")
        # scikit-learn >=1.8 expresses L2 through l1_ratio=0; its legacy
        # ``penalty='l2'`` argument now emits a deprecation warning.
        config["l1_ratio"] = 0.0
        class_weight, evidence = resolve_positive_weight(policy, training_labels)
        estimator = LogisticRegression(
            **config,
            class_weight=class_weight,
            random_state=random_seed,
        )
        estimator._argus_regularization = "l2_via_l1_ratio_0"  # type: ignore[attr-defined]
        implementation = "sklearn.linear_model.LogisticRegression"
    elif model_family == "random_forest":
        policy = str(config.pop("class_weight_policy"))
        class_weight, evidence = resolve_positive_weight(policy, training_labels)
        estimator = RandomForestClassifier(
            **config,
            class_weight=class_weight,
            random_state=random_seed,
            n_jobs=n_jobs,
        )
        implementation = "sklearn.ensemble.RandomForestClassifier"
    elif model_family == "lightgbm":
        policy = str(config.pop("scale_pos_weight_policy"))
        resolved, evidence = resolve_positive_weight(policy, training_labels)
        if isinstance(resolved, Mapping):
            scale_pos_weight = float(resolved[1])
        elif resolved is None:
            scale_pos_weight = 1.0
        else:
            raise RefinementModelError(
                "LightGBM requires none, sqrt_train_ratio, or capped_100 weighting"
            )
        estimator = LGBMClassifier(
            **config,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            boost_from_average=True,
            deterministic=True,
            force_col_wise=True,
            random_state=random_seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
        implementation = "lightgbm.LGBMClassifier"
    else:
        raise RefinementModelError(f"Unknown refinement model family: {model_family!r}")

    return CandidateModel(
        model_family=model_family,
        candidate_id=candidate_id,
        estimator=estimator,
        implementation=implementation,
        effective_parameters=dict(estimator.get_params()),
        imbalance_policy=evidence,
    )


def fit_candidate(candidate: CandidateModel, matrix: np.ndarray, labels: np.ndarray) -> float:
    """Fit one candidate and return measured wall-clock seconds."""

    started = time.perf_counter()
    candidate.estimator.fit(matrix, labels)
    return time.perf_counter() - started


def predict_dual_scores(
    estimator: Any,
    matrix: np.ndarray,
    *,
    batch_rows: int,
) -> DualScores:
    """Predict raw ranking margins and probabilities in bounded batches."""

    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
        raise RefinementModelError("batch_rows must be a positive integer")
    rows = int(matrix.shape[0])
    ranking = np.empty(rows, dtype=np.float64)
    probability = np.empty(rows, dtype=np.float64)
    if isinstance(estimator, LGBMClassifier):
        score_type = "lightgbm_raw_margin"
    elif hasattr(estimator, "decision_function"):
        score_type = "decision_function_margin"
    else:
        score_type = "positive_class_probability"

    started = time.perf_counter()
    for start in range(0, rows, batch_rows):
        stop = min(rows, start + batch_rows)
        batch = matrix[start:stop]
        probabilities = np.asarray(estimator.predict_proba(batch), dtype=np.float64)
        if probabilities.shape != (stop - start, 2):
            raise RefinementModelError("predict_proba returned an unexpected shape")
        probability[start:stop] = probabilities[:, 1]
        if isinstance(estimator, LGBMClassifier):
            raw = estimator.predict(batch, raw_score=True)
            ranking[start:stop] = np.asarray(raw, dtype=np.float64)
        elif hasattr(estimator, "decision_function"):
            raw = np.asarray(estimator.decision_function(batch), dtype=np.float64)
            ranking[start:stop] = raw.reshape(-1)
        else:
            ranking[start:stop] = probability[start:stop]

    for name, values in (("ranking", ranking), ("probability", probability)):
        if not np.isfinite(values).all():
            raise RefinementModelError(f"{name} scores contain non-finite values")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise RefinementModelError("Probability scores fall outside [0, 1]")
    return DualScores(
        ranking_score=ranking,
        probability=probability,
        ranking_score_type=score_type,
        runtime_seconds=time.perf_counter() - started,
    )


def convergence_evidence(estimator: Any, warning_messages: list[str]) -> dict[str, Any]:
    """Return explicit convergence evidence for solver-bearing logistic models."""

    if not isinstance(estimator, LogisticRegression):
        return {"applicable": False, "converged": None}
    iterations = np.asarray(estimator.n_iter_, dtype=np.int64)
    maximum = int(estimator.max_iter)
    convergence_warnings = [message for message in warning_messages if "converg" in message.lower()]
    observed = int(iterations.max())
    return {
        "applicable": True,
        "solver": estimator.solver,
        "regularization": getattr(estimator, "_argus_regularization", estimator.penalty),
        "C": float(estimator.C),
        "configured_max_iter": maximum,
        "observed_n_iter": observed,
        "stopped_before_max_iter": observed < maximum,
        "convergence_warnings": convergence_warnings,
        "converged": observed < maximum and not convergence_warnings,
        "coefficient_l2_norm": float(np.linalg.norm(estimator.coef_)),
        "maximum_absolute_coefficient": float(np.max(np.abs(estimator.coef_))),
    }


def lightgbm_leaf_evidence(estimator: Any) -> dict[str, Any]:
    """Summarize fitted LightGBM leaf magnitudes without treating them as importance."""

    if not isinstance(estimator, LGBMClassifier):
        return {"applicable": False}
    dumped = estimator.booster_.dump_model()
    per_tree: list[float] = []

    def visit(node: Mapping[str, Any], values: list[float]) -> None:
        if "leaf_value" in node:
            values.append(float(node["leaf_value"]))
            return
        visit(node["left_child"], values)
        visit(node["right_child"], values)

    for tree in dumped["tree_info"]:
        values: list[float] = []
        visit(tree["tree_structure"], values)
        per_tree.append(max(abs(value) for value in values))
    return {
        "applicable": True,
        "tree_count": len(per_tree),
        "maximum_absolute_leaf_value": float(max(per_tree)),
        "trees_with_absolute_leaf_over_100": int(sum(value > 100 for value in per_tree)),
        "trees_with_absolute_leaf_over_1m": int(sum(value > 1_000_000 for value in per_tree)),
        "per_tree_maximum_absolute_leaf": per_tree,
    }
