from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from argus.modeling.refinement_protocol import (
    RefinementProtocolError,
    TemporalFold,
    select_outer_validation_champion,
    summarize_and_select_candidates,
)


def _fold() -> TemporalFold:
    start = datetime(2022, 1, 1)
    return TemporalFold(
        fold=1,
        train_minimum_timestamp=start,
        train_maximum_timestamp=start + timedelta(hours=1),
        validation_minimum_timestamp=start + timedelta(hours=2),
        validation_maximum_timestamp=start + timedelta(hours=3),
        train_rows=100,
        train_positive_labels=2,
        validation_rows=50,
        validation_positive_labels=1,
    )


def test_temporal_fold_enforces_strict_chronology_and_both_classes() -> None:
    assert _fold().to_dict()["timestamp_groups_split"] is False
    with pytest.raises(RefinementProtocolError, match="strictly chronological"):
        TemporalFold(
            **{
                **_fold().__dict__,
                "validation_minimum_timestamp": _fold().train_maximum_timestamp,
            }
        )
    with pytest.raises(RefinementProtocolError, match="both classes"):
        TemporalFold(**{**_fold().__dict__, "train_positive_labels": 0})


def test_candidate_selection_uses_complete_inner_folds_and_deterministic_ties() -> None:
    trials = pd.DataFrame(
        [
            {
                "model_family": family,
                "candidate_id": candidate,
                "fold": fold,
                "average_precision": score,
                "eligible": True,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
                "partition": "inner_validation",
            }
            for family, candidate, score in (
                ("lightgbm", "b", 0.2),
                ("lightgbm", "a", 0.2),
                ("random_forest", "r", 0.3),
            )
            for fold in (1, 2, 3)
        ]
    )
    summary, selected = summarize_and_select_candidates(trials, expected_folds=3)
    assert len(summary) == 3
    assert selected["lightgbm"]["candidate_id"] == "a"
    assert selected["random_forest"]["candidate_id"] == "r"
    assert all(not value["test_metrics_used"] for value in selected.values())


def test_candidate_selection_rejects_test_or_missing_fold_evidence() -> None:
    incomplete = pd.DataFrame(
        [
            {
                "model_family": "lightgbm",
                "candidate_id": "a",
                "fold": 1,
                "average_precision": 0.2,
                "eligible": True,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
                "partition": "inner_validation",
            }
        ]
    )
    with pytest.raises(RefinementProtocolError, match="exactly 3 folds"):
        summarize_and_select_candidates(incomplete, expected_folds=3)
    incomplete["test_metric"] = 0.9
    with pytest.raises(RefinementProtocolError, match="Test metrics"):
        summarize_and_select_candidates(incomplete, expected_folds=1)


def test_outer_champion_is_validation_only() -> None:
    results = {
        "a": {"ranking_metrics": {"average_precision": 0.2}},
        "b": {"ranking_metrics": {"average_precision": 0.3}},
    }
    champion = select_outer_validation_champion(results, partition="validation")
    assert champion["champion_model"] == "b"
    assert champion["test_metrics_used"] is False
    with pytest.raises(RefinementProtocolError, match="validation-only"):
        select_outer_validation_champion(results, partition="test")
