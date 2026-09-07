from __future__ import annotations

import math

import pytest

from argus.modeling.selection import (
    ChampionSelectionError,
    select_transaction_baseline_champion,
)


def test_champion_is_selected_by_validation_average_precision_only() -> None:
    selection = select_transaction_baseline_champion(
        {
            "logistic_regression": 0.12,
            "random_forest": 0.18,
            "lightgbm": 0.15,
        }
    )

    assert selection["champion_model"] == "random_forest"
    assert selection["selection_partition"] == "validation"
    assert selection["selection_metric"] == "average_precision"
    assert selection["selection_metric_value"] == 0.18
    assert selection["test_metrics_used"] is False
    assert [row["model_name"] for row in selection["ranked_candidates"]] == [
        "random_forest",
        "lightgbm",
        "logistic_regression",
    ]


def test_exact_validation_ap_tie_is_broken_by_model_name_ascending() -> None:
    selection = select_transaction_baseline_champion(
        {"random_forest": 0.2, "lightgbm": 0.2, "logistic_regression": 0.1}
    )

    assert selection["champion_model"] == "lightgbm"
    assert selection["tie_break_rule"] == "model_name_ascending"


def test_selection_rejects_every_explicit_test_metrics_input() -> None:
    with pytest.raises(ChampionSelectionError, match="Final-test metrics"):
        select_transaction_baseline_champion(
            {"logistic_regression": 0.1},
            test_metrics={},
        )


def test_selection_rejects_non_validation_partition() -> None:
    with pytest.raises(ChampionSelectionError, match="validation results only"):
        select_transaction_baseline_champion(
            {"logistic_regression": 0.1},
            partition="test",
        )


@pytest.mark.parametrize("value", [None, True, -0.1, 1.1, math.nan])
def test_selection_rejects_invalid_average_precision(value: object) -> None:
    with pytest.raises(ChampionSelectionError):
        select_transaction_baseline_champion({"logistic_regression": value})  # type: ignore[dict-item]
