from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argus.modeling.preprocessing import (
    ALWAYS_FORBIDDEN_PREDICTORS,
    CURRENT_TRANSACTION_NUMERIC_FEATURES,
    FEATURE_FAMILY_CONTRACTS,
    FORBIDDEN_PREDICTORS,
    GRAPH_HISTORY_FEATURES,
    HISTORY_NUMERIC_FEATURES,
    LOW_CARDINALITY_CATEGORICAL_FEATURES,
    NULLABLE_NUMERIC_FEATURES,
    TEMPORAL_NUMERIC_FEATURES,
    TRANSACTION_NULLABLE_NUMERIC_FEATURES,
    TRANSACTION_ONLY_FAMILY,
    TRANSACTION_TEMPORAL_HISTORY_FAMILY,
    TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY,
    FeatureContract,
    FittedPreprocessor,
    PreprocessingError,
    feature_contract_for_family,
)


def _family_fixture() -> pd.DataFrame:
    graph_contract = feature_contract_for_family(TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY)
    rows = 4
    data: dict[str, object] = {
        column: np.arange(1, rows + 1, dtype="float64") + index
        for index, column in enumerate(graph_contract.numeric_features)
    }
    for column in graph_contract.missing_indicator_features:
        values = np.arange(1, rows + 1, dtype="float64")
        values[0] = np.nan
        data[column] = values
    data.update(
        {
            "receiving_currency": ["USD", "EUR", "usd", "GBP"],
            "payment_currency": ["USD", "EUR", "usd", "GBP"],
            "payment_format": ["Wire", "Cash", "wire", "Cheque"],
            "from_bank": ["1", "1", "2", "3"],
            "to_bank": ["10", "11", "10", "12"],
        }
    )
    return pd.DataFrame(data)


def test_reviewed_feature_families_are_strictly_nested() -> None:
    transaction = feature_contract_for_family(TRANSACTION_ONLY_FAMILY)
    temporal = feature_contract_for_family(TRANSACTION_TEMPORAL_HISTORY_FAMILY)
    graph = feature_contract_for_family(TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY)

    transaction_predictors = set(transaction.predictor_columns)
    temporal_predictors = set(temporal.predictor_columns)
    graph_predictors = set(graph.predictor_columns)

    assert transaction_predictors < temporal_predictors < graph_predictors
    assert temporal_predictors - transaction_predictors == set(
        (*TEMPORAL_NUMERIC_FEATURES, *HISTORY_NUMERIC_FEATURES)
    )
    assert graph_predictors - temporal_predictors == set(GRAPH_HISTORY_FEATURES)


def test_family_contracts_have_expected_transformed_fixture_counts() -> None:
    frame = _family_fixture()
    expected_counts = {
        TRANSACTION_ONLY_FAMILY: 21,
        TRANSACTION_TEMPORAL_HISTORY_FAMILY: 46,
        TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY: 51,
    }

    for family, expected_count in expected_counts.items():
        fitted = FittedPreprocessor.fit_pandas(
            frame,
            contract=feature_contract_for_family(family),
        )
        transformed = fitted.transform_pandas(frame)

        assert transformed.shape == (len(frame), expected_count)
        assert len(fitted.feature_names) == expected_count
        assert np.isfinite(transformed).all()


def test_core_identity_and_target_fields_cannot_be_enabled_or_unforbidden() -> None:
    for field in ALWAYS_FORBIDDEN_PREDICTORS:
        with pytest.raises(PreprocessingError, match="cannot be removed"):
            FeatureContract(
                forbidden_predictors=tuple(
                    candidate for candidate in ALWAYS_FORBIDDEN_PREDICTORS if candidate != field
                )
            )

        with pytest.raises(PreprocessingError, match="Forbidden predictors"):
            FeatureContract(
                numeric_features=(*CURRENT_TRANSACTION_NUMERIC_FEATURES, field),
                missing_indicator_features=TRANSACTION_NULLABLE_NUMERIC_FEATURES,
                forbidden_predictors=ALWAYS_FORBIDDEN_PREDICTORS,
            )


def test_graph_family_only_relaxes_reviewed_graph_exclusion() -> None:
    graph = feature_contract_for_family(TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY)

    assert set(graph.forbidden_predictors) == set(ALWAYS_FORBIDDEN_PREDICTORS)
    assert set(graph.numeric_features).intersection(GRAPH_HISTORY_FEATURES) == set(
        GRAPH_HISTORY_FEATURES
    )
    assert set(ALWAYS_FORBIDDEN_PREDICTORS).isdisjoint(graph.predictor_columns)


def test_sprint2_default_remains_temporal_history_without_graph() -> None:
    default = FeatureContract.default()
    sprint2_family = feature_contract_for_family(TRANSACTION_TEMPORAL_HISTORY_FAMILY)

    assert default == sprint2_family
    assert default.numeric_features == (
        *CURRENT_TRANSACTION_NUMERIC_FEATURES,
        *TEMPORAL_NUMERIC_FEATURES,
        *HISTORY_NUMERIC_FEATURES,
    )
    assert default.low_cardinality_categories == LOW_CARDINALITY_CATEGORICAL_FEATURES
    assert default.missing_indicator_features == NULLABLE_NUMERIC_FEATURES
    assert default.forbidden_predictors == FORBIDDEN_PREDICTORS
    assert set(GRAPH_HISTORY_FEATURES).isdisjoint(default.predictor_columns)


def test_public_family_factory_is_normalized_but_closed() -> None:
    assert set(FEATURE_FAMILY_CONTRACTS) == {
        TRANSACTION_ONLY_FAMILY,
        TRANSACTION_TEMPORAL_HISTORY_FAMILY,
        TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY,
    }
    assert (
        feature_contract_for_family(" TRANSACTION_ONLY ")
        == FEATURE_FAMILY_CONTRACTS[TRANSACTION_ONLY_FAMILY]
    )
    with pytest.raises(PreprocessingError, match="Unknown feature family"):
        feature_contract_for_family("unreviewed_family")
    with pytest.raises(TypeError):
        FEATURE_FAMILY_CONTRACTS["unreviewed_family"] = FeatureContract()  # type: ignore[index]
