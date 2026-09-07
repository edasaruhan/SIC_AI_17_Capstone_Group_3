from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from argus.modeling.preprocessing import (
    GRAPH_HISTORY_FEATURES,
    FeatureContract,
    FittedPreprocessor,
    PreprocessingError,
    iter_duckdb_transformed_batches,
)


def _feature_frame(rows: int = 4) -> pd.DataFrame:
    contract = FeatureContract.default()
    data: dict[str, object] = {
        column: np.arange(1, rows + 1, dtype="float64") + index
        for index, column in enumerate(contract.numeric_features)
    }
    for column in contract.missing_indicator_features:
        values = np.arange(1, rows + 1, dtype="float64")
        values[0] = np.nan
        data[column] = values
    data.update(
        {
            "receiving_currency": ["USD", "EUR", "usd", "GBP"][:rows],
            "payment_currency": ["USD", "EUR", "USD", "GBP"][:rows],
            "payment_format": ["Wire", "Cash", "wire", "Cheque"][:rows],
            "from_bank": ["1", "1", "2", "3"][:rows],
            "to_bank": ["10", "11", "10", "12"][:rows],
            "transaction_id": [f"tx-{index}" for index in range(rows)],
            "source_row_number": np.arange(2, rows + 2, dtype="int64"),
            "is_laundering": np.array([0, 1, 0, 0], dtype="int8")[:rows],
            "timestamp": pd.date_range("2022-01-01", periods=rows, freq="h"),
        }
    )
    return pd.DataFrame(data)


def test_default_contract_is_an_explicit_transaction_only_allowlist() -> None:
    contract = FeatureContract.default()

    assert len(contract.numeric_features) == 31
    assert contract.low_cardinality_categories == (
        "receiving_currency",
        "payment_currency",
        "payment_format",
    )
    assert contract.bank_frequency_categories == ("from_bank", "to_bank")
    assert set(GRAPH_HISTORY_FEATURES).isdisjoint(contract.predictor_columns)
    assert set(contract.forbidden_predictors).isdisjoint(contract.predictor_columns)
    assert "is_laundering" not in contract.predictor_columns
    assert "transaction_id" not in contract.predictor_columns
    assert "timestamp" not in contract.predictor_columns


def test_transform_uses_train_state_and_handles_unknown_categories() -> None:
    train = _feature_frame(3)
    fitted = FittedPreprocessor.fit_pandas(train)
    original_digest = fitted.state_sha256
    validation = _feature_frame(1)
    validation.loc[0, "receiving_currency"] = "NEW CURRENCY"
    validation.loc[0, "payment_currency"] = "NEW PAYMENT CURRENCY"
    validation.loc[0, "payment_format"] = "NEW FORMAT"
    validation.loc[0, "from_bank"] = "999999"
    validation.loc[0, "to_bank"] = "888888"

    transformed = fitted.transform_pandas(validation)

    assert transformed.dtype == np.float32
    assert transformed.shape == (1, len(fitted.feature_names))
    assert np.isfinite(transformed).all()
    for column in fitted.contract.low_cardinality_categories:
        indices = [
            index
            for index, name in enumerate(fitted.feature_names)
            if name.startswith(f"onehot__{column}=")
        ]
        assert transformed[0, indices].sum() == 0.0
    assert transformed[0, fitted.feature_names.index("frequency__from_bank")] == 0.0
    assert transformed[0, fitted.feature_names.index("frequency__to_bank")] == 0.0
    assert fitted.state_sha256 == original_digest


def test_numeric_state_is_fitted_after_train_median_imputation() -> None:
    train = _feature_frame(3)
    column = "sender_seconds_since_previous"
    train[column] = [np.nan, 10.0, 30.0]

    fitted = FittedPreprocessor.fit_pandas(train)
    transformed_train = fitted.transform_pandas(train)
    index = fitted.feature_names.index(f"numeric__{column}")
    missing_index = fitted.feature_names.index(f"missing__{column}")

    assert fitted.numeric_medians[column] == 20.0
    assert fitted.numeric_means[column] == 20.0
    assert np.isclose(transformed_train[:, index].mean(), 0.0)
    assert transformed_train[:, missing_index].tolist() == [1.0, 0.0, 0.0]


def test_fit_rejects_non_train_partition_and_forbidden_custom_contract() -> None:
    mixed = _feature_frame(3)
    mixed["partition"] = ["train", "validation", "train"]
    with pytest.raises(PreprocessingError, match="train rows only"):
        FittedPreprocessor.fit_pandas(mixed)

    with pytest.raises(PreprocessingError, match="Forbidden predictors"):
        FeatureContract(numeric_features=("is_laundering",))
    with pytest.raises(PreprocessingError, match="cannot be removed"):
        FeatureContract(forbidden_predictors=())


def test_fitted_state_round_trip_and_corruption_detection(tmp_path) -> None:
    fitted = FittedPreprocessor.fit_pandas(_feature_frame(3))
    state_path = fitted.save_state(tmp_path / "preprocessing_state.json")
    restored = FittedPreprocessor.load_state(state_path)

    assert restored.state_sha256 == fitted.state_sha256
    assert restored.feature_names == fitted.feature_names
    np.testing.assert_allclose(
        restored.transform_pandas(_feature_frame(1)),
        fitted.transform_pandas(_feature_frame(1)),
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["numeric_medians"]["amount_paid"] += 1.0
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreprocessingError, match="fingerprint mismatch"):
        FittedPreprocessor.load_state(state_path)


def test_duckdb_fit_matches_pandas_train_and_batches_frozen_validation(tmp_path) -> None:
    train = _feature_frame(3)
    validation = _feature_frame(1)
    validation["transaction_id"] = ["validation-only"]
    validation["source_row_number"] = [10]
    validation["timestamp"] = [pd.Timestamp("2022-01-02")]
    validation["receiving_currency"] = ["UNSEEN"]
    validation["from_bank"] = ["999999"]
    features = pd.concat([train, validation], ignore_index=True)
    split = features[["transaction_id", "timestamp"]].copy()
    split["partition"] = ["train", "train", "train", "validation"]
    feature_path = tmp_path / "features.parquet"
    split_path = tmp_path / "split.parquet"
    connection = duckdb.connect()
    connection.from_df(features).write_parquet(str(feature_path))
    connection.from_df(split).write_parquet(str(split_path))

    pandas_fitted = FittedPreprocessor.fit_pandas(train)
    duckdb_fitted = FittedPreprocessor.fit_duckdb(connection, feature_path, split_path)

    assert duckdb_fitted.fitted_train_rows == 3
    assert duckdb_fitted.feature_names == pandas_fitted.feature_names
    assert duckdb_fitted.category_vocabularies == pandas_fitted.category_vocabularies
    assert duckdb_fitted.bank_frequencies == pandas_fitted.bank_frequencies
    for column in duckdb_fitted.contract.numeric_features:
        assert duckdb_fitted.numeric_medians[column] == pytest.approx(
            pandas_fitted.numeric_medians[column]
        )
        assert duckdb_fitted.numeric_means[column] == pytest.approx(
            pandas_fitted.numeric_means[column]
        )
        assert duckdb_fitted.numeric_scales[column] == pytest.approx(
            pandas_fitted.numeric_scales[column]
        )

    batches = list(
        iter_duckdb_transformed_batches(
            connection,
            duckdb_fitted,
            feature_path,
            split_path,
            "validation",
            batch_size=1,
        )
    )
    assert sum(batch.rows for batch in batches) == 1
    assert batches[0].transaction_ids.tolist() == ["validation-only"]
    assert batches[0].matrix[0, duckdb_fitted.feature_names.index("frequency__from_bank")] == 0
