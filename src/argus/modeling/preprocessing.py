"""Train-only preprocessing for transaction-level baseline models.

The full Sprint 1 feature table deliberately contains identifiers, provenance,
the target, and a small graph-history family alongside eligible transaction
signals.  This module uses an explicit allow-list rather than a drop-list so a
new column cannot silently become a predictor.  Every learned value (numeric
statistics, category vocabularies, and bank frequencies) is fitted on training
rows only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd


class PreprocessingError(ValueError):
    """Raised when a baseline feature contract or transform is unsafe."""


CURRENT_TRANSACTION_NUMERIC_FEATURES = (
    "amount_paid",
    "amount_received",
    "log_amount_paid",
    "log_amount_received",
    "same_bank",
    "currency_match",
    "amount_difference_same_currency",
    "amount_ratio_same_currency",
)

TEMPORAL_NUMERIC_FEATURES = (
    "hour",
    "day_of_week",
    "is_weekend",
    "sender_seconds_since_previous",
    "receiver_seconds_since_previous",
)

# Retained as a public compatibility alias for Sprint 2 callers and artifacts.
TRANSACTION_AND_TIME_NUMERIC_FEATURES = (
    *CURRENT_TRANSACTION_NUMERIC_FEATURES,
    *TEMPORAL_NUMERIC_FEATURES,
)

HISTORY_NUMERIC_FEATURES = (
    "sender_previous_transaction_count",
    "sender_previous_outgoing_amount",
    "sender_previous_unique_counterparties",
    "sender_burst_count_1h",
    "sender_rolling_outgoing_amount_1h",
    "sender_burst_count_24h",
    "sender_rolling_outgoing_amount_24h",
    "sender_burst_count_7d",
    "sender_rolling_outgoing_amount_7d",
    "receiver_previous_transaction_count",
    "receiver_previous_incoming_amount",
    "receiver_previous_unique_counterparties",
    "receiver_burst_count_1h",
    "receiver_rolling_incoming_amount_1h",
    "receiver_burst_count_24h",
    "receiver_rolling_incoming_amount_24h",
    "receiver_burst_count_7d",
    "receiver_rolling_incoming_amount_7d",
)

LOW_CARDINALITY_CATEGORICAL_FEATURES = (
    "payment_currency",
    "receiving_currency",
    "payment_format",
)

BANK_FREQUENCY_FEATURES = ("from_bank", "to_bank")

TRANSACTION_NULLABLE_NUMERIC_FEATURES = (
    "amount_difference_same_currency",
    "amount_ratio_same_currency",
)

TEMPORAL_NULLABLE_NUMERIC_FEATURES = (
    "sender_seconds_since_previous",
    "receiver_seconds_since_previous",
)

NULLABLE_NUMERIC_FEATURES = (
    *TRANSACTION_NULLABLE_NUMERIC_FEATURES,
    *TEMPORAL_NULLABLE_NUMERIC_FEATURES,
)

GRAPH_HISTORY_FEATURES = (
    "sender_prior_fan_out_degree",
    "sender_prior_fan_in_degree",
    "receiver_prior_fan_out_degree",
    "receiver_prior_fan_in_degree",
    "pair_previous_transfer_count",
)

ALWAYS_FORBIDDEN_PREDICTORS = (
    "is_laundering",
    "partition",
    "transaction_id",
    "source_file",
    "source_row_number",
    "timestamp",
    "from_bank_raw",
    "to_bank_raw",
    "from_account",
    "to_account",
    "from_node_id",
    "to_node_id",
)

# Sprint 2's transaction baseline excluded graph history.  Keep this combined
# name for compatibility, while the always-forbidden safety boundary remains
# separate so Sprint 3 can deliberately include the reviewed graph family.
FORBIDDEN_PREDICTORS = (
    *ALWAYS_FORBIDDEN_PREDICTORS,
    *GRAPH_HISTORY_FEATURES,
)

TRANSACTION_ONLY_FAMILY = "transaction_only"
TRANSACTION_TEMPORAL_HISTORY_FAMILY = "transaction_temporal_history"
TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY = "transaction_temporal_history_graph"

_MISSING_CATEGORY = "__ARGUS_MISSING__"
_STATE_VERSION = 1
_DUCKDB_VECTOR_SIZE = 2048


@dataclass(frozen=True)
class FeatureContract:
    """Immutable allow-list with an invariant identity/target safety boundary."""

    numeric_features: tuple[str, ...] = (
        *TRANSACTION_AND_TIME_NUMERIC_FEATURES,
        *HISTORY_NUMERIC_FEATURES,
    )
    low_cardinality_categories: tuple[str, ...] = LOW_CARDINALITY_CATEGORICAL_FEATURES
    bank_frequency_categories: tuple[str, ...] = BANK_FREQUENCY_FEATURES
    missing_indicator_features: tuple[str, ...] = NULLABLE_NUMERIC_FEATURES
    forbidden_predictors: tuple[str, ...] = FORBIDDEN_PREDICTORS

    def __post_init__(self) -> None:
        missing_safety_guards = sorted(
            set(ALWAYS_FORBIDDEN_PREDICTORS).difference(self.forbidden_predictors)
        )
        if missing_safety_guards:
            raise PreprocessingError(
                f"Core forbidden predictors cannot be removed: {missing_safety_guards}"
            )
        groups = {
            "numeric_features": self.numeric_features,
            "low_cardinality_categories": self.low_cardinality_categories,
            "bank_frequency_categories": self.bank_frequency_categories,
        }
        for name, values in groups.items():
            if not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise PreprocessingError(f"{name} must contain non-empty column names")
            if len(values) != len(set(values)):
                raise PreprocessingError(f"{name} contains duplicate columns")

        predictor_groups = tuple(groups.values())
        all_predictors = tuple(value for group in predictor_groups for value in group)
        if len(all_predictors) != len(set(all_predictors)):
            raise PreprocessingError("Feature groups must not overlap")

        unsafe = sorted(set(all_predictors).intersection(self.forbidden_predictors))
        if unsafe:
            raise PreprocessingError(f"Forbidden predictors requested: {unsafe}")

        unknown_indicators = sorted(
            set(self.missing_indicator_features).difference(self.numeric_features)
        )
        if unknown_indicators:
            raise PreprocessingError(
                f"Missing indicators must reference numeric features: {unknown_indicators}"
            )
        if len(self.missing_indicator_features) != len(set(self.missing_indicator_features)):
            raise PreprocessingError("missing_indicator_features contains duplicates")

    @classmethod
    def default(cls) -> FeatureContract:
        """Return the reviewed transaction/time/history-only contract."""

        return cls()

    @property
    def predictor_columns(self) -> tuple[str, ...]:
        """Return source columns selected from the Sprint 1 feature table."""

        return (
            *self.numeric_features,
            *self.low_cardinality_categories,
            *self.bank_frequency_categories,
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return a JSON-safe representation with deterministic group ordering."""

        return {
            "numeric_features": list(self.numeric_features),
            "low_cardinality_categories": list(self.low_cardinality_categories),
            "bank_frequency_categories": list(self.bank_frequency_categories),
            "missing_indicator_features": list(self.missing_indicator_features),
            "forbidden_predictors": list(self.forbidden_predictors),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FeatureContract:
        """Restore a contract from serialized state."""

        required = {
            "numeric_features",
            "low_cardinality_categories",
            "bank_frequency_categories",
            "missing_indicator_features",
            "forbidden_predictors",
        }
        if set(values) != required:
            raise PreprocessingError(
                f"Serialized feature contract keys differ: expected={sorted(required)}, "
                f"actual={sorted(values)}"
            )
        return cls(**{key: tuple(values[key]) for key in required})


FEATURE_FAMILY_CONTRACTS = MappingProxyType(
    {
        TRANSACTION_ONLY_FAMILY: FeatureContract(
            numeric_features=CURRENT_TRANSACTION_NUMERIC_FEATURES,
            missing_indicator_features=TRANSACTION_NULLABLE_NUMERIC_FEATURES,
            forbidden_predictors=FORBIDDEN_PREDICTORS,
        ),
        TRANSACTION_TEMPORAL_HISTORY_FAMILY: FeatureContract.default(),
        TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY: FeatureContract(
            numeric_features=(
                *TRANSACTION_AND_TIME_NUMERIC_FEATURES,
                *HISTORY_NUMERIC_FEATURES,
                *GRAPH_HISTORY_FEATURES,
            ),
            missing_indicator_features=NULLABLE_NUMERIC_FEATURES,
            forbidden_predictors=ALWAYS_FORBIDDEN_PREDICTORS,
        ),
    }
)


def feature_contract_for_family(family: str) -> FeatureContract:
    """Return one of the three reviewed, strictly nested Sprint 3 contracts."""

    if not isinstance(family, str) or not family.strip():
        raise PreprocessingError("Feature family must be a non-empty string")
    key = family.strip().lower()
    try:
        return FEATURE_FAMILY_CONTRACTS[key]
    except KeyError as exc:
        raise PreprocessingError(
            f"Unknown feature family {family!r}; expected one of {sorted(FEATURE_FAMILY_CONTRACTS)}"
        ) from exc


def _normalise_category(values: pd.Series) -> pd.Series:
    normalised = values.astype("string").str.strip().str.upper()
    return normalised.mask(normalised.isna() | normalised.eq(""), _MISSING_CATEGORY)


def _check_required_columns(frame: pd.DataFrame, contract: FeatureContract) -> None:
    missing = sorted(set(contract.predictor_columns).difference(frame.columns))
    if missing:
        raise PreprocessingError(f"Required predictor columns are missing: {missing}")


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(
            f"Numeric predictor {column!r} contains non-numeric values"
        ) from exc
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(values).any():
        raise PreprocessingError(f"Numeric predictor {column!r} contains infinite values")
    return values


def _state_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FittedPreprocessor:
    """Fully fitted, immutable preprocessing state.

    Construct instances through :meth:`fit_pandas` or :meth:`fit_duckdb`; the
    resulting object has no method that updates learned state during transform.
    """

    contract: FeatureContract
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]
    category_vocabularies: dict[str, tuple[str, ...]]
    bank_frequencies: dict[str, dict[str, float]]
    fitted_train_rows: int
    fit_source: str

    def __post_init__(self) -> None:
        if self.fitted_train_rows <= 0:
            raise PreprocessingError("fitted_train_rows must be positive")
        numeric = set(self.contract.numeric_features)
        for name, values in (
            ("numeric_medians", self.numeric_medians),
            ("numeric_means", self.numeric_means),
            ("numeric_scales", self.numeric_scales),
        ):
            if set(values) != numeric:
                raise PreprocessingError(f"{name} does not match numeric feature contract")
            if any(not math.isfinite(float(value)) for value in values.values()):
                raise PreprocessingError(f"{name} must contain only finite values")
        if any(float(value) <= 0 for value in self.numeric_scales.values()):
            raise PreprocessingError("Numeric scales must be strictly positive")
        if set(self.category_vocabularies) != set(self.contract.low_cardinality_categories):
            raise PreprocessingError("Category vocabularies do not match feature contract")
        if set(self.bank_frequencies) != set(self.contract.bank_frequency_categories):
            raise PreprocessingError("Bank-frequency maps do not match feature contract")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise PreprocessingError("Transformed feature names are not unique")

    @classmethod
    def fit_pandas(
        cls,
        train: pd.DataFrame,
        contract: FeatureContract | None = None,
    ) -> FittedPreprocessor:
        """Fit all learned state from a pandas training frame only."""

        selected_contract = contract or FeatureContract.default()
        _check_required_columns(train, selected_contract)
        if train.empty:
            raise PreprocessingError("Cannot fit preprocessing on an empty training frame")
        if "partition" in train.columns:
            partitions = set(train["partition"].astype("string").str.casefold().dropna())
            if partitions != {"train"}:
                raise PreprocessingError(
                    "fit_pandas accepts train rows only when a partition column is present"
                )

        medians: dict[str, float] = {}
        numeric_columns: list[np.ndarray] = []
        for column in selected_contract.numeric_features:
            values = _numeric_values(train, column)
            finite = values[~np.isnan(values)]
            if not finite.size:
                raise PreprocessingError(
                    f"Numeric predictor {column!r} has no observed training values"
                )
            median = float(np.median(finite))
            medians[column] = median
            numeric_columns.append(np.where(np.isnan(values), median, values))

        means = {
            column: float(values.mean(dtype="float64"))
            for column, values in zip(
                selected_contract.numeric_features, numeric_columns, strict=True
            )
        }
        scales: dict[str, float] = {}
        for column, values in zip(selected_contract.numeric_features, numeric_columns, strict=True):
            scale = float(values.std(dtype="float64", ddof=0))
            scales[column] = scale if math.isfinite(scale) and scale > 0.0 else 1.0

        vocabularies: dict[str, tuple[str, ...]] = {}
        for column in selected_contract.low_cardinality_categories:
            vocabularies[column] = tuple(sorted(_normalise_category(train[column]).unique()))

        frequencies: dict[str, dict[str, float]] = {}
        train_rows = len(train)
        for column in selected_contract.bank_frequency_categories:
            counts = _normalise_category(train[column]).value_counts(dropna=False, sort=False)
            frequencies[column] = {
                str(value): float(count / train_rows)
                for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
            }

        return cls(
            contract=selected_contract,
            numeric_medians=medians,
            numeric_means=means,
            numeric_scales=scales,
            category_vocabularies=vocabularies,
            bank_frequencies=frequencies,
            fitted_train_rows=train_rows,
            fit_source="pandas_train_frame",
        )

    @classmethod
    def fit_duckdb(
        cls,
        connection: Any,
        feature_parquet: str | Path,
        split_parquet: str | Path,
        contract: FeatureContract | None = None,
    ) -> FittedPreprocessor:
        """Fit exact train-only statistics out of core with DuckDB.

        Only rows labelled ``train`` in the frozen Sprint 1 split manifest enter
        these SQL aggregates. No validation or test statistic is queried.
        """

        selected_contract = contract or FeatureContract.default()
        feature_path = str(Path(feature_parquet).expanduser().resolve())
        split_path = str(Path(split_parquet).expanduser().resolve())

        described = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [feature_path]
        ).fetchall()
        available = {str(row[0]) for row in described}
        required = {*selected_contract.predictor_columns, "transaction_id"}
        missing = sorted(required.difference(available))
        if missing:
            raise PreprocessingError(f"Feature Parquet is missing required columns: {missing}")

        source_sql = (
            "FROM read_parquet(?) AS feature "
            "INNER JOIN read_parquet(?) AS split USING (transaction_id) "
            "WHERE split.partition = 'train'"
        )
        parameters = [feature_path, split_path]
        train_rows = int(
            connection.execute(f"SELECT count(*) {source_sql}", parameters).fetchone()[0]
        )
        if train_rows <= 0:
            raise PreprocessingError("Frozen split contains no training rows")

        numeric_identifiers = [
            _quote_identifier(name) for name in selected_contract.numeric_features
        ]
        invalid_expressions = [
            f"sum(CASE WHEN {name} IS NOT NULL AND NOT isfinite(CAST({name} AS DOUBLE)) "
            "THEN 1 ELSE 0 END)"
            for name in numeric_identifiers
        ]
        invalid_counts = connection.execute(
            f"SELECT {', '.join(invalid_expressions)} {source_sql}", parameters
        ).fetchone()
        invalid = {
            column: int(count or 0)
            for column, count in zip(
                selected_contract.numeric_features, invalid_counts, strict=True
            )
            if int(count or 0) > 0
        }
        if invalid:
            raise PreprocessingError(f"Non-finite numeric training values found: {invalid}")

        median_expressions = [f"median(CAST({name} AS DOUBLE))" for name in numeric_identifiers]
        median_row = connection.execute(
            f"SELECT {', '.join(median_expressions)} {source_sql}", parameters
        ).fetchone()
        medians: dict[str, float] = {}
        for column, value in zip(selected_contract.numeric_features, median_row, strict=True):
            if value is None or not math.isfinite(float(value)):
                raise PreprocessingError(
                    f"Numeric predictor {column!r} has no observed training values"
                )
            medians[column] = float(value)

        imputed = [
            f"coalesce(CAST({name} AS DOUBLE), {_sql_float(medians[column])})"
            for column, name in zip(
                selected_contract.numeric_features, numeric_identifiers, strict=True
            )
        ]
        moment_expressions = [
            expression
            for value in imputed
            for expression in (f"avg({value})", f"stddev_pop({value})")
        ]
        moment_row = connection.execute(
            f"SELECT {', '.join(moment_expressions)} {source_sql}", parameters
        ).fetchone()
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for index, column in enumerate(selected_contract.numeric_features):
            mean = float(moment_row[index * 2])
            scale = float(moment_row[index * 2 + 1])
            means[column] = mean
            scales[column] = scale if math.isfinite(scale) and scale > 0.0 else 1.0

        vocabularies: dict[str, tuple[str, ...]] = {}
        for column in selected_contract.low_cardinality_categories:
            expression = _duckdb_normalised_category(column)
            rows = connection.execute(
                f"SELECT {expression} AS value {source_sql} GROUP BY value ORDER BY value",
                parameters,
            ).fetchall()
            vocabularies[column] = tuple(str(row[0]) for row in rows)

        frequencies: dict[str, dict[str, float]] = {}
        for column in selected_contract.bank_frequency_categories:
            expression = _duckdb_normalised_category(column)
            rows = connection.execute(
                f"SELECT {expression} AS value, count(*) AS frequency {source_sql} "
                "GROUP BY value ORDER BY value",
                parameters,
            ).fetchall()
            frequencies[column] = {str(value): float(count) / train_rows for value, count in rows}

        return cls(
            contract=selected_contract,
            numeric_medians=medians,
            numeric_means=means,
            numeric_scales=scales,
            category_vocabularies=vocabularies,
            bank_frequencies=frequencies,
            fitted_train_rows=train_rows,
            fit_source="duckdb_frozen_train_partition",
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names in exact transformed-matrix column order."""

        numeric = tuple(f"numeric__{name}" for name in self.contract.numeric_features)
        indicators = tuple(f"missing__{name}" for name in self.contract.missing_indicator_features)
        one_hot = tuple(
            f"onehot__{column}={value}"
            for column in self.contract.low_cardinality_categories
            for value in self.category_vocabularies[column]
        )
        bank_frequency = tuple(
            f"frequency__{column}" for column in self.contract.bank_frequency_categories
        )
        return (*numeric, *indicators, *one_hot, *bank_frequency)

    def transform_pandas(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform without mutating or expanding fitted state."""

        _check_required_columns(frame, self.contract)
        matrix = np.zeros((len(frame), len(self.feature_names)), dtype=np.float32)

        offset = 0
        missing_by_column: dict[str, np.ndarray] = {}
        for column in self.contract.numeric_features:
            values = _numeric_values(frame, column)
            missing = np.isnan(values)
            if column in self.contract.missing_indicator_features:
                missing_by_column[column] = missing
            values = np.where(missing, self.numeric_medians[column], values)
            transformed = (values - self.numeric_means[column]) / self.numeric_scales[column]
            if not np.isfinite(transformed).all():
                raise PreprocessingError(
                    f"Numeric transform for {column!r} produced non-finite values"
                )
            matrix[:, offset] = transformed.astype(np.float32, copy=False)
            offset += 1

        for column in self.contract.missing_indicator_features:
            matrix[:, offset] = missing_by_column[column].astype(np.float32, copy=False)
            offset += 1

        row_indices = np.arange(len(frame))
        for column in self.contract.low_cardinality_categories:
            vocabulary = self.category_vocabularies[column]
            codes = pd.Categorical(_normalise_category(frame[column]), categories=vocabulary).codes
            known = codes >= 0
            matrix[row_indices[known], offset + codes[known]] = 1.0
            offset += len(vocabulary)

        for column in self.contract.bank_frequency_categories:
            mapped = _normalise_category(frame[column]).map(self.bank_frequencies[column])
            matrix[:, offset] = mapped.fillna(0.0).to_numpy(dtype=np.float32)
            offset += 1

        if offset != matrix.shape[1]:
            raise RuntimeError("Internal transformed feature offset mismatch")
        return matrix

    transform = transform_pandas

    def transform_batch(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str = "is_laundering",
    ) -> TransformedBatch:
        """Transform predictors and retain row identity/target outside the matrix."""

        required_metadata = {"transaction_id", "source_row_number", target_column}
        missing = sorted(required_metadata.difference(frame.columns))
        if missing:
            raise PreprocessingError(f"Batch metadata columns are missing: {missing}")
        target = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype="int8")
        if not np.isin(target, (0, 1)).all():
            raise PreprocessingError("Target values must be binary 0/1")
        return TransformedBatch(
            matrix=self.transform_pandas(frame),
            target=target,
            transaction_ids=frame["transaction_id"].astype("string").to_numpy(),
            source_row_numbers=pd.to_numeric(frame["source_row_number"], errors="raise").to_numpy(
                dtype="int64"
            ),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "state_version": _STATE_VERSION,
            "contract": self.contract.to_dict(),
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_scales": self.numeric_scales,
            "category_vocabularies": {
                key: list(values) for key, values in self.category_vocabularies.items()
            },
            "bank_frequencies": self.bank_frequencies,
            "fitted_train_rows": self.fitted_train_rows,
            "fit_source": self.fit_source,
        }

    @property
    def state_sha256(self) -> str:
        """Fingerprint exact fitted preprocessing state."""

        return _state_digest(self._payload_without_digest())

    def to_state_dict(self) -> dict[str, Any]:
        """Return complete machine-readable fitted state."""

        payload = self._payload_without_digest()
        payload["state_sha256"] = _state_digest(payload)
        return payload

    def manifest(self) -> dict[str, Any]:
        """Return a compact manifest without high-cardinality bank maps."""

        return {
            "state_version": _STATE_VERSION,
            "state_sha256": self.state_sha256,
            "fit_scope": "train_only",
            "fit_source": self.fit_source,
            "fitted_train_rows": self.fitted_train_rows,
            "contract": self.contract.to_dict(),
            "transformed_feature_count": len(self.feature_names),
            "transformed_feature_names": list(self.feature_names),
            "numeric_imputation": "train_median",
            "numeric_scaling": "train_mean_and_population_standard_deviation_after_imputation",
            "unknown_low_cardinality_policy": "all_zero_one_hot",
            "unknown_bank_policy": "frequency_zero",
            "category_normalization": "strip_then_uppercase; missing uses reserved token",
            "category_vocabularies": {
                key: list(values) for key, values in self.category_vocabularies.items()
            },
            "bank_frequency_category_counts": {
                key: len(values) for key, values in self.bank_frequencies.items()
            },
        }

    def save_state(self, path: str | Path) -> Path:
        """Serialize complete fitted state, including bank frequency maps."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_state_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load_state(cls, path: str | Path) -> FittedPreprocessor:
        """Load fitted state and reject corrupted or incompatible content."""

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreprocessingError(f"Could not load preprocessing state: {exc}") from exc
        expected_digest = payload.pop("state_sha256", None)
        if payload.get("state_version") != _STATE_VERSION:
            raise PreprocessingError("Unsupported preprocessing state version")
        if not isinstance(expected_digest, str) or _state_digest(payload) != expected_digest:
            raise PreprocessingError("Preprocessing state fingerprint mismatch")
        payload.pop("state_version")
        contract = FeatureContract.from_dict(payload.pop("contract"))
        payload["category_vocabularies"] = {
            key: tuple(values) for key, values in payload["category_vocabularies"].items()
        }
        return cls(contract=contract, **payload)


@dataclass(frozen=True)
class TransformedBatch:
    """Bounded model matrix plus identity and labels kept outside predictors."""

    matrix: np.ndarray
    target: np.ndarray
    transaction_ids: np.ndarray
    source_row_numbers: np.ndarray

    def __post_init__(self) -> None:
        rows = self.matrix.shape[0]
        if self.matrix.ndim != 2:
            raise PreprocessingError("Transformed matrix must be two-dimensional")
        if any(len(values) != rows for values in self.metadata_arrays):
            raise PreprocessingError("Transformed batch metadata lengths differ")

    @property
    def metadata_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.target, self.transaction_ids, self.source_row_numbers)

    @property
    def rows(self) -> int:
        return self.matrix.shape[0]


def iter_duckdb_transformed_batches(
    connection: Any,
    preprocessor: FittedPreprocessor,
    feature_parquet: str | Path,
    split_parquet: str | Path,
    partition: str,
    *,
    batch_size: int = 100_000,
) -> Any:
    """Yield deterministic, bounded float32 batches for one frozen partition."""

    if partition not in {"train", "validation", "test"}:
        raise PreprocessingError("partition must be train, validation, or test")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise PreprocessingError("batch_size must be a positive integer")

    feature_path = str(Path(feature_parquet).expanduser().resolve())
    split_path = str(Path(split_parquet).expanduser().resolve())
    bounds = connection.execute(
        "SELECT min(timestamp), max(timestamp), count(*) FROM read_parquet(?) WHERE partition = ?",
        [split_path, partition],
    ).fetchone()
    minimum, maximum, expected_rows = bounds
    if minimum is None or maximum is None or int(expected_rows) <= 0:
        raise PreprocessingError(f"Frozen split partition {partition!r} is empty")

    selected = (
        "transaction_id",
        "source_row_number",
        "is_laundering",
        *preprocessor.contract.predictor_columns,
    )
    projection = ", ".join(_quote_identifier(column) for column in selected)
    query = (
        f"SELECT {projection} FROM read_parquet(?) "
        "WHERE timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp, source_row_number"
    )
    cursor = connection.execute(query, [feature_path, minimum, maximum])
    vectors_per_chunk = max(1, math.ceil(batch_size / _DUCKDB_VECTOR_SIZE))
    actual_rows = 0
    while True:
        frame = cursor.fetch_df_chunk(vectors_per_chunk=vectors_per_chunk)
        if frame.empty:
            break
        actual_rows += len(frame)
        yield preprocessor.transform_batch(frame)
    if actual_rows != int(expected_rows):
        raise PreprocessingError(
            f"Partition row count differs from frozen split: expected={expected_rows}, "
            f"actual={actual_rows}"
        )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_float(value: float) -> str:
    if not math.isfinite(value):
        raise PreprocessingError("Cannot serialize a non-finite SQL numeric literal")
    return format(value, ".17g")


def _duckdb_normalised_category(column: str) -> str:
    identifier = _quote_identifier(column)
    missing = _MISSING_CATEGORY.replace("'", "''")
    return f"coalesce(nullif(upper(trim(CAST({identifier} AS VARCHAR))), ''), '{missing}')"
