"""Leakage-safe graph sampling and node-feature construction for Sprint 4."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from argus.modeling.preprocessing import FittedPreprocessor


class Sprint4DataError(ValueError):
    """Raised when the sampled GraphSAGE data contract is violated."""


_METADATA_COLUMNS = (
    "transaction_id",
    "source_row_number",
    "timestamp",
    "from_node_id",
    "to_node_id",
    "amount_paid",
    "is_laundering",
)


@dataclass(frozen=True)
class NodeFeatureNormalizer:
    """Train-context statistics applied unchanged to later graph views."""

    means: tuple[float, ...]
    scales: tuple[float, ...]
    feature_names: tuple[str, ...] = (
        "log1p_in_degree",
        "log1p_out_degree",
    )

    def __post_init__(self) -> None:
        if len(self.means) != len(self.feature_names) or len(self.scales) != len(
            self.feature_names
        ):
            raise Sprint4DataError("Node normalizer dimensions do not match its feature names")
        if not np.isfinite(self.means).all() or not np.isfinite(self.scales).all():
            raise Sprint4DataError("Node normalizer values must be finite")
        if any(scale <= 0.0 for scale in self.scales):
            raise Sprint4DataError("Node normalizer scales must be positive")

    def transform(self, structural: np.ndarray) -> np.ndarray:
        values = np.asarray(structural, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise Sprint4DataError("Structural node matrix has an invalid shape")
        result = (values - np.asarray(self.means, dtype=np.float32)) / np.asarray(
            self.scales, dtype=np.float32
        )
        if not np.isfinite(result).all():
            raise Sprint4DataError("Normalized node features contain non-finite values")
        return result.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "fit_scope": "sampled_training_context_nodes_with_observed_edges_only",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NodeFeatureNormalizer:
        return cls(
            means=tuple(float(value) for value in payload["means"]),
            scales=tuple(float(value) for value in payload["scales"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
        )


@dataclass(frozen=True)
class GraphView:
    """One directed sampled graph and its deterministic node feature matrix."""

    node_ids: tuple[str, ...]
    node_features: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    node_to_index: Mapping[str, int]
    normalizer: NodeFeatureNormalizer
    context_edge_rows: int

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def feature_count(self) -> int:
        return int(self.node_features.shape[1])


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def sample_context_edges(
    connection: duckdb.DuckDBPyConnection,
    feature_table: str | Path,
    split_table: str | Path,
    *,
    maximum_edges: int,
    expected_population_rows: int,
    maximum_timestamp: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a label-agnostic deterministic subset from outer-train graph edges."""

    if maximum_edges <= 0 or maximum_edges >= expected_population_rows:
        raise Sprint4DataError("Context maximum_edges must be a strict positive subset")
    timestamp_clause = ""
    parameters: list[Any] = [_resolved(feature_table), _resolved(split_table)]
    if maximum_timestamp is not None:
        timestamp_clause = " AND f.timestamp <= CAST(? AS TIMESTAMP)"
        parameters.append(maximum_timestamp)
    population = connection.execute(
        f"""
        SELECT count(*)
        FROM read_parquet(?) AS f
        JOIN read_parquet(?) AS s USING (transaction_id)
        WHERE s.partition = 'train'{timestamp_clause}
        """,
        parameters,
    ).fetchone()[0]
    if int(population) != expected_population_rows:
        raise Sprint4DataError(
            f"Context population changed: expected={expected_population_rows}, actual={population}"
        )
    frame = connection.execute(
        f"""
        SELECT
            f.transaction_id,
            f.source_row_number,
            f.timestamp,
            f.from_node_id,
            f.to_node_id,
            CAST(f.amount_paid AS DOUBLE) AS amount_paid,
            CAST(f.amount_received AS DOUBLE) AS amount_received,
            f.payment_currency,
            f.receiving_currency
        FROM read_parquet(?) AS f
        JOIN read_parquet(?) AS s USING (transaction_id)
        WHERE s.partition = 'train'{timestamp_clause}
        ORDER BY md5(f.transaction_id), f.transaction_id
        LIMIT ?
        """,
        [*parameters, maximum_edges],
    ).fetch_df()
    if len(frame) != maximum_edges or frame["transaction_id"].nunique() != maximum_edges:
        raise Sprint4DataError("Context edge sample is incomplete or contains duplicate IDs")
    if frame[["from_node_id", "to_node_id"]].isna().any(axis=None):
        raise Sprint4DataError("Context edge sample contains empty endpoints")
    return frame, {
        "partition": "train",
        "population_rows": int(population),
        "sampled_rows": len(frame),
        "coverage_fraction": len(frame) / int(population),
        "sampling_method": "md5_order_without_replacement",
        "selection_uses_target": False,
        "maximum_timestamp": maximum_timestamp,
        "minimum_sample_timestamp": frame["timestamp"].min().isoformat(),
        "maximum_sample_timestamp": frame["timestamp"].max().isoformat(),
        "directed": True,
        "repeated_edges_retained": True,
    }


def sample_supervised_training_edges(
    connection: duckdb.DuckDBPyConnection,
    feature_table: str | Path,
    split_table: str | Path,
    preprocessor: FittedPreprocessor,
    *,
    strictly_after: str,
    expected_population_rows: int,
    maximum_negative_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Keep every later train positive and a deterministic bounded negative sample."""

    selected = tuple(dict.fromkeys((*_METADATA_COLUMNS, *preprocessor.contract.predictor_columns)))
    projection = ", ".join(f"f.{_quoted(column)}" for column in selected)
    paths = [_resolved(feature_table), _resolved(split_table), strictly_after]
    counts = connection.execute(
        """
        SELECT count(*), sum(CAST(f.is_laundering AS BIGINT))
        FROM read_parquet(?) AS f
        JOIN read_parquet(?) AS s USING (transaction_id)
        WHERE s.partition = 'train' AND f.timestamp > CAST(? AS TIMESTAMP)
        """,
        paths,
    ).fetchone()
    population, positives = int(counts[0]), int(counts[1])
    if population != expected_population_rows:
        raise Sprint4DataError(
            "Supervised population changed: "
            f"expected={expected_population_rows}, actual={population}"
        )
    if positives <= 0:
        raise Sprint4DataError("Supervised training interval has no positive transactions")
    if maximum_negative_rows <= 0 or maximum_negative_rows >= population - positives:
        raise Sprint4DataError("Training negative sample must be a strict positive subset")
    frame = connection.execute(
        f"""
        WITH eligible AS (
            SELECT {projection}
            FROM read_parquet(?) AS f
            JOIN read_parquet(?) AS s USING (transaction_id)
            WHERE s.partition = 'train' AND f.timestamp > CAST(? AS TIMESTAMP)
        ), sampled_negative AS (
            SELECT * FROM eligible
            WHERE is_laundering = 0
            ORDER BY md5(transaction_id), transaction_id
            LIMIT ?
        )
        SELECT * FROM eligible WHERE is_laundering = 1
        UNION ALL BY NAME
        SELECT * FROM sampled_negative
        ORDER BY timestamp, source_row_number
        """,
        [*paths, maximum_negative_rows],
    ).fetch_df()
    actual_positives = int(frame["is_laundering"].sum())
    actual_negatives = int(len(frame) - actual_positives)
    if actual_positives != positives or actual_negatives != maximum_negative_rows:
        raise Sprint4DataError("Supervised sample does not match its all-positive contract")
    matrix = preprocessor.transform_pandas(frame)
    return (
        frame,
        matrix,
        {
            "partition": "train",
            "population_rows": population,
            "population_positives": positives,
            "population_negatives": population - positives,
            "sampled_rows": len(frame),
            "sampled_positives": actual_positives,
            "sampled_negatives": actual_negatives,
            "all_population_positives_included": True,
            "negative_sampling_method": "md5_order_without_replacement",
            "minimum_timestamp": frame["timestamp"].min().isoformat(),
            "maximum_timestamp": frame["timestamp"].max().isoformat(),
            "strictly_after": strictly_after,
            "validation_rows_used": False,
            "test_rows_used": False,
        },
    )


def validation_nodes(
    connection: duckdb.DuckDBPyConnection,
    feature_table: str | Path,
    split_table: str | Path,
) -> tuple[str, ...]:
    """Return validation endpoint identities without labels or edge aggregation."""

    rows = connection.execute(
        """
        WITH endpoints AS (
            SELECT f.from_node_id AS node_id
            FROM read_parquet(?) AS f JOIN read_parquet(?) AS s USING (transaction_id)
            WHERE s.partition = 'validation'
            UNION
            SELECT f.to_node_id AS node_id
            FROM read_parquet(?) AS f JOIN read_parquet(?) AS s USING (transaction_id)
            WHERE s.partition = 'validation'
        )
        SELECT node_id FROM endpoints ORDER BY node_id
        """,
        [
            _resolved(feature_table),
            _resolved(split_table),
            _resolved(feature_table),
            _resolved(split_table),
        ],
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def iter_validation_frames(
    connection: duckdb.DuckDBPyConnection,
    feature_table: str | Path,
    split_table: str | Path,
    preprocessor: FittedPreprocessor,
    *,
    batch_rows: int,
) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Yield exact outer-validation rows in chronological order; test is impossible."""

    if batch_rows <= 0:
        raise Sprint4DataError("batch_rows must be positive")
    selected = tuple(dict.fromkeys((*_METADATA_COLUMNS, *preprocessor.contract.predictor_columns)))
    projection = ", ".join(f"f.{_quoted(column)}" for column in selected)
    cursor = connection.execute(
        f"""
        SELECT {projection}
        FROM read_parquet(?) AS f
        JOIN read_parquet(?) AS s USING (transaction_id)
        WHERE s.partition = 'validation'
        ORDER BY f.timestamp, f.source_row_number
        """,
        [_resolved(feature_table), _resolved(split_table)],
    )
    vectors = max(1, math.ceil(batch_rows / 2048))
    while True:
        frame = cursor.fetch_df_chunk(vectors_per_chunk=vectors)
        if frame.empty:
            return
        yield frame, preprocessor.transform_pandas(frame)


def _hash_pair(value: str) -> tuple[float, float]:
    integer = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
    angle = (integer / float(2**64)) * (2.0 * math.pi)
    return math.sin(angle), math.cos(angle)


def _structural_features(
    node_count: int,
    sources: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    incoming_degree = np.bincount(targets, minlength=node_count).astype(np.float64)
    outgoing_degree = np.bincount(sources, minlength=node_count).astype(np.float64)
    return np.column_stack(
        (
            np.log1p(incoming_degree),
            np.log1p(outgoing_degree),
        )
    ).astype(np.float32)


def build_graph_view(
    context_edges: pd.DataFrame,
    *,
    required_node_ids: Sequence[str],
    normalizer: NodeFeatureNormalizer | None = None,
) -> GraphView:
    """Create a directed graph view; fit normalization only on context-observed nodes."""

    required = {"from_node_id", "to_node_id"}
    missing = sorted(required.difference(context_edges.columns))
    if missing:
        raise Sprint4DataError(f"Context edges are missing columns: {missing}")
    endpoint_values = np.concatenate(
        (
            context_edges["from_node_id"].astype(str).to_numpy(),
            context_edges["to_node_id"].astype(str).to_numpy(),
            np.asarray(tuple(required_node_ids), dtype=object),
        )
    )
    if endpoint_values.size == 0 or any(not str(value).strip() for value in endpoint_values):
        raise Sprint4DataError("Graph node identifiers must be non-empty")
    node_ids_array = np.unique(endpoint_values.astype(str))
    node_ids = tuple(node_ids_array.tolist())
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    sources = context_edges["from_node_id"].astype(str).map(node_to_index).to_numpy(np.int64)
    targets = context_edges["to_node_id"].astype(str).map(node_to_index).to_numpy(np.int64)
    structural = _structural_features(len(node_ids), sources, targets)
    observed = (structural[:, 0] > 0.0) | (structural[:, 1] > 0.0)
    if not np.any(observed):
        raise Sprint4DataError("Sampled context graph has no observed nodes")
    if normalizer is None:
        means = structural[observed].mean(axis=0, dtype=np.float64)
        scales = structural[observed].std(axis=0, dtype=np.float64)
        scales[scales == 0.0] = 1.0
        normalizer = NodeFeatureNormalizer(tuple(means.tolist()), tuple(scales.tolist()))
    normalized = normalizer.transform(structural)
    identity = np.empty((len(node_ids), 4), dtype=np.float32)
    for index, node_id in enumerate(node_ids):
        bank, separator, account = node_id.partition("::")
        if not separator or not bank or not account:
            raise Sprint4DataError(f"Invalid composite node identity: {node_id!r}")
        identity[index, :2] = _hash_pair(bank)
        identity[index, 2:] = _hash_pair(account)
    features = np.concatenate((normalized, identity), axis=1).astype(np.float32, copy=False)
    return GraphView(
        node_ids=node_ids,
        node_features=features,
        edge_sources=sources,
        edge_targets=targets,
        node_to_index=node_to_index,
        normalizer=normalizer,
        context_edge_rows=len(context_edges),
    )


def endpoint_indices(
    frame: pd.DataFrame, node_to_index: Mapping[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Map transaction endpoints into a graph view with actionable OOV errors."""

    sender = frame["from_node_id"].astype(str).map(node_to_index)
    receiver = frame["to_node_id"].astype(str).map(node_to_index)
    if sender.isna().any() or receiver.isna().any():
        raise Sprint4DataError("Graph view omitted required transaction endpoint nodes")
    return sender.to_numpy(np.int64), receiver.to_numpy(np.int64)


__all__ = [
    "GraphView",
    "NodeFeatureNormalizer",
    "Sprint4DataError",
    "build_graph_view",
    "endpoint_indices",
    "iter_validation_frames",
    "sample_context_edges",
    "sample_supervised_training_edges",
    "validation_nodes",
]
