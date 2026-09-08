"""Frozen, one-shot final-test inference and artifact publication.

The module has deliberately narrow powers.  It can load already-fitted models,
apply already-fitted preprocessing state, and evaluate already-frozen raw-score
thresholds.  It exposes no training, model-selection, threshold-search, or
feature-selection entry point.

The final partition is first mentioned in SQL only after an exclusive opening
receipt has been created with ``O_EXCL``.  A receipt is never removed, even if
inference or artifact publication fails, so a failed attempt cannot silently be
retried.
"""

from __future__ import annotations

import math
import os
import platform
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import torch

from argus.config import get_path, load_config
from argus.final_evaluation.contract import (
    FinalEvaluationContractError,
    build_freeze_contract,
    create_exclusive_access_receipt,
    load_json_object,
    mapping,
    project_path,
)
from argus.final_evaluation.evaluation import evaluate_frozen_models, stable_sigmoid
from argus.gnn.adjacency import build_directed_mean_adjacency
from argus.gnn.determinism import configure_deterministic_cpu
from argus.gnn.model import GraphSAGEEdgeClassifier
from argus.modeling.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_fingerprint,
    write_run_manifest,
)
from argus.modeling.baseline import _configure_duckdb
from argus.modeling.preprocessing import FittedPreprocessor, feature_contract_for_family
from argus.sprint4.graph_data import (
    NodeFeatureNormalizer,
    build_graph_view,
    endpoint_indices,
)

_FROZEN_CHAMPION = "graph_enhanced_lightgbm"
_MODEL_NAMES = (
    "graph_enhanced_lightgbm",
    "refined_transaction_lightgbm",
    "graphsage_edge_classifier",
)
_TREE_MODEL_NAMES = _MODEL_NAMES[:2]
_MODEL_VERSIONS = {
    "graph_enhanced_lightgbm": "Sprint 3 graph-enhanced LightGBM (frozen champion)",
    "refined_transaction_lightgbm": "Sprint 3 refined transaction LightGBM",
    "graphsage_edge_classifier": "Sprint 4 sampled GraphSAGE edge classifier",
}
_COMPLETION_FILENAME = "FINAL_TEST_COMPLETED.json"
_ROOT_REPORT_FILENAME = "FINAL_EVALUATION_STATUS.md"
_WORK_DIRECTORY_NAME = ".one_shot_work"
_DUCKDB_VECTOR_SIZE = 2048


class FinalEvaluationError(RuntimeError):
    """Raised when the authorized one-shot final evaluation cannot complete."""


@dataclass(frozen=True)
class FinalEvaluationRunResult:
    """Locations and immutable manifest for the completed final evaluation."""

    run_dir: Path
    manifest_path: Path
    report_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _FrozenRuntime:
    tree_models: Mapping[str, lightgbm.LGBMClassifier]
    preprocessors: Mapping[str, FittedPreprocessor]
    graphsage_model: GraphSAGEEdgeClassifier
    context_edges: pd.DataFrame
    node_normalizer: NodeFeatureNormalizer
    inference_manifest: Mapping[str, Any]
    training_manifest: Mapping[str, Any]
    determinism: Mapping[str, Any]


@dataclass(frozen=True)
class _ScoredFinalTest:
    metadata: pd.DataFrame
    raw_scores: Mapping[str, np.ndarray]
    identity_audit: Mapping[str, Any]
    graph_audit: Mapping[str, Any]
    scoring_audit: Mapping[str, Any]


class _StageRecorder:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.current_stage = "preflight"

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        self.current_stage = name
        started = time.perf_counter()
        print(f"[final-evaluation] {name} ...", flush=True)
        try:
            yield
        except BaseException:
            elapsed = time.perf_counter() - started
            self.seconds[name] = elapsed
            print(f"[final-evaluation] {name} FAILED after {elapsed:.2f}s", flush=True)
            raise
        elapsed = time.perf_counter() - started
        self.seconds[name] = elapsed
        print(f"[final-evaluation] {name} completed in {elapsed:.2f}s", flush=True)


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def _run_child(run_dir: Path, value: object, name: str) -> Path:
    """Resolve a configured run artifact while preventing path traversal."""

    if not isinstance(value, str) or not value.strip():
        raise FinalEvaluationError(f"{name} must be a non-empty relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise FinalEvaluationError(f"{name} must be relative to the Sprint 5 run directory")
    result = (run_dir / raw).resolve()
    try:
        result.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise FinalEvaluationError(f"{name} escapes the Sprint 5 run directory") from exc
    return result


def _assert_fresh_one_shot_target(
    run_dir: Path,
    *,
    freeze_path: Path,
    receipt_path: Path,
) -> None:
    """Refuse materialized prior attempts without deleting or replacing evidence."""

    if receipt_path.exists():
        raise FinalEvaluationError(
            "FINAL_TEST_OPENED.json already exists; final-test authorization was consumed "
            "and this run cannot be retried"
        )
    if not run_dir.exists():
        return
    permitted = {freeze_path.resolve()}
    unexpected = sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != ".gitkeep" and path.resolve() not in permitted
    )
    if unexpected:
        raise FinalEvaluationError(
            "Sprint 5 contains prior materialized output but no opening receipt; refusing "
            f"to overwrite or auto-clean it: {unexpected[:5]}"
        )


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load a hash-verified checkpoint using the safest supported Torch API."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility for older supported Torch
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise FinalEvaluationError("Frozen GraphSAGE checkpoint is not a mapping")
    return checkpoint


def _load_frozen_runtime(
    connection: duckdb.DuckDBPyConnection,
    config: Mapping[str, Any],
    freeze_contract: Mapping[str, Any],
) -> _FrozenRuntime:
    """Deserialize only hash-verified, pre-test artifacts.

    The sole Parquet query in this pre-receipt step reads the frozen train-only
    GraphSAGE context artifact.  Neither the feature store nor split manifest is
    queried here.
    """

    settings = mapping(config["sprint5"], "sprint5")
    models = mapping(settings["models"], "sprint5.models")
    contract_models = mapping(freeze_contract["models"], "freeze_contract.models")
    preprocessors: dict[str, FittedPreprocessor] = {}
    state_cache: dict[str, FittedPreprocessor] = {}
    tree_models: dict[str, lightgbm.LGBMClassifier] = {}

    for name in _MODEL_NAMES:
        contract_model = mapping(contract_models[name], f"freeze_contract.models.{name}")
        paths = mapping(contract_model["resolved_paths"], f"freeze_contract.models.{name}.paths")
        state_path = Path(str(paths["preprocessor_state"])).resolve()
        cache_key = str(state_path)
        preprocessor = state_cache.get(cache_key)
        if preprocessor is None:
            preprocessor = FittedPreprocessor.load_state(state_path)
            state_cache[cache_key] = preprocessor
        if preprocessor.fitted_train_rows != 3_554_957:
            raise FinalEvaluationError(f"Frozen preprocessor row count changed for {name}")
        configured_family = str(mapping(models[name], f"sprint5.models.{name}")["feature_family"])
        preprocessor_family = (
            "transaction_temporal_history"
            if name == "graphsage_edge_classifier"
            else configured_family
        )
        if preprocessor.contract != feature_contract_for_family(preprocessor_family):
            raise FinalEvaluationError(f"Frozen preprocessor feature family changed for {name}")
        preprocessors[name] = preprocessor

        if name in _TREE_MODEL_NAMES:
            estimator = joblib.load(Path(str(paths["model"])))
            if not isinstance(estimator, lightgbm.LGBMClassifier):
                raise FinalEvaluationError(f"Frozen estimator is not LightGBM for {name}")
            if not getattr(estimator, "fitted_", False):
                raise FinalEvaluationError(f"Frozen LightGBM estimator is not fitted for {name}")
            if int(estimator.n_features_in_) != len(preprocessor.feature_names):
                raise FinalEvaluationError(
                    f"Frozen estimator/preprocessor width mismatch for {name}: "
                    f"model={estimator.n_features_in_}, "
                    f"preprocessor={len(preprocessor.feature_names)}"
                )
            tree_models[name] = estimator

    graph_contract = mapping(
        contract_models["graphsage_edge_classifier"],
        "freeze_contract.models.graphsage_edge_classifier",
    )
    checkpoint_path = Path(str(mapping(graph_contract["resolved_paths"], "graph paths")["model"]))
    graph_model = GraphSAGEEdgeClassifier.from_checkpoint(
        _load_torch_checkpoint(checkpoint_path), map_location="cpu"
    )
    transaction_preprocessor = preprocessors["graphsage_edge_classifier"]
    if graph_model.config.transaction_feature_dim != len(transaction_preprocessor.feature_names):
        raise FinalEvaluationError("GraphSAGE transaction feature width changed")

    graph_contract_paths = mapping(
        mapping(freeze_contract["graphsage_inference"], "freeze graph")["resolved_paths"],
        "freeze graph paths",
    )
    inference_manifest = load_json_object(
        Path(str(graph_contract_paths["inference_manifest"])), "GraphSAGE inference manifest"
    )
    training_manifest = load_json_object(
        Path(str(graph_contract_paths["training_graph_manifest"])),
        "GraphSAGE training graph manifest",
    )
    if inference_manifest.get("normalizer") != training_manifest.get("normalizer"):
        raise FinalEvaluationError("Frozen GraphSAGE normalizer differs across manifests")
    node_normalizer = NodeFeatureNormalizer.from_dict(
        mapping(inference_manifest.get("normalizer"), "inference_manifest.normalizer")
    )
    if graph_model.config.node_input_dim != 6:
        raise FinalEvaluationError("Frozen GraphSAGE node input width changed")

    context_path = Path(str(graph_contract_paths["context"])).resolve()
    context_edges = connection.execute(
        """
        SELECT from_node_id, to_node_id, source_row_number
        FROM read_parquet(?)
        ORDER BY source_row_number
        """,
        [_resolved(context_path)],
    ).fetch_df()
    expected_context_rows = int(
        mapping(settings["graphsage_inference"], "sprint5.graphsage_inference")["context_max_edges"]
    )
    if len(context_edges) != expected_context_rows:
        raise FinalEvaluationError(
            "Frozen GraphSAGE message context row count changed: "
            f"expected={expected_context_rows}, actual={len(context_edges)}"
        )
    if context_edges[["from_node_id", "to_node_id"]].isna().any(axis=None):
        raise FinalEvaluationError("Frozen GraphSAGE context contains empty endpoint identities")

    determinism = configure_deterministic_cpu(
        int(mapping(config["project"], "project")["random_seed"]),
        num_threads=int(settings["torch_threads"]),
    )
    graph_model.eval()
    return _FrozenRuntime(
        tree_models=tree_models,
        preprocessors=preprocessors,
        graphsage_model=graph_model,
        context_edges=context_edges,
        node_normalizer=node_normalizer,
        inference_manifest=inference_manifest,
        training_manifest=training_manifest,
        determinism=determinism,
    )


def _timestamp_iso(value: object) -> str:
    if value is None or pd.isna(value):
        raise FinalEvaluationError("Final-test timestamp bound is missing")
    return pd.Timestamp(value).isoformat()


def _audit_exact_test_join(
    connection: duckdb.DuckDBPyConnection,
    feature_table: Path,
    split_table: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove exact final membership and one-to-one feature identity after access opens."""

    feature_path = _resolved(feature_table)
    split_path = _resolved(split_table)
    split_row = connection.execute(
        """
        SELECT
            count(*) AS rows,
            count(DISTINCT transaction_id) AS unique_transaction_ids,
            min(timestamp) AS minimum_timestamp,
            max(timestamp) AS maximum_timestamp
        FROM read_parquet(?)
        WHERE partition = 'test'
        """,
        [split_path],
    ).fetchone()
    expected_rows = int(expected["rows"])
    if int(split_row[0]) != expected_rows or int(split_row[1]) != expected_rows:
        raise FinalEvaluationError(
            "Frozen test split membership is incomplete or duplicated: "
            f"rows={split_row[0]}, unique_ids={split_row[1]}, expected={expected_rows}"
        )

    match_row = connection.execute(
        """
        WITH test_split AS (
            SELECT transaction_id
            FROM read_parquet(?)
            WHERE partition = 'test'
        ), feature_matches AS (
            SELECT s.transaction_id, count(f.transaction_id) AS feature_match_count
            FROM test_split AS s
            LEFT JOIN read_parquet(?) AS f
              ON f.transaction_id = s.transaction_id
            GROUP BY s.transaction_id
        )
        SELECT
            count(*) FILTER (WHERE feature_match_count = 0) AS missing_feature_rows,
            count(*) FILTER (WHERE feature_match_count > 1) AS duplicate_feature_matches
        FROM feature_matches
        """,
        [split_path, feature_path],
    ).fetchone()
    if int(match_row[0]) or int(match_row[1]):
        raise FinalEvaluationError(
            "Test split to feature-store identity is not one-to-one: "
            f"missing={match_row[0]}, duplicate_matches={match_row[1]}"
        )

    joined_row = connection.execute(
        """
        SELECT
            count(*) AS rows,
            count(DISTINCT f.transaction_id) AS unique_transaction_ids,
            count(DISTINCT f.source_row_number) AS unique_source_rows,
            sum(CAST(f.is_laundering AS BIGINT)) AS positive_labels,
            count(*) FILTER (
                WHERE f.is_laundering IS NULL OR f.is_laundering NOT IN (0, 1)
            ) AS invalid_labels,
            count(*) FILTER (WHERE f.source_row_number IS NULL) AS empty_source_rows,
            count(*) FILTER (
                WHERE f.timestamp IS DISTINCT FROM s.timestamp
            ) AS timestamp_mismatches,
            min(f.timestamp) AS minimum_timestamp,
            max(f.timestamp) AS maximum_timestamp
        FROM read_parquet(?) AS f
        INNER JOIN read_parquet(?) AS s
          ON f.transaction_id = s.transaction_id
        WHERE s.partition = 'test'
        """,
        [feature_path, split_path],
    ).fetchone()
    expected_positives = int(expected["positives"])
    actual_minimum = _timestamp_iso(joined_row[7])
    actual_maximum = _timestamp_iso(joined_row[8])
    errors = {
        "joined_rows": int(joined_row[0]) != expected_rows,
        "unique_transaction_ids": int(joined_row[1]) != expected_rows,
        "unique_source_rows": int(joined_row[2]) != expected_rows,
        "positive_labels": int(joined_row[3]) != expected_positives,
        "invalid_labels": int(joined_row[4]) != 0,
        "empty_source_rows": int(joined_row[5]) != 0,
        "timestamp_mismatches": int(joined_row[6]) != 0,
        "minimum_timestamp": actual_minimum != str(expected["minimum_timestamp"]),
        "maximum_timestamp": actual_maximum != str(expected["maximum_timestamp"]),
    }
    failed = sorted(key for key, mismatch in errors.items() if mismatch)
    if failed:
        raise FinalEvaluationError(f"Exact final-test identity audit failed: {failed}")
    return {
        "schema": "argus.final_evaluation.test_identity_audit.v1",
        "status": "PASS",
        "evaluation_partition": "test",
        "membership_key": "transaction_id",
        "feature_identity_key": "source_row_number",
        "exact_split_membership_join_performed_inside_authorized_one_shot_run": True,
        "selection_method": "inner_join_feature_store_to_frozen_split_manifest_by_transaction_id",
        "timestamp_range_filter_used_as_partition_substitute": False,
        "split_manifest_rows": int(split_row[0]),
        "split_manifest_unique_transaction_ids": int(split_row[1]),
        "feature_join_rows": int(joined_row[0]),
        "feature_join_unique_transaction_ids": int(joined_row[1]),
        "feature_join_unique_source_rows": int(joined_row[2]),
        "positive_labels": int(joined_row[3]),
        "negative_labels": int(joined_row[0]) - int(joined_row[3]),
        "minimum_timestamp": actual_minimum,
        "maximum_timestamp": actual_maximum,
        "missing_feature_rows": int(match_row[0]),
        "duplicate_feature_matches": int(match_row[1]),
        "timestamp_mismatches": int(joined_row[6]),
        "raw_test_reopened_by_post_run_verifier": False,
    }


def _test_endpoint_identities(
    connection: duckdb.DuckDBPyConnection,
    feature_table: Path,
    split_table: Path,
) -> tuple[str, ...]:
    """Read endpoint identities, but no labels or test edges for message passing."""

    rows = connection.execute(
        """
        WITH test_endpoints AS (
            SELECT f.from_node_id AS node_id
            FROM read_parquet(?) AS f
            INNER JOIN read_parquet(?) AS s
              ON f.transaction_id = s.transaction_id
            WHERE s.partition = 'test'
            UNION
            SELECT f.to_node_id AS node_id
            FROM read_parquet(?) AS f
            INNER JOIN read_parquet(?) AS s
              ON f.transaction_id = s.transaction_id
            WHERE s.partition = 'test'
        )
        SELECT node_id
        FROM test_endpoints
        ORDER BY node_id
        """,
        [
            _resolved(feature_table),
            _resolved(split_table),
            _resolved(feature_table),
            _resolved(split_table),
        ],
    ).fetchall()
    result = tuple(str(row[0]) for row in rows)
    if not result or any(not value.strip() for value in result):
        raise FinalEvaluationError("Final-test endpoint identity set is empty or invalid")
    return result


def _score_tree_raw(
    estimator: lightgbm.LGBMClassifier,
    matrix: np.ndarray,
    *,
    model_name: str,
) -> np.ndarray:
    values = np.asarray(estimator.predict(matrix, raw_score=True), dtype=np.float64).reshape(-1)
    if values.size != len(matrix) or not np.isfinite(values).all():
        raise FinalEvaluationError(f"Frozen raw LightGBM scores are invalid for {model_name}")
    return values


def _score_final_test(
    connection: duckdb.DuckDBPyConnection,
    config: Mapping[str, Any],
    freeze_contract: Mapping[str, Any],
    runtime: _FrozenRuntime,
) -> _ScoredFinalTest:
    """Score all three frozen models while streaming one exact test-membership query."""

    settings = mapping(config["sprint5"], "sprint5")
    project_root = Path(str(mapping(config["_meta"], "_meta")["project_root"])).resolve()
    frozen = mapping(settings["frozen_references"], "sprint5.frozen_references")
    feature_table = project_path(
        project_root,
        mapping(frozen["feature_store"], "frozen.feature_store")["path"],
        "sprint5.frozen_references.feature_store.path",
    )
    split_table = project_path(
        project_root,
        mapping(frozen["split_manifest"], "frozen.split_manifest")["path"],
        "sprint5.frozen_references.split_manifest.path",
    )
    expected = mapping(
        mapping(freeze_contract["test_partition_metadata_only"], "freeze test"),
        "freeze test",
    )
    identity_audit = _audit_exact_test_join(connection, feature_table, split_table, expected)

    test_nodes = _test_endpoint_identities(connection, feature_table, split_table)
    graph_view = build_graph_view(
        runtime.context_edges,
        required_node_ids=test_nodes,
        normalizer=runtime.node_normalizer,
    )
    if graph_view.feature_count != runtime.graphsage_model.config.node_input_dim:
        raise FinalEvaluationError("Frozen GraphSAGE node feature width changed at final inference")
    edge_index = torch.from_numpy(
        np.vstack((graph_view.edge_sources, graph_view.edge_targets))
    ).long()
    adjacency = build_directed_mean_adjacency(edge_index, num_nodes=graph_view.node_count)
    node_features = torch.from_numpy(graph_view.node_features)
    runtime.graphsage_model.eval()
    with torch.no_grad():
        node_embeddings = runtime.graphsage_model.encode_nodes(
            adjacency, node_features=node_features
        )

    selected_columns: list[str] = [
        "transaction_id",
        "source_row_number",
        "timestamp",
        "from_node_id",
        "to_node_id",
        "is_laundering",
    ]
    for name in _MODEL_NAMES:
        for column in runtime.preprocessors[name].contract.predictor_columns:
            if column not in selected_columns:
                selected_columns.append(column)
    projection = ", ".join(f"f.{_quoted(column)}" for column in selected_columns)
    cursor = connection.execute(
        f"""
        SELECT {projection}
        FROM read_parquet(?) AS f
        INNER JOIN read_parquet(?) AS s
          ON f.transaction_id = s.transaction_id
        WHERE s.partition = 'test'
        ORDER BY f.timestamp, f.source_row_number
        """,
        [_resolved(feature_table), _resolved(split_table)],
    )
    vectors_per_chunk = max(1, math.ceil(int(settings["batch_rows"]) / _DUCKDB_VECTOR_SIZE))
    metadata_chunks: list[pd.DataFrame] = []
    score_chunks: dict[str, list[np.ndarray]] = {name: [] for name in _MODEL_NAMES}
    batch_count = 0
    scored_rows = 0
    expected_rows = int(expected["rows"])
    started = time.perf_counter()
    while True:
        frame = cursor.fetch_df_chunk(vectors_per_chunk=vectors_per_chunk)
        if frame.empty:
            break
        batch_count += 1
        scored_rows += len(frame)
        if scored_rows > expected_rows:
            raise FinalEvaluationError("Final-test scoring stream exceeded frozen row count")
        metadata_chunks.append(
            frame[["transaction_id", "source_row_number", "timestamp", "is_laundering"]].copy()
        )
        for name in _TREE_MODEL_NAMES:
            matrix = runtime.preprocessors[name].transform_pandas(frame)
            score_chunks[name].append(
                _score_tree_raw(runtime.tree_models[name], matrix, model_name=name)
            )

        transaction_matrix = runtime.preprocessors["graphsage_edge_classifier"].transform_pandas(
            frame
        )
        sender, receiver = endpoint_indices(frame, graph_view.node_to_index)
        with torch.no_grad():
            logits = runtime.graphsage_model.classify_edges(
                node_embeddings,
                torch.from_numpy(sender).long(),
                torch.from_numpy(receiver).long(),
                torch.from_numpy(np.asarray(transaction_matrix, dtype=np.float32)),
            )
        graph_values = logits.detach().cpu().numpy().astype(np.float64, copy=False)
        if graph_values.size != len(frame) or not np.isfinite(graph_values).all():
            raise FinalEvaluationError("Frozen GraphSAGE final-test logits are invalid")
        score_chunks["graphsage_edge_classifier"].append(graph_values)
        print(
            f"[final-evaluation] all three models scored {scored_rows:,}/{expected_rows:,}",
            flush=True,
        )

    if scored_rows != expected_rows or not metadata_chunks:
        raise FinalEvaluationError(
            f"Final-test scoring row mismatch: expected={expected_rows}, actual={scored_rows}"
        )
    metadata = pd.concat(metadata_chunks, ignore_index=True)
    labels = pd.to_numeric(metadata["is_laundering"], errors="raise").to_numpy(np.int8)
    source_rows = pd.to_numeric(metadata["source_row_number"], errors="raise").to_numpy(np.int64)
    if int(labels.sum()) != int(expected["positives"]):
        raise FinalEvaluationError("Final-test positive count changed during scoring")
    if metadata["transaction_id"].nunique(dropna=False) != expected_rows:
        raise FinalEvaluationError("Final-test transaction identities are not unique")
    if np.unique(source_rows).size != expected_rows:
        raise FinalEvaluationError("Final-test source row identities are not unique")
    if _timestamp_iso(metadata["timestamp"].min()) != str(expected["minimum_timestamp"]):
        raise FinalEvaluationError("Final-test minimum timestamp changed during scoring")
    if _timestamp_iso(metadata["timestamp"].max()) != str(expected["maximum_timestamp"]):
        raise FinalEvaluationError("Final-test maximum timestamp changed during scoring")

    raw_scores = {
        name: np.concatenate(score_chunks[name]).astype(np.float64, copy=False)
        for name in _MODEL_NAMES
    }
    if any(values.size != expected_rows for values in raw_scores.values()):
        raise FinalEvaluationError("One or more frozen models did not score every final row")
    context_nodes = set(runtime.context_edges["from_node_id"].astype(str)).union(
        runtime.context_edges["to_node_id"].astype(str)
    )
    appended_test_nodes = len(set(test_nodes).difference(context_nodes))
    graph_audit = {
        "schema": "argus.final_evaluation.graph_inference.v1",
        "message_context_partition": "train",
        "message_context_rows": len(runtime.context_edges),
        "message_graph_directed": True,
        "repeated_context_edges_retained": True,
        "message_graph_node_count": graph_view.node_count,
        "message_graph_edge_count": adjacency.edge_count,
        "test_endpoint_identity_count": len(test_nodes),
        "test_endpoint_identities_appended": appended_test_nodes,
        "test_endpoint_features": "deterministic_identity_plus_zero_context_degree_when_unseen",
        "normalizer_reused_from_sprint4": True,
        "normalizer_fit_scope": runtime.node_normalizer.to_dict()["fit_scope"],
        "validation_edges_used_for_message_passing": False,
        "test_edges_used_for_message_passing": False,
        "test_labels_used_for_graph_construction": False,
        "unsupported_account_level_label_created": False,
    }
    scoring_audit = {
        "schema": "argus.final_evaluation.scoring_audit.v1",
        "evaluation_partition": "test",
        "rows": scored_rows,
        "batches": batch_count,
        "models": list(_MODEL_NAMES),
        "models_scored_per_batch": 3,
        "same_authorized_one_shot_run": True,
        "exact_split_manifest_join": True,
        "test_rows_used_for_preprocessor_fit": False,
        "test_rows_used_for_training": False,
        "test_rows_used_for_model_selection": False,
        "test_rows_used_for_threshold_selection": False,
        "predict_seconds": time.perf_counter() - started,
    }
    return _ScoredFinalTest(
        metadata=metadata,
        raw_scores=raw_scores,
        identity_audit=identity_audit,
        graph_audit=graph_audit,
        scoring_audit=scoring_audit,
    )


def _write_parquet_atomic(
    frame: pd.DataFrame,
    destination: Path,
    *,
    compression: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    relation = f"argus_final_predictions_{uuid.uuid4().hex}"
    connection = duckdb.connect()
    target = temporary.resolve().as_posix().replace("'", "''")
    try:
        connection.register(relation, frame)
        connection.execute(
            f"COPY (SELECT * FROM {relation} ORDER BY timestamp, source_row_number) "
            f"TO '{target}' (FORMAT PARQUET, COMPRESSION {compression.upper()})"
        )
        connection.unregister(relation)
        connection.close()
        os.replace(temporary, destination)
    finally:
        try:
            connection.close()
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def _prediction_frame(
    scored: _ScoredFinalTest,
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    frame = scored.metadata.copy()
    frame["transaction_id"] = frame["transaction_id"].astype("string")
    frame["source_row_number"] = pd.to_numeric(frame["source_row_number"], errors="raise").astype(
        "int64"
    )
    frame["is_laundering"] = pd.to_numeric(frame["is_laundering"], errors="raise").astype("int8")
    frame.insert(3, "evaluation_partition", "test")
    for name in _MODEL_NAMES:
        raw = np.asarray(scored.raw_scores[name], dtype=np.float64)
        frame[f"raw_score_{name}"] = raw
        frame[f"probability_{name}"] = stable_sigmoid(raw)
        frame[f"alert_{name}"] = raw >= float(thresholds[name])
    expected_columns = {
        "transaction_id",
        "source_row_number",
        "timestamp",
        "evaluation_partition",
        "is_laundering",
        *{
            f"{prefix}_{name}"
            for name in _MODEL_NAMES
            for prefix in ("raw_score", "probability", "alert")
        },
    }
    if set(frame.columns) != expected_columns:
        raise FinalEvaluationError("Internal final prediction schema mismatch")
    return frame


def _prevalence_payload(
    freeze_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    validation = mapping(freeze_contract["validation_partition"], "freeze validation")
    test = mapping(freeze_contract["test_partition_metadata_only"], "freeze test")
    validation_rate = float(validation["positive_rate"])
    test_rate = float(test["positive_rate"])
    payload = {
        "schema": "argus.final_evaluation.prevalence_shift.v1",
        "validation": {
            "rows": int(validation["rows"]),
            "positives": int(validation["positives"]),
            "negatives": int(validation["negatives"]),
            "positive_rate": validation_rate,
            "source": "pre_test_frozen_split_metadata",
        },
        "test": {
            "rows": int(test["rows"]),
            "positives": int(test["positives"]),
            "negatives": int(test["negatives"]),
            "positive_rate": test_rate,
            "source": "authorized_exact_test_identity_audit",
        },
        "absolute_positive_rate_change": test_rate - validation_rate,
        "test_to_validation_positive_rate_ratio": test_rate / validation_rate,
        "split_boundaries_changed": False,
        "prevalence_rebalanced": False,
        "thresholds_adjusted_for_test_prevalence": False,
        "interpretation": (
            "Precision and fixed-threshold alert volume are prevalence-sensitive; the "
            "observed chronological shift was reported without resampling or threshold changes."
        ),
    }
    rows = pd.DataFrame(
        [
            {"partition": "validation", **payload["validation"]},
            {"partition": "test", **payload["test"]},
        ]
    )
    return payload, rows


def _product_comparison(
    comparison: Sequence[Mapping[str, Any]],
    top_k_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    display_k = 1000
    available = {int(row["requested_k"]) for row in top_k_rows}
    if display_k not in available:
        display_k = max(available)
    display = {str(row["model"]): row for row in top_k_rows if int(row["requested_k"]) == display_k}
    records: list[dict[str, Any]] = []
    for original in comparison:
        row = dict(original)
        model = str(row["model"])
        top_k = display[model]
        row.update(
            {
                "version": _MODEL_VERSIONS[model],
                "is_frozen_champion": model == _FROZEN_CHAMPION,
                "fpr": row["false_positive_rate"],
                "alerts": row["alert_count"],
                "recall_at_k": top_k["recall_at_k"],
                "precision_at_k": top_k["precision_at_k"],
                "top_k_for_display": display_k,
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def _atomic_write_text(text: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _render_report(
    comparison: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    lines = [
        "# ARGUS AI - Frozen One-Shot Final Evaluation",
        "",
        "Status: **PASS**",
        "",
        (
            f"The exact untouched test partition contained {summary['test_row_count']:,} "
            f"transactions and {summary['test_positive_count']:,} positive labels "
            f"({summary['test_positive_rate']:.6%})."
        ),
        "",
        "The champion remained graph_enhanced_lightgbm, frozen on validation before test "
        "access. Test metrics were confirmatory only and performed no selection or tuning.",
        "",
        "| Model | Frozen role | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            "| {model} | {role} | {pr_auc:.8f} | {roc_auc:.8f} | {precision:.8f} | "
            "{recall:.8f} | {f1:.8f} | {false_positive_rate:.8f} | {alert_count:,} |".format(**row)
        )
    lines.extend(
        [
            "",
            "All values above were generated from the sealed prediction artifact using "
            "validation-selected raw-score thresholds. No post-test retraining or tuning was "
            "performed.",
            "",
        ]
    )
    return "\n".join(lines)


def _persist_success(
    config: Mapping[str, Any],
    freeze_contract: Mapping[str, Any],
    opening_receipt: Mapping[str, Any],
    scored: _ScoredFinalTest,
    recorder: _StageRecorder,
    *,
    started_at: datetime,
    started_monotonic: float,
) -> FinalEvaluationRunResult:
    settings = mapping(config["sprint5"], "sprint5")
    outputs = mapping(settings["outputs"], "sprint5.outputs")
    run_dir = get_path(config, "run_dir").resolve()
    report_path = get_path(config, "generated_report").resolve()
    product_dir = run_dir / "product"
    product_dir.mkdir(parents=True, exist_ok=True)
    models = mapping(freeze_contract["models"], "freeze_contract.models")
    thresholds = {
        name: float(mapping(models[name], f"freeze model {name}")["frozen_raw_threshold"])
        for name in _MODEL_NAMES
    }
    roles = {
        name: str(mapping(models[name], f"freeze model {name}")["role"]) for name in _MODEL_NAMES
    }
    validation_ap = {
        name: float(mapping(models[name], f"freeze model {name}")["validation_average_precision"])
        for name in _MODEL_NAMES
    }
    labels = scored.metadata["is_laundering"].to_numpy(np.int8)
    source_rows = scored.metadata["source_row_number"].to_numpy(np.int64)
    comparison, top_k_rows, curves = evaluate_frozen_models(
        labels,
        source_rows,
        scored.raw_scores,
        frozen_thresholds=thresholds,
        model_roles=roles,
        validation_average_precision=validation_ap,
        top_k=tuple(int(value) for value in freeze_contract["top_k"]),
        frozen_champion=_FROZEN_CHAMPION,
    )
    comparison_by_model = {str(row["model"]): row for row in comparison}
    champion_metrics = comparison_by_model[_FROZEN_CHAMPION]

    predictions = _prediction_frame(scored, thresholds)
    prediction_path = _run_child(
        run_dir,
        outputs["final_test_predictions_parquet"],
        "sprint5.outputs.final_test_predictions_parquet",
    )
    _write_parquet_atomic(
        predictions,
        prediction_path,
        compression=str(settings["parquet_compression"]),
    )

    comparison_payload = {
        "schema": "argus.final_evaluation.model_comparison.v1",
        "partition": "test",
        "evaluation_role": "one_shot_confirmatory",
        "frozen_champion": _FROZEN_CHAMPION,
        "champion_frozen_before_test": True,
        "presentation_order": "frozen_role_then_model_name_not_test_metric",
        "test_metrics_used_for_model_selection": False,
        "test_metrics_used_for_threshold_selection": False,
        "models": comparison,
    }
    comparison_json_path = _run_child(
        run_dir,
        outputs["final_model_comparison_json"],
        "sprint5.outputs.final_model_comparison_json",
    )
    comparison_csv_path = _run_child(
        run_dir,
        outputs["final_model_comparison_csv"],
        "sprint5.outputs.final_model_comparison_csv",
    )
    atomic_write_json(comparison_payload, comparison_json_path)
    atomic_write_csv(pd.DataFrame(comparison), comparison_csv_path)

    final_metrics_payload = {
        "schema": "argus.final_evaluation.final_metrics.v1",
        "evaluation_partition": "test",
        "evaluation_role": "one_shot_confirmatory",
        "frozen_champion": _FROZEN_CHAMPION,
        "champion_frozen_before_test": True,
        "model_selection_used_test": False,
        "threshold_tuned_on_test": False,
        "post_test_tuning_or_retraining": False,
        "metrics": champion_metrics,
    }
    final_metrics_json_path = _run_child(
        run_dir, outputs["final_metrics_json"], "sprint5.outputs.final_metrics_json"
    )
    final_metrics_csv_path = _run_child(
        run_dir, outputs["final_metrics_csv"], "sprint5.outputs.final_metrics_csv"
    )
    atomic_write_json(final_metrics_payload, final_metrics_json_path)
    atomic_write_csv(pd.DataFrame([champion_metrics]), final_metrics_csv_path)

    top_k_payload = {
        "schema": "argus.final_evaluation.top_k_metrics.v1",
        "evaluation_partition": "test",
        "ranking_order": "raw_score_descending_then_source_row_number_ascending",
        "rows": top_k_rows,
    }
    top_k_json_path = _run_child(
        run_dir,
        outputs["final_top_k_metrics_json"],
        "sprint5.outputs.final_top_k_metrics_json",
    )
    top_k_csv_path = _run_child(
        run_dir,
        outputs["final_top_k_metrics_csv"],
        "sprint5.outputs.final_top_k_metrics_csv",
    )
    atomic_write_json(top_k_payload, top_k_json_path)
    top_k_frame = pd.DataFrame(top_k_rows)
    atomic_write_csv(top_k_frame, top_k_csv_path)

    curves_path = _run_child(
        run_dir,
        outputs["final_pr_curves_csv"],
        "sprint5.outputs.final_pr_curves_csv",
    )
    atomic_write_csv(curves, curves_path)
    confusion = pd.DataFrame(
        [
            {
                "model": row["model"],
                "evaluation_partition": "test",
                "frozen_raw_threshold": row["threshold"],
                "true_negative": row["true_negative"],
                "false_positive": row["false_positive"],
                "false_negative": row["false_negative"],
                "true_positive": row["true_positive"],
            }
            for row in comparison
        ]
    )
    atomic_write_csv(confusion, run_dir / "confusion_matrices.csv")

    threshold_details = {
        "schema": "argus.final_evaluation.frozen_thresholds.v1",
        "selection_partition": "validation",
        "evaluation_partition": "test",
        "threshold_comparison": "raw_score_greater_than_or_equal",
        "test_labels_used_for_threshold_selection": False,
        "models": {
            name: {
                "frozen_raw_threshold": thresholds[name],
                "role": roles[name],
                "source": mapping(models[name], f"freeze model {name}")["threshold_source"],
            }
            for name in _MODEL_NAMES
        },
    }
    atomic_write_json(threshold_details, run_dir / "frozen_threshold_details.json")
    model_metadata = {
        "schema": "argus.final_evaluation.model_metadata.v1",
        "models": {
            name: {
                "role": roles[name],
                "version": _MODEL_VERSIONS[name],
                "feature_family": mapping(models[name], f"freeze model {name}")["feature_family"],
                "model": mapping(models[name], f"freeze model {name}")["model"],
                "preprocessor_state": mapping(models[name], f"freeze model {name}")[
                    "preprocessor_state"
                ],
                "preprocessor_fit_scope": "train_only",
                "training_performed_in_final_evaluation": False,
            }
            for name in _MODEL_NAMES
        },
        "model_selection_partition": "validation",
        "final_evaluation_partition": "test",
        "model_selection_used_test": False,
        "post_test_retraining": False,
    }
    atomic_write_json(model_metadata, run_dir / "model_metadata.json")
    atomic_write_json(dict(scored.identity_audit), run_dir / "test_identity_audit.json")
    atomic_write_json(dict(scored.graph_audit), run_dir / "final_graph_inference_manifest.json")
    atomic_write_json(dict(scored.scoring_audit), run_dir / "scoring_audit.json")

    prevalence, prevalence_frame = _prevalence_payload(freeze_contract)
    atomic_write_json(prevalence, run_dir / "prevalence_shift.json")
    prevalence_json_path = _run_child(
        run_dir,
        outputs["prevalence_comparison_json"],
        "sprint5.outputs.prevalence_comparison_json",
    )
    prevalence_csv_path = _run_child(
        run_dir,
        outputs["prevalence_comparison_csv"],
        "sprint5.outputs.prevalence_comparison_csv",
    )
    atomic_write_json(prevalence, prevalence_json_path)
    atomic_write_csv(prevalence_frame, prevalence_csv_path)

    validation = mapping(freeze_contract["validation_partition"], "freeze validation")
    test = mapping(freeze_contract["test_partition_metadata_only"], "freeze test")
    summary = {
        "schema": "argus.final_evaluation.product_summary.v1",
        "evaluation_partition": "test",
        "final_test_opened": True,
        "one_shot_final_evaluation": True,
        "champion_frozen_before_test": True,
        "test_used_for_model_selection": False,
        "model_selection_used_test": False,
        "tuning_after_test": False,
        "post_test_tuning_performed": False,
        "final_test_access_count": 1,
        "frozen_champion_model": _FROZEN_CHAMPION,
        "test_row_count": int(test["rows"]),
        "test_positive_count": int(test["positives"]),
        "test_negative_count": int(test["negatives"]),
        "test_positive_rate": float(test["positive_rate"]),
        "validation_row_count": int(validation["rows"]),
        "validation_positive_count": int(validation["positives"]),
        "validation_positive_rate": float(validation["positive_rate"]),
        "test_to_validation_prevalence_ratio": prevalence["test_to_validation_positive_rate_ratio"],
        "champion_metrics": champion_metrics,
    }
    summary_path = _run_child(
        run_dir,
        outputs["final_test_summary_json"],
        "sprint5.outputs.final_test_summary_json",
    )
    atomic_write_json(summary, summary_path)
    atomic_write_json(summary, product_dir / "final_test_summary.json")

    product_comparison = _product_comparison(comparison, top_k_rows)
    atomic_write_csv(product_comparison, product_dir / "final_model_comparison.csv")
    atomic_write_csv(curves, product_dir / "final_pr_curves.csv")
    atomic_write_csv(top_k_frame, product_dir / "final_top_k_metrics.csv")

    sprint4_dir = get_path(config, "sprint4_run_dir").resolve()
    relative_validation_root = Path(os.path.relpath(sprint4_dir, start=run_dir)).as_posix()
    validation_reference = {
        "schema": "argus.final_evaluation.validation_reference.v1",
        "validation_artifact_root": relative_validation_root,
        "evaluation_partition": "validation",
        "final_evaluation_partition": "test",
        "validation_metrics_used_for_model_selection": True,
        "test_metrics_used_for_model_selection": False,
        "frozen_champion": _FROZEN_CHAMPION,
        "sprint4_manifest": freeze_contract["frozen_references"]["sprint4_manifest"],
    }
    validation_reference_path = _run_child(
        run_dir,
        outputs["validation_reference_json"],
        "sprint5.outputs.validation_reference_json",
    )
    atomic_write_json(validation_reference, validation_reference_path)
    # The application resolves this value against the Sprint 5 root, even though
    # the reference itself lives below product/.
    atomic_write_json(validation_reference, product_dir / "validation_artifact_reference.json")

    acceptance = {
        "explicit_final_test_authorization_recorded": True,
        "one_shot_access_receipt_created_before_test_query": True,
        "frozen_sprint4_checkpoint_and_artifact_hashes_verified": True,
        "graph_enhanced_lightgbm_frozen_as_pre_test_champion": True,
        "exact_train_fitted_preprocessors_reused": True,
        "exact_validation_selected_thresholds_reused": True,
        "full_untouched_test_partition_scored": True,
        "three_models_scored_in_same_authorized_run": True,
        "all_required_final_metrics_saved": True,
        "prevalence_shift_reported_without_rebalancing": True,
        "graphsage_uses_frozen_train_context_without_test_message_edges": True,
        "machine_readable_artifacts_saved": True,
        "streamlit_final_artifact_contract_saved": True,
        "test_used_for_selection_or_tuning": False,
        "post_test_tuning_or_retraining_performed": False,
        "unsupported_account_level_label_created": False,
    }
    quality_report = {
        "schema": "argus.final_evaluation.quality.v1",
        "status": "PASS"
        if (
            all(
                value
                for key, value in acceptance.items()
                if key
                not in {
                    "test_used_for_selection_or_tuning",
                    "post_test_tuning_or_retraining_performed",
                    "unsupported_account_level_label_created",
                }
            )
            and not any(
                acceptance[key]
                for key in (
                    "test_used_for_selection_or_tuning",
                    "post_test_tuning_or_retraining_performed",
                    "unsupported_account_level_label_created",
                )
            )
        )
        else "FAIL",
        "acceptance": acceptance,
        "independent_saved_artifact_verification": "AVAILABLE_VIA_verify_final_evaluation",
    }
    quality_path = _run_child(
        run_dir, outputs["quality_report_json"], "sprint5.outputs.quality_report_json"
    )
    atomic_write_json(quality_report, quality_path)

    report = _render_report(comparison, summary)
    root_report_path = run_dir / _ROOT_REPORT_FILENAME
    _atomic_write_text(report, root_report_path)
    _atomic_write_text(report, report_path)

    completion = {
        "schema": "argus.final_evaluation.completion_receipt.v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "opening_receipt": file_fingerprint(run_dir / "FINAL_TEST_OPENED.json"),
        "final_test_predictions": file_fingerprint(prediction_path),
        "frozen_champion": _FROZEN_CHAMPION,
        "models_scored": list(_MODEL_NAMES),
        "final_test_row_count": int(test["rows"]),
        "final_test_positive_count": int(test["positives"]),
        "final_test_access_count": 1,
        "retry_permitted": False,
        "tuning_after_test": False,
        "retraining_after_test": False,
    }
    completion_path = run_dir / _COMPLETION_FILENAME
    atomic_write_json(completion, completion_path)

    finished_at = datetime.now(UTC)
    manifest_payload: dict[str, Any] = {
        "schema": "argus.final_evaluation.run_manifest.v1",
        "status": "PASS",
        "sprint": 5,
        "scope": "frozen one-shot final-test evaluation; no selection or tuning",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "runtime_seconds": time.perf_counter() - started_monotonic,
        "runtime_seconds_by_stage": dict(recorder.seconds),
        "configuration": {
            "file": str(mapping(config["_meta"], "_meta")["config_file"]),
            "random_seed": int(mapping(config["project"], "project")["random_seed"]),
            "evaluation_partition": "test",
            "selection_partition": "validation",
            "ranking_score_type": "raw_model_score",
            "ranking_tie_break": settings["ranking_tie_break"],
            "top_k": list(settings["top_k"]),
        },
        "frozen_champion": _FROZEN_CHAMPION,
        "champion_frozen_before_test": True,
        "final_test_execution_count": 1,
        "final_test_reexecution_permitted": False,
        "freeze_contract": file_fingerprint(run_dir / "freeze_contract.json"),
        "opening_receipt": dict(opening_receipt),
        "completion_receipt": completion,
        "test_identity_audit": dict(scored.identity_audit),
        "graph_inference": dict(scored.graph_audit),
        "scoring_audit": dict(scored.scoring_audit),
        "final_model_comparison": comparison,
        "champion_final_metrics": champion_metrics,
        "split_prevalence": prevalence,
        "post_test_actions": {
            "model_selection": False,
            "hyperparameter_tuning": False,
            "feature_selection": False,
            "threshold_selection": False,
            "preprocessing_fit": False,
            "model_retraining": False,
            "prevalence_rebalancing": False,
        },
        "libraries": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "joblib": joblib.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "acceptance": acceptance,
        "core_artifact_paths": [
            "freeze_contract.json",
            "FINAL_TEST_OPENED.json",
            _COMPLETION_FILENAME,
            prediction_path.relative_to(run_dir).as_posix(),
            comparison_json_path.relative_to(run_dir).as_posix(),
            final_metrics_json_path.relative_to(run_dir).as_posix(),
            top_k_json_path.relative_to(run_dir).as_posix(),
            curves_path.relative_to(run_dir).as_posix(),
            "confusion_matrices.csv",
            "frozen_threshold_details.json",
            "model_metadata.json",
            "test_identity_audit.json",
            "prevalence_shift.json",
            "product/final_test_summary.json",
            "product/final_model_comparison.csv",
            "product/final_pr_curves.csv",
            "product/final_top_k_metrics.csv",
            _ROOT_REPORT_FILENAME,
        ],
    }
    manifest_path = _run_child(
        run_dir, outputs["run_manifest_json"], "sprint5.outputs.run_manifest_json"
    )
    manifest = write_run_manifest(
        manifest_payload,
        run_dir,
        manifest_path=manifest_path,
        # These are downstream reports.  They are deliberately outside the
        # immutable inference inventory so publishing verification/quality
        # evidence cannot invalidate or silently re-baseline the run manifest.
        exclude_paths=[
            run_dir / _WORK_DIRECTORY_NAME,
            quality_path,
            run_dir / str(outputs["verification_report_json"]),
        ],
    )
    return FinalEvaluationRunResult(run_dir, manifest_path, report_path, manifest)


def _write_failure_marker(
    config: Mapping[str, Any],
    *,
    receipt_path: Path,
    stage: str,
    error: BaseException,
) -> None:
    """Record a consumed, non-retryable attempt without removing any evidence."""

    run_dir = get_path(config, "run_dir").resolve()
    outputs = mapping(mapping(config["sprint5"], "sprint5")["outputs"], "sprint5.outputs")
    path = _run_child(
        run_dir, outputs["failure_marker_json"], "sprint5.outputs.failure_marker_json"
    )
    payload = {
        "schema": "argus.final_evaluation.failure.v1",
        "status": "FAILED_AFTER_FINAL_TEST_ACCESS_OPENED",
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "opening_receipt": file_fingerprint(receipt_path),
        "final_test_access_count": 1,
        "authorization_consumed": True,
        "retry_permitted": False,
        "receipt_removed": False,
        "partial_artifacts_auto_cleaned": False,
        "operator_action": (
            "Do not rerun. Preserve this directory and begin a separately authorized, "
            "versioned protocol only after documenting the failure."
        ),
    }
    atomic_write_json(payload, path)


def run_final_evaluation(
    config_path: str | Path = "configs/final_evaluation.yaml",
) -> FinalEvaluationRunResult:
    """Run the only authorized final-test inference after freezing all decisions.

    This function intentionally performs no cleanup and has no retry option.  A
    failure after receipt creation writes a durable failure marker and leaves the
    exclusive opening receipt in place.
    """

    started_at = datetime.now(UTC)
    started_monotonic = time.perf_counter()
    recorder = _StageRecorder()
    config = load_config(config_path)
    settings = mapping(config["sprint5"], "sprint5")
    gate = mapping(settings["one_shot_gate"], "sprint5.one_shot_gate")
    run_dir = get_path(config, "run_dir").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = _run_child(
        run_dir, gate["freeze_contract_path"], "sprint5.one_shot_gate.freeze_contract_path"
    )
    receipt_path = _run_child(run_dir, gate["receipt_path"], "sprint5.one_shot_gate.receipt_path")
    _assert_fresh_one_shot_target(run_dir, freeze_path=freeze_path, receipt_path=receipt_path)

    connection: duckdb.DuckDBPyConnection | None = None
    receipt_opened = False
    try:
        with recorder.measure("preflight_freeze_contract_in_memory"):
            freeze_contract = build_freeze_contract(config)

        with recorder.measure("load_frozen_models_and_train_context"):
            connection = duckdb.connect()
            _configure_duckdb(connection, settings, run_dir / _WORK_DIRECTORY_NAME)
            runtime = _load_frozen_runtime(connection, config, freeze_contract)

        with recorder.measure("seal_contract_and_open_exclusive_final_test_access"):
            # All static artifacts have now loaded successfully.  Persist the
            # contract immediately before the exclusive receipt; no SQL above
            # references the feature store, split manifest, or final partition.
            atomic_write_json(freeze_contract, freeze_path)
            opening_receipt = create_exclusive_access_receipt(
                receipt_path,
                freeze_contract_path=freeze_path,
                freeze_contract=freeze_contract,
            )
            receipt_opened = True

        with recorder.measure("score_exact_final_test_once"):
            scored = _score_final_test(connection, config, freeze_contract, runtime)
        connection.close()
        connection = None

        with recorder.measure("publish_immutable_final_artifacts"):
            result = _persist_success(
                config,
                freeze_contract,
                opening_receipt,
                scored,
                recorder,
                started_at=started_at,
                started_monotonic=started_monotonic,
            )
        print(
            "[final-evaluation] PASS: frozen champion remained "
            f"{_FROZEN_CHAMPION}; manifest={result.manifest_path}",
            flush=True,
        )
        return result
    except BaseException as exc:
        if receipt_opened:
            try:
                _write_failure_marker(
                    config,
                    receipt_path=receipt_path,
                    stage=recorder.current_stage,
                    error=exc,
                )
            except BaseException as marker_error:
                print(
                    "[final-evaluation] CRITICAL: failure marker could not be written: "
                    f"{marker_error}",
                    flush=True,
                )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, FinalEvaluationError):
            raise
        if isinstance(exc, FinalEvaluationContractError):
            raise FinalEvaluationError(str(exc)) from exc
        raise FinalEvaluationError(
            f"Final evaluation failed during {recorder.current_stage}: {exc}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()


__all__ = [
    "FinalEvaluationError",
    "FinalEvaluationRunResult",
    "run_final_evaluation",
]
