"""Executable Sprint 4 GraphSAGE experiment and saved-artifact product pipeline.

The final test partition is deliberately inaccessible in this module. GraphSAGE
training uses predeclared deterministic subsets of outer-train edges, while model
comparison scores every row in the frozen outer-validation partition. The two
accepted Sprint 3 LightGBM score vectors are treated as immutable references.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import shutil
import sys
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
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch

from argus.app.artifacts import load_dashboard_artifacts
from argus.config import get_path, load_config
from argus.gnn.adjacency import DirectedMeanAdjacency, build_directed_mean_adjacency
from argus.gnn.determinism import configure_deterministic_cpu
from argus.gnn.explain import explain_edge_inputs
from argus.gnn.model import GraphSAGEConfig, GraphSAGEEdgeClassifier
from argus.modeling.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_fingerprint,
    sha256_file,
    write_run_manifest,
)
from argus.modeling.baseline import _configure_duckdb, _source_snapshot, _verify_frozen_inputs
from argus.modeling.metrics import compute_average_precision
from argus.modeling.preprocessing import FittedPreprocessor
from argus.modeling.refinement import git_is_ancestor, subprocess_checkpoint
from argus.product.cases import build_case_from_frames
from argus.product.schemas import validate_case_payload
from argus.sprint4.evaluation import (
    build_pr_curve_table,
    compare_validation_models,
    graphsage_score_diagnostics,
)
from argus.sprint4.graph_data import (
    GraphView,
    build_graph_view,
    endpoint_indices,
    iter_validation_frames,
    sample_context_edges,
    sample_supervised_training_edges,
    validation_nodes,
)
from argus.sprint4.reporting import render_sprint4_report

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_WORK_PREFIX = ".argus_sprint4_work_"


class Sprint4PipelineError(RuntimeError):
    """Raised when an executable Sprint 4 protocol invariant fails."""


@dataclass(frozen=True)
class Sprint4RunResult:
    """Locations and manifest for one completed offline Sprint 4 run."""

    run_dir: Path
    manifest_path: Path
    report_path: Path
    manifest: dict[str, Any]


class _StageRecorder:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        print(f"[sprint4] {name} ...", flush=True)
        started = time.perf_counter()
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - started
            self.seconds[name] = elapsed
            print(f"[sprint4] {name} FAILED after {elapsed:.2f}s", flush=True)
            raise
        elapsed = time.perf_counter() - started
        self.seconds[name] = elapsed
        print(f"[sprint4] {name} completed in {elapsed:.2f}s", flush=True)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Sprint4PipelineError(f"{name} must be a mapping")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sprint4PipelineError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Sprint4PipelineError(f"{label} must contain a JSON object")
    return payload


def _prepare_run_directory(run_dir: Path) -> None:
    """Clean only generated Sprint 4 content and preserve the tracked marker."""

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir.resolve()
    if resolved.name != "sprint4" or resolved.parent.name != "artifacts":
        raise Sprint4PipelineError(f"Refusing to clean unexpected run directory: {resolved}")
    for child in run_dir.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _safe_cleanup_work_directory(work_dir: Path, run_dir: Path) -> None:
    resolved_work = work_dir.resolve()
    resolved_run = run_dir.resolve()
    if resolved_work.parent != resolved_run or not resolved_work.name.startswith(_WORK_PREFIX):
        raise Sprint4PipelineError(f"Refusing to remove unexpected work directory: {work_dir}")
    if work_dir.exists():
        shutil.rmtree(work_dir)


def _atomic_torch_save(payload: object, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_numpy_save(array: np.ndarray, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _write_frame_parquet(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    destination: Path,
    *,
    compression: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    relation_name = f"argus_frame_{uuid.uuid4().hex}"
    connection.register(relation_name, frame)
    target = temporary.resolve().as_posix().replace("'", "''")
    try:
        connection.execute(
            f"COPY (SELECT * FROM {relation_name}) TO '{target}' "
            f"(FORMAT PARQUET, COMPRESSION {compression.upper()})"
        )
        os.replace(temporary, destination)
    finally:
        connection.unregister(relation_name)
        temporary.unlink(missing_ok=True)
    return destination


def _verify_sprint3_reference(
    project_root: Path,
    sprint3_dir: Path,
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = sprint3_dir / "run_manifest.json"
    actual_sha = sha256_file(manifest_path)
    if actual_sha != frozen["sprint3_manifest_sha256"]:
        raise Sprint4PipelineError(
            "Frozen Sprint 3 manifest hash mismatch: "
            f"expected={frozen['sprint3_manifest_sha256']}, actual={actual_sha}"
        )
    manifest = _load_json(manifest_path, "Sprint 3 run manifest")
    if manifest.get("status") != "PASS" or not all(manifest.get("acceptance", {}).values()):
        raise Sprint4PipelineError("Frozen Sprint 3 run is not fully accepted")
    expected_commit = str(frozen["sprint3_commit"])
    head = subprocess_checkpoint(project_root)
    if head != expected_commit and not git_is_ancestor(project_root, expected_commit, head):
        raise Sprint4PipelineError("Frozen Sprint 3 checkpoint is absent from Git history")
    conclusion = _mapping(
        _mapping(manifest.get("ablation"), "sprint3.ablation").get("graph_value_conclusion"),
        "sprint3.graph_value_conclusion",
    )
    if not math.isclose(
        float(conclusion["b_average_precision"]),
        float(frozen["refined_baseline_average_precision"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        float(conclusion["c_average_precision"]),
        float(frozen["graph_enhanced_average_precision"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise Sprint4PipelineError("Frozen Sprint 3 reference metrics changed")
    return manifest, {
        "sprint3_manifest": file_fingerprint(manifest_path),
        "sprint3_commit": expected_commit,
        "current_git_head": head,
        "checkpoint_is_ancestor": True,
        "reference_metrics_verified": True,
    }


def _graph_adjacency(view: GraphView) -> DirectedMeanAdjacency:
    edge_index = torch.from_numpy(np.vstack((view.edge_sources, view.edge_targets))).long()
    return build_directed_mean_adjacency(edge_index, num_nodes=view.node_count)


def _train_graphsage(
    train_view: GraphView,
    supervised: pd.DataFrame,
    transaction_matrix: np.ndarray,
    settings: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[GraphSAGEEdgeClassifier, DirectedMeanAdjacency, dict[str, Any]]:
    model_settings = _mapping(settings["model"], "sprint4.model")
    determinism = configure_deterministic_cpu(seed, num_threads=int(settings["torch_threads"]))
    adjacency = _graph_adjacency(train_view)
    sender, receiver = endpoint_indices(supervised, train_view.node_to_index)
    senders = torch.from_numpy(sender).long()
    receivers = torch.from_numpy(receiver).long()
    tx_features = torch.from_numpy(np.asarray(transaction_matrix, dtype=np.float32))
    labels = torch.from_numpy(supervised["is_laundering"].to_numpy(np.float32))
    node_features = torch.from_numpy(train_view.node_features)

    config = GraphSAGEConfig(
        num_nodes=train_view.node_count,
        transaction_feature_dim=transaction_matrix.shape[1],
        node_input_dim=train_view.feature_count,
        hidden_dim=int(model_settings["node_hidden_dim"]),
        embedding_dim=int(model_settings["node_embedding_dim"]),
        num_layers=2,
        classifier_hidden_dim=int(model_settings["edge_hidden_dim"]),
        dropout=float(model_settings["dropout"]),
        seed=seed,
    )
    model = GraphSAGEEdgeClassifier(config)
    positives = int(labels.sum().item())
    negatives = int(labels.numel() - positives)
    if positives <= 0 or negatives <= 0:
        raise Sprint4PipelineError("GraphSAGE supervised sample must contain both classes")
    positive_weight = math.sqrt(negatives / positives)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_settings["learning_rate"]),
        weight_decay=float(model_settings["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(model_settings["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            adjacency,
            senders,
            receivers,
            tx_features,
            node_features=node_features,
        )
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise Sprint4PipelineError(f"Non-finite GraphSAGE loss at epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        record = {
            "epoch": epoch,
            "loss": float(loss.detach().item()),
            "gradient_norm_before_clip": float(gradient_norm),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(
            f"[sprint4] epoch={epoch:02d} loss={record['loss']:.7f} "
            f"seconds={record['seconds']:.2f}",
            flush=True,
        )
    model.eval()
    return (
        model,
        adjacency,
        {
            "architecture": config.to_dict(),
            "determinism": determinism,
            "target_unit": "transaction_edge",
            "unsupported_account_level_label_created": False,
            "training_rows": int(labels.numel()),
            "training_positives": positives,
            "training_negatives": negatives,
            "positive_weight": positive_weight,
            "positive_weight_policy": "sqrt_sample_negative_to_positive_ratio",
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "gradient_clip_max_norm": 5.0,
            "epochs_completed": len(history),
            "history": history,
            "fit_seconds": time.perf_counter() - started,
            "outer_validation_used_for_fit_or_early_stopping": False,
            "final_test_used": False,
        },
    )


def _score_full_validation(
    connection: duckdb.DuckDBPyConnection,
    model: GraphSAGEEdgeClassifier,
    inference_view: GraphView,
    preprocessor: FittedPreprocessor,
    feature_table: Path,
    split_table: Path,
    *,
    expected_rows: int,
    expected_positives: int,
    batch_rows: int,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    DirectedMeanAdjacency,
    dict[str, Any],
]:
    adjacency = _graph_adjacency(inference_view)
    node_features = torch.from_numpy(inference_view.node_features)
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode_nodes(adjacency, node_features=node_features)
    source_rows = np.empty(expected_rows, dtype=np.int64)
    labels = np.empty(expected_rows, dtype=np.int8)
    raw_logits = np.empty(expected_rows, dtype=np.float64)
    probabilities = np.empty(expected_rows, dtype=np.float64)
    offset = 0
    batches = 0
    started = time.perf_counter()
    with torch.no_grad():
        for frame, matrix in iter_validation_frames(
            connection,
            feature_table,
            split_table,
            preprocessor,
            batch_rows=batch_rows,
        ):
            stop = offset + len(frame)
            if stop > expected_rows:
                raise Sprint4PipelineError("Validation iterator exceeded frozen row count")
            sender, receiver = endpoint_indices(frame, inference_view.node_to_index)
            logits = model.classify_edges(
                node_embeddings,
                torch.from_numpy(sender).long(),
                torch.from_numpy(receiver).long(),
                torch.from_numpy(np.asarray(matrix, dtype=np.float32)),
            )
            source_rows[offset:stop] = frame["source_row_number"].to_numpy(np.int64)
            labels[offset:stop] = frame["is_laundering"].to_numpy(np.int8)
            raw_logits[offset:stop] = logits.detach().cpu().numpy()
            probabilities[offset:stop] = torch.sigmoid(logits).detach().cpu().numpy()
            offset = stop
            batches += 1
            print(f"[sprint4] validation scored {offset:,}/{expected_rows:,}", flush=True)
    if offset != expected_rows or int(labels.sum()) != expected_positives:
        raise Sprint4PipelineError(
            "Full validation scoring mismatch: "
            f"rows={offset}, positives={int(labels[:offset].sum())}"
        )
    if np.unique(source_rows).size != expected_rows:
        raise Sprint4PipelineError("Full validation source rows are not unique")
    if not np.isfinite(raw_logits).all() or not np.isfinite(probabilities).all():
        raise Sprint4PipelineError("GraphSAGE validation scores contain non-finite values")
    return (
        {
            "source_row_number": source_rows,
            "is_laundering": labels,
            "raw_score_graphsage_edge_classifier": raw_logits,
            "probability_graphsage_edge_classifier": probabilities,
        },
        node_embeddings.detach().cpu(),
        adjacency,
        {
            "partition": "validation",
            "rows": expected_rows,
            "positive_labels": expected_positives,
            "batches": batches,
            "predict_seconds": time.perf_counter() - started,
            "message_graph_partition": "train",
            "validation_edges_used_for_message_passing": False,
            "test_rows_transformed_or_scored": False,
        },
    )


def _write_validation_predictions(
    connection: duckdb.DuckDBPyConnection,
    values: Mapping[str, np.ndarray],
    *,
    feature_table: Path,
    split_table: Path,
    sprint3_predictions: Path,
    destination: Path,
    compression: str,
) -> dict[str, Any]:
    frame = pd.DataFrame(values)
    connection.register("argus_sprint4_gnn_scores", frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    target = temporary.resolve().as_posix().replace("'", "''")
    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    f.transaction_id,
                    CAST(g.source_row_number AS BIGINT) AS source_row_number,
                    'validation' AS evaluation_partition,
                    CAST(g.is_laundering AS TINYINT) AS is_laundering,
                    CAST(g.raw_score_graphsage_edge_classifier AS DOUBLE)
                        AS raw_score_graphsage_edge_classifier,
                    CAST(g.probability_graphsage_edge_classifier AS DOUBLE)
                        AS probability_graphsage_edge_classifier,
                    CAST(r.raw_score_refined_lightgbm AS DOUBLE)
                        AS raw_score_refined_transaction_lightgbm,
                    CAST(r.probability_refined_lightgbm AS DOUBLE)
                        AS probability_refined_transaction_lightgbm,
                    CAST(r.raw_score_ablation_transaction_temporal_history_graph AS DOUBLE)
                        AS raw_score_graph_enhanced_lightgbm,
                    CAST(r.probability_ablation_transaction_temporal_history_graph AS DOUBLE)
                        AS probability_graph_enhanced_lightgbm
                FROM argus_sprint4_gnn_scores AS g
                JOIN read_parquet(?) AS f
                  ON f.source_row_number = g.source_row_number
                 AND f.is_laundering = g.is_laundering
                JOIN read_parquet(?) AS s
                  ON s.transaction_id = f.transaction_id
                 AND s.partition = 'validation'
                JOIN read_parquet(?) AS r
                  ON r.source_row_number = g.source_row_number
                 AND r.is_laundering = g.is_laundering
                ORDER BY g.source_row_number
            ) TO '{target}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
            """,
            [
                str(feature_table.resolve()),
                str(split_table.resolve()),
                str(sprint3_predictions.resolve()),
            ],
        )
        os.replace(temporary, destination)
    finally:
        connection.unregister("argus_sprint4_gnn_scores")
        temporary.unlink(missing_ok=True)
    row = connection.execute(
        """
        SELECT
            count(*), count(DISTINCT transaction_id), count(DISTINCT source_row_number),
            sum(is_laundering), count(DISTINCT evaluation_partition),
            min(evaluation_partition),
            count(*) FILTER (
                WHERE NOT isfinite(raw_score_graphsage_edge_classifier)
                   OR NOT isfinite(probability_graphsage_edge_classifier)
                   OR NOT isfinite(raw_score_refined_transaction_lightgbm)
                   OR NOT isfinite(raw_score_graph_enhanced_lightgbm)
            )
        FROM read_parquet(?)
        """,
        [str(destination.resolve())],
    ).fetchone()
    return {
        "partition": str(row[5]),
        "rows": int(row[0]),
        "unique_transaction_ids": int(row[1]),
        "unique_source_rows": int(row[2]),
        "positive_labels": int(row[3]),
        "distinct_partition_values": int(row[4]),
        "nonfinite_score_rows": int(row[6]),
        "model_score_columns": {
            "graphsage_edge_classifier": {
                "raw": "raw_score_graphsage_edge_classifier",
                "probability": "probability_graphsage_edge_classifier",
            },
            "refined_transaction_lightgbm": {
                "raw": "raw_score_refined_transaction_lightgbm",
                "probability": "probability_refined_transaction_lightgbm",
            },
            "graph_enhanced_lightgbm": {
                "raw": "raw_score_graph_enhanced_lightgbm",
                "probability": "probability_graph_enhanced_lightgbm",
            },
        },
        "test_predictions_included": False,
    }


def _evaluate_saved_predictions(
    connection: duckdb.DuckDBPyConnection,
    predictions_path: Path,
    settings: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
    dict[str, Any],
]:
    frame = connection.execute(
        """
        SELECT
            source_row_number, is_laundering,
            raw_score_graphsage_edge_classifier,
            probability_graphsage_edge_classifier,
            raw_score_refined_transaction_lightgbm,
            raw_score_graph_enhanced_lightgbm
        FROM read_parquet(?)
        WHERE evaluation_partition = 'validation'
        ORDER BY source_row_number
        """,
        [str(predictions_path.resolve())],
    ).fetch_df()
    score_vectors = {
        "graphsage_edge_classifier": frame["raw_score_graphsage_edge_classifier"].to_numpy(),
        "refined_transaction_lightgbm": frame["raw_score_refined_transaction_lightgbm"].to_numpy(),
        "graph_enhanced_lightgbm": frame["raw_score_graph_enhanced_lightgbm"].to_numpy(),
    }
    labels = frame["is_laundering"].to_numpy(np.int8)
    source_rows = frame["source_row_number"].to_numpy(np.int64)
    threshold_settings = _mapping(
        settings["threshold_optimization"], "sprint4.threshold_optimization"
    )
    comparison, thresholds, top_k = compare_validation_models(
        labels,
        source_rows,
        score_vectors,
        top_k=tuple(int(value) for value in settings["top_k"]),
        alert_budget=int(threshold_settings["alert_budget"]),
        fpr_ceiling=float(threshold_settings["fpr_ceiling"]),
    )
    frozen = _mapping(settings["frozen_references"], "sprint4.frozen_references")
    reproduced = {
        "refined_transaction_lightgbm": compute_average_precision(
            labels, score_vectors["refined_transaction_lightgbm"]
        ),
        "graph_enhanced_lightgbm": compute_average_precision(
            labels, score_vectors["graph_enhanced_lightgbm"]
        ),
    }
    expected = {
        "refined_transaction_lightgbm": float(frozen["refined_baseline_average_precision"]),
        "graph_enhanced_lightgbm": float(frozen["graph_enhanced_average_precision"]),
    }
    for name in expected:
        if not math.isclose(float(reproduced[name]), expected[name], rel_tol=0.0, abs_tol=1e-15):
            raise Sprint4PipelineError(
                f"Frozen {name} validation AP did not reproduce: "
                f"expected={expected[name]}, actual={reproduced[name]}"
            )

    by_model_k = {(str(row["model"]), int(row["requested_k"])): row for row in top_k}
    preferred_k = 1000 if 1000 in settings["top_k"] else int(settings["top_k"][-1])
    validation_leader = max(
        comparison, key=lambda row: (float(row["average_precision"]), str(row["model"]))
    )["model"]
    for row in comparison:
        item = by_model_k[(str(row["model"]), preferred_k)]
        row.update(
            {
                "version": {
                    "refined_transaction_lightgbm": "Sprint 3 refined transaction baseline",
                    "graph_enhanced_lightgbm": "Sprint 3 graph-enhanced LightGBM",
                    "graphsage_edge_classifier": "Sprint 4 sampled GraphSAGE edge classifier",
                }[str(row["model"])],
                "ranking_score_type": "raw_model_score",
                "recall_at_k": item["recall_at_k"],
                "precision_at_k": item["precision_at_k"],
                "top_k_for_display": preferred_k,
                "is_validation_leader": row["model"] == validation_leader,
                "is_champion": row["model"] == validation_leader,
                "final_test_opened": False,
            }
        )
    comparison_frame = pd.DataFrame(comparison).sort_values(
        ["average_precision", "model"], ascending=[False, True], kind="mergesort"
    )
    probability = frame["probability_graphsage_edge_classifier"].to_numpy()
    saturation = graphsage_score_diagnostics(
        labels,
        score_vectors["graphsage_edge_classifier"],
        probability,
        source_rows,
        top_k=tuple(int(value) for value in settings["top_k"]),
    )
    pr_curves = build_pr_curve_table(labels, score_vectors)
    for row in top_k:
        row["evaluation_partition"] = "validation"
        row["test_metrics_used"] = False
    audit = {
        "frozen_reference_average_precision_reproduced": reproduced,
        "frozen_reference_tolerance": 1e-15,
        "graphsage_saturation": saturation,
        "comparison_score_representation": "raw_model_scores",
        "same_full_validation_rows": True,
        "validation_leader": validation_leader,
        "selection_is_final_test_independent": True,
    }
    return comparison_frame, comparison, thresholds, top_k, pr_curves, audit


def _plot_validation_results(
    comparison: pd.DataFrame,
    pr_curves: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    ordered = comparison.sort_values("average_precision", kind="mergesort")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered["model"], ordered["average_precision"], color="#0f766e")
    axis.set_xlabel("Validation PR-AUC (average precision)")
    axis.set_title("Frozen validation comparison")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "validation_pr_auc_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for model_name, values in pr_curves.groupby("model", sort=True):
        ordered_curve = values.sort_values("recall", kind="mergesort")
        axis.plot(ordered_curve["recall"], ordered_curve["precision"], label=model_name)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Full validation precision-recall curves")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "validation_precision_recall_curves.png", dpi=160)
    plt.close(fig)


def _model_contribution_records(
    names: Sequence[str],
    contributions: np.ndarray,
    observed_values: np.ndarray,
    *,
    maximum: int,
) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(contributions), kind="stable")[:maximum]
    return [
        {
            "feature": str(names[index]),
            "contribution": float(contributions[index]),
            "observed_feature_value": float(observed_values[index]),
        }
        for index in order
    ]


def _build_product_artifacts(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_dir: Path,
    sprint3_dir: Path,
    predictions_path: Path,
    feature_table: Path,
    split_table: Path,
    inference_context: pd.DataFrame,
    inference_view: GraphView,
    inference_adjacency: DirectedMeanAdjacency,
    node_embeddings: torch.Tensor,
    model: GraphSAGEEdgeClassifier,
    transaction_preprocessor: FittedPreprocessor,
    thresholds: Mapping[str, Any],
    comparison: pd.DataFrame,
    top_k_rows: Sequence[Mapping[str, Any]],
    pr_curves: pd.DataFrame,
    settings: Mapping[str, Any],
    compression: str,
) -> dict[str, Any]:
    product_dir = run_dir / "product"
    product_dir.mkdir(parents=True, exist_ok=True)
    case_settings = _mapping(settings["case_builder"], "sprint4.case_builder")
    explanation_settings = _mapping(settings["explanations"], "sprint4.explanations")
    maximum_cases = int(case_settings["maximum_cases"])
    selected = connection.execute(
        """
        SELECT
            transaction_id, source_row_number,
            raw_score_graphsage_edge_classifier,
            probability_graphsage_edge_classifier,
            raw_score_refined_transaction_lightgbm,
            raw_score_graph_enhanced_lightgbm
        FROM read_parquet(?)
        WHERE evaluation_partition = 'validation'
        ORDER BY raw_score_graphsage_edge_classifier DESC, source_row_number ASC
        LIMIT ?
        """,
        [str(predictions_path.resolve()), maximum_cases],
    ).fetch_df()
    selected["validation_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
    connection.register("argus_case_sources", selected[["source_row_number", "validation_rank"]])
    try:
        case_frame = connection.execute(
            """
            SELECT f.* EXCLUDE (is_laundering), 'validation' AS partition, c.validation_rank
            FROM read_parquet(?) AS f
            JOIN argus_case_sources AS c USING (source_row_number)
            JOIN read_parquet(?) AS s USING (transaction_id)
            WHERE s.partition = 'validation'
            ORDER BY c.validation_rank
            """,
            [str(feature_table.resolve()), str(split_table.resolve())],
        ).fetch_df()
    finally:
        connection.unregister("argus_case_sources")
    if len(case_frame) != maximum_cases or "is_laundering" in case_frame.columns:
        raise Sprint4PipelineError("Case source export violated row count or target exclusion")
    case_frame = case_frame.merge(
        selected.drop(columns="validation_rank"),
        on=["transaction_id", "source_row_number"],
        how="left",
        validate="one_to_one",
    ).sort_values("validation_rank", kind="mergesort")

    sender, receiver = endpoint_indices(case_frame, inference_view.node_to_index)
    transaction_matrix = transaction_preprocessor.transform_pandas(case_frame)
    attribution = explain_edge_inputs(
        model,
        inference_adjacency,
        torch.from_numpy(sender).long(),
        torch.from_numpy(receiver).long(),
        torch.from_numpy(transaction_matrix.astype(np.float32, copy=False)),
        node_features=torch.from_numpy(inference_view.node_features),
    )
    saved_raw = case_frame["raw_score_graphsage_edge_classifier"].to_numpy(np.float64)
    gnn_raw_error = float(
        np.max(np.abs(attribution.logits.detach().numpy().astype(np.float64) - saved_raw))
    )
    if gnn_raw_error > 1e-5:
        raise Sprint4PipelineError(f"GraphSAGE case explanation score mismatch: {gnn_raw_error}")
    embedding_values_sender = node_embeddings.index_select(0, torch.from_numpy(sender)).numpy()
    embedding_values_receiver = node_embeddings.index_select(0, torch.from_numpy(receiver)).numpy()
    gnn_names = [
        *[f"sender_embedding_{index:02d}" for index in range(model.config.embedding_dim)],
        *[f"receiver_embedding_{index:02d}" for index in range(model.config.embedding_dim)],
        *[f"transaction::{name}" for name in transaction_preprocessor.feature_names],
    ]
    gnn_contributions = np.concatenate(
        (
            attribution.sender_gradient_x_input.numpy(),
            attribution.receiver_gradient_x_input.numpy(),
            attribution.transaction_gradient_x_input.numpy(),
        ),
        axis=1,
    )
    gnn_observed = np.concatenate(
        (embedding_values_sender, embedding_values_receiver, transaction_matrix), axis=1
    )

    tree_preprocessor = FittedPreprocessor.load_state(
        sprint3_dir
        / "preprocessing"
        / "outer_train"
        / "transaction_temporal_history_graph"
        / "fitted_state.json"
    )
    tree_model = joblib.load(
        sprint3_dir / "models" / "ablation_transaction_temporal_history_graph" / "model.joblib"
    )
    tree_matrix = tree_preprocessor.transform_pandas(case_frame)
    booster = tree_model.booster_
    tree_contributions = np.asarray(booster.predict(tree_matrix, pred_contrib=True))
    tree_raw = np.asarray(booster.predict(tree_matrix, raw_score=True)).reshape(-1)
    if tree_contributions.shape != (len(case_frame), tree_matrix.shape[1] + 1):
        raise Sprint4PipelineError("LightGBM pred_contrib returned an unexpected shape")
    tree_reference = case_frame["raw_score_graph_enhanced_lightgbm"].to_numpy(np.float64)
    tree_reference_error = float(np.max(np.abs(tree_raw - tree_reference)))
    tree_additivity_error = float(
        np.max(
            np.abs(tree_contributions[:, :-1].sum(axis=1) + tree_contributions[:, -1] - tree_raw)
        )
    )
    if tree_reference_error > 1e-10 or tree_additivity_error > 1e-8:
        raise Sprint4PipelineError(
            "TreeSHAP reproduction/additivity check failed: "
            f"reference={tree_reference_error}, additivity={tree_additivity_error}"
        )

    threshold_raw = float(
        thresholds["graphsage_edge_classifier"]["summaries"][
            "predeclared_joint_primary_constraint"
        ]["operating_point"]["threshold"]
    )
    threshold_probability = (
        1.0 / (1.0 + math.exp(-threshold_raw))
        if threshold_raw >= 0.0
        else math.exp(threshold_raw) / (1.0 + math.exp(threshold_raw))
    )
    context = inference_context.copy()
    context["partition"] = "train"
    cases: list[dict[str, Any]] = []
    tree_explanations: list[dict[str, Any]] = []
    gnn_explanations: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    maximum_contributions = int(explanation_settings["tree_top_features_per_case"])
    for index, row in case_frame.reset_index(drop=True).iterrows():
        endpoints = {str(row["from_node_id"]), str(row["to_node_id"])}
        local = context[
            context["from_node_id"].astype(str).isin(endpoints)
            | context["to_node_id"].astype(str).isin(endpoints)
        ].copy()
        contributions = _model_contribution_records(
            gnn_names,
            gnn_contributions[index],
            gnn_observed[index],
            maximum=maximum_contributions,
        )
        case = build_case_from_frames(
            row,
            local,
            model_score=float(row["probability_graphsage_edge_classifier"]),
            model_name="graphsage_edge_classifier",
            threshold=threshold_probability,
            rank=int(row["validation_rank"]),
            score_name="graphsage_uncalibrated_sigmoid_score_ranked_by_raw_logit",
            feature_contributions=contributions,
            max_feature_contributions=maximum_contributions,
            max_neighborhood_edges=int(case_settings["maximum_display_edges"]),
            llm=None,
        )
        payload = case.to_dict()
        validate_case_payload(payload)
        cases.append(payload)
        tree_items = _model_contribution_records(
            tree_preprocessor.feature_names,
            tree_contributions[index, :-1],
            tree_matrix[index],
            maximum=maximum_contributions,
        )
        tree_explanations.append(
            {
                "case_id": payload["case_id"],
                "transaction_id": payload["transaction_id"],
                "model": "graph_enhanced_lightgbm",
                "method": "lightgbm_native_pred_contrib_treeshap",
                "raw_score": float(tree_raw[index]),
                "base_value": float(tree_contributions[index, -1]),
                "sum_contributions_plus_base": float(tree_contributions[index].sum()),
                "additivity_absolute_error": float(
                    abs(tree_contributions[index].sum() - tree_raw[index])
                ),
                "top_contributions": tree_items,
                "basis": "model",
                "causal_claim": False,
            }
        )
        gnn_explanations.append(
            {
                "case_id": payload["case_id"],
                "transaction_id": payload["transaction_id"],
                "model": "graphsage_edge_classifier",
                "method": "local_gradient_x_input_sensitivity",
                "claim_scope": "sensitivity_not_shap",
                "raw_logit": float(saved_raw[index]),
                "probability": float(row["probability_graphsage_edge_classifier"]),
                "calibrated_probability": False,
                "score_interpretation": "uncalibrated_sigmoid_ranking_score",
                "top_contributions": contributions,
                "basis": "model",
                "causal_claim": False,
            }
        )
        amounts = [edge.get("amount") for edge in payload["transactions"]]
        total_flow = sum(float(value) for value in amounts if value is not None)
        queue.append(
            {
                "priority": payload["priority"]["band"],
                "case_id": payload["case_id"],
                "risk_score": float(row["probability_graphsage_edge_classifier"]),
                "score_type": "uncalibrated_sigmoid_ranking_score",
                "account_count": len(payload["network"]["nodes"]),
                "transaction_count": len(payload["transactions"]),
                "total_flow": total_flow,
                "major_pattern": "high_graphsage_transaction_score",
                "evidence_count": len(payload["observed_evidence"]),
                "validation_rank": int(row["validation_rank"]),
            }
        )

    cases_payload = {
        "schema": "argus.investigation_cases.v1",
        "partition": "validation",
        "model": "graphsage_edge_classifier",
        "selection_score": "raw_logit_descending_then_source_row_number_ascending",
        "target_fields_included": False,
        "test_rows_included": False,
        "cases": cases,
    }
    serialized_cases = json.dumps(cases_payload, ensure_ascii=False).lower()
    if '"is_laundering"' in serialized_cases:
        raise Sprint4PipelineError("Case artifacts contain a forbidden target field")
    atomic_write_json(cases_payload, product_dir / "cases.json")
    atomic_write_csv(pd.DataFrame(queue), product_dir / "investigation_queue.csv")
    atomic_write_json(
        {
            "method": "lightgbm_native_pred_contrib_treeshap",
            "partition": "validation",
            "model": "graph_enhanced_lightgbm",
            "maximum_reference_score_error": tree_reference_error,
            "maximum_additivity_absolute_error": tree_additivity_error,
            "explanations": tree_explanations,
            "test_rows_used": False,
        },
        product_dir / "tree_shap_explanations.json",
    )
    atomic_write_json(
        {
            "method": "local_gradient_x_input_sensitivity",
            "claim_scope": "sensitivity_not_shap",
            "partition": "validation",
            "maximum_saved_logit_error": gnn_raw_error,
            "explanations": gnn_explanations,
            "test_rows_used": False,
        },
        product_dir / "graphsage_explanations.json",
    )

    product_comparison = comparison.copy()
    atomic_write_csv(product_comparison, product_dir / "model_comparison.csv")
    atomic_write_csv(pd.DataFrame(top_k_rows), product_dir / "top_k_metrics.csv")
    atomic_write_csv(pr_curves, product_dir / "pr_curves.csv")
    ablation = pd.read_csv(sprint3_dir / "ablation" / "feature_family_ablation.csv")
    atomic_write_csv(ablation, product_dir / "feature_family_ablation.csv")
    leader = product_comparison.loc[product_comparison["is_validation_leader"]].iloc[0]
    graphsage_row = product_comparison.loc[
        product_comparison["model"].eq("graphsage_edge_classifier")
    ].iloc[0]
    summary = {
        "partition": "validation",
        "transactions_analyzed": int(leader["validation_row_count"]),
        "flagged_transactions": int(graphsage_row["alert_count"]),
        "case_source_model": str(graphsage_row["model"]),
        "case_source_alert_volume": int(graphsage_row["alert_count"]),
        "case_source_score_calibrated": False,
        "high_priority_cases": sum(item["priority"].startswith("elevated") for item in queue),
        "validation_pr_auc": float(leader["average_precision"]),
        "validation_recall_at_k": float(leader["recall_at_k"]),
        "validation_fpr": float(leader["false_positive_rate"]),
        "alert_volume": int(leader["alert_count"]),
        "validation_leader": str(leader["model"]),
        "final_test_opened": False,
    }
    atomic_write_json(summary, product_dir / "dashboard_summary.json")
    loaded = load_dashboard_artifacts(run_dir)
    if len(loaded.cases) != maximum_cases or len(loaded.queue) != maximum_cases:
        raise Sprint4PipelineError("Saved Streamlit artifact round-trip failed")
    observed_counts = [len(item["observed_evidence"]) for item in cases]
    return {
        "cases": {
            "count": len(cases),
            "minimum_observed_evidence": min(observed_counts),
            "maximum_observed_evidence": max(observed_counts),
            "deterministic_fallback_notes": sum(
                item["analyst_note"]["mode"] == "deterministic_fallback" for item in cases
            ),
            "all_validation_only": True,
            "target_fields_included": False,
        },
        "explanations": {
            "tree_method": "lightgbm_native_pred_contrib_treeshap",
            "tree_maximum_additivity_error": tree_additivity_error,
            "tree_maximum_saved_score_error": tree_reference_error,
            "gnn_method": "local_gradient_x_input_sensitivity",
            "gnn_claim_scope": "sensitivity_not_shap",
            "gnn_maximum_saved_score_error": gnn_raw_error,
        },
        "streamlit": {
            "artifact_round_trip": True,
            "page_load_training": False,
            "screens": [
                "Executive Dashboard",
                "Investigation Queue",
                "Case Investigator",
                "Model Comparison",
            ],
        },
    }


def run_sprint4_pipeline(config_path: str | Path = "configs/sprint4.yaml") -> Sprint4RunResult:
    """Run sampled GraphSAGE training, full validation scoring, and product export."""

    started_clock = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    config = load_config(config_path)
    settings = _mapping(config["sprint4"], "sprint4")
    baseline_settings = _mapping(config["baseline"], "baseline")
    frozen = _mapping(settings["frozen_references"], "sprint4.frozen_references")
    graph_sampling = _mapping(settings["graph_sampling"], "sprint4.graph_sampling")
    supervised_settings = _mapping(settings["supervised_training"], "sprint4.supervised_training")
    project_root = Path(config["_meta"]["project_root"])
    feature_table = get_path(config, "feature_table")
    split_table = get_path(config, "split_table")
    split_metadata_path = get_path(config, "split_metadata")
    upstream_manifest_path = get_path(config, "upstream_manifest")
    run_dir = get_path(config, "run_dir")
    sprint3_dir = get_path(config, "sprint3_run_dir")
    report_path = get_path(config, "generated_report")
    manifest_path = run_dir / "run_manifest.json"
    sprint3_predictions = sprint3_dir / "validation_predictions.parquet"
    compression = str(settings["parquet_compression"])
    recorder = _StageRecorder()
    connection: duckdb.DuckDBPyConnection | None = None
    work_dir = run_dir / f"{_WORK_PREFIX}{uuid.uuid4().hex}"
    succeeded = False

    # Immutable input and Git checks precede cleanup so a bad dependency cannot
    # destroy the last successful Sprint 4 evidence set.
    with recorder.measure("frozen_input_preflight"):
        provenance, prevalence, _ = _verify_frozen_inputs(
            settings=baseline_settings,
            feature_table=feature_table,
            split_table=split_table,
            split_metadata_path=split_metadata_path,
            upstream_manifest_path=upstream_manifest_path,
        )
        _sprint3_manifest, sprint3_evidence = _verify_sprint3_reference(
            project_root, sprint3_dir, frozen
        )
        source_snapshot = _source_snapshot(project_root)
        if not sprint3_predictions.is_file():
            raise FileNotFoundError(
                f"Sprint 3 validation predictions missing: {sprint3_predictions}"
            )

    _prepare_run_directory(run_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        atomic_write_json(config, run_dir / "resolved_config.json")
        atomic_write_json(source_snapshot, run_dir / "source_snapshot.json")
        atomic_write_json(provenance, run_dir / "input_provenance.json")
        atomic_write_json(prevalence, run_dir / "split_prevalence.json")
        atomic_write_json(sprint3_evidence, run_dir / "frozen_sprint3_reference.json")

        connection = duckdb.connect()
        _configure_duckdb(connection, settings, work_dir / "spill")
        transaction_preprocessor = FittedPreprocessor.load_state(
            sprint3_dir
            / "preprocessing"
            / "outer_train"
            / "transaction_temporal_history"
            / "fitted_state.json"
        )

        with recorder.measure("deterministic_training_sampling"):
            training_context, training_context_audit = sample_context_edges(
                connection,
                feature_table,
                split_table,
                maximum_edges=int(graph_sampling["training_context_max_edges"]),
                expected_population_rows=int(graph_sampling["training_context_population_rows"]),
                maximum_timestamp=str(graph_sampling["training_context_end"]),
            )
            supervised, supervised_matrix, supervised_audit = sample_supervised_training_edges(
                connection,
                feature_table,
                split_table,
                transaction_preprocessor,
                strictly_after=str(graph_sampling["training_context_end"]),
                expected_population_rows=int(supervised_settings["population_rows"]),
                maximum_negative_rows=int(supervised_settings["maximum_negative_rows"]),
            )
            _write_frame_parquet(
                connection,
                training_context,
                run_dir / "sampling" / "training_context_edges.parquet",
                compression=compression,
            )
            supervised_export = supervised[
                [
                    "transaction_id",
                    "source_row_number",
                    "timestamp",
                    "from_node_id",
                    "to_node_id",
                    "amount_paid",
                    "is_laundering",
                ]
            ]
            _write_frame_parquet(
                connection,
                supervised_export,
                run_dir / "sampling" / "supervised_training_edges.parquet",
                compression=compression,
            )
            atomic_write_json(
                {
                    "training_context": training_context_audit,
                    "supervised_training": supervised_audit,
                    "selection_target_used_for_context": False,
                    "test_rows_used": False,
                },
                run_dir / "sampling" / "training_sampling_manifest.json",
            )

        with recorder.measure("training_graph_construction"):
            train_required_nodes = tuple(
                pd.unique(
                    pd.concat(
                        [supervised["from_node_id"], supervised["to_node_id"]],
                        ignore_index=True,
                    ).astype(str)
                )
            )
            training_view = build_graph_view(
                training_context,
                required_node_ids=train_required_nodes,
            )
            training_normalizer = training_view.normalizer
            atomic_write_json(
                {
                    "node_count": training_view.node_count,
                    "context_edge_count": training_view.context_edge_rows,
                    "node_feature_count": training_view.feature_count,
                    "node_features": [
                        *settings["node_features"]["structural_features"],
                        *settings["node_features"]["deterministic_identity_features"],
                    ],
                    "monetary_node_aggregates_used": False,
                    "amount_aggregation_policy": settings["node_features"][
                        "amount_aggregation_policy"
                    ],
                    "normalizer": training_view.normalizer.to_dict(),
                    "required_supervised_endpoint_nodes": len(train_required_nodes),
                    "directed": True,
                    "validation_edges_used": False,
                    "test_edges_used": False,
                },
                run_dir / "model" / "training_graph_manifest.json",
            )

        with recorder.measure("graphsage_training"):
            seed = int(config["project"]["random_seed"])
            model, training_adjacency, training_evidence = _train_graphsage(
                training_view,
                supervised,
                supervised_matrix,
                settings,
                seed=seed,
            )
            _atomic_torch_save(model.export_checkpoint(), run_dir / "model" / "graphsage.pt")
            atomic_write_json(training_evidence, run_dir / "model" / "training_history.json")
            reloaded = GraphSAGEEdgeClassifier.from_checkpoint(
                torch.load(
                    run_dir / "model" / "graphsage.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            if any(
                not torch.equal(model.state_dict()[name], reloaded.state_dict()[name])
                for name in model.state_dict()
            ):
                raise Sprint4PipelineError("Saved GraphSAGE checkpoint did not round-trip")
            del reloaded, training_adjacency, training_view, supervised_matrix
            gc.collect()

        with recorder.measure("inference_graph_construction"):
            inference_context, inference_context_audit = sample_context_edges(
                connection,
                feature_table,
                split_table,
                maximum_edges=int(graph_sampling["inference_context_max_edges"]),
                expected_population_rows=int(graph_sampling["inference_context_population_rows"]),
                maximum_timestamp=None,
            )
            validation_node_ids = validation_nodes(connection, feature_table, split_table)
            inference_view = build_graph_view(
                inference_context,
                required_node_ids=validation_node_ids,
                normalizer=training_normalizer,
            )
            _write_frame_parquet(
                connection,
                inference_context,
                run_dir / "sampling" / "inference_context_edges.parquet",
                compression=compression,
            )
            node_frame = pd.DataFrame(
                {
                    "node_index": np.arange(inference_view.node_count, dtype=np.int64),
                    "node_id": inference_view.node_ids,
                }
            )
            _write_frame_parquet(
                connection,
                node_frame,
                run_dir / "model" / "inference_nodes.parquet",
                compression=compression,
            )
            inference_graph_manifest = {
                "partition": "train_message_context_plus_validation_endpoint_identities",
                "message_context_partition": "train",
                "node_count": inference_view.node_count,
                "validation_endpoint_node_count": len(validation_node_ids),
                "context_edge_count": inference_view.context_edge_rows,
                "node_feature_count": inference_view.feature_count,
                "normalizer_fit_scope": "sampled_training_context_nodes_only",
                "normalizer": training_normalizer.to_dict(),
                "monetary_node_aggregates_used": False,
                "amount_aggregation_policy": settings["node_features"]["amount_aggregation_policy"],
                "validation_labels_used_for_graph_construction": False,
                "validation_edges_used_for_message_passing": False,
                "test_nodes_or_edges_used": False,
            }
            atomic_write_json(
                inference_graph_manifest, run_dir / "model" / "inference_graph_manifest.json"
            )

        with recorder.measure("full_validation_scoring"):
            score_values, node_embeddings, inference_adjacency, scoring_audit = (
                _score_full_validation(
                    connection,
                    model,
                    inference_view,
                    transaction_preprocessor,
                    feature_table,
                    split_table,
                    expected_rows=int(frozen["validation_rows"]),
                    expected_positives=int(frozen["validation_positives"]),
                    batch_rows=int(settings["batch_rows"]),
                )
            )
            _atomic_numpy_save(
                node_embeddings.numpy(), run_dir / "model" / "inference_node_embeddings.npy"
            )
            _atomic_torch_save(
                inference_adjacency.state_dict(),
                run_dir / "model" / "inference_adjacency.pt",
            )
            predictions_path = run_dir / "validation_predictions.parquet"
            prediction_manifest = _write_validation_predictions(
                connection,
                score_values,
                feature_table=feature_table,
                split_table=split_table,
                sprint3_predictions=sprint3_predictions,
                destination=predictions_path,
                compression=compression,
            )
            if (
                prediction_manifest["rows"] != int(frozen["validation_rows"])
                or prediction_manifest["positive_labels"] != int(frozen["validation_positives"])
                or prediction_manifest["unique_transaction_ids"] != int(frozen["validation_rows"])
                or prediction_manifest["partition"] != "validation"
                or prediction_manifest["distinct_partition_values"] != 1
                or prediction_manifest["nonfinite_score_rows"] != 0
            ):
                raise Sprint4PipelineError(
                    f"Validation prediction artifact failed verification: {prediction_manifest}"
                )
            atomic_write_json(prediction_manifest, run_dir / "validation_predictions_manifest.json")
            del score_values
            gc.collect()

        with recorder.measure("validation_only_evaluation"):
            (
                comparison_frame,
                comparison_rows,
                thresholds,
                top_k_rows,
                pr_curves,
                evaluation_audit,
            ) = _evaluate_saved_predictions(connection, predictions_path, settings)
            atomic_write_csv(comparison_frame, run_dir / "model_comparison.csv")
            atomic_write_json(
                {
                    "partition": "validation",
                    "primary_metric": "average_precision",
                    "ranking_score_type": "raw_model_score",
                    "models": comparison_rows,
                    "validation_leader": evaluation_audit["validation_leader"],
                    "test_metrics_used": False,
                },
                run_dir / "model_comparison.json",
            )
            atomic_write_json(
                {"partition": "validation", "models": thresholds, "test_rows_used": False},
                run_dir / "thresholds.json",
            )
            atomic_write_csv(pd.DataFrame(top_k_rows), run_dir / "top_k_metrics.csv")
            atomic_write_json(
                {"partition": "validation", "rows": top_k_rows, "test_rows_used": False},
                run_dir / "top_k_metrics.json",
            )
            atomic_write_csv(pr_curves, run_dir / "pr_curves.csv")
            atomic_write_json(evaluation_audit, run_dir / "graphsage_score_diagnostics.json")
            _plot_validation_results(comparison_frame, pr_curves, run_dir / "figures")

        with recorder.measure("case_evidence_and_product_export"):
            product_evidence = _build_product_artifacts(
                connection,
                run_dir=run_dir,
                sprint3_dir=sprint3_dir,
                predictions_path=predictions_path,
                feature_table=feature_table,
                split_table=split_table,
                inference_context=inference_context,
                inference_view=inference_view,
                inference_adjacency=inference_adjacency,
                node_embeddings=node_embeddings,
                model=model,
                transaction_preprocessor=transaction_preprocessor,
                thresholds=thresholds,
                comparison=comparison_frame,
                top_k_rows=top_k_rows,
                pr_curves=pr_curves,
                settings=settings,
                compression=compression,
            )

        final_test_policy = {
            "partition": "test",
            "status": "SEALED",
            "metadata_counts_inherited_from_sprint1": True,
            "feature_rows_loaded": False,
            "feature_transform_called": False,
            "graph_nodes_or_edges_constructed": False,
            "model_predictions_generated": False,
            "metrics_computed": False,
            "model_selection_or_tuning_used_test": False,
            "allowed_next_action": "explicit_final_evaluation_authorization_required",
        }
        atomic_write_json(final_test_policy, run_dir / "final_test_policy.json")
        sampling_evidence = {
            "full_graph_training_attempted": bool(graph_sampling["full_graph_training_attempted"]),
            "full_graph_training_exclusion_reason": str(
                graph_sampling["full_graph_training_exclusion_reason"]
            ),
            "sampling_method": "md5_order_without_replacement",
            "training_context": training_context_audit,
            "supervised_training": supervised_audit,
            "inference_context": inference_context_audit,
            "outer_validation_evaluation": {
                "rows": int(frozen["validation_rows"]),
                "sampled": False,
                "coverage_fraction": 1.0,
            },
            "claimed_as_full_graph_training": False,
        }
        atomic_write_json(sampling_evidence, run_dir / "sampling" / "sampling_disclosure.json")
        acceptance = {
            "frozen_sprint3_references_preserved": True,
            "transaction_edge_target_only": True,
            "unsupported_account_level_label_not_created": True,
            "deterministic_sampled_graphsage_training_completed": True,
            "full_graph_limitation_explicitly_disclosed": True,
            "full_frozen_validation_scored": prediction_manifest["rows"]
            == int(frozen["validation_rows"]),
            "same_validation_rows_used_for_three_model_comparison": evaluation_audit[
                "same_full_validation_rows"
            ],
            "validation_only_threshold_optimization_completed": True,
            "final_test_remained_sealed": not any(
                value is True
                for key, value in final_test_policy.items()
                if key
                in {
                    "feature_rows_loaded",
                    "feature_transform_called",
                    "graph_nodes_or_edges_constructed",
                    "model_predictions_generated",
                    "metrics_computed",
                    "model_selection_or_tuning_used_test",
                }
            ),
            "graphsage_checkpoint_and_node_embeddings_saved": (
                (run_dir / "model" / "graphsage.pt").is_file()
                and (run_dir / "model" / "inference_node_embeddings.npy").is_file()
            ),
            "real_validation_cases_built": product_evidence["cases"]["count"]
            == int(settings["case_builder"]["maximum_cases"]),
            "minimum_three_observed_evidence_per_case": product_evidence["cases"][
                "minimum_observed_evidence"
            ]
            >= 3,
            "observed_and_model_evidence_separated": True,
            "tree_shap_additivity_verified": product_evidence["explanations"][
                "tree_maximum_additivity_error"
            ]
            <= 1e-8,
            "gnn_explanation_labeled_as_sensitivity_not_shap": product_evidence["explanations"][
                "gnn_claim_scope"
            ]
            == "sensitivity_not_shap",
            "deterministic_no_llm_fallback_completed": product_evidence["cases"][
                "deterministic_fallback_notes"
            ]
            == product_evidence["cases"]["count"],
            "streamlit_saved_artifact_round_trip_passed": product_evidence["streamlit"][
                "artifact_round_trip"
            ],
            "streamlit_page_load_training_disabled": not product_evidence["streamlit"][
                "page_load_training"
            ],
        }
        if not all(acceptance.values()):
            failed = sorted(name for name, passed in acceptance.items() if not passed)
            raise Sprint4PipelineError(f"Sprint 4 acceptance failed: {failed}")
        atomic_write_json(acceptance, run_dir / "acceptance_checklist.json")

        finished_at = datetime.now(UTC).isoformat()
        runtime_seconds = time.perf_counter() - started_clock
        libraries = {
            "python": sys.version.split()[0],
            "duckdb": duckdb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "torch": torch.__version__,
            "joblib": joblib.__version__,
        }
        atomic_write_json(libraries, run_dir / "library_versions.json")
        manifest_payload: dict[str, Any] = {
            "sprint": 4,
            "status": "PASS",
            "scope": "GraphSAGE plus product layer; stopped before final-test evaluation",
            "experiment_scope": settings["experiment_scope"],
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "runtime_seconds": runtime_seconds,
            "runtime_seconds_by_stage": recorder.seconds,
            "target_design": {
                "unit": "transaction_edge",
                "label_source": "IBM HI-Small transaction is_laundering",
                "unsupported_account_label_created": False,
                "node_embeddings_used_for_edge_classification": True,
                "sender_and_receiver_roles_kept_separate": True,
            },
            "score_semantics": {
                "graphsage_ranking_score": "raw_logit",
                "graphsage_sigmoid_score_calibrated": False,
                "reason": "sampled_negatives_and_positive_weight_change_class_prior",
            },
            "node_feature_design": {
                "features": [
                    *settings["node_features"]["structural_features"],
                    *settings["node_features"]["deterministic_identity_features"],
                ],
                "monetary_node_aggregates_used": False,
                "amount_aggregation_policy": settings["node_features"]["amount_aggregation_policy"],
            },
            "configuration": {
                "config_path": str(Path(config_path)),
                "random_seed": int(config["project"]["random_seed"]),
                "full_data_transaction_rows": int(provenance["expected_full_rows"]),
            },
            "provenance": provenance,
            "split_prevalence": prevalence,
            "frozen_sprint3": sprint3_evidence,
            "sampling": sampling_evidence,
            "training": training_evidence,
            "inference_graph": inference_graph_manifest,
            "validation_scoring": scoring_audit,
            "prediction_evidence": prediction_manifest,
            "model_comparison": comparison_rows,
            "evaluation_audit": evaluation_audit,
            "product": product_evidence,
            "final_test_policy": final_test_policy,
            "data_access_audit": {
                "train_feature_rows_used": True,
                "validation_feature_rows_used_for_inference_and_evaluation": True,
                "validation_labels_used_for_training_or_message_passing": False,
                "test_feature_rows_loaded": False,
                "test_labels_loaded": False,
                "test_predictions_generated": False,
                "test_metrics_computed": False,
            },
            "libraries": libraries,
            "platform": {
                "python_implementation": platform.python_implementation(),
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "configured_device": "cpu",
            },
            "acceptance": acceptance,
            "quality": {},
            "core_artifact_paths": [
                "artifacts/sprint4/model/graphsage.pt",
                "artifacts/sprint4/model/inference_node_embeddings.npy",
                "artifacts/sprint4/validation_predictions.parquet",
                "artifacts/sprint4/model_comparison.csv",
                "artifacts/sprint4/product/investigation_queue.csv",
                "artifacts/sprint4/product/cases.json",
                "artifacts/sprint4/product/tree_shap_explanations.json",
                "artifacts/sprint4/product/graphsage_explanations.json",
                "artifacts/sprint4/product/dashboard_summary.json",
                "reports/generated/SPRINT_4_STATUS.md",
            ],
        }
        render_sprint4_report(manifest_payload, run_dir / "SPRINT_4_STATUS.md")
        manifest = write_run_manifest(
            manifest_payload,
            run_dir,
            exclude_paths=[work_dir],
        )
        render_sprint4_report(manifest, report_path)
        succeeded = True
        return Sprint4RunResult(run_dir, manifest_path, report_path, manifest)
    finally:
        if connection is not None:
            connection.close()
        if succeeded:
            _safe_cleanup_work_directory(work_dir, run_dir)
        elif work_dir.exists():
            print(f"[sprint4] Work files retained after failure: {work_dir}", flush=True)


__all__ = ["Sprint4PipelineError", "Sprint4RunResult", "run_sprint4_pipeline"]
