"""Temporal-fold and validation-only selection contracts for Sprint 3."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


class RefinementProtocolError(ValueError):
    """Raised when temporal or selection boundaries are unsafe."""


@dataclass(frozen=True)
class TemporalFold:
    """One expanding-window fold contained entirely in frozen outer train."""

    fold: int
    train_minimum_timestamp: datetime
    train_maximum_timestamp: datetime
    validation_minimum_timestamp: datetime
    validation_maximum_timestamp: datetime
    train_rows: int
    train_positive_labels: int
    validation_rows: int
    validation_positive_labels: int

    def __post_init__(self) -> None:
        if self.fold <= 0:
            raise RefinementProtocolError("fold must be positive")
        if not (
            self.train_minimum_timestamp
            <= self.train_maximum_timestamp
            < self.validation_minimum_timestamp
            <= self.validation_maximum_timestamp
        ):
            raise RefinementProtocolError("Fold timestamps are not strictly chronological")
        if min(self.train_rows, self.validation_rows) <= 0:
            raise RefinementProtocolError("Fold partitions must be non-empty")
        if not (0 < self.train_positive_labels < self.train_rows):
            raise RefinementProtocolError("Fold train must contain both classes")
        if not (0 < self.validation_positive_labels < self.validation_rows):
            raise RefinementProtocolError("Fold validation must contain both classes")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
        payload["train_positive_rate"] = self.train_positive_labels / self.train_rows
        payload["validation_positive_rate"] = self.validation_positive_labels / self.validation_rows
        payload["strict_chronology"] = True
        payload["timestamp_groups_split"] = False
        return payload


def _validated_quantiles(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not 0.0 < value <= 1.0 for value in result):
        raise RefinementProtocolError(f"{name} values must be in (0, 1]")
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise RefinementProtocolError(f"{name} values must be strictly increasing")
    return result


def derive_expanding_window_folds(
    connection: duckdb.DuckDBPyConnection,
    feature_table: str | Path,
    split_table: str | Path,
    *,
    train_quantiles: Sequence[float],
    validation_end_quantiles: Sequence[float],
) -> list[TemporalFold]:
    """Derive target-independent timestamp-group-safe folds inside outer train."""

    train_q = _validated_quantiles(train_quantiles, "train_quantiles")
    validation_q = _validated_quantiles(validation_end_quantiles, "validation_end_quantiles")
    if len(train_q) != len(validation_q):
        raise RefinementProtocolError("Temporal fold quantile lists must have equal length")
    if any(train >= validation for train, validation in zip(train_q, validation_q, strict=True)):
        raise RefinementProtocolError("Each train quantile must precede its validation end")

    split_path = str(Path(split_table).resolve())
    feature_path = str(Path(feature_table).resolve())
    all_quantiles = sorted(set((*train_q, *validation_q)))
    literal = "[" + ",".join(format(value, ".17g") for value in all_quantiles) + "]"
    boundary_values = connection.execute(
        f"SELECT quantile_disc(timestamp, {literal}) "
        "FROM read_parquet(?) WHERE partition = 'train'",
        [split_path],
    ).fetchone()[0]
    boundaries = dict(zip(all_quantiles, boundary_values, strict=True))

    folds: list[TemporalFold] = []
    for index, (train_fraction, validation_fraction) in enumerate(
        zip(train_q, validation_q, strict=True), start=1
    ):
        train_end = boundaries[train_fraction]
        validation_end = boundaries[validation_fraction]
        row = connection.execute(
            """
            SELECT
                min(CASE WHEN split.timestamp <= ? THEN split.timestamp END),
                max(CASE WHEN split.timestamp <= ? THEN split.timestamp END),
                min(CASE WHEN split.timestamp > ? AND split.timestamp <= ?
                         THEN split.timestamp END),
                max(CASE WHEN split.timestamp > ? AND split.timestamp <= ?
                         THEN split.timestamp END),
                count(*) FILTER (WHERE split.timestamp <= ?),
                sum(feature.is_laundering) FILTER (WHERE split.timestamp <= ?),
                count(*) FILTER (WHERE split.timestamp > ? AND split.timestamp <= ?),
                sum(feature.is_laundering) FILTER (
                    WHERE split.timestamp > ? AND split.timestamp <= ?
                )
            FROM read_parquet(?) split
            INNER JOIN read_parquet(?) feature USING (transaction_id)
            WHERE split.partition = 'train'
            """,
            [
                train_end,
                train_end,
                train_end,
                validation_end,
                train_end,
                validation_end,
                train_end,
                train_end,
                train_end,
                validation_end,
                train_end,
                validation_end,
                split_path,
                feature_path,
            ],
        ).fetchone()
        folds.append(
            TemporalFold(
                fold=index,
                train_minimum_timestamp=row[0],
                train_maximum_timestamp=row[1],
                validation_minimum_timestamp=row[2],
                validation_maximum_timestamp=row[3],
                train_rows=int(row[4]),
                train_positive_labels=int(row[5]),
                validation_rows=int(row[6]),
                validation_positive_labels=int(row[7]),
            )
        )
    return folds


def write_fold_split_manifest(
    connection: duckdb.DuckDBPyConnection,
    split_table: str | Path,
    fold: TemporalFold,
    destination: str | Path,
    *,
    compression: str = "zstd",
) -> Path:
    """Persist only fold train/validation membership; no target is consulted."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary_sql = temporary.resolve().as_posix().replace("'", "''")
    split_path = str(Path(split_table).resolve())
    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    transaction_id,
                    timestamp,
                    CASE
                        WHEN timestamp <= ? THEN 'train'
                        WHEN timestamp <= ? THEN 'validation'
                    END AS partition
                FROM read_parquet(?)
                WHERE partition = 'train' AND timestamp <= ?
                ORDER BY timestamp, transaction_id
            ) TO '{temporary_sql}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
            """,
            [
                fold.train_maximum_timestamp,
                fold.validation_maximum_timestamp,
                split_path,
                fold.validation_maximum_timestamp,
            ],
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    counts = dict(
        connection.execute(
            "SELECT partition, count(*) FROM read_parquet(?) GROUP BY partition",
            [str(output.resolve())],
        ).fetchall()
    )
    if counts != {"train": fold.train_rows, "validation": fold.validation_rows}:
        raise RefinementProtocolError(f"Persisted fold membership differs: {counts}")
    return output


def summarize_and_select_candidates(
    trials: pd.DataFrame,
    *,
    expected_folds: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Select each family by mean temporal-fold AP, never outer validation/test."""

    required = {
        "model_family",
        "candidate_id",
        "fold",
        "average_precision",
        "eligible",
    }
    missing = required - set(trials.columns)
    if missing:
        raise RefinementProtocolError(f"Candidate trials lack columns: {sorted(missing)}")
    if "partition" in trials and set(trials["partition"]) != {"inner_validation"}:
        raise RefinementProtocolError("Tuning trials must be inner-validation only")
    if "test_metric" in trials:
        raise RefinementProtocolError("Test metrics are forbidden in candidate selection")

    rows: list[dict[str, Any]] = []
    for (family, candidate_id), group in trials.groupby(
        ["model_family", "candidate_id"], sort=True
    ):
        if len(group) != expected_folds or group["fold"].nunique() != expected_folds:
            raise RefinementProtocolError(
                f"Candidate {candidate_id} does not have exactly {expected_folds} folds"
            )
        scores = group["average_precision"].to_numpy(dtype=float)
        rows.append(
            {
                "model_family": family,
                "candidate_id": candidate_id,
                "folds": expected_folds,
                "mean_average_precision": float(np.mean(scores)),
                "std_average_precision": float(np.std(scores, ddof=0)),
                "minimum_fold_average_precision": float(np.min(scores)),
                "all_folds_eligible": bool(group["eligible"].all()),
                "total_fit_seconds": float(group["fit_seconds"].sum()),
                "total_predict_seconds": float(group["predict_seconds"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    selected: dict[str, dict[str, Any]] = {}
    for family, group in summary.groupby("model_family", sort=True):
        eligible = group[group["all_folds_eligible"]]
        if eligible.empty:
            raise RefinementProtocolError(f"No eligible candidate remains for {family}")
        winner = eligible.sort_values(
            ["mean_average_precision", "candidate_id"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        selected[str(family)] = {
            "candidate_id": str(winner["candidate_id"]),
            "selection_partition": "inner_temporal_validation_folds",
            "selection_metric": "mean_average_precision",
            "selection_metric_value": float(winner["mean_average_precision"]),
            "test_metrics_used": False,
        }
    return summary.sort_values(
        ["model_family", "mean_average_precision", "candidate_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True), selected


def select_outer_validation_champion(
    results: Mapping[str, Mapping[str, Any]],
    *,
    partition: str,
) -> dict[str, Any]:
    """Select a refined transaction champion from outer validation AP only."""

    if partition != "validation":
        raise RefinementProtocolError("Refined champion selection is validation-only")
    if not results:
        raise RefinementProtocolError("At least one refined candidate is required")
    ranked = sorted(
        (
            (float(payload["ranking_metrics"]["average_precision"]), name)
            for name, payload in results.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return {
        "champion_model": ranked[0][1],
        "selection_partition": "validation",
        "selection_metric": "average_precision_on_raw_ranking_score",
        "selection_metric_value": ranked[0][0],
        "test_metrics_used": False,
        "candidate_scores": {name: score for score, name in ranked},
    }
