"""Deterministic train/validation/test splits with strict time separation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TemporalSplitError(ValueError):
    """Raised when a strict chronological three-way split is impossible."""


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological partitions and their machine-readable evidence."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any]

    def __iter__(self):
        yield self.train
        yield self.validation
        yield self.test


def _timestamp_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "minimum_timestamp": frame["timestamp"].min().isoformat(),
        "maximum_timestamp": frame["timestamp"].max().isoformat(),
        "unique_timestamps": int(frame["timestamp"].nunique()),
        "positive_labels": (
            int(frame["is_laundering"].sum()) if "is_laundering" in frame else None
        ),
    }


def _write_metadata(metadata: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def chronological_split(
    transactions: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    train_end: str | pd.Timestamp | None = None,
    validation_end: str | pd.Timestamp | None = None,
    metadata_path: str | Path | None = None,
) -> TemporalSplit:
    """Split rows chronologically while keeping equal timestamps together.

    Fractional boundaries target row proportions and snap to timestamp-group
    boundaries. This can make actual proportions differ slightly from requested
    values but guarantees strict ``train < validation < test`` timestamps.
    """

    if "timestamp" not in transactions:
        raise TemporalSplitError("timestamp column is required")
    if "transaction_id" not in transactions:
        raise TemporalSplitError("transaction_id is required for overlap checks")

    ordered = transactions.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["timestamp"].isna().any():
        raise TemporalSplitError("All timestamps must be valid before splitting")
    if ordered["transaction_id"].duplicated().any():
        raise TemporalSplitError("transaction_id must be unique before splitting")
    ordered = ordered.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(
        drop=True
    )
    unique_timestamps = ordered["timestamp"].drop_duplicates().reset_index(drop=True)
    if len(unique_timestamps) < 3:
        raise TemporalSplitError(
            "At least three distinct timestamps are required for strict train/validation/test"
        )

    explicit = train_end is not None or validation_end is not None
    if explicit:
        if train_end is None or validation_end is None:
            raise TemporalSplitError("train_end and validation_end must be provided together")
        train_boundary = pd.Timestamp(train_end)
        validation_boundary = pd.Timestamp(validation_end)
        if train_boundary >= validation_boundary:
            raise TemporalSplitError("train_end must be strictly earlier than validation_end")
        train_mask = ordered["timestamp"].le(train_boundary)
        validation_mask = ordered["timestamp"].gt(train_boundary) & ordered["timestamp"].le(
            validation_boundary
        )
        test_mask = ordered["timestamp"].gt(validation_boundary)
        requested = {
            "mode": "explicit_boundaries",
            "train_end": train_boundary.isoformat(),
            "validation_end": validation_boundary.isoformat(),
        }
    else:
        fractions = np.array([train_fraction, validation_fraction, test_fraction], dtype=float)
        if (~np.isfinite(fractions)).any() or (fractions <= 0).any():
            raise TemporalSplitError("All split fractions must be finite and positive")
        if not np.isclose(fractions.sum(), 1.0):
            raise TemporalSplitError("Split fractions must sum to 1.0")

        group_counts = ordered.groupby("timestamp", sort=True, observed=True).size()
        cumulative = group_counts.cumsum().to_numpy()
        group_total = len(group_counts)
        train_groups = int(np.searchsorted(cumulative, len(ordered) * train_fraction) + 1)
        validation_groups = int(
            np.searchsorted(cumulative, len(ordered) * (train_fraction + validation_fraction)) + 1
        )
        train_groups = min(max(train_groups, 1), group_total - 2)
        validation_groups = min(max(validation_groups, train_groups + 1), group_total - 1)
        validation_start = unique_timestamps.iloc[train_groups]
        test_start = unique_timestamps.iloc[validation_groups]
        train_mask = ordered["timestamp"].lt(validation_start)
        validation_mask = ordered["timestamp"].ge(validation_start) & ordered["timestamp"].lt(
            test_start
        )
        test_mask = ordered["timestamp"].ge(test_start)
        requested = {
            "mode": "fractions_snapped_to_timestamp_groups",
            "train_fraction": float(train_fraction),
            "validation_fraction": float(validation_fraction),
            "test_fraction": float(test_fraction),
        }

    train = ordered.loc[train_mask].reset_index(drop=True)
    validation = ordered.loc[validation_mask].reset_index(drop=True)
    test = ordered.loc[test_mask].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise TemporalSplitError(
            "Chronological boundaries produced an empty partition; adjust the config"
        )
    if not train["timestamp"].max() < validation["timestamp"].min():
        raise TemporalSplitError("Train and validation timestamps are not strictly separated")
    if not validation["timestamp"].max() < test["timestamp"].min():
        raise TemporalSplitError("Validation and test timestamps are not strictly separated")

    identity_sets = [set(part["transaction_id"].astype(str)) for part in (train, validation, test)]
    if (
        identity_sets[0] & identity_sets[1]
        or identity_sets[0] & identity_sets[2]
        or identity_sets[1] & identity_sets[2]
    ):
        raise TemporalSplitError("A transaction identity appears in more than one partition")

    metadata: dict[str, Any] = {
        "strategy": "chronological",
        "timestamp_groups_kept_intact": True,
        "strict_boundaries_verified": True,
        "no_transaction_overlap_verified": True,
        "requested": requested,
        "total_rows": len(ordered),
        "partitions": {
            "train": _timestamp_summary(train),
            "validation": _timestamp_summary(validation),
            "test": _timestamp_summary(test),
        },
    }
    if metadata_path is not None:
        _write_metadata(metadata, Path(metadata_path))
    return TemporalSplit(train=train, validation=validation, test=test, metadata=metadata)
