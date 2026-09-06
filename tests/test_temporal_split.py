from __future__ import annotations

import json

import pandas as pd
import pytest

from argus.data.split import TemporalSplitError, chronological_split


def _frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 02:00",
            "2024-01-01 03:00",
            "2024-01-01 04:00",
            "2024-01-01 05:00",
            "2024-01-01 06:00",
            "2024-01-01 07:00",
            "2024-01-01 08:00",
        ]
    )
    return pd.DataFrame(
        {
            "transaction_id": [f"t{index}" for index in range(len(timestamps))],
            "timestamp": timestamps,
            "is_laundering": [0] * len(timestamps),
        }
    )


def test_fractional_split_is_strict_disjoint_and_writes_metadata(tmp_path) -> None:
    metadata_path = tmp_path / "split.json"

    split = chronological_split(_frame(), metadata_path=metadata_path)

    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    train_ids = set(split.train["transaction_id"])
    validation_ids = set(split.validation["transaction_id"])
    test_ids = set(split.test["transaction_id"])
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids
    assert split.train["timestamp"].value_counts().max() == 2
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved["strict_boundaries_verified"] is True
    assert saved["no_transaction_overlap_verified"] is True
    assert sum(part["rows"] for part in saved["partitions"].values()) == len(_frame())


def test_split_is_deterministic_under_input_row_shuffle() -> None:
    first = chronological_split(_frame())
    shuffled = _frame().sample(frac=1, random_state=123).reset_index(drop=True)
    second = chronological_split(shuffled)

    for left, right in zip(first, second, strict=True):
        assert left["transaction_id"].tolist() == right["transaction_id"].tolist()


def test_equal_timestamps_never_cross_partitions() -> None:
    split = chronological_split(_frame())
    membership: dict[pd.Timestamp, set[str]] = {}
    for name, part in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for timestamp in part["timestamp"]:
            membership.setdefault(timestamp, set()).add(name)
    assert all(len(partitions) == 1 for partitions in membership.values())


def test_too_few_timestamp_groups_is_rejected() -> None:
    frame = _frame().iloc[:3].copy()
    with pytest.raises(TemporalSplitError, match="three distinct timestamps"):
        chronological_split(frame)
