from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argus.config import load_config
from argus.modeling.baseline import (
    BaselinePipelineError,
    _close_memmap,
    _feature_contract,
    _prevalence_from_sprint1_metadata,
    _require_model_partition,
)
from argus.modeling.preprocessing import GRAPH_HISTORY_FEATURES


def test_model_partition_gate_rejects_final_test() -> None:
    assert _require_model_partition("train") == "train"
    assert _require_model_partition("validation") == "validation"
    with pytest.raises(BaselinePipelineError, match="restricted to train and validation"):
        _require_model_partition("test")


def test_memmap_is_explicitly_closed_for_windows_cleanup(tmp_path) -> None:
    path = tmp_path / "matrix.npy"
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(2, 2))
    matrix[:] = 1.0

    _close_memmap(matrix)
    path.unlink()

    assert not path.exists()


def test_configured_contract_excludes_all_graph_history() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "baseline.yaml")
    contract = _feature_contract(config["baseline"])

    assert not set(contract.predictor_columns).intersection(GRAPH_HISTORY_FEATURES)
    assert set(GRAPH_HISTORY_FEATURES).issubset(contract.forbidden_predictors)


def test_prevalence_comes_from_frozen_metadata_without_rebalancing() -> None:
    metadata = {
        "strategy": "chronological",
        "strict_boundaries_verified": True,
        "no_transaction_overlap_verified": True,
        "partitions": {
            "train": {
                "rows": 4,
                "positive_labels": 1,
                "minimum_timestamp": "2022-01-01T00:00:00",
                "maximum_timestamp": "2022-01-01T00:03:00",
            },
            "validation": {
                "rows": 2,
                "positive_labels": 1,
                "minimum_timestamp": "2022-01-01T00:04:00",
                "maximum_timestamp": "2022-01-01T00:05:00",
            },
            "test": {
                "rows": 2,
                "positive_labels": 0,
                "minimum_timestamp": "2022-01-01T00:06:00",
                "maximum_timestamp": "2022-01-01T00:07:00",
            },
        },
    }

    prevalence = _prevalence_from_sprint1_metadata(metadata, expected_rows=8)

    assert prevalence["train"]["positive_rate"] == 0.25
    assert prevalence["validation"]["positive_rate"] == 0.5
    assert prevalence["test"]["source"] == "pre_existing_sprint1_split_metadata"
