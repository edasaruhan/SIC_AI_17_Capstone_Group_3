from __future__ import annotations

import json
from types import SimpleNamespace

import duckdb
import joblib
import numpy as np
import pytest

import argus.modeling.baseline as baseline_module
from argus.modeling.artifacts import sha256_file
from argus.modeling.baseline import (
    BaselinePipelineError,
    _atomic_joblib_dump,
    _close_memmap,
    _materialize_partition,
    _verify_frozen_inputs,
    _write_validation_predictions,
)


def _split_metadata() -> dict[str, object]:
    return {
        "strategy": "chronological",
        "strict_boundaries_verified": True,
        "no_transaction_overlap_verified": True,
        "partitions": {
            "train": {
                "rows": 2,
                "positive_labels": 1,
                "minimum_timestamp": "2022-01-01T00:00:00",
                "maximum_timestamp": "2022-01-01T00:01:00",
            },
            "validation": {
                "rows": 1,
                "positive_labels": 0,
                "minimum_timestamp": "2022-01-01T00:02:00",
                "maximum_timestamp": "2022-01-01T00:02:00",
            },
            "test": {
                "rows": 1,
                "positive_labels": 1,
                "minimum_timestamp": "2022-01-01T00:03:00",
                "maximum_timestamp": "2022-01-01T00:03:00",
            },
        },
    }


def _scores() -> dict[str, np.ndarray]:
    return {
        "logistic_regression": np.array([0.1, 0.8]),
        "random_forest": np.array([0.2, 0.7]),
        "lightgbm": np.array([0.3, 0.9]),
    }


def test_frozen_input_verification_returns_auditable_provenance(tmp_path) -> None:
    feature_table = tmp_path / "features.parquet"
    split_table = tmp_path / "splits.parquet"
    split_metadata_path = tmp_path / "split_metadata.json"
    upstream_manifest_path = tmp_path / "manifest.json"
    feature_table.write_bytes(b"feature fixture")
    split_table.write_bytes(b"split fixture")
    split_metadata_path.write_text(json.dumps(_split_metadata()), encoding="utf-8")
    upstream_manifest_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "sprint": "sprint1",
                "features": {"rows": 4},
                "provenance": {
                    "dataset": "synthetic-test",
                    "dataset_is_synthetic": True,
                    "transactions": {"sha256": "transactions"},
                    "accounts": {"sha256": "accounts"},
                },
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "frozen_upstream": {
            "feature_table_sha256": sha256_file(feature_table),
            "split_table_sha256": sha256_file(split_table),
            "split_metadata_sha256": sha256_file(split_metadata_path),
            "expected_rows": 4,
        }
    }

    provenance, prevalence, upstream = _verify_frozen_inputs(
        settings=settings,
        feature_table=feature_table,
        split_table=split_table,
        split_metadata_path=split_metadata_path,
        upstream_manifest_path=upstream_manifest_path,
    )

    assert provenance["dataset"] == "synthetic-test"
    assert provenance["frozen_inputs"]["features.parquet"]["sha256"] == sha256_file(
        feature_table
    )
    assert prevalence["test"]["rows"] == 1
    assert upstream["status"] == "PASS"


def test_frozen_input_verification_rejects_tampering(tmp_path) -> None:
    feature_table = tmp_path / "features.parquet"
    split_table = tmp_path / "splits.parquet"
    metadata = tmp_path / "split_metadata.json"
    upstream = tmp_path / "manifest.json"
    for path in (feature_table, split_table, metadata):
        path.write_bytes(b"current")
    upstream.write_text("{}", encoding="utf-8")
    settings = {
        "frozen_upstream": {
            "feature_table_sha256": "0" * 64,
            "split_table_sha256": sha256_file(split_table),
            "split_metadata_sha256": sha256_file(metadata),
            "expected_rows": 4,
        }
    }

    with pytest.raises(BaselinePipelineError, match="hash mismatch"):
        _verify_frozen_inputs(
            settings=settings,
            feature_table=feature_table,
            split_table=split_table,
            split_metadata_path=metadata,
            upstream_manifest_path=upstream,
        )


def test_materialized_validation_partition_is_disk_backed_and_complete(
    tmp_path, monkeypatch
) -> None:
    batches = [
        SimpleNamespace(
            rows=2,
            matrix=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            target=np.array([0, 1], dtype=np.int8),
            source_row_numbers=np.array([10, 11], dtype=np.int64),
        ),
        SimpleNamespace(
            rows=1,
            matrix=np.array([[5.0, 6.0]], dtype=np.float32),
            target=np.array([0], dtype=np.int8),
            source_row_numbers=np.array([12], dtype=np.int64),
        ),
    ]
    monkeypatch.setattr(
        baseline_module, "iter_duckdb_transformed_batches", lambda *args, **kwargs: iter(batches)
    )

    result = _materialize_partition(
        duckdb.connect(),
        SimpleNamespace(feature_names=("amount", "hour")),
        tmp_path / "features.parquet",
        tmp_path / "splits.parquet",
        partition="validation",
        expected_rows=3,
        expected_positives=1,
        batch_rows=2,
        work_dir=tmp_path,
    )

    assert result.rows == 3
    assert result.batches == 2
    assert result.matrix.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert result.source_row_numbers is not None
    assert result.source_row_numbers.tolist() == [10, 11, 12]
    _close_memmap(result.source_row_numbers)
    _close_memmap(result.target)
    _close_memmap(result.matrix)


def test_materialization_failure_closes_all_memmaps(tmp_path, monkeypatch) -> None:
    duplicate_rows = [
        SimpleNamespace(
            rows=2,
            matrix=np.ones((2, 1), dtype=np.float32),
            target=np.array([0, 1], dtype=np.int8),
            source_row_numbers=np.array([7, 7], dtype=np.int64),
        )
    ]
    monkeypatch.setattr(
        baseline_module,
        "iter_duckdb_transformed_batches",
        lambda *args, **kwargs: iter(duplicate_rows),
    )

    with pytest.raises(BaselinePipelineError, match="not unique"):
        _materialize_partition(
            duckdb.connect(),
            SimpleNamespace(feature_names=("amount",)),
            tmp_path / "features.parquet",
            tmp_path / "splits.parquet",
            partition="validation",
            expected_rows=2,
            expected_positives=1,
            batch_rows=2,
            work_dir=tmp_path,
        )

    for filename in ("X_validation.npy", "y_validation.npy", "source_rows_validation.npy"):
        path = tmp_path / filename
        path.unlink()
        assert not path.exists()


def test_validation_predictions_are_sorted_and_audited(tmp_path) -> None:
    connection = duckdb.connect()
    destination = tmp_path / "nested" / "validation_predictions.parquet"

    audit = _write_validation_predictions(
        connection,
        source_row_numbers=np.array([8, 3]),
        labels=np.array([1, 0]),
        scores_by_model=_scores(),
        source_filename="source'file.csv",
        destination=destination,
        compression="zstd",
    )

    rows = connection.execute(
        "SELECT transaction_id, source_row_number FROM read_parquet(?) ORDER BY source_row_number",
        [str(destination)],
    ).fetchall()
    assert rows == [("source'file.csv:row-3", 3), ("source'file.csv:row-8", 8)]
    assert audit["rows"] == 2
    assert audit["unique_transaction_ids"] == 2
    assert audit["test_predictions_included"] is False


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (
            {"logistic_regression": np.array([0.1, 0.2])},
            "exactly the three reviewed baseline models",
        ),
        (
            {**_scores(), "lightgbm": np.array([0.3, np.nan])},
            "finite probabilities",
        ),
        (
            {**_scores(), "lightgbm": np.array([0.3])},
            "score length differs",
        ),
    ],
)
def test_validation_predictions_reject_invalid_scores(tmp_path, scores, message) -> None:
    destination = tmp_path / "validation_predictions.parquet"

    with pytest.raises(BaselinePipelineError, match=message):
        _write_validation_predictions(
            duckdb.connect(),
            source_row_numbers=np.array([1, 2]),
            labels=np.array([0, 1]),
            scores_by_model=scores,
            source_filename="source.csv",
            destination=destination,
            compression="zstd",
        )

    assert not destination.exists()


def test_atomic_model_write_preserves_previous_file_on_replace_failure(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "model.joblib"
    joblib.dump({"version": "old"}, destination)

    def fail_replace(source, target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(baseline_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _atomic_joblib_dump({"version": "new"}, destination, compression=0)

    assert joblib.load(destination) == {"version": "old"}
    assert list(tmp_path.glob(".*.tmp")) == []
