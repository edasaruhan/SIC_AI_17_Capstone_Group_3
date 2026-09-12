from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import argus.sprint4.pipeline as sprint4_module
from argus.sprint4.graph_data import build_graph_view
from argus.sprint4.pipeline import (
    Sprint4PipelineError,
    _create_staging_run_directory,
    _publish_staged_run,
    _train_graphsage,
    _write_frame_parquet,
    _write_validation_predictions,
    run_sprint4_pipeline,
)


def _settings() -> dict[str, object]:
    return {
        "torch_threads": 1,
        "model": {
            "node_hidden_dim": 6,
            "node_embedding_dim": 4,
            "edge_hidden_dim": 5,
            "dropout": 0.0,
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
        },
    }


def test_staged_run_publish_replaces_complete_package(tmp_path: Path) -> None:
    published = tmp_path / "artifacts" / "sprint4"
    published.mkdir(parents=True)
    (published / ".gitkeep").touch()
    (published / "old.txt").write_text("old", encoding="utf-8")
    staging = _create_staging_run_directory(published)
    (staging / "run_manifest.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (staging / "new.txt").write_text("new", encoding="utf-8")

    _publish_staged_run(staging, published)

    assert not staging.exists()
    assert not (published / "old.txt").exists()
    assert (published / "new.txt").read_text(encoding="utf-8") == "new"
    assert list(published.parent.glob(".argus_sprint4_backup_*")) == []


def test_staged_run_publish_failure_restores_previous_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "artifacts" / "sprint4"
    published.mkdir(parents=True)
    (published / "sentinel.txt").write_text("previous-pass", encoding="utf-8")
    staging = _create_staging_run_directory(published)
    (staging / "run_manifest.json").write_text('{"status":"PASS"}', encoding="utf-8")
    real_replace = sprint4_module.os.replace

    def fail_staging_swap(source: object, destination: object) -> None:
        if Path(source).resolve() == staging.resolve():
            raise OSError("simulated staging swap failure")
        real_replace(source, destination)

    monkeypatch.setattr(sprint4_module.os, "replace", fail_staging_swap)

    with pytest.raises(Sprint4PipelineError, match="could not be published"):
        _publish_staged_run(staging, published)

    assert (published / "sentinel.txt").read_text(encoding="utf-8") == "previous-pass"
    assert staging.is_dir()
    assert list(published.parent.glob(".argus_sprint4_backup_*")) == []


def test_staged_run_publish_rejects_incomplete_or_failed_stage(tmp_path: Path) -> None:
    published = tmp_path / "artifacts" / "sprint4"
    staging = _create_staging_run_directory(published)

    with pytest.raises(Sprint4PipelineError, match="incomplete"):
        _publish_staged_run(staging, published)

    (staging / "run_manifest.json").write_text('{"status":"FAIL"}', encoding="utf-8")
    with pytest.raises(Sprint4PipelineError, match="status is not PASS"):
        _publish_staged_run(staging, published)


def test_staged_run_publish_rejects_retained_work_directory(tmp_path: Path) -> None:
    published = tmp_path / "artifacts" / "sprint4"
    staging = _create_staging_run_directory(published)
    (staging / "run_manifest.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (staging / ".argus_sprint4_work_incomplete").mkdir()

    with pytest.raises(Sprint4PipelineError, match="still contains work directories"):
        _publish_staged_run(staging, published)


def test_pipeline_failure_after_preflight_preserves_published_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "artifacts" / "sprint4"
    published.mkdir(parents=True)
    sentinel = published / "run_manifest.json"
    sentinel.write_text('{"status":"PASS","run":"previous"}', encoding="utf-8")
    sprint3_dir = tmp_path / "artifacts" / "sprint3"
    sprint3_dir.mkdir(parents=True)
    (sprint3_dir / "validation_predictions.parquet").touch()
    paths = {
        "feature_table": tmp_path / "features.parquet",
        "split_table": tmp_path / "splits.parquet",
        "split_metadata": tmp_path / "split.json",
        "upstream_manifest": tmp_path / "upstream.json",
        "run_dir": published,
        "sprint3_run_dir": sprint3_dir,
        "generated_report": tmp_path / "report.md",
    }
    config = {
        "_meta": {"project_root": str(tmp_path)},
        "project": {"random_seed": 7},
        "baseline": {},
        "sprint4": {
            "frozen_references": {},
            "graph_sampling": {},
            "supervised_training": {},
            "parquet_compression": "zstd",
        },
    }
    monkeypatch.setattr(sprint4_module, "load_config", lambda _path: config)
    monkeypatch.setattr(sprint4_module, "get_path", lambda _config, key: paths[key])
    monkeypatch.setattr(sprint4_module, "_verify_frozen_inputs", lambda **_kwargs: ({}, {}, {}))
    monkeypatch.setattr(sprint4_module, "_verify_sprint3_reference", lambda *_args: ({}, {}))
    monkeypatch.setattr(sprint4_module, "_source_snapshot", lambda _root: {})

    def fail_duckdb_configuration(*_args: object) -> None:
        raise RuntimeError("simulated post-preflight failure")

    monkeypatch.setattr(sprint4_module, "_configure_duckdb", fail_duckdb_configuration)

    with pytest.raises(RuntimeError, match="post-preflight failure"):
        run_sprint4_pipeline(tmp_path / "sprint4.yaml")

    assert sentinel.read_text(encoding="utf-8") == '{"status":"PASS","run":"previous"}'
    staged = list(published.parent.glob(".argus_sprint4_stage_*"))
    assert len(staged) == 1
    assert (staged[0] / "resolved_config.json").is_file()


def test_small_graphsage_training_is_transaction_edge_supervision_only() -> None:
    context = pd.DataFrame(
        {
            "from_node_id": ["001::A", "002::B", "003::C"],
            "to_node_id": ["002::B", "003::C", "001::A"],
            "amount_paid": [10.0, 20.0, 30.0],
        }
    )
    supervised = pd.DataFrame(
        {
            "from_node_id": ["001::A", "002::B", "003::C", "001::A"],
            "to_node_id": ["002::B", "003::C", "001::A", "003::C"],
            "is_laundering": [0, 1, 0, 1],
        }
    )
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    view = build_graph_view(context, required_node_ids=())

    model, adjacency, evidence = _train_graphsage(view, supervised, matrix, _settings(), seed=7)

    assert model.config.transaction_feature_dim == 2
    assert adjacency.edge_count == 3
    assert evidence["target_unit"] == "transaction_edge"
    assert evidence["unsupported_account_level_label_created"] is False
    assert evidence["epochs_completed"] == 2
    assert evidence["outer_validation_used_for_fit_or_early_stopping"] is False
    assert evidence["final_test_used"] is False


def test_validation_prediction_writer_requires_validation_join_and_aligns_references(
    tmp_path: Path,
) -> None:
    connection = duckdb.connect()
    try:
        feature = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2", "T3"],
                "source_row_number": [1, 2, 3],
                "is_laundering": [0, 1, 0],
            }
        )
        split = pd.DataFrame(
            {"transaction_id": ["T1", "T2", "T3"], "partition": ["validation"] * 3}
        )
        reference = pd.DataFrame(
            {
                "source_row_number": [1, 2, 3],
                "is_laundering": [0, 1, 0],
                "raw_score_refined_lightgbm": [0.1, 0.8, 0.2],
                "probability_refined_lightgbm": [0.2, 0.7, 0.3],
                "raw_score_ablation_transaction_temporal_history_graph": [0.0, 0.9, 0.1],
                "probability_ablation_transaction_temporal_history_graph": [0.1, 0.8, 0.2],
            }
        )
        feature_path = _write_frame_parquet(
            connection, feature, tmp_path / "feature.parquet", compression="zstd"
        )
        split_path = _write_frame_parquet(
            connection, split, tmp_path / "split.parquet", compression="zstd"
        )
        reference_path = _write_frame_parquet(
            connection, reference, tmp_path / "reference.parquet", compression="zstd"
        )
        values = {
            "source_row_number": np.array([1, 2, 3]),
            "is_laundering": np.array([0, 1, 0]),
            "raw_score_graphsage_edge_classifier": np.array([-1.0, 2.0, 0.0]),
            "probability_graphsage_edge_classifier": np.array([0.2, 0.9, 0.5]),
        }

        audit = _write_validation_predictions(
            connection,
            values,
            feature_table=feature_path,
            split_table=split_path,
            sprint3_predictions=reference_path,
            destination=tmp_path / "predictions.parquet",
            compression="zstd",
        )

        assert audit["partition"] == "validation"
        assert audit["rows"] == 3
        assert audit["positive_labels"] == 1
        assert audit["out_of_range_probability_rows"] == 0
        assert audit["test_predictions_included"] is False
        columns = (
            connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(tmp_path / "predictions.parquet")],
            )
            .fetch_df()["column_name"]
            .tolist()
        )
        assert "raw_score_graph_enhanced_lightgbm" in columns
        assert "raw_score_graphsage_edge_classifier" in columns
    finally:
        connection.close()


def test_validation_prediction_writer_preserves_previous_artifact_on_bad_reference(
    tmp_path: Path,
) -> None:
    connection = duckdb.connect()
    try:
        feature_path = _write_frame_parquet(
            connection,
            pd.DataFrame(
                {"transaction_id": ["T1"], "source_row_number": [1], "is_laundering": [0]}
            ),
            tmp_path / "feature.parquet",
            compression="zstd",
        )
        split_path = _write_frame_parquet(
            connection,
            pd.DataFrame({"transaction_id": ["T1"], "partition": ["validation"]}),
            tmp_path / "split.parquet",
            compression="zstd",
        )
        reference_path = _write_frame_parquet(
            connection,
            pd.DataFrame(
                {
                    "source_row_number": [1],
                    "is_laundering": [0],
                    "raw_score_refined_lightgbm": [0.1],
                    "probability_refined_lightgbm": [np.nan],
                    "raw_score_ablation_transaction_temporal_history_graph": [0.2],
                    "probability_ablation_transaction_temporal_history_graph": [0.4],
                }
            ),
            tmp_path / "reference.parquet",
            compression="zstd",
        )
        destination = _write_frame_parquet(
            connection,
            pd.DataFrame({"sentinel": [42]}),
            tmp_path / "predictions.parquet",
            compression="zstd",
        )
        values = {
            "source_row_number": np.array([1]),
            "is_laundering": np.array([0]),
            "raw_score_graphsage_edge_classifier": np.array([0.3]),
            "probability_graphsage_edge_classifier": np.array([0.6]),
        }

        with pytest.raises(Sprint4PipelineError, match="pre-commit verification"):
            _write_validation_predictions(
                connection,
                values,
                feature_table=feature_path,
                split_table=split_path,
                sprint3_predictions=reference_path,
                destination=destination,
                compression="zstd",
            )

        assert connection.execute(
            "SELECT sentinel FROM read_parquet(?)", [str(destination)]
        ).fetchall() == [(42,)]
        assert list(tmp_path.glob(".predictions.parquet.*.tmp")) == []
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("invalid_case", "message"),
    [
        ("missing_column", "exactly the reviewed columns"),
        ("duplicate_source", "not unique"),
        ("nonfinite_raw", "raw scores must be finite"),
        ("probability_out_of_range", "probabilities must be bounded"),
    ],
)
def test_validation_prediction_writer_rejects_invalid_graphsage_values(
    tmp_path: Path, invalid_case: str, message: str
) -> None:
    values = {
        "source_row_number": np.array([1, 2]),
        "is_laundering": np.array([0, 1]),
        "raw_score_graphsage_edge_classifier": np.array([-1.0, 2.0]),
        "probability_graphsage_edge_classifier": np.array([0.2, 0.9]),
    }
    if invalid_case == "missing_column":
        values.pop("raw_score_graphsage_edge_classifier")
    elif invalid_case == "duplicate_source":
        values["source_row_number"] = np.array([1, 1])
    elif invalid_case == "nonfinite_raw":
        values["raw_score_graphsage_edge_classifier"] = np.array([np.inf, 2.0])
    else:
        values["probability_graphsage_edge_classifier"] = np.array([0.2, 1.1])

    with pytest.raises(Sprint4PipelineError, match=message):
        _write_validation_predictions(
            duckdb.connect(),
            values,
            feature_table=tmp_path / "feature.parquet",
            split_table=tmp_path / "split.parquet",
            sprint3_predictions=tmp_path / "reference.parquet",
            destination=tmp_path / "predictions.parquet",
            compression="zstd",
        )
