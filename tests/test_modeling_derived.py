from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from argus.modeling import derived
from argus.modeling.artifacts import build_artifact_inventory


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.unlink(missing_ok=True)
    connection = duckdb.connect()
    try:
        connection.register("prediction_source", frame)
        connection.execute("COPY prediction_source TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


def _seed_run(tmp_path: Path) -> tuple[dict[str, Any], Path, pd.DataFrame]:
    run_dir = tmp_path / "sprint2"
    run_dir.mkdir()
    prediction = pd.DataFrame(
        {
            "transaction_id": [f"synthetic.csv:row-{index}" for index in range(4)],
            "source_row_number": [0, 1, 2, 3],
            "is_laundering": [0, 1, 0, 1],
            "score_logistic_regression": [0.8, 0.7, 0.6, 0.5],
            "score_random_forest": [0.1, 0.9, 0.8, 0.7],
            "score_lightgbm": [0.1, 0.9, 0.2, 0.8],
        }
    )
    _write_parquet(prediction, run_dir / "validation_predictions.parquet")
    for model_name in ("logistic_regression", "random_forest", "lightgbm"):
        _write_json(
            run_dir / "models" / model_name / "metadata.json",
            {
                "model": model_name,
                "implementation": f"Synthetic{model_name}",
                "threshold_basis": "fixed_from_config_not_optimized",
                "runtime_seconds": {"fit": 0.1, "validation_predict": 0.01},
            },
        )
    upstream_path = tmp_path / "upstream_manifest.json"
    _write_json(
        upstream_path,
        {
            "provenance": {
                "transactions": {"sha256": "1" * 64},
                "accounts": {"sha256": "2" * 64},
            }
        },
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "status": "PASS",
        "split_prevalence": {"validation": {"rows": 4, "positive_labels": 2}},
        "provenance": {
            "dataset": "synthetic",
            "frozen_inputs": {
                "transaction_features.parquet": {"sha256": "3" * 64},
                "split_manifest.parquet": {"sha256": "4" * 64},
                "split_metadata.json": {"sha256": "5" * 64},
            },
        },
        "feature_contract": {"state_sha256": "6" * 64},
        "configuration": {"random_seed": 42},
        "libraries": {"scikit_learn": "synthetic", "lightgbm": "synthetic"},
    }
    _write_json(manifest_path, manifest)
    manifest["artifacts"] = build_artifact_inventory(
        run_dir,
        exclude_paths=[manifest_path],
    )
    manifest["artifact_count_excluding_manifest"] = len(manifest["artifacts"])
    _write_json(manifest_path, manifest)
    config = {
        "paths": {
            "run_dir": str(run_dir),
            "upstream_manifest": str(upstream_path),
        },
        "baseline": {
            "decision_threshold": 0.5,
            "top_k": [2],
            "feature_scope": "transaction_only",
        },
    }
    return config, run_dir, prediction


def test_refresh_rebuilds_metrics_without_refitting_or_test_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run_dir, _ = _seed_run(tmp_path)
    plots: list[Path] = []
    monkeypatch.setattr(derived, "load_config", lambda _: config)
    monkeypatch.setattr(
        derived,
        "plot_metric_comparison",
        lambda _frame, path: plots.append(Path(path)),
    )
    monkeypatch.setattr(
        derived,
        "plot_precision_recall_curves",
        lambda _labels, _scores, path: plots.append(Path(path)),
    )

    result = derived.refresh_sprint2_derived_artifacts("synthetic.yaml")

    assert result["status"] == "PASS"
    assert result["model_refit_performed"] is False
    assert result["validation_rows"] == 4
    assert result["champion"] == "lightgbm"
    assert len(plots) == 2
    comparison = json.loads((run_dir / "model_comparison.json").read_text(encoding="utf-8"))
    assert comparison["evaluation_partition"] == "validation"
    assert comparison["accuracy_is_primary"] is False
    assert set(comparison["model_results"]) == {
        "logistic_regression",
        "random_forest",
        "lightgbm",
    }
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    evidence = manifest["derived_artifacts"]
    assert evidence["model_refit_performed"] is False
    assert evidence["test_inference_performed"] is False
    assert evidence["precision_recall_figure"]["numeric_metrics_use_all_validation_rows"] is True
    for model_name in comparison["model_results"]:
        metadata = json.loads(
            (run_dir / "models" / model_name / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["metric_schema_version"] == 2
        assert metadata["top_k_cutoff_tie_diagnostics"] is True


def test_refresh_refuses_to_rebaseline_a_changed_prediction_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run_dir, prediction = _seed_run(tmp_path)
    monkeypatch.setattr(derived, "load_config", lambda _: config)
    tampered = prediction.copy()
    tampered.loc[0, "score_lightgbm"] = 0.99
    _write_parquet(tampered, run_dir / "validation_predictions.parquet")

    with pytest.raises(RuntimeError, match="fingerprint differs"):
        derived.refresh_sprint2_derived_artifacts("synthetic.yaml")

    assert not (run_dir / "models" / "lightgbm" / "validation_metrics.json").exists()
    unchanged = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert "derived_artifacts" not in unchanged


def test_refresh_rejects_incomplete_run_and_split_count_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run_dir, _ = _seed_run(tmp_path)
    monkeypatch.setattr(derived, "load_config", lambda _: config)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "RUNNING"
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="No completed Sprint 2"):
        derived.refresh_sprint2_derived_artifacts("synthetic.yaml")

    manifest["status"] = "PASS"
    manifest["split_prevalence"]["validation"]["rows"] = 5
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="frozen split metadata"):
        derived.refresh_sprint2_derived_artifacts("synthetic.yaml")


def test_refresh_requires_frozen_prediction_inventory_and_json_objects(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="no frozen artifact inventory"):
        derived._verify_frozen_prediction_fingerprint(run_dir, {})
    with pytest.raises(RuntimeError, match="exactly one"):
        derived._verify_frozen_prediction_fingerprint(run_dir, {"artifacts": []})

    array = tmp_path / "array.json"
    _write_json(array, ["not", "an", "object"])
    with pytest.raises(RuntimeError, match="Expected a JSON object"):
        derived._load_json(array)
