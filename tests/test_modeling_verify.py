from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from argus.modeling import verify
from argus.modeling.artifacts import build_artifact_inventory
from argus.modeling.metrics import evaluate_binary_predictions
from argus.modeling.reporting import PR_CURVE_MAX_PLOT_POINTS


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_parquet(frame: pd.DataFrame, path: Path, *, relation: str) -> None:
    connection = duckdb.connect()
    try:
        connection.register(relation, frame)
        connection.execute(f"COPY {relation} TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


def _manifest_for_entrypoint() -> dict[str, Any]:
    return {
        "status": "PASS",
        "derived_artifacts": {
            "precision_recall_figure": {
                "numeric_metrics_use_all_validation_rows": True,
                "display_only_decimation": "deterministic_even_index_with_endpoints",
                "maximum_points_per_model": PR_CURVE_MAX_PLOT_POINTS,
            }
        },
    }


def _closed_policy() -> dict[str, bool]:
    return {
        "final_test_used_for_training": False,
        "final_test_used_for_preprocessing_fit": False,
        "final_test_used_for_model_selection": False,
        "final_test_inference_performed": False,
        "test_feature_rows_materialized": False,
        "test_prediction_artifact_exists": False,
    }


def _patch_entrypoint_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
) -> None:
    config = {"paths": {"run_dir": str(run_dir)}}
    metrics = {
        name: {"average_precision": score, "roc_auc": 0.8}
        for name, score in (
            ("logistic_regression", 0.1),
            ("random_forest", 0.2),
            ("lightgbm", 0.3),
        )
    }
    monkeypatch.setattr(verify, "load_config", lambda _: config)
    monkeypatch.setattr(verify, "_verify_saved_inventory", lambda *_: 4)
    monkeypatch.setattr(
        verify,
        "_verify_prediction_table",
        lambda *_: (
            {
                "rows": 12,
                "positive_labels": 2,
                "unique_transaction_ids": 12,
                "unique_source_row_numbers": 12,
                "source_filename": "synthetic.csv",
                "evaluation_partition": "validation",
                "frozen_split_membership": {"validation": 12},
                "test_rows_present": False,
            },
            metrics,
        ),
    )
    monkeypatch.setattr(
        verify,
        "_verify_champion",
        lambda *_: {
            "champion_model": "lightgbm",
            "selection_partition": "validation",
        },
    )
    monkeypatch.setattr(
        verify,
        "_verify_comparison_artifacts",
        lambda *_: {
            "partition": "validation",
            "provenance_embedded": True,
            "top_k_cutoff_tie_diagnostics": True,
            "models": 3,
        },
    )
    monkeypatch.setattr(
        verify,
        "write_run_manifest",
        lambda payload, _run_dir: {
            **payload,
            "artifact_count_excluding_manifest": 6,
        },
    )


def test_saved_inventory_accepts_exact_files_and_prior_verification_report(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifact.txt").write_text("frozen", encoding="utf-8")
    inventory = build_artifact_inventory(run_dir)
    manifest = {
        "artifacts": inventory,
        "artifact_count_excluding_manifest": len(inventory),
    }
    _write_json(run_dir / "verification_report.json", {"status": "PASS"})

    assert verify._verify_saved_inventory(run_dir, manifest) == 1


def test_saved_inventory_rejects_count_hash_and_unmanifested_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "artifact.txt"
    artifact.write_text("frozen", encoding="utf-8")
    inventory = build_artifact_inventory(run_dir)
    manifest = {
        "artifacts": inventory,
        "artifact_count_excluding_manifest": len(inventory),
    }

    wrong_count = {**manifest, "artifact_count_excluding_manifest": 99}
    with pytest.raises(verify.BaselineVerificationError, match="count is inconsistent"):
        verify._verify_saved_inventory(run_dir, wrong_count)

    artifact.write_text("tampered artifact", encoding="utf-8")
    with pytest.raises(verify.BaselineVerificationError, match="size differs"):
        verify._verify_saved_inventory(run_dir, manifest)

    artifact.write_text("frozen", encoding="utf-8")
    (run_dir / "rogue.bin").write_bytes(b"rogue")
    with pytest.raises(verify.BaselineVerificationError, match="Unmanifested artifacts"):
        verify._verify_saved_inventory(run_dir, manifest)


def test_prediction_table_reconciles_partition_metrics_and_model_classes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source_rows = np.arange(4, dtype=np.int64)
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    prediction = pd.DataFrame(
        {
            "transaction_id": [f"synthetic.csv:row-{row}" for row in source_rows],
            "source_row_number": source_rows,
            "is_laundering": labels,
            "score_logistic_regression": [0.1, 0.8, 0.2, 0.7],
            "score_random_forest": [0.2, 0.9, 0.1, 0.6],
            "score_lightgbm": [0.05, 0.95, 0.3, 0.65],
        }
    )
    split = pd.DataFrame(
        {
            "transaction_id": prediction["transaction_id"],
            "partition": ["validation"] * len(prediction),
        }
    )
    prediction_path = run_dir / "validation_predictions.parquet"
    split_path = tmp_path / "split.parquet"
    _write_parquet(prediction, prediction_path, relation="prediction_source")
    _write_parquet(split, split_path, relation="split_source")
    config = {
        "paths": {"split_table": str(split_path)},
        "baseline": {"decision_threshold": 0.5, "top_k": [2]},
    }
    manifest = {
        "split_prevalence": {
            "validation": {"rows": 4, "positive_labels": 2},
        }
    }
    model_types = {
        "logistic_regression": SGDClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=2, random_state=42),
        "lightgbm": LGBMClassifier(n_estimators=2, random_state=42),
    }
    for model_name, model in model_types.items():
        model_dir = run_dir / "models" / model_name
        model_dir.mkdir(parents=True)
        scores = prediction[f"score_{model_name}"].to_numpy(dtype=np.float64)
        metrics = evaluate_binary_predictions(
            labels,
            scores,
            source_rows,
            threshold=0.5,
            top_k_values=[2],
        )
        _write_json(model_dir / "validation_metrics.json", metrics)
        joblib.dump(model, model_dir / "model.joblib")

    evidence, metrics = verify._verify_prediction_table(run_dir, manifest, config)

    assert evidence["rows"] == 4
    assert evidence["positive_labels"] == 2
    assert evidence["source_filename"] == "synthetic.csv"
    assert evidence["frozen_split_membership"] == {"validation": 4}
    assert evidence["test_rows_present"] is False
    assert set(metrics) == set(model_types)

    lightgbm_metrics_path = run_dir / "models" / "lightgbm" / "validation_metrics.json"
    tampered = json.loads(lightgbm_metrics_path.read_text(encoding="utf-8"))
    tampered["average_precision"] = 0.0
    _write_json(lightgbm_metrics_path, tampered)
    with pytest.raises(verify.BaselineVerificationError, match="metrics differ for lightgbm"):
        verify._verify_prediction_table(run_dir, manifest, config)


def test_verification_entrypoint_persists_recomputed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_manifest.json", _manifest_for_entrypoint())
    _write_json(run_dir / "final_test_policy.json", _closed_policy())
    _patch_entrypoint_dependencies(monkeypatch, run_dir)

    result = verify.verify_sprint2_run("synthetic.yaml")

    assert result["status"] == "PASS"
    assert result["saved_inventory_entries_verified"] == 4
    assert result["validation_predictions"]["rows"] == 12
    assert result["champion_recomputed"]["champion_model"] == "lightgbm"
    assert result["final_test_policy_verified_closed"] is True
    assert result["test_prediction_artifacts_found"] == 0
    assert result["refreshed_artifact_count_excluding_manifest"] == 6
    persisted = json.loads((run_dir / "verification_report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "PASS"
    assert set(persisted["validation_metrics_recomputed"]) == {
        "logistic_regression",
        "random_forest",
        "lightgbm",
    }


def test_verification_entrypoint_rejects_open_test_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_manifest.json", _manifest_for_entrypoint())
    policy = _closed_policy()
    policy["final_test_inference_performed"] = True
    _write_json(run_dir / "final_test_policy.json", policy)
    _patch_entrypoint_dependencies(monkeypatch, run_dir)

    with pytest.raises(verify.BaselineVerificationError, match="policy is not closed"):
        verify.verify_sprint2_run("synthetic.yaml")


def test_verification_entrypoint_rejects_test_prediction_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_manifest.json", _manifest_for_entrypoint())
    _write_json(run_dir / "final_test_policy.json", _closed_policy())
    (run_dir / "unexpected_test_prediction.csv").write_text("score\n0.5\n", encoding="utf-8")
    _patch_entrypoint_dependencies(monkeypatch, run_dir)

    with pytest.raises(verify.BaselineVerificationError, match="Unexpected test prediction"):
        verify.verify_sprint2_run("synthetic.yaml")


@pytest.mark.parametrize(
    "manifest",
    [
        {**_manifest_for_entrypoint(), "status": "RUNNING"},
        {
            **_manifest_for_entrypoint(),
            "derived_artifacts": {"precision_recall_figure": {}},
        },
    ],
)
def test_verification_entrypoint_requires_verifiable_status_and_plot_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, Any],
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_manifest.json", manifest)
    monkeypatch.setattr(
        verify,
        "load_config",
        lambda _: {"paths": {"run_dir": str(run_dir)}},
    )

    with pytest.raises(verify.BaselineVerificationError):
        verify.verify_sprint2_run("synthetic.yaml")


def test_comparison_verification_rejects_wrong_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "model_comparison.json",
        {"provenance": {"dataset_name": "wrong"}},
    )
    manifest = {
        "provenance": {
            "dataset": "synthetic",
            "raw_sources": {"transactions": {"sha256": "1" * 64}},
            "frozen_inputs": {
                "transaction_features.parquet": {"sha256": "2" * 64},
                "split_manifest.parquet": {"sha256": "3" * 64},
                "split_metadata.json": {"sha256": "4" * 64},
            },
        },
        "feature_contract": {"state_sha256": "5" * 64},
        "configuration": {"random_seed": 42},
        "libraries": {"scikit_learn": "1", "lightgbm": "1"},
    }
    config = {"baseline": {"feature_scope": "transaction_only", "top_k": [10]}}

    with pytest.raises(verify.BaselineVerificationError, match="provenance differs"):
        verify._verify_comparison_artifacts(run_dir, manifest, config, {}, {})


def test_verification_json_loader_reports_invalid_or_non_object_payload(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    with pytest.raises(verify.BaselineVerificationError, match="Could not read JSON"):
        verify._load_object(invalid)

    array = tmp_path / "array.json"
    _write_json(array, ["not", "an", "object"])
    with pytest.raises(verify.BaselineVerificationError, match="is not an object"):
        verify._load_object(array)
