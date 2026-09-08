from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from argus.final_evaluation import verify as final_verify
from argus.final_evaluation.contract import (
    FinalEvaluationContractError,
    create_exclusive_access_receipt,
)
from argus.final_evaluation.evaluation import (
    FinalEvaluationMetricError,
    evaluate_frozen_models,
    stable_sigmoid,
)
from argus.modeling.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    build_artifact_inventory,
    file_fingerprint,
)

_CHAMPION = "graph_enhanced_lightgbm"
_MODELS = (
    _CHAMPION,
    "refined_transaction_lightgbm",
    "graphsage_edge_classifier",
)


def _metric_inputs() -> dict[str, Any]:
    return {
        "labels": np.asarray([1, 0, 1, 0]),
        "source_rows": np.asarray([40, 10, 30, 20]),
        "raw_scores": {
            _CHAMPION: np.asarray([0.0, 0.0, -1.0, 1.0]),
            "refined_transaction_lightgbm": np.asarray([2.0, -2.0, 1.0, -1.0]),
        },
        "frozen_thresholds": {_CHAMPION: 0.0, "refined_transaction_lightgbm": 0.0},
        "model_roles": {
            _CHAMPION: "frozen_champion",
            "refined_transaction_lightgbm": "comparator",
        },
        "validation_average_precision": {
            _CHAMPION: 0.7,
            "refined_transaction_lightgbm": 0.6,
        },
        "top_k": (1, 2),
        "frozen_champion": _CHAMPION,
    }


def test_stable_sigmoid_handles_extreme_logits_without_clipping_rank() -> None:
    values = np.asarray([-1_000.0, -2.0, 0.0, 2.0, 1_000.0])

    probabilities = stable_sigmoid(values)

    assert np.isfinite(probabilities).all()
    assert np.all(np.diff(probabilities) > 0.0)
    assert probabilities[0] == 0.0
    assert probabilities[2] == pytest.approx(0.5)
    assert probabilities[-1] == 1.0


@pytest.mark.parametrize(
    "values",
    [[], [[0.0]], [0.0, float("nan")], [0.0, float("inf")], ["not-numeric"]],
)
def test_stable_sigmoid_rejects_invalid_vectors(values: object) -> None:
    with pytest.raises(FinalEvaluationMetricError):
        stable_sigmoid(values)  # type: ignore[arg-type]


def test_frozen_metrics_use_inclusive_threshold_and_predeclared_presentation_order() -> None:
    comparison, top_k, curves = evaluate_frozen_models(**_metric_inputs())

    assert [row["model"] for row in comparison] == [
        _CHAMPION,
        "refined_transaction_lightgbm",
    ]
    champion = comparison[0]
    assert champion["true_positive"] == 1
    assert champion["false_positive"] == 2
    assert champion["false_negative"] == 1
    assert champion["true_negative"] == 0
    assert champion["precision"] == pytest.approx(1 / 3)
    assert champion["recall"] == pytest.approx(1 / 2)
    assert champion["f1"] == pytest.approx(0.4)
    assert champion["threshold_comparison"] == "score_greater_than_or_equal"
    assert champion["model_selection_used_test"] is False
    assert comparison[1]["average_precision"] > champion["average_precision"]
    assert comparison[0]["role"] == "frozen_champion"
    assert top_k[1]["tie_break_material_at_cutoff"] is True
    assert top_k[1]["true_positives"] == 0
    assert set(curves["partition"]) == {"test"}
    assert set(curves["evaluation_role"]) == {"one_shot_confirmatory"}


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("labels", np.asarray([257, 0, 1, 0]), "binary"),
        ("source_rows", np.asarray([40.5, 10, 30, 20]), "integer"),
        ("frozen_thresholds", {_CHAMPION: True}, "threshold"),
        ("validation_average_precision", {_CHAMPION: float("nan")}, "average precision"),
        ("model_roles", {_CHAMPION: "test_winner"}, "roles"),
    ],
)
def test_frozen_metrics_reject_lossy_or_unfrozen_inputs(
    field: str,
    value: object,
    match: str,
) -> None:
    arguments = _metric_inputs()
    if field in {"frozen_thresholds", "validation_average_precision", "model_roles"}:
        arguments[field] = {**arguments[field], **value}  # type: ignore[arg-type]
    else:
        arguments[field] = value

    with pytest.raises(FinalEvaluationMetricError, match=match):
        evaluate_frozen_models(**arguments)


def test_exclusive_receipt_uses_one_winner_and_never_overwrites(tmp_path: Path) -> None:
    freeze_path = tmp_path / "freeze_contract.json"
    receipt_path = tmp_path / "FINAL_TEST_OPENED.json"
    freeze = {"frozen_champion": _CHAMPION}
    atomic_write_json(freeze, freeze_path)

    def attempt() -> dict[str, Any] | FinalEvaluationContractError:
        try:
            return create_exclusive_access_receipt(
                receipt_path,
                freeze_contract_path=freeze_path,
                freeze_contract=freeze,
            )
        except FinalEvaluationContractError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _: attempt(), range(8)))

    successes = [result for result in outcomes if isinstance(result, dict)]
    failures = [result for result in outcomes if isinstance(result, FinalEvaluationContractError)]
    assert len(successes) == 1
    assert len(failures) == 7
    original = receipt_path.read_bytes()
    assert json.loads(original)["retry_permitted"] is False

    with pytest.raises(FinalEvaluationContractError, match="already exists"):
        create_exclusive_access_receipt(
            receipt_path,
            freeze_contract_path=freeze_path,
            freeze_contract=freeze,
        )
    assert receipt_path.read_bytes() == original


def _write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    connection = duckdb.connect()
    try:
        connection.register("synthetic_predictions", frame)
        connection.execute(
            "COPY synthetic_predictions TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(destination)],
        )
    finally:
        connection.close()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _refresh_inventory(run_dir: Path) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = _json(manifest_path)
    inventory = build_artifact_inventory(run_dir, exclude_paths=(manifest_path,))
    manifest["artifacts"] = inventory
    manifest["artifact_count_excluding_manifest"] = len(inventory)
    atomic_write_json(manifest, manifest_path)


def _build_synthetic_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    timestamps = pd.date_range("2025-01-01", periods=6, freq="h")
    labels = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.int8)
    source_rows = np.asarray([101, 102, 103, 104, 105, 106], dtype=np.int64)
    raw_scores = {
        _CHAMPION: np.asarray([2.0, -2.0, 1.0, -1.0, 0.5, -0.5]),
        "refined_transaction_lightgbm": np.asarray([0.2, 1.2, -0.2, -1.0, 2.0, 0.0]),
        "graphsage_edge_classifier": np.asarray([-1.0, 1.0, 2.0, -2.0, 0.3, -0.3]),
    }
    thresholds = {name: 0.0 for name in _MODELS}
    roles = {name: ("frozen_champion" if name == _CHAMPION else "comparator") for name in _MODELS}
    validation_ap = {_CHAMPION: 0.75, _MODELS[1]: 0.65, _MODELS[2]: 0.55}

    frame_data: dict[str, Any] = {
        "transaction_id": [f"TX-{index}" for index in range(len(labels))],
        "source_row_number": source_rows,
        "timestamp": timestamps,
        "evaluation_partition": ["test"] * len(labels),
        "is_laundering": labels,
    }
    for name in _MODELS:
        frame_data[f"raw_score_{name}"] = raw_scores[name]
        frame_data[f"probability_{name}"] = stable_sigmoid(raw_scores[name])
        frame_data[f"alert_{name}"] = raw_scores[name] >= thresholds[name]
    predictions = pd.DataFrame(frame_data)

    freeze = {
        "schema": "argus.final_evaluation.freeze_contract.v1",
        "created_before_final_test_access": True,
        "frozen_champion": _CHAMPION,
        "selection_partition": "validation",
        "evaluation_partition": "test",
        "model_selection_locked_before_test": True,
        "thresholds_locked_before_test": True,
        "final_test_rows_read": False,
        "final_test_labels_read": False,
        "validation_partition": {"positive_rate": 0.25},
        "test_partition_metadata_only": {
            "rows": len(labels),
            "positives": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "minimum_timestamp": timestamps.min().isoformat(),
            "maximum_timestamp": timestamps.max().isoformat(),
        },
        "models": {
            name: {
                "role": roles[name],
                "frozen_raw_threshold": thresholds[name],
                "validation_average_precision": validation_ap[name],
            }
            for name in _MODELS
        },
        "top_k": [1, 3, 10],
    }
    freeze_path = run_dir / "freeze_contract.json"
    atomic_write_json(freeze, freeze_path)
    predictions_path = run_dir / "final_test_predictions.parquet"
    _write_parquet(predictions, predictions_path)

    comparison, top_k, curves = evaluate_frozen_models(
        labels,
        source_rows,
        raw_scores,
        frozen_thresholds=thresholds,
        model_roles=roles,
        validation_average_precision=validation_ap,
        top_k=freeze["top_k"],
        frozen_champion=_CHAMPION,
    )
    comparison_payload = {
        "partition": "test",
        "presentation_order": "frozen_role_then_model_name_not_test_metric",
        "frozen_champion": _CHAMPION,
        "champion_frozen_before_test": True,
        "test_metrics_used_for_model_selection": False,
        "test_metrics_used_for_threshold_selection": False,
        "models": comparison,
    }
    atomic_write_json(comparison_payload, run_dir / "final_model_comparison.json")
    atomic_write_csv(pd.DataFrame(comparison), run_dir / "final_model_comparison.csv")
    champion_metrics = next(row for row in comparison if row["model"] == _CHAMPION)
    atomic_write_json(
        {
            "frozen_champion": _CHAMPION,
            "champion_frozen_before_test": True,
            "model_selection_used_test": False,
            "threshold_tuned_on_test": False,
            "post_test_tuning_or_retraining": False,
            "metrics": champion_metrics,
        },
        run_dir / "final_metrics.json",
    )
    atomic_write_csv(pd.DataFrame([champion_metrics]), run_dir / "final_metrics.csv")
    atomic_write_json({"rows": top_k}, run_dir / "final_top_k_metrics.json")
    atomic_write_csv(pd.DataFrame(top_k), run_dir / "final_top_k_metrics.csv")
    atomic_write_csv(curves, run_dir / "final_pr_curves.csv")
    confusion = pd.DataFrame(
        [
            {
                "model": row["model"],
                "true_positive": row["true_positive"],
                "false_positive": row["false_positive"],
                "false_negative": row["false_negative"],
                "true_negative": row["true_negative"],
            }
            for row in comparison
        ]
    )
    atomic_write_csv(confusion, run_dir / "confusion_matrices.csv")
    atomic_write_json(
        {
            "threshold_source_partition": "validation",
            "models": [
                {"model": name, "frozen_raw_threshold": thresholds[name]} for name in _MODELS
            ],
        },
        run_dir / "frozen_threshold_details.json",
    )
    atomic_write_json(
        {"frozen_champion": _CHAMPION, "models": list(_MODELS)},
        run_dir / "model_metadata.json",
    )
    atomic_write_json(
        {
            "message_context_partition": "train",
            "message_context_rows": 300_000,
            "test_labels_used_for_graph_construction": False,
            "validation_edges_used_for_message_passing": False,
            "test_edges_used_for_message_passing": False,
        },
        run_dir / "final_graph_inference_manifest.json",
    )
    atomic_write_json(
        {
            "status": "PASS",
            "exact_split_membership_join_performed_inside_authorized_one_shot_run": True,
            "raw_test_reopened_by_post_run_verifier": False,
        },
        run_dir / "test_identity_audit.json",
    )
    atomic_write_json(
        {
            "validation": {"positive_rate": 0.25},
            "test": {"positive_rate": float(labels.mean())},
            "split_boundaries_changed": False,
            "prevalence_rebalanced": False,
        },
        run_dir / "prevalence_shift.json",
    )
    atomic_write_json(
        {
            "evaluation_partition": "test",
            "one_shot_final_evaluation": True,
            "final_test_opened": True,
            "frozen_champion_model": _CHAMPION,
            "champion_frozen_before_test": True,
            "test_used_for_model_selection": False,
            "tuning_after_test": False,
            "final_test_access_count": 1,
        },
        run_dir / "product" / "final_test_summary.json",
    )

    opened_path = run_dir / "FINAL_TEST_OPENED.json"
    create_exclusive_access_receipt(
        opened_path,
        freeze_contract_path=freeze_path,
        freeze_contract=freeze,
    )
    atomic_write_json(
        {
            "schema": "argus.final_test_completion_receipt.v1",
            "status": "COMPLETED",
            "opened_receipt": file_fingerprint(opened_path),
            "predictions": file_fingerprint(predictions_path),
            "rows": len(labels),
            "positives": int(labels.sum()),
            "frozen_champion": _CHAMPION,
            "retry_permitted": False,
            "tuning_after_test": False,
            "retraining_after_test": False,
        },
        run_dir / "FINAL_TEST_COMPLETED.json",
    )

    manifest_path = run_dir / "run_manifest.json"
    inventory = build_artifact_inventory(run_dir, exclude_paths=(manifest_path,))
    acceptance = {key: True for key in final_verify._POSITIVE_ACCEPTANCE_KEYS}
    acceptance.update({key: False for key in final_verify._NEGATIVE_ACCEPTANCE_KEYS})
    atomic_write_json(
        {
            "schema": "argus.final_evaluation.run_manifest.v1",
            "sprint": 5,
            "status": "PASS",
            "frozen_champion": _CHAMPION,
            "final_test_execution_count": 1,
            "final_test_reexecution_permitted": False,
            "final_model_comparison": comparison,
            "acceptance": acceptance,
            "post_test_actions": {"threshold_tuning": False, "model_retraining": False},
            "artifacts": inventory,
            "artifact_count_excluding_manifest": len(inventory),
        },
        manifest_path,
    )
    return run_dir


@pytest.fixture
def synthetic_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run_dir = _build_synthetic_run(tmp_path / "synthetic-final-run")
    monkeypatch.setattr(final_verify, "load_config", lambda _: {"synthetic": True})
    monkeypatch.setattr(final_verify, "get_path", lambda _config, _key: run_dir)
    return run_dir


def _verify_synthetic(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir.parent / "intentionally-absent-final-config.yaml"
    assert not config_path.exists()
    return final_verify.verify_final_evaluation(config_path)


def test_verifier_accepts_self_consistent_synthetic_artifacts(synthetic_run: Path) -> None:
    result = _verify_synthetic(synthetic_run)

    assert result["status"] == "PASS"
    assert result["predictions"]["rows"] == 6
    assert result["metrics"]["models_recomputed"] == list(_MODELS)


def test_verifier_rejects_file_tampering_before_recomputation(synthetic_run: Path) -> None:
    target = synthetic_run / "model_metadata.json"
    target.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(final_verify.FinalEvaluationVerificationError, match="changed"):
        _verify_synthetic(synthetic_run)


def test_verifier_rejects_duplicate_prediction_identities(synthetic_run: Path) -> None:
    frame = final_verify._load_predictions(synthetic_run)
    frame.loc[1, "transaction_id"] = frame.loc[0, "transaction_id"]
    _write_parquet(frame, synthetic_run / "final_test_predictions.parquet")
    _refresh_inventory(synthetic_run)

    with pytest.raises(
        final_verify.FinalEvaluationVerificationError, match="Duplicate transaction"
    ):
        _verify_synthetic(synthetic_run)


def test_verifier_rejects_duplicate_model_rows(synthetic_run: Path) -> None:
    path = synthetic_run / "final_model_comparison.json"
    payload = _json(path)
    payload["models"][1]["model"] = payload["models"][0]["model"]
    atomic_write_json(payload, path)
    _refresh_inventory(synthetic_run)

    with pytest.raises(final_verify.FinalEvaluationVerificationError, match="duplicate model"):
        _verify_synthetic(synthetic_run)


def test_verifier_rejects_probability_not_derived_from_raw_score(synthetic_run: Path) -> None:
    frame = final_verify._load_predictions(synthetic_run)
    frame.loc[0, f"probability_{_CHAMPION}"] += 0.01
    _write_parquet(frame, synthetic_run / "final_test_predictions.parquet")
    _refresh_inventory(synthetic_run)

    with pytest.raises(final_verify.FinalEvaluationVerificationError, match="probability mismatch"):
        _verify_synthetic(synthetic_run)


def test_verifier_rejects_non_boolean_saved_alerts(synthetic_run: Path) -> None:
    frame = final_verify._load_predictions(synthetic_run)
    column = f"alert_{_CHAMPION}"
    frame[column] = frame[column].map({True: "true", False: "false"})
    _write_parquet(frame, synthetic_run / "final_test_predictions.parquet")
    _refresh_inventory(synthetic_run)

    with pytest.raises(final_verify.FinalEvaluationVerificationError, match="not boolean"):
        _verify_synthetic(synthetic_run)


def test_verifier_rejects_frozen_threshold_alert_mismatch(synthetic_run: Path) -> None:
    frame = final_verify._load_predictions(synthetic_run)
    column = f"alert_{_CHAMPION}"
    frame.loc[0, column] = not bool(frame.loc[0, column])
    _write_parquet(frame, synthetic_run / "final_test_predictions.parquet")
    _refresh_inventory(synthetic_run)

    with pytest.raises(final_verify.FinalEvaluationVerificationError, match="threshold alert"):
        _verify_synthetic(synthetic_run)


def test_verifier_rejects_rewritten_metric(synthetic_run: Path) -> None:
    path = synthetic_run / "final_metrics.json"
    payload = _json(path)
    payload["metrics"]["average_precision"] -= 0.1
    atomic_write_json(payload, path)
    _refresh_inventory(synthetic_run)

    with pytest.raises(final_verify.FinalEvaluationVerificationError, match="average_precision"):
        _verify_synthetic(synthetic_run)


def test_verifier_rejects_rewritten_threshold_metric(synthetic_run: Path) -> None:
    path = synthetic_run / "final_model_comparison.json"
    payload = _json(path)
    payload["models"][0]["threshold"] = 0.25
    atomic_write_json(payload, path)
    _refresh_inventory(synthetic_run)

    with pytest.raises(final_verify.FinalEvaluationVerificationError, match="threshold"):
        _verify_synthetic(synthetic_run)


def test_verifier_rejects_false_positive_acceptance_claim(synthetic_run: Path) -> None:
    path = synthetic_run / "run_manifest.json"
    payload = _json(path)
    payload["acceptance"]["test_used_for_selection_or_tuning"] = True
    atomic_write_json(payload, path)

    with pytest.raises(final_verify.FinalEvaluationVerificationError, match="Negative acceptance"):
        _verify_synthetic(synthetic_run)
