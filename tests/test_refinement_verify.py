from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from argus.modeling.artifacts import atomic_write_json
from argus.modeling.metrics import compute_average_precision, evaluate_binary_predictions
from argus.modeling.preprocessing import FEATURE_FAMILY_CONTRACTS
from argus.modeling.refinement_metrics import (
    analyze_validation_score_saturation,
    optimize_validation_thresholds,
)
from argus.modeling.refinement_verify import (
    ALL_ABLATION_FEATURE_FAMILIES,
    PRIMARY_FEATURE_FAMILIES,
    RefinementVerificationError,
    recompute_threshold_artifact,
    validate_ablation_comparability,
    validate_ablation_model_metadata,
    validate_closed_test_policy,
    validate_feature_family_manifests,
    validate_graph_value_conclusion,
    validate_temporal_folds,
    validate_threshold_analysis,
    validate_threshold_artifact,
    validate_tuning_artifacts,
    verify_ablation_metrics_from_predictions,
    verify_core_model_artifacts,
    verify_duplicate_graph_evidence,
    verify_folds_against_frozen_inputs,
    verify_validation_predictions,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _write_parquet(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    destination: Path,
    *,
    relation_name: str,
) -> None:
    connection.register(relation_name, frame)
    try:
        connection.execute(
            f"COPY {relation_name} TO ? (FORMAT PARQUET)",
            [str(destination)],
        )
    finally:
        connection.unregister(relation_name)


def _selected_threshold(payload: dict[str, object], scores: np.ndarray) -> float:
    primary = payload["summaries"]["predeclared_joint_primary_constraint"]  # type: ignore[index]
    point = primary["operating_point"]  # type: ignore[index]
    if point is not None and point["threshold"] is not None:
        return float(point["threshold"])
    return float(np.nextafter(np.max(scores), np.inf))


def _closed_policy() -> dict[str, object]:
    return {
        "final_test_used_for_training": False,
        "final_test_used_for_preprocessing_fit": False,
        "final_test_used_for_tuning": False,
        "final_test_used_for_model_selection": False,
        "final_test_used_for_threshold_selection": False,
        "final_test_inference_performed": False,
        "test_feature_rows_materialized": False,
        "test_prediction_artifact_exists": False,
        "permitted_model_partitions": ["train", "validation"],
        "selection_partition": "validation",
        "test_metadata_reported": True,
        "test_metadata_source": "pre_existing_sprint1_split_metadata.json",
    }


def _access_audit() -> dict[str, object]:
    return {
        "evaluated_partitions": ["inner_validation", "validation"],
        "preprocessor_fit_partitions": ["train", "inner_train"],
        "transformed_partitions": ["train", "validation"],
        "test_access": "pre_existing_metadata_only",
    }


def _fold_payload() -> dict[str, object]:
    start = datetime(2022, 1, 1)
    folds = []
    for number, train_end, validation_start, validation_end in (
        (1, 1, 2, 3),
        (2, 3, 4, 5),
        (3, 5, 6, 7),
    ):
        train_rows = train_end + 1
        validation_rows = validation_end - validation_start + 1
        folds.append(
            {
                "fold": number,
                "train_minimum_timestamp": start.isoformat(),
                "train_maximum_timestamp": (start + timedelta(hours=train_end)).isoformat(),
                "validation_minimum_timestamp": (
                    start + timedelta(hours=validation_start)
                ).isoformat(),
                "validation_maximum_timestamp": (
                    start + timedelta(hours=validation_end)
                ).isoformat(),
                "train_rows": train_rows,
                "train_positive_labels": train_rows // 2,
                "validation_rows": validation_rows,
                "validation_positive_labels": 1,
                "train_positive_rate": (train_rows // 2) / train_rows,
                "validation_positive_rate": 0.5,
                "strict_chronology": True,
                "timestamp_groups_split": False,
            }
        )
    return {"strategy": "expanding_window", "folds": folds}


def _outer_train() -> dict[str, object]:
    return {
        "minimum_timestamp": "2022-01-01T00:00:00",
        "maximum_timestamp": "2022-01-01T07:00:00",
        "rows": 8,
    }


def _manifest_for(family: str) -> dict[str, object]:
    contract = FEATURE_FAMILY_CONTRACTS[family]
    names = [f"numeric__{name}" for name in contract.numeric_features]
    names.extend(f"missing__{name}" for name in contract.missing_indicator_features)
    names.extend(f"onehot__{name}=VALUE" for name in contract.low_cardinality_categories)
    names.extend(f"frequency__{name}" for name in contract.bank_frequency_categories)
    return {
        "fit_scope": "train_only",
        "contract": contract.to_dict(),
        "transformed_feature_names": names,
        "transformed_feature_count": len(names),
    }


def _ablation() -> pd.DataFrame:
    parameters_digest = _digest({"n_estimators": 10, "random_state": 42})
    feature_counts = {
        "transaction_only": 49,
        "transaction_temporal_history": 74,
        "transaction_temporal_history_graph": 79,
        "transaction_temporal_history_graph_novel3": 77,
    }
    return pd.DataFrame(
        [
            {
                "feature_family": family,
                "model_family": "lightgbm",
                "candidate_id": "lgb_stable_sqrt_weight",
                "effective_parameters_sha256": parameters_digest,
                "random_seed": 42,
                "evaluation_partition": "validation",
                "split_manifest_sha256": "split-sha",
                "evaluation_protocol_sha256": "protocol-sha",
                "average_precision": score,
                "roc_auc": roc_auc,
                "transformed_feature_count": feature_counts[family],
                "test_metrics_used": False,
            }
            for family, score, roc_auc in (
                ("transaction_only", 0.10, 0.60),
                ("transaction_temporal_history", 0.20, 0.70),
                ("transaction_temporal_history_graph", 0.25, 0.75),
                ("transaction_temporal_history_graph_novel3", 0.22, 0.72),
            )
        ]
    )


def test_closed_test_policy_requires_explicit_false_flags_and_no_test_artifacts() -> None:
    evidence = validate_closed_test_policy(
        _closed_policy(),
        _access_audit(),
        ["models/lightgbm/model.joblib", "final_test_policy.json"],
    )
    assert evidence["test_derived_artifacts_found"] == 0
    assert evidence["test_access"] == "pre_existing_metadata_only"

    policy = _closed_policy()
    policy["final_test_inference_performed"] = True
    with pytest.raises(RefinementVerificationError, match="explicitly False"):
        validate_closed_test_policy(policy, _access_audit(), [])

    audit = _access_audit()
    audit["evaluated_partitions"] = ["validation", "test"]
    with pytest.raises(RefinementVerificationError, match="Test access recorded"):
        validate_closed_test_policy(_closed_policy(), audit, [])

    with pytest.raises(RefinementVerificationError, match="Test-derived artifact"):
        validate_closed_test_policy(
            _closed_policy(), _access_audit(), ["predictions/test_predictions.parquet"]
        )


def test_temporal_fold_validator_enforces_expansion_and_outer_train_boundary() -> None:
    result = validate_temporal_folds(_fold_payload(), outer_train=_outer_train())
    assert result == {
        "status": "PASS",
        "strategy": "expanding_window",
        "fold_count": 3,
        "inside_frozen_outer_train": True,
        "strict_chronology": True,
        "timestamp_groups_split": False,
    }

    overlapping = _fold_payload()
    overlapping["folds"][1]["validation_minimum_timestamp"] = "2022-01-01T03:00:00"  # type: ignore[index]
    with pytest.raises(RefinementVerificationError, match="strictly chronological"):
        validate_temporal_folds(overlapping, outer_train=_outer_train())

    outside = _fold_payload()
    outside["folds"][-1]["validation_maximum_timestamp"] = "2022-01-01T08:00:00"  # type: ignore[index]
    with pytest.raises(RefinementVerificationError, match="frozen outer train"):
        validate_temporal_folds(outside, outer_train=_outer_train())


def test_temporal_fold_recount_uses_only_frozen_outer_train(tmp_path: Path) -> None:
    start = datetime(2022, 1, 1)
    ids = [f"row-{index}" for index in range(8)]
    split = pd.DataFrame(
        {
            "transaction_id": ids,
            "timestamp": [start + timedelta(hours=index) for index in range(8)],
            "partition": ["train"] * 8,
        }
    )
    feature = pd.DataFrame(
        {
            "transaction_id": ids,
            "is_laundering": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    split_path = tmp_path / "split.parquet"
    feature_path = tmp_path / "features.parquet"
    fold_manifest_dir = tmp_path / "folds"
    fold_manifest_dir.mkdir()
    connection = duckdb.connect()
    try:
        _write_parquet(connection, split, split_path, relation_name="split_frame")
        _write_parquet(connection, feature, feature_path, relation_name="feature_frame")
        for fold in _fold_payload()["folds"]:  # type: ignore[index]
            train_end = datetime.fromisoformat(fold["train_maximum_timestamp"])
            validation_end = datetime.fromisoformat(fold["validation_maximum_timestamp"])
            saved = split.loc[split["timestamp"] <= validation_end].copy()
            saved["partition"] = np.where(saved["timestamp"] <= train_end, "train", "validation")
            _write_parquet(
                connection,
                saved,
                fold_manifest_dir / f"fold_{fold['fold']}.parquet",
                relation_name=f"fold_{fold['fold']}",
            )
    finally:
        connection.close()

    result = verify_folds_against_frozen_inputs(
        _fold_payload(),
        feature_table=feature_path,
        split_table=split_path,
        fold_manifest_dir=fold_manifest_dir,
    )
    assert result["folds_recounted"] == 3
    assert result["saved_fold_manifests_reconciled"] == 3
    assert result["query_partition"] == "train"
    assert result["test_rows_queried"] is False

    tampered = _fold_payload()
    tampered["folds"][0]["train_rows"] = 3  # type: ignore[index]
    with pytest.raises(RefinementVerificationError, match="recount differs"):
        verify_folds_against_frozen_inputs(
            tampered, feature_table=feature_path, split_table=split_path
        )


def test_feature_family_manifests_match_reviewed_strictly_nested_contracts() -> None:
    manifests = {family: _manifest_for(family) for family in PRIMARY_FEATURE_FAMILIES}
    result = validate_feature_family_manifests(manifests)
    assert result["strictly_nested"] is True
    assert result["c_minus_b"] == [
        "sender_prior_fan_out_degree",
        "sender_prior_fan_in_degree",
        "receiver_prior_fan_out_degree",
        "receiver_prior_fan_in_degree",
        "pair_previous_transfer_count",
    ]

    corrupted = deepcopy(manifests)
    corrupted["transaction_temporal_history_graph"]["contract"]["numeric_features"].pop()  # type: ignore[index,union-attr]
    with pytest.raises(RefinementVerificationError, match="contract differs"):
        validate_feature_family_manifests(corrupted)


def test_ablation_comparison_is_validation_only_and_changes_features_alone() -> None:
    result = validate_ablation_comparability(
        _ablation(), expected_split_sha256="split-sha", expected_random_seed=42
    )
    assert result["same_model_family"] is True
    assert result["same_effective_parameters"] is True
    assert result["same_evaluation_protocol"] is True

    changed_model = _ablation()
    changed_model.loc[2, "candidate_id"] = "different"
    with pytest.raises(RefinementVerificationError, match="non-feature factor"):
        validate_ablation_comparability(
            changed_model, expected_split_sha256="split-sha", expected_random_seed=42
        )

    test_leak = _ablation().assign(test_average_precision=0.99)
    with pytest.raises(RefinementVerificationError, match="test-derived"):
        validate_ablation_comparability(
            test_leak, expected_split_sha256="split-sha", expected_random_seed=42
        )


def test_ablation_metadata_and_saved_score_metrics_are_reconciled() -> None:
    parameters = {"n_estimators": 10, "random_state": 42}
    digest = _digest(parameters)
    ablation = _ablation().set_index("feature_family")
    metadata = {
        family: {
            "feature_family": family,
            "model_family": "lightgbm",
            "candidate_id": "lgb_stable_sqrt_weight",
            "effective_parameters": parameters,
            "effective_parameters_sha256": digest,
            "random_seed": 42,
            "split_manifest_sha256": "split-sha",
            "ranking_score_type": "lightgbm_raw_margin",
            "configured_top_k": [100, 500, 1000],
            "transformed_feature_count": int(ablation.loc[family, "transformed_feature_count"]),
            "evaluation_partition": "validation",
            "test_predictions_generated": False,
            "test_metrics": None,
        }
        for family in ALL_ABLATION_FEATURE_FAMILIES
    }
    result = validate_ablation_model_metadata(metadata, _ablation())
    assert result["same_fitted_model_family_candidate_parameters_seed_split_and_protocol"] is True

    scores = {
        "ablation_transaction_only": {"average_precision": 0.10, "roc_auc": 0.60},
        "refined_lightgbm": {"average_precision": 0.20, "roc_auc": 0.70},
        "ablation_transaction_temporal_history_graph": {
            "average_precision": 0.25,
            "roc_auc": 0.75,
        },
        "ablation_transaction_temporal_history_graph_novel3": {
            "average_precision": 0.22,
            "roc_auc": 0.72,
        },
    }
    metric_result = verify_ablation_metrics_from_predictions(_ablation(), scores)
    assert metric_result["rows_recomputed"] == 4

    changed = deepcopy(metadata)
    changed["transaction_temporal_history_graph"]["random_seed"] = 7
    with pytest.raises(RefinementVerificationError, match="model metadata differ"):
        validate_ablation_model_metadata(changed, _ablation())

    bad_scores = deepcopy(scores)
    bad_scores["ablation_transaction_temporal_history_graph"]["average_precision"] = 0.99
    with pytest.raises(RefinementVerificationError, match="average precision differs"):
        verify_ablation_metrics_from_predictions(_ablation(), bad_scores)


def test_graph_value_conclusion_recomputes_delta_and_rejects_causal_claims() -> None:
    conclusion = {
        "evaluation_partition": "validation",
        "test_metrics_used": False,
        "comparability": {
            "same_model_family": True,
            "same_candidate": True,
            "same_effective_parameters": True,
            "same_frozen_split": True,
            "same_evaluation_protocol": True,
        },
        "baseline_average_precision": 0.20,
        "graph_average_precision": 0.25,
        "absolute_average_precision_delta": 0.05,
        "interpretation": (
            "Associational validation evidence only; feature importance is not causal."
        ),
    }
    result = validate_graph_value_conclusion(conclusion, _ablation())
    assert result["absolute_average_precision_delta"] == pytest.approx(0.05)

    wrong = {**conclusion, "absolute_average_precision_delta": 0.5}
    with pytest.raises(RefinementVerificationError, match="delta differs"):
        validate_graph_value_conclusion(wrong, _ablation())

    causal = {**conclusion, "interpretation": "Graph fields caused the improvement."}
    with pytest.raises(RefinementVerificationError, match="causal"):
        validate_graph_value_conclusion(causal, _ablation())


def test_duplicate_graph_evidence_is_recounted_without_test_rows(tmp_path: Path) -> None:
    ids = [f"row-{index}" for index in range(5)]
    split = pd.DataFrame(
        {
            "transaction_id": ids,
            "partition": ["train", "train", "validation", "validation", "test"],
        }
    )
    # The sealed test row deliberately differs.  A validation-safe recount must
    # exclude it while checking every permitted train/validation row.
    feature = pd.DataFrame(
        {
            "transaction_id": ids,
            "sender_prior_fan_out_degree": [0, 1, 2, 3, 999],
            "sender_previous_unique_counterparties": [0, 1, 2, 3, -1],
            "receiver_prior_fan_in_degree": [1, 2, 3, 4, 999],
            "receiver_previous_unique_counterparties": [1, 2, 3, 4, -1],
        }
    )
    split_path = tmp_path / "split.parquet"
    feature_path = tmp_path / "feature.parquet"
    connection = duckdb.connect()
    try:
        _write_parquet(connection, split, split_path, relation_name="duplicate_split")
        _write_parquet(connection, feature, feature_path, relation_name="duplicate_feature")
    finally:
        connection.close()
    saved = {
        "sender_prior_fan_out_degree": "sender_previous_unique_counterparties",
        "receiver_prior_fan_in_degree": "receiver_previous_unique_counterparties",
        "rows_checked": 4,
        "sender_mismatch_rows": 0,
        "receiver_mismatch_rows": 0,
        "mismatch_rows_full_train_plus_validation": 0,
        "partitions_checked": ["train", "validation"],
        "test_rows_checked": False,
    }
    result = verify_duplicate_graph_evidence(
        saved,
        feature_table=feature_path,
        split_table=split_path,
        expected_rows=4,
    )
    assert result["rows_checked"] == 4
    assert result["test_rows_checked"] is False

    tampered = {**saved, "sender_mismatch_rows": 1}
    with pytest.raises(RefinementVerificationError, match="differs"):
        verify_duplicate_graph_evidence(
            tampered,
            feature_table=feature_path,
            split_table=split_path,
            expected_rows=4,
        )


def _trials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_family": family,
                "candidate_id": candidate,
                "fold": fold,
                "average_precision": score,
                "eligible": True,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
                "partition": "inner_validation",
                "feature_family": "transaction_temporal_history",
            }
            for family, candidate, score in (
                ("logistic_regression", "lr_a", 0.10),
                ("logistic_regression", "lr_b", 0.20),
                ("lightgbm", "lgb_a", 0.30),
            )
            for fold in (1, 2, 3)
        ]
    )


def test_tuning_verifier_reselects_candidates_from_inner_folds_only() -> None:
    selected = {
        "logistic_regression": {
            "candidate_id": "lr_b",
            "selection_partition": "inner_temporal_validation_folds",
            "test_metrics_used": False,
        },
        "lightgbm": {
            "candidate_id": "lgb_a",
            "selection_partition": "inner_temporal_validation_folds",
            "test_metrics_used": False,
        },
    }
    config = {
        "logistic_regression": [{"candidate_id": "lr_a"}, {"candidate_id": "lr_b"}],
        "lightgbm": [{"candidate_id": "lgb_a"}],
    }
    result = validate_tuning_artifacts(
        _trials(), selected, expected_folds=3, configured_candidates=config
    )
    assert result["selected_candidates"] == {
        "lightgbm": "lgb_a",
        "logistic_regression": "lr_b",
    }

    leaked = _trials().assign(test_metric=0.99)
    with pytest.raises(RefinementVerificationError, match="test-derived"):
        validate_tuning_artifacts(leaked, selected, expected_folds=3, configured_candidates=config)

    outer_validation = _trials()
    outer_validation["partition"] = "validation"
    with pytest.raises(RefinementVerificationError, match="inner-validation-only"):
        validate_tuning_artifacts(
            outer_validation, selected, expected_folds=3, configured_candidates=config
        )


def test_threshold_verifier_recomputes_validation_only_whole_tie_frontier() -> None:
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)
    scores = np.array([0.9, 0.9, 0.5, 0.4, 0.4, 0.1], dtype=np.float64)
    artifact = optimize_validation_thresholds(
        labels,
        scores,
        partition="validation",
        fpr_ceiling=0.5,
        alert_budget=4,
        primary_fpr_ceiling=0.5,
        primary_alert_budget=4,
    )
    result = validate_threshold_artifact(artifact, fpr_ceiling=0.5, alert_budget=4)
    assert result["whole_equal_score_groups"] is True
    assert result["test_partition_used"] is False
    recompute_threshold_artifact(artifact, labels, scores, fpr_ceiling=0.5, alert_budget=4)
    wrapper = {
        "partition": "validation",
        "score_type": "raw_ranking_score",
        "primary_rule": "maximize_recall_subject_to_joint_constraints",
        "tie_policy": "include_complete_equal_score_group",
        "models": {"logistic_regression": artifact},
        "test_metrics_used": False,
    }
    wrapper_result = validate_threshold_analysis(wrapper, fpr_ceiling=0.5, alert_budget=4)
    assert wrapper_result["models"]["logistic_regression"]["status"] == "PASS"

    tampered = deepcopy(artifact)
    tampered["summaries"]["predeclared_joint_primary_constraint"]["operating_point"][  # type: ignore[index]
        "alert_count"
    ] = 999
    with pytest.raises(RefinementVerificationError, match="exceeds alert budget"):
        validate_threshold_artifact(tampered, fpr_ceiling=0.5, alert_budget=4)
    with pytest.raises(RefinementVerificationError, match="differs from recomputation"):
        recompute_threshold_artifact(tampered, labels, scores, fpr_ceiling=0.5, alert_budget=4)


def test_validation_prediction_verifier_recomputes_all_model_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    model_root = run_dir / "models"
    labels = np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.int8)
    source_rows = np.arange(1, labels.size + 1, dtype=np.int64)
    transaction_ids = np.array([f"source.csv:row-{row}" for row in source_rows])
    split = pd.DataFrame(
        {
            "transaction_id": [*transaction_ids, "source.csv:row-9"],
            "timestamp": pd.date_range("2022-01-01", periods=9, freq="min"),
            "partition": ["validation"] * labels.size + ["test"],
        }
    )
    feature = pd.DataFrame(
        {
            "transaction_id": [*transaction_ids, "source.csv:row-9"],
            "source_row_number": [*source_rows, 9],
            "is_laundering": [*labels, 1],
        }
    )
    prediction_data: dict[str, object] = {
        "transaction_id": transaction_ids,
        "source_row_number": source_rows,
        "is_laundering": labels,
    }
    score_specs = {
        "refined_logistic_regression": (
            "logistic_regression",
            "transaction_temporal_history",
        ),
        "refined_random_forest": ("random_forest", "transaction_temporal_history"),
        "refined_lightgbm": ("lightgbm", "transaction_temporal_history"),
        "ablation_transaction_only": (
            "ablation_transaction_only",
            "transaction_only",
        ),
        "ablation_transaction_temporal_history_graph": (
            "ablation_transaction_temporal_history_graph",
            "transaction_temporal_history_graph",
        ),
        "ablation_transaction_temporal_history_graph_novel3": (
            "ablation_transaction_temporal_history_graph_novel3",
            "transaction_temporal_history_graph_novel3",
        ),
    }
    score_columns: dict[str, dict[str, str]] = {}
    base_raw = np.array([-1.2, 1.8, -0.8, 0.2, 1.1, -0.4, 0.7, -1.5])
    for index, (score_key, (artifact_key, feature_family)) in enumerate(score_specs.items()):
        raw = base_raw + index * np.linspace(-0.03, 0.03, labels.size)
        probability = 1.0 / (1.0 + np.exp(-raw))
        raw_column = f"raw_score_{score_key}"
        probability_column = f"probability_{score_key}"
        prediction_data[raw_column] = raw
        prediction_data[probability_column] = probability
        score_columns[score_key] = {
            "raw_score": raw_column,
            "probability": probability_column,
        }
        thresholds = optimize_validation_thresholds(
            labels,
            raw,
            partition="validation",
            fpr_ceiling=0.5,
            alert_budget=4,
            primary_fpr_ceiling=0.5,
            primary_alert_budget=4,
        )
        metrics = evaluate_binary_predictions(
            labels,
            raw,
            source_rows,
            threshold=_selected_threshold(thresholds, raw),
            top_k_values=[1, 3],
        )
        saturation = analyze_validation_score_saturation(
            labels,
            probabilities=probability,
            raw_scores=raw,
            source_row_number=source_rows,
            top_k_values=[1, 3],
            partition="validation",
        )
        saturation["raw_average_precision"] = compute_average_precision(labels, raw)
        saturation["probability_average_precision"] = compute_average_precision(labels, probability)
        model_dir = model_root / artifact_key
        atomic_write_json(thresholds, model_dir / "thresholds.json")
        atomic_write_json(metrics, model_dir / "validation_metrics.json")
        atomic_write_json(saturation, model_dir / "saturation.json")
        atomic_write_json(
            {
                "ranking_metrics": metrics,
                "threshold_optimization": thresholds,
                "saturation": saturation,
                "feature_family": feature_family,
                "evaluation_partition": "validation",
                "test_predictions_generated": False,
                "test_metrics": None,
            },
            model_dir / "metadata.json",
        )

    split_path = tmp_path / "split.parquet"
    feature_path = tmp_path / "feature.parquet"
    prediction_path = run_dir / "validation_predictions.parquet"
    connection = duckdb.connect()
    try:
        _write_parquet(connection, split, split_path, relation_name="prediction_split")
        _write_parquet(connection, feature, feature_path, relation_name="prediction_feature")
        _write_parquet(
            connection,
            pd.DataFrame(prediction_data),
            prediction_path,
            relation_name="saved_predictions",
        )
    finally:
        connection.close()
    manifest = {
        "partition": "validation",
        "rows": labels.size,
        "unique_transaction_ids": labels.size,
        "positive_labels": int(labels.sum()),
        "model_score_columns": score_columns,
        "test_predictions_included": False,
    }
    evidence = verify_validation_predictions(
        prediction_path,
        manifest,
        feature_table=feature_path,
        split_table=split_path,
        expected_rows=labels.size,
        expected_positive_labels=int(labels.sum()),
        top_k_values=[1, 3],
        run_dir=run_dir,
        fpr_ceiling=0.5,
        alert_budget=4,
    )
    assert set(evidence["models_recomputed"]) == {
        "logistic_regression",
        "random_forest",
        "lightgbm",
    }
    assert evidence["saturation_artifacts_recomputed"] == 6
    assert evidence["frozen_validation_identity_and_labels_verified"] is True
    assert evidence["test_rows_present"] is False

    tampered_frame = pd.DataFrame(prediction_data)
    tampered_frame.loc[[0, 1], "is_laundering"] = tampered_frame.loc[
        [1, 0], "is_laundering"
    ].to_numpy()
    tampered_path = run_dir / "tampered_validation_predictions.parquet"
    connection = duckdb.connect()
    try:
        _write_parquet(
            connection,
            tampered_frame,
            tampered_path,
            relation_name="tampered_predictions",
        )
    finally:
        connection.close()
    with pytest.raises(RefinementVerificationError, match="frozen validation features"):
        verify_validation_predictions(
            tampered_path,
            manifest,
            feature_table=feature_path,
            split_table=split_path,
            expected_rows=labels.size,
            expected_positive_labels=int(labels.sum()),
            top_k_values=[1, 3],
            run_dir=run_dir,
            fpr_ceiling=0.5,
            alert_budget=4,
        )


def test_core_model_artifacts_deserialize_and_prove_logistic_convergence(
    tmp_path: Path,
) -> None:
    matrix = np.array([[float(index), float(index % 3)] for index in range(30)], dtype=np.float64)
    labels = np.array([0] * 15 + [1] * 15, dtype=np.int8)
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=500, tol=1e-3).fit(matrix, labels),
        "random_forest": RandomForestClassifier(n_estimators=3, random_state=42).fit(
            matrix, labels
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=3,
            min_child_samples=1,
            random_state=42,
            verbosity=-1,
        ).fit(matrix, labels),
    }
    selected: dict[str, dict[str, object]] = {}
    for model_name, estimator in estimators.items():
        candidate_id = f"selected_{model_name}"
        selected[model_name] = {"candidate_id": candidate_id}
        parameters = {"candidate_id": candidate_id, "random_seed": 42}
        metadata: dict[str, object] = {
            "candidate_id": candidate_id,
            "selection_source": "inner_temporal_validation_folds",
            "feature_family": "transaction_temporal_history",
            "test_predictions_generated": False,
            "test_metrics": None,
            "effective_parameters": parameters,
            "effective_parameters_sha256": _digest(parameters),
        }
        if model_name == "logistic_regression":
            observed = int(np.asarray(estimator.n_iter_).max())
            assert observed < estimator.max_iter
            metadata["convergence"] = {
                "converged": True,
                "observed_n_iter": observed,
                "configured_max_iter": estimator.max_iter,
                "convergence_warnings": [],
            }
        model_dir = tmp_path / "models" / model_name
        model_dir.mkdir(parents=True)
        joblib.dump(estimator, model_dir / "model.joblib")
        atomic_write_json(metadata, model_dir / "metadata.json")

    evidence = verify_core_model_artifacts(tmp_path, selected)
    logistic = evidence["models_deserialized"]["logistic_regression"]
    assert logistic["converged"] is True
    assert logistic["observed_n_iter"] < logistic["configured_max_iter"]

    metadata_path = tmp_path / "models" / "logistic_regression" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["convergence"]["converged"] = False
    atomic_write_json(metadata, metadata_path)
    with pytest.raises(RefinementVerificationError, match="did not converge"):
        verify_core_model_artifacts(tmp_path, selected)
