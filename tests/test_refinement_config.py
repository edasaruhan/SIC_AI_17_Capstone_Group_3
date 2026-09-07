from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from argus.config import ConfigError, load_config, validate_config


def _config() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    return load_config(project_root / "configs" / "refinement.yaml")


def _invalid(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_refinement_config_loads_as_full_closed_test_protocol() -> None:
    config = _config()

    assert config["sprint3"]["full_data"] is True
    assert config["sprint3"]["sampled"] is False
    assert config["sprint3"]["final_test_access"] is False
    assert config["sprint3"]["top_k"] == config["baseline"]["top_k"]
    assert set(config["sprint3"]["tuning"]) >= {
        "logistic_regression",
        "random_forest",
        "lightgbm",
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("protocol_version", "1", "protocol_version"),
        ("full_data", False, "full_data=true"),
        ("sampled", True, "sampled=false"),
        ("selection_metric", "roc_auc", "selection_metric"),
        ("ranking_score_type", "probability", "ranking_score_type"),
        ("final_test_access", True, "final_test_access=false"),
        ("final_test_policy", "inference_allowed", "final_test_policy"),
        ("ranking_tie_break", "arbitrary", "ranking_tie_break"),
    ],
)
def test_refinement_protocol_identity_is_immutable(key: str, value: object, message: str) -> None:
    config = _config()
    config["sprint3"][key] = value

    _invalid(config, message)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("top_k", [100, 1000, 500], "top_k"),
        ("top_k", [100, True, 1000], "top_k"),
        ("top_k", [100, 500], "match baseline.top_k"),
        ("batch_rows", 0, "batch_rows"),
        ("threads", True, "threads"),
        ("model_threads", -1, "model_threads"),
        ("memory_limit", "4 GiB", "memory_limit"),
        ("max_temp_directory_size", "unlimited", "max_temp_directory_size"),
        ("parquet_compression", "gzip", "parquet_compression"),
        ("model_serialization_compression", 10, "model_serialization_compression"),
        ("model_serialization_compression", True, "model_serialization_compression"),
        ("retain_work_matrices", 0, "retain_work_matrices"),
    ],
)
def test_refinement_resource_and_ranking_settings_are_bounded(
    key: str, value: object, message: str
) -> None:
    config = _config()
    config["sprint3"][key] = value

    _invalid(config, message)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("strategy", "random_kfold", "strategy"),
        ("timestamp_groups_must_remain_intact", False, "timestamp_groups"),
        ("selection_aggregation", "best_fold", "selection_aggregation"),
        ("cumulative_train_quantiles", [0.4, 0.6], "at least three"),
        ("cumulative_train_quantiles", [0.4, 0.8, 0.6], "strictly increasing"),
        ("cumulative_train_quantiles", [0.4, 0.6, float("nan")], "finite number"),
        ("validation_end_quantiles", [0.6, 0.8, 0.9, 1.0], "equal length"),
        ("validation_end_quantiles", [0.3, 0.8, 1.0], "must precede"),
        ("validation_end_quantiles", [0.6, 0.9, 1.0], "Adjacent"),
        ("validation_end_quantiles", [0.6, 0.8, 0.95], "must be 1.0"),
    ],
)
def test_temporal_cv_contract_rejects_unsafe_folds(key: str, value: object, message: str) -> None:
    config = _config()
    config["sprint3"]["temporal_cv"][key] = value

    _invalid(config, message)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("partition", "test"),
        ("score_type", "probability"),
        ("primary_rule", "maximize_f1"),
        ("tie_policy", "truncate_ties"),
        ("alert_budget", 0),
        ("alert_budget", True),
        ("fpr_ceiling", 0.0),
        ("fpr_ceiling", 1.0),
    ],
)
def test_threshold_policy_is_validation_only_and_whole_tie(key: str, value: object) -> None:
    config = _config()
    config["sprint3"]["threshold_optimization"][key] = value

    _invalid(config, f"threshold_optimization.{key}")


def test_tuning_requires_exact_model_families_and_candidate_counts() -> None:
    renamed = _config()
    tuning = renamed["sprint3"]["tuning"]
    tuning["xgboost"] = tuning.pop("lightgbm")
    _invalid(renamed, "tuning keys differ")

    shortened = _config()
    shortened["sprint3"]["tuning"]["logistic_regression"].pop()
    _invalid(shortened, "exactly 4 candidates")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("feature_family", "transaction_only"),
        ("candidate_tie_break", "declaration_order"),
        ("reject_nonconverged_logistic", False),
        ("reject_nonfinite_scores", 1),
    ],
)
def test_tuning_safety_flags_and_selection_scope_are_fixed(key: str, value: object) -> None:
    config = _config()
    config["sprint3"]["tuning"][key] = value

    _invalid(config, f"tuning.{key}")


def test_tuning_candidate_ids_are_valid_and_globally_unique() -> None:
    malformed = _config()
    malformed["sprint3"]["tuning"]["random_forest"][0]["candidate_id"] = "RF candidate"
    _invalid(malformed, "candidate_id")

    duplicate = _config()
    duplicate["sprint3"]["tuning"]["random_forest"][0]["candidate_id"] = duplicate["sprint3"][
        "tuning"
    ]["logistic_regression"][0]["candidate_id"]
    _invalid(duplicate, "globally unique")


@pytest.mark.parametrize(
    ("family", "candidate_index", "key", "value", "message"),
    [
        ("logistic_regression", 0, "solver", "liblinear", "solver"),
        ("logistic_regression", 0, "max_iter", 20, "max_iter"),
        ("logistic_regression", 0, "C", 0.0, r"\.C"),
        ("random_forest", 0, "bootstrap", False, "bootstrap"),
        ("random_forest", 0, "max_samples", 1.5, "max_samples"),
        ("lightgbm", 0, "scale_pos_weight_policy", "balanced", "policy"),
        ("lightgbm", 0, "reg_lambda", 0.0, "reg_lambda"),
        ("lightgbm", 0, "max_delta_step", 0.0, "max_delta_step"),
    ],
)
def test_candidate_parameters_are_supported_and_bounded(
    family: str,
    candidate_index: int,
    key: str,
    value: object,
    message: str,
) -> None:
    config = _config()
    config["sprint3"]["tuning"][family][candidate_index][key] = value

    _invalid(config, message)


def test_candidate_schema_rejects_misspelled_parameter() -> None:
    config = _config()
    candidate = config["sprint3"]["tuning"]["lightgbm"][0]
    candidate["learningrate"] = candidate.pop("learning_rate")

    _invalid(config, "keys differ")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("model_family", "random_forest", "model_family"),
        (
            "primary_families",
            [
                "transaction_only",
                "transaction_temporal_history_graph",
                "transaction_temporal_history",
            ],
            "primary_families",
        ),
        ("sensitivity_family", "transaction_temporal_history_graph", "sensitivity_family"),
        ("duplicate_graph_pairs", {}, "duplicate_graph_pairs"),
    ],
)
def test_ablation_contract_preserves_a_b_c_and_novel3(
    key: str, value: object, message: str
) -> None:
    config = _config()
    config["sprint3"]["ablation"][key] = value

    _invalid(config, message)


def test_refinement_validation_does_not_mutate_the_loaded_config() -> None:
    config = _config()
    before = deepcopy(config)

    validate_config(config)

    assert config == before
