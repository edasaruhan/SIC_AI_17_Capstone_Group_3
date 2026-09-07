"""Configuration loading for reproducible ARGUS pipeline runs.

Configuration files are ordinary YAML mappings.  A file may inherit from one or
more YAML files with ``extends``; relative inheritance paths are interpreted
relative to the file that declares them.  Values in the top-level ``paths``
mapping are resolved against the project root (the parent of ``configs/`` by
default), so command behaviour does not depend on the caller's current working
directory.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeAlias

import yaml

ConfigDict: TypeAlias = dict[str, Any]

_POSITIVE_SIZE_PATTERN = re.compile(r"[1-9]\d*(?:\.\d+)?(?:KB|MB|GB|TB)")
_CANDIDATE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


class ConfigError(ValueError):
    """Raised when an ARGUS configuration is missing or internally invalid."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge mappings without mutating either input."""

    merged: ConfigDict = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> ConfigDict:
    """Read a YAML mapping with actionable error messages."""

    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")
    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, MutableMapping):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    return dict(loaded)


def _load_with_extends(path: Path, active: tuple[Path, ...] = ()) -> ConfigDict:
    """Load ``path`` and recursively merge its declared base configurations."""

    resolved = path.expanduser().resolve()
    if resolved in active:
        chain = " -> ".join(str(item) for item in (*active, resolved))
        raise ConfigError(f"Circular configuration inheritance detected: {chain}")

    current = _read_yaml(resolved)
    declared_bases = current.pop("extends", None)
    if declared_bases is None:
        return current
    if isinstance(declared_bases, str):
        base_names = [declared_bases]
    elif isinstance(declared_bases, list) and all(isinstance(item, str) for item in declared_bases):
        base_names = declared_bases
    else:
        raise ConfigError(f"'extends' must be a path string or list of path strings: {resolved}")

    combined: ConfigDict = {}
    for base_name in base_names:
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        inherited = _load_with_extends(base_path, (*active, resolved))
        combined = _deep_merge(combined, inherited)
    return _deep_merge(combined, current)


def _default_project_root(config_path: Path) -> Path:
    """Infer the repository root from a conventional ``configs`` directory."""

    parent = config_path.resolve().parent
    return parent.parent if parent.name.casefold() == "configs" else parent


def _resolve_config_paths(config: ConfigDict, project_root: Path) -> None:
    """Resolve top-level path values in place against ``project_root``."""

    paths = config.get("paths")
    if paths is None:
        return
    if not isinstance(paths, MutableMapping):
        raise ConfigError("'paths' must be a mapping")

    for key, raw_value in list(paths.items()):
        if raw_value is None:
            continue
        if not isinstance(raw_value, (str, Path)):
            raise ConfigError(f"paths.{key} must be a string or null")
        value = Path(raw_value).expanduser()
        if not value.is_absolute():
            value = project_root / value
        paths[key] = str(value.resolve())


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing or invalid '{key}' configuration section")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    """Reject omissions and misspelled/unsupported schema keys."""

    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ConfigError(f"{name} keys differ; missing={missing}, unexpected={unexpected}")


def _require_positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConfigError(f"{name} must be a finite number")
    if minimum is not None and (numeric < minimum if minimum_inclusive else numeric <= minimum):
        relation = ">=" if minimum_inclusive else ">"
        raise ConfigError(f"{name} must be {relation} {minimum}")
    if maximum is not None and (numeric > maximum if maximum_inclusive else numeric >= maximum):
        relation = "<=" if maximum_inclusive else "<"
        raise ConfigError(f"{name} must be {relation} {maximum}")
    return numeric


def _require_positive_size(value: object, name: str) -> None:
    if not isinstance(value, str) or _POSITIVE_SIZE_PATTERN.fullmatch(value) is None:
        raise ConfigError(f"{name} must be a positive size such as '1GB'")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate settings required by the Sprint 1 data pipeline."""

    project = _require_mapping(config, "project")
    seed = project.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("project.random_seed must be a non-negative integer")

    paths = _require_mapping(config, "paths")
    for key in ("raw_data", "raw_accounts", "run_dir"):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise ConfigError(f"paths.{key} must be a non-empty path string")

    data = _require_mapping(config, "data")
    column_mapping = data.get("column_mapping")
    if not isinstance(column_mapping, Mapping) or not column_mapping:
        raise ConfigError("data.column_mapping must be a non-empty mapping")

    required_canonical = {
        "timestamp",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    }
    mapped_names = {str(value) for value in column_mapping.values()}
    missing_mappings = sorted(required_canonical - mapped_names)
    if missing_mappings:
        raise ConfigError(
            "data.column_mapping is missing canonical fields: " + ", ".join(missing_mappings)
        )

    sampling = data.get("sampling", {})
    if not isinstance(sampling, Mapping):
        raise ConfigError("data.sampling must be a mapping")
    max_rows = sampling.get("max_rows")
    if max_rows is not None and (
        not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0
    ):
        raise ConfigError("data.sampling.max_rows must be null or a positive integer")
    if sampling.get("method") != "chronological_prefix":
        raise ConfigError("data.sampling.method must be 'chronological_prefix'")

    split = _require_mapping(config, "split")
    if split.get("strategy") != "chronological":
        raise ConfigError("split.strategy must be 'chronological'")

    explicit_boundaries = (
        split.get("train_end") is not None or split.get("validation_end") is not None
    )
    if explicit_boundaries:
        if split.get("train_end") is None or split.get("validation_end") is None:
            raise ConfigError("split.train_end and split.validation_end must be set together")
    else:
        fraction_keys = ("train_fraction", "validation_fraction", "test_fraction")
        fractions: list[float] = []
        for key in fraction_keys:
            value = split.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(f"split.{key} must be numeric")
            numeric_value = float(value)
            if not 0.0 < numeric_value < 1.0:
                raise ConfigError(f"split.{key} must be strictly between 0 and 1")
            fractions.append(numeric_value)
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ConfigError("Chronological split fractions must sum to 1.0")

    full_pipeline = config.get("full_pipeline")
    if full_pipeline is not None:
        if not isinstance(full_pipeline, Mapping):
            raise ConfigError("full_pipeline must be a mapping")
        if full_pipeline.get("engine") != "duckdb":
            raise ConfigError("full_pipeline.engine must be 'duckdb'")
        for key in ("memory_limit", "max_temp_directory_size"):
            value = full_pipeline.get(key)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[1-9]\d*(?:\.\d+)?(?:KB|MB|GB|TB)", value) is None
            ):
                raise ConfigError(f"full_pipeline.{key} must be a positive size such as '1GB'")
        threads = full_pipeline.get("threads")
        if not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0:
            raise ConfigError("full_pipeline.threads must be a positive integer")
        if full_pipeline.get("parquet_compression") not in {"snappy", "zstd"}:
            raise ConfigError("full_pipeline.parquet_compression must be 'snappy' or 'zstd'")
        sample_rows = full_pipeline.get("eda_sample_rows")
        if not isinstance(sample_rows, int) or isinstance(sample_rows, bool) or sample_rows <= 0:
            raise ConfigError("full_pipeline.eda_sample_rows must be a positive integer")

    baseline = config.get("baseline")
    if baseline is not None:
        _validate_baseline_config(baseline)

    sprint3 = config.get("sprint3")
    if sprint3 is not None:
        _validate_refinement_config(sprint3, baseline=baseline)


def _validate_baseline_config(baseline: object) -> None:
    """Validate the immutable Sprint 2 model-selection contract."""

    if not isinstance(baseline, Mapping):
        raise ConfigError("baseline must be a mapping")
    if baseline.get("selection_partition") != "validation":
        raise ConfigError("baseline.selection_partition must be 'validation'")
    if baseline.get("selection_metric") != "average_precision":
        raise ConfigError("baseline.selection_metric must be 'average_precision'")
    if baseline.get("final_test_access") is not False:
        raise ConfigError("Sprint 2 requires baseline.final_test_access=false")
    if baseline.get("final_test_policy") != "metadata_only_no_model_inference":
        raise ConfigError("baseline.final_test_policy must be 'metadata_only_no_model_inference'")

    threshold = baseline.get("decision_threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ConfigError("baseline.decision_threshold must be numeric in [0, 1]")
    top_k = baseline.get("top_k")
    if (
        not isinstance(top_k, list)
        or not top_k
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in top_k
        )
        or len(set(top_k)) != len(top_k)
    ):
        raise ConfigError("baseline.top_k must contain unique positive integers")

    features = baseline.get("features")
    if not isinstance(features, Mapping):
        raise ConfigError("baseline.features must be a mapping")
    selected_groups = ("numeric", "one_hot_categorical", "frequency_categorical")
    selected: list[str] = []
    for key in selected_groups:
        values = features.get(key)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise ConfigError(f"baseline.features.{key} must be a non-empty string list")
        selected.extend(values)
    if len(selected) != len(set(selected)):
        raise ConfigError("A baseline input feature may appear in only one encoding group")
    forbidden: list[str] = []
    for key in ("forbidden_graph_history", "forbidden_identity_or_target"):
        values = features.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ConfigError(f"baseline.features.{key} must be a string list")
        forbidden.extend(values)
    overlap = sorted(set(selected) & set(forbidden))
    if overlap:
        raise ConfigError("Forbidden fields selected as model features: " + ", ".join(overlap))

    models = baseline.get("models")
    expected_models = {"logistic_regression", "random_forest", "lightgbm"}
    if not isinstance(models, Mapping) or set(models) != expected_models:
        raise ConfigError(
            "baseline.models must define exactly logistic_regression, random_forest, lightgbm"
        )
    if any(not isinstance(models[name], Mapping) for name in expected_models):
        raise ConfigError("Each baseline model configuration must be a mapping")

    frozen = baseline.get("frozen_upstream")
    if not isinstance(frozen, Mapping):
        raise ConfigError("baseline.frozen_upstream must be a mapping")
    for key in ("feature_table_sha256", "split_table_sha256", "split_metadata_sha256"):
        value = frozen.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ConfigError(f"baseline.frozen_upstream.{key} must be a lowercase SHA-256")


def _validate_refinement_config(sprint3: object, *, baseline: object) -> None:
    """Validate the complete, predeclared Sprint 3 experiment protocol.

    Sprint 3 is intentionally not a generic estimator configuration surface.  Its
    evidence is comparable only while the full-data scope, closed-test policy,
    expanding folds, tuning grid, and same-model feature ablation remain intact.
    Strict keys also turn misspelled settings into load-time errors rather than
    silently ignored experimental changes.
    """

    if not isinstance(sprint3, Mapping):
        raise ConfigError("sprint3 must be a mapping")
    _require_exact_keys(
        sprint3,
        {
            "protocol_version",
            "full_data",
            "sampled",
            "selection_metric",
            "ranking_score_type",
            "final_test_access",
            "final_test_policy",
            "top_k",
            "ranking_tie_break",
            "batch_rows",
            "memory_limit",
            "threads",
            "model_threads",
            "max_temp_directory_size",
            "parquet_compression",
            "model_serialization_compression",
            "retain_work_matrices",
            "temporal_cv",
            "threshold_optimization",
            "tuning",
            "ablation",
        },
        "sprint3",
    )

    if sprint3.get("protocol_version") != 1 or isinstance(sprint3.get("protocol_version"), bool):
        raise ConfigError("sprint3.protocol_version must be integer 1")
    if sprint3.get("full_data") is not True or sprint3.get("sampled") is not False:
        raise ConfigError("Sprint 3 requires sprint3.full_data=true and sprint3.sampled=false")
    if sprint3.get("selection_metric") != "average_precision":
        raise ConfigError("sprint3.selection_metric must be 'average_precision'")
    if sprint3.get("ranking_score_type") != "raw_margin_where_available":
        raise ConfigError("sprint3.ranking_score_type must be 'raw_margin_where_available'")
    if sprint3.get("final_test_access") is not False:
        raise ConfigError("Sprint 3 requires sprint3.final_test_access=false")
    if sprint3.get("final_test_policy") != "metadata_only_no_transform_no_inference":
        raise ConfigError(
            "sprint3.final_test_policy must be 'metadata_only_no_transform_no_inference'"
        )
    if sprint3.get("ranking_tie_break") != "source_row_number_ascending":
        raise ConfigError("sprint3.ranking_tie_break must be 'source_row_number_ascending'")

    top_k = sprint3.get("top_k")
    if (
        not isinstance(top_k, list)
        or not top_k
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in top_k
        )
        or top_k != sorted(set(top_k))
    ):
        raise ConfigError("sprint3.top_k must be strictly increasing unique positive integers")
    if not isinstance(baseline, Mapping):
        raise ConfigError("Sprint 3 requires the inherited baseline contract")
    if top_k != baseline.get("top_k"):
        raise ConfigError("sprint3.top_k must match baseline.top_k for comparable evaluation")
    if sprint3.get("ranking_tie_break") != baseline.get("ranking_tie_break"):
        raise ConfigError("sprint3.ranking_tie_break must match baseline.ranking_tie_break")

    _require_positive_integer(sprint3.get("batch_rows"), "sprint3.batch_rows")
    _require_positive_integer(sprint3.get("threads"), "sprint3.threads")
    _require_positive_integer(sprint3.get("model_threads"), "sprint3.model_threads")
    _require_positive_size(sprint3.get("memory_limit"), "sprint3.memory_limit")
    _require_positive_size(
        sprint3.get("max_temp_directory_size"), "sprint3.max_temp_directory_size"
    )
    if sprint3.get("parquet_compression") not in {"snappy", "zstd"}:
        raise ConfigError("sprint3.parquet_compression must be 'snappy' or 'zstd'")
    compression = sprint3.get("model_serialization_compression")
    if (
        isinstance(compression, bool)
        or not isinstance(compression, int)
        or not 0 <= compression <= 9
    ):
        raise ConfigError("sprint3.model_serialization_compression must be an integer in [0, 9]")
    if not isinstance(sprint3.get("retain_work_matrices"), bool):
        raise ConfigError("sprint3.retain_work_matrices must be boolean")

    _validate_refinement_temporal_cv(sprint3.get("temporal_cv"))
    _validate_refinement_threshold(sprint3.get("threshold_optimization"))
    _validate_refinement_tuning(sprint3.get("tuning"))
    _validate_refinement_ablation(sprint3.get("ablation"))


def _validated_refinement_quantiles(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 3:
        raise ConfigError(f"{name} must contain at least three numeric quantiles")
    quantiles = [
        _require_finite_number(
            item,
            f"{name}[{index}]",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        for index, item in enumerate(value)
    ]
    if quantiles != sorted(set(quantiles)):
        raise ConfigError(f"{name} must be strictly increasing")
    return quantiles


def _validate_refinement_temporal_cv(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError("sprint3.temporal_cv must be a mapping")
    _require_exact_keys(
        value,
        {
            "strategy",
            "cumulative_train_quantiles",
            "validation_end_quantiles",
            "timestamp_groups_must_remain_intact",
            "selection_aggregation",
        },
        "sprint3.temporal_cv",
    )
    if value.get("strategy") != "expanding_window":
        raise ConfigError("sprint3.temporal_cv.strategy must be 'expanding_window'")
    if value.get("selection_aggregation") != "mean_fold_average_precision":
        raise ConfigError(
            "sprint3.temporal_cv.selection_aggregation must be 'mean_fold_average_precision'"
        )
    if value.get("timestamp_groups_must_remain_intact") is not True:
        raise ConfigError("sprint3.temporal_cv.timestamp_groups_must_remain_intact must be true")
    train = _validated_refinement_quantiles(
        value.get("cumulative_train_quantiles"),
        "sprint3.temporal_cv.cumulative_train_quantiles",
    )
    validation = _validated_refinement_quantiles(
        value.get("validation_end_quantiles"),
        "sprint3.temporal_cv.validation_end_quantiles",
    )
    if len(train) != len(validation):
        raise ConfigError("Sprint 3 temporal-CV quantile lists must have equal length")
    if any(
        train_end >= validation_end
        for train_end, validation_end in zip(train, validation, strict=True)
    ):
        raise ConfigError("Each Sprint 3 train quantile must precede its validation end")
    if any(
        not math.isclose(validation[index], train[index + 1], abs_tol=1e-12)
        for index in range(len(train) - 1)
    ):
        raise ConfigError(
            "Adjacent Sprint 3 folds must join at validation_end[i] == train_end[i+1]"
        )
    if not math.isclose(validation[-1], 1.0, abs_tol=1e-12):
        raise ConfigError("The final Sprint 3 temporal-CV validation quantile must be 1.0")


def _validate_refinement_threshold(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError("sprint3.threshold_optimization must be a mapping")
    _require_exact_keys(
        value,
        {
            "partition",
            "score_type",
            "primary_rule",
            "alert_budget",
            "fpr_ceiling",
            "tie_policy",
        },
        "sprint3.threshold_optimization",
    )
    expected_strings = {
        "partition": "validation",
        "score_type": "raw_ranking_score",
        "primary_rule": "maximize_recall_subject_to_joint_constraints",
        "tie_policy": "include_complete_equal_score_group",
    }
    for key, expected in expected_strings.items():
        if value.get(key) != expected:
            raise ConfigError(f"sprint3.threshold_optimization.{key} must be {expected!r}")
    _require_positive_integer(
        value.get("alert_budget"), "sprint3.threshold_optimization.alert_budget"
    )
    _require_finite_number(
        value.get("fpr_ceiling"),
        "sprint3.threshold_optimization.fpr_ceiling",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )


def _validate_refinement_tuning(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError("sprint3.tuning must be a mapping")
    model_families = {"logistic_regression", "random_forest", "lightgbm"}
    _require_exact_keys(
        value,
        {
            "feature_family",
            "candidate_tie_break",
            "reject_nonconverged_logistic",
            "reject_nonfinite_scores",
            *model_families,
        },
        "sprint3.tuning",
    )
    if value.get("feature_family") != "transaction_temporal_history":
        raise ConfigError("sprint3.tuning.feature_family must be 'transaction_temporal_history'")
    if value.get("candidate_tie_break") != "candidate_id_ascending":
        raise ConfigError("sprint3.tuning.candidate_tie_break must be 'candidate_id_ascending'")
    for key in ("reject_nonconverged_logistic", "reject_nonfinite_scores"):
        if value.get(key) is not True:
            raise ConfigError(f"sprint3.tuning.{key} must be true")

    expected_counts = {"logistic_regression": 4, "random_forest": 2, "lightgbm": 4}
    candidate_ids: list[str] = []
    policies: dict[str, set[str]] = {}
    solvers: set[str] = set()
    for family in sorted(model_families):
        candidates = value.get(family)
        expected_count = expected_counts[family]
        if not isinstance(candidates, list) or len(candidates) != expected_count:
            raise ConfigError(
                f"sprint3.tuning.{family} must contain exactly {expected_count} candidates"
            )
        family_policies: set[str] = set()
        for index, candidate in enumerate(candidates):
            name = f"sprint3.tuning.{family}[{index}]"
            if not isinstance(candidate, Mapping):
                raise ConfigError(f"{name} must be a mapping")
            _validate_refinement_candidate(family, candidate, name)
            candidate_id = candidate.get("candidate_id")
            assert isinstance(candidate_id, str)
            candidate_ids.append(candidate_id)
            policy_key = (
                "scale_pos_weight_policy" if family == "lightgbm" else "class_weight_policy"
            )
            family_policies.add(str(candidate[policy_key]))
            if family == "logistic_regression":
                solvers.add(str(candidate["solver"]))
        policies[family] = family_policies
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ConfigError("Sprint 3 candidate_id values must be globally unique")
    if solvers != {"newton-cholesky", "saga"}:
        raise ConfigError("Sprint 3 Logistic Regression grid must cover newton-cholesky and saga")
    if policies["logistic_regression"] != {"none", "sqrt_train_ratio"}:
        raise ConfigError(
            "Sprint 3 Logistic Regression grid must cover none and sqrt_train_ratio weighting"
        )
    if policies["random_forest"] != {"balanced_subsample"}:
        raise ConfigError("Sprint 3 Random Forest grid requires balanced_subsample weighting")
    if policies["lightgbm"] != {"none", "sqrt_train_ratio", "capped_100"}:
        raise ConfigError(
            "Sprint 3 LightGBM grid must cover none, sqrt_train_ratio, and capped_100 weighting"
        )


def _validate_refinement_candidate(family: str, candidate: Mapping[str, Any], name: str) -> None:
    schemas = {
        "logistic_regression": {
            "candidate_id",
            "solver",
            "penalty",
            "C",
            "class_weight_policy",
            "max_iter",
            "tol",
        },
        "random_forest": {
            "candidate_id",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight_policy",
            "bootstrap",
            "max_samples",
        },
        "lightgbm": {
            "candidate_id",
            "scale_pos_weight_policy",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "max_delta_step",
        },
    }
    _require_exact_keys(candidate, schemas[family], name)
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or _CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None:
        raise ConfigError(f"{name}.candidate_id must be a lowercase snake-case identifier")

    if family == "logistic_regression":
        if candidate.get("solver") not in {"newton-cholesky", "saga"}:
            raise ConfigError(f"{name}.solver is unsupported")
        if candidate.get("penalty") != "l2":
            raise ConfigError(f"{name}.penalty must be 'l2'")
        if candidate.get("class_weight_policy") not in {"none", "sqrt_train_ratio"}:
            raise ConfigError(f"{name}.class_weight_policy is unsupported")
        _require_finite_number(
            candidate.get("C"), f"{name}.C", minimum=0.0, minimum_inclusive=False
        )
        _require_positive_integer(candidate.get("max_iter"), f"{name}.max_iter", minimum=100)
        _require_finite_number(
            candidate.get("tol"),
            f"{name}.tol",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
        return

    if family == "random_forest":
        for key in ("n_estimators", "max_depth", "min_samples_leaf"):
            _require_positive_integer(candidate.get(key), f"{name}.{key}")
        if candidate.get("max_features") != "sqrt":
            raise ConfigError(f"{name}.max_features must be 'sqrt'")
        if candidate.get("class_weight_policy") != "balanced_subsample":
            raise ConfigError(f"{name}.class_weight_policy must be 'balanced_subsample'")
        if candidate.get("bootstrap") is not True:
            raise ConfigError(f"{name}.bootstrap must be true")
        _require_finite_number(
            candidate.get("max_samples"),
            f"{name}.max_samples",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        return

    if candidate.get("scale_pos_weight_policy") not in {
        "none",
        "sqrt_train_ratio",
        "capped_100",
    }:
        raise ConfigError(f"{name}.scale_pos_weight_policy is unsupported")
    for key in ("n_estimators", "max_depth", "min_child_samples"):
        _require_positive_integer(candidate.get(key), f"{name}.{key}")
    _require_positive_integer(candidate.get("num_leaves"), f"{name}.num_leaves", minimum=2)
    _require_finite_number(
        candidate.get("learning_rate"),
        f"{name}.learning_rate",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _require_finite_number(
        candidate.get("min_child_weight"), f"{name}.min_child_weight", minimum=0.0
    )
    _require_finite_number(candidate.get("reg_alpha"), f"{name}.reg_alpha", minimum=0.0)
    _require_finite_number(
        candidate.get("reg_lambda"),
        f"{name}.reg_lambda",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_finite_number(
        candidate.get("max_delta_step"),
        f"{name}.max_delta_step",
        minimum=0.0,
        minimum_inclusive=False,
    )


def _validate_refinement_ablation(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError("sprint3.ablation must be a mapping")
    _require_exact_keys(
        value,
        {
            "model_family",
            "primary_families",
            "sensitivity_family",
            "duplicate_graph_pairs",
        },
        "sprint3.ablation",
    )
    if value.get("model_family") != "lightgbm":
        raise ConfigError("sprint3.ablation.model_family must be 'lightgbm'")
    expected_primary = [
        "transaction_only",
        "transaction_temporal_history",
        "transaction_temporal_history_graph",
    ]
    if value.get("primary_families") != expected_primary:
        raise ConfigError(
            "sprint3.ablation.primary_families must be ordered transaction-only -> "
            "temporal/history -> graph"
        )
    if value.get("sensitivity_family") != "transaction_temporal_history_graph_novel3":
        raise ConfigError(
            "sprint3.ablation.sensitivity_family must be "
            "'transaction_temporal_history_graph_novel3'"
        )
    expected_duplicates = {
        "sender_prior_fan_out_degree": "sender_previous_unique_counterparties",
        "receiver_prior_fan_in_degree": "receiver_previous_unique_counterparties",
    }
    duplicate_pairs = value.get("duplicate_graph_pairs")
    if not isinstance(duplicate_pairs, Mapping) or dict(duplicate_pairs) != expected_duplicates:
        raise ConfigError(
            "sprint3.ablation.duplicate_graph_pairs must declare the two reviewed exact pairs"
        )


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    resolve_paths: bool = True,
) -> ConfigDict:
    """Load, merge, resolve, and validate an ARGUS YAML configuration.

    Args:
        path: YAML file to load. Relative paths use the caller's working directory.
        project_root: Base for values under ``paths``. By default this is inferred
            as the parent of the conventional ``configs`` directory.
        resolve_paths: If false, keep YAML path strings unchanged.

    Returns:
        A new plain dictionary. When ``resolve_paths`` is true, values below the
        top-level ``paths`` key are absolute strings. ``_meta`` records the source
        configuration and inferred project root for reproducibility.
    """

    config_path = Path(path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else _default_project_root(config_path)
    )
    config = _load_with_extends(config_path)
    if resolve_paths:
        _resolve_config_paths(config, root)
    config["_meta"] = {
        "config_file": str(config_path),
        "project_root": str(root),
        "paths_resolved": resolve_paths,
    }
    validate_config(config)
    return config


def get_path(config: Mapping[str, Any], key: str) -> Path:
    """Return a configured path as ``Path`` with a clear missing-key error."""

    paths = _require_mapping(config, "paths")
    value = paths.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"Missing configured path: paths.{key}")
    return Path(value)
