"""Configuration loading for reproducible ARGUS pipeline runs.

Configuration files are ordinary YAML mappings.  A file may inherit from one or
more YAML files with ``extends``; relative inheritance paths are interpreted
relative to the file that declares them.  Values in the top-level ``paths``
mapping are resolved against the project root (the parent of ``configs/`` by
default), so command behaviour does not depend on the caller's current working
directory.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeAlias

import yaml

ConfigDict: TypeAlias = dict[str, Any]


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
