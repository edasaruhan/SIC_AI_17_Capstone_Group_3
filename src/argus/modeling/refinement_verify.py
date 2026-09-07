"""Independent verification of persisted Sprint 3 refinement evidence.

The verifier deliberately consumes saved artifacts rather than trusting the
run summary.  Its small, pure validators are also used by unit tests to keep
the temporal, feature-family, test-access, and graph-comparison contracts
executable before a full five-million-row run is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from argus.config import get_path, load_config
from argus.modeling.artifacts import (
    atomic_write_json,
    build_artifact_inventory,
    file_fingerprint,
    sha256_file,
    write_run_manifest,
)
from argus.modeling.metrics import compute_average_precision, evaluate_binary_predictions
from argus.modeling.preprocessing import (
    ALWAYS_FORBIDDEN_PREDICTORS,
    FEATURE_FAMILY_CONTRACTS,
    GRAPH_HISTORY_FEATURES,
    TRANSACTION_ONLY_FAMILY,
    TRANSACTION_TEMPORAL_HISTORY_FAMILY,
    TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY,
    FeatureContract,
    FittedPreprocessor,
)
from argus.modeling.refinement_metrics import (
    analyze_validation_score_saturation,
    optimize_validation_thresholds,
)
from argus.modeling.refinement_protocol import (
    select_outer_validation_champion,
    summarize_and_select_candidates,
)


class RefinementVerificationError(RuntimeError):
    """Raised when saved Sprint 3 evidence is absent or contradictory."""


PRIMARY_FEATURE_FAMILIES = (
    TRANSACTION_ONLY_FAMILY,
    TRANSACTION_TEMPORAL_HISTORY_FAMILY,
    TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY,
)
NOVEL_GRAPH_SENSITIVITY_FAMILY = "transaction_temporal_history_graph_novel3"
NOVEL_GRAPH_FEATURES = (
    "sender_prior_fan_in_degree",
    "receiver_prior_fan_out_degree",
    "pair_previous_transfer_count",
)
ALL_ABLATION_FEATURE_FAMILIES = (*PRIMARY_FEATURE_FAMILIES, NOVEL_GRAPH_SENSITIVITY_FAMILY)

_TEST_POLICY_FALSE_FIELDS = (
    "final_test_used_for_training",
    "final_test_used_for_preprocessing_fit",
    "final_test_used_for_tuning",
    "final_test_used_for_model_selection",
    "final_test_used_for_threshold_selection",
    "final_test_inference_performed",
    "test_feature_rows_materialized",
    "test_prediction_artifact_exists",
)

_FORBIDDEN_TEST_ARTIFACT = re.compile(
    r"(?:test.*(?:prediction|score|metric|feature|matrix)|"
    r"(?:prediction|score|metric|feature|matrix).*test)",
    flags=re.IGNORECASE,
)

_PREDICTION_SCORE_ARTIFACTS = {
    "refined_logistic_regression": ("logistic_regression", TRANSACTION_TEMPORAL_HISTORY_FAMILY),
    "refined_random_forest": ("random_forest", TRANSACTION_TEMPORAL_HISTORY_FAMILY),
    "refined_lightgbm": ("lightgbm", TRANSACTION_TEMPORAL_HISTORY_FAMILY),
    "ablation_transaction_only": ("ablation_transaction_only", TRANSACTION_ONLY_FAMILY),
    "ablation_transaction_temporal_history_graph": (
        "ablation_transaction_temporal_history_graph",
        TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY,
    ),
    "ablation_transaction_temporal_history_graph_novel3": (
        "ablation_transaction_temporal_history_graph_novel3",
        "transaction_temporal_history_graph_novel3",
    ),
}


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefinementVerificationError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RefinementVerificationError(f"JSON artifact is not an object: {path}")
    return payload


def _iso_timestamp(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise RefinementVerificationError(f"{name} must be an ISO timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RefinementVerificationError(f"{name} is not a valid ISO timestamp") from exc
    # The source timestamps are timezone-naive.  Normalize an explicit UTC
    # value to naive UTC solely so it can be compared with DuckDB values.
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _require_bool(payload: Mapping[str, Any], key: str, expected: bool) -> None:
    if payload.get(key) is not expected:
        raise RefinementVerificationError(f"{key} must be explicitly {expected}")


def _assert_close(actual: Any, expected: Any, *, name: str) -> None:
    try:
        matches = math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)
    except (TypeError, ValueError) as exc:
        raise RefinementVerificationError(f"{name} is not numeric") from exc
    if not matches:
        raise RefinementVerificationError(
            f"{name} differs: saved={actual!r}, recomputed={expected!r}"
        )


def _stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contains_test_partition(value: object) -> bool:
    return bool(re.search(r"(?:^|_)test(?:$|_)", str(value).strip().lower()))


def validate_closed_test_policy(
    policy: Mapping[str, Any],
    data_access_audit: Mapping[str, Any],
    artifact_paths: Sequence[str],
) -> dict[str, Any]:
    """Reject evidence of any Sprint 3 test transform, inference, or selection."""

    for key in _TEST_POLICY_FALSE_FIELDS:
        _require_bool(policy, key, False)
    if policy.get("selection_partition", "validation") != "validation":
        raise RefinementVerificationError("Final model selection must be validation-only")
    permitted = policy.get(
        "permitted_model_partitions", policy.get("permitted_transformed_partitions")
    )
    if not isinstance(permitted, list) or "test" in permitted:
        raise RefinementVerificationError("Test is present in permitted model partitions")
    if policy.get("test_metadata_reported") is not True:
        raise RefinementVerificationError("Only pre-existing test metadata may be reported")
    metadata_source = str(policy.get("test_metadata_source", "")).lower()
    if "metadata" not in metadata_source:
        raise RefinementVerificationError("Test metadata source is not documented")

    audited_lists = 0
    for key, value in data_access_audit.items():
        if key.endswith("partitions"):
            if not isinstance(value, list):
                raise RefinementVerificationError(f"{key} must be a list")
            audited_lists += 1
            if any(_contains_test_partition(partition) for partition in value):
                raise RefinementVerificationError(f"Test access recorded in {key}")
    if audited_lists == 0:
        raise RefinementVerificationError("Data access audit has no partition lists")
    test_access = str(data_access_audit.get("test_access", "")).lower()
    if "metadata" not in test_access or any(
        word in test_access for word in ("transform", "inference", "prediction", "label")
    ):
        raise RefinementVerificationError("Data access audit does not restrict test to metadata")

    suspicious = [
        path
        for path in artifact_paths
        if _FORBIDDEN_TEST_ARTIFACT.search(Path(path).as_posix())
        and Path(path).name != "final_test_policy.json"
    ]
    if suspicious:
        raise RefinementVerificationError(
            f"Test-derived artifact names were found: {sorted(suspicious)}"
        )
    return {
        "status": "PASS",
        "explicit_false_policy_fields": len(_TEST_POLICY_FALSE_FIELDS),
        "partition_lists_checked": audited_lists,
        "test_derived_artifacts_found": 0,
        "test_access": "pre_existing_metadata_only",
    }


def _fold_rows(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        if payload.get("strategy") != "expanding_window":
            raise RefinementVerificationError("Temporal CV strategy is not expanding_window")
        values = payload.get("folds")
    else:
        values = payload
    if not isinstance(values, list) or not values:
        raise RefinementVerificationError("Temporal fold artifact contains no folds")
    if not all(isinstance(value, dict) for value in values):
        raise RefinementVerificationError("Temporal fold entries must be objects")
    return values


def validate_temporal_folds(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    outer_train: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate expanding, non-overlapping folds wholly inside frozen outer train."""

    rows = _fold_rows(payload)
    outer_min = _iso_timestamp(outer_train.get("minimum_timestamp"), name="outer train minimum")
    outer_max = _iso_timestamp(outer_train.get("maximum_timestamp"), name="outer train maximum")
    expected_numbers = list(range(1, len(rows) + 1))
    if [int(row.get("fold", -1)) for row in rows] != expected_numbers:
        raise RefinementVerificationError("Temporal folds are not consecutively numbered")

    previous: dict[str, Any] | None = None
    common_train_minimum: datetime | None = None
    for row in rows:
        number = int(row["fold"])
        train_min = _iso_timestamp(
            row.get("train_minimum_timestamp"), name=f"fold {number} train minimum"
        )
        train_max = _iso_timestamp(
            row.get("train_maximum_timestamp"), name=f"fold {number} train maximum"
        )
        validation_min = _iso_timestamp(
            row.get("validation_minimum_timestamp"), name=f"fold {number} validation minimum"
        )
        validation_max = _iso_timestamp(
            row.get("validation_maximum_timestamp"), name=f"fold {number} validation maximum"
        )
        if not outer_min <= train_min <= train_max < validation_min <= validation_max <= outer_max:
            raise RefinementVerificationError(
                f"Fold {number} is not strictly chronological inside frozen outer train"
            )
        if row.get("strict_chronology") is not True:
            raise RefinementVerificationError(f"Fold {number} lacks strict chronology evidence")
        if row.get("timestamp_groups_split") is not False:
            raise RefinementVerificationError(f"Fold {number} splits equal timestamp groups")
        for prefix in ("train", "validation"):
            count = int(row.get(f"{prefix}_rows", 0))
            positives = int(row.get(f"{prefix}_positive_labels", -1))
            if count <= 0 or not 0 < positives < count:
                raise RefinementVerificationError(
                    f"Fold {number} {prefix} must be non-empty and contain both classes"
                )
            rate = row.get(f"{prefix}_positive_rate")
            if rate is not None:
                _assert_close(
                    rate,
                    positives / count,
                    name=f"fold {number} {prefix} positive rate",
                )

        if common_train_minimum is None:
            common_train_minimum = train_min
        elif train_min != common_train_minimum:
            raise RefinementVerificationError("Expanding folds do not share one train origin")
        if previous is not None:
            prior_train_max = previous["train_max"]
            prior_validation_max = previous["validation_max"]
            if train_max <= prior_train_max:
                raise RefinementVerificationError("Expanding train windows do not grow")
            if train_max < prior_validation_max:
                raise RefinementVerificationError(
                    "A later train window does not include the preceding validation window"
                )
            if validation_min <= prior_validation_max:
                raise RefinementVerificationError("Temporal validation windows overlap")
            if int(row["train_rows"]) <= int(previous["train_rows"]):
                raise RefinementVerificationError("Expanding train row counts do not grow")
        previous = {
            "train_max": train_max,
            "validation_max": validation_max,
            "train_rows": int(row["train_rows"]),
        }
    return {
        "status": "PASS",
        "strategy": "expanding_window",
        "fold_count": len(rows),
        "inside_frozen_outer_train": True,
        "strict_chronology": True,
        "timestamp_groups_split": False,
    }


def verify_folds_against_frozen_inputs(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    feature_table: Path,
    split_table: Path,
    fold_manifest_dir: Path | None = None,
) -> dict[str, Any]:
    """Recount folds and, when supplied, reconcile exact saved memberships."""

    rows = _fold_rows(payload)
    feature_path = str(feature_table.resolve())
    split_path = str(split_table.resolve())
    connection = duckdb.connect()
    try:
        for row in rows:
            train_end = _iso_timestamp(row["train_maximum_timestamp"], name="fold train maximum")
            validation_end = _iso_timestamp(
                row["validation_maximum_timestamp"], name="fold validation maximum"
            )
            recomputed = connection.execute(
                """
                SELECT
                    min(split.timestamp) FILTER (WHERE split.timestamp <= ?),
                    max(split.timestamp) FILTER (WHERE split.timestamp <= ?),
                    min(split.timestamp) FILTER (
                        WHERE split.timestamp > ? AND split.timestamp <= ?
                    ),
                    max(split.timestamp) FILTER (
                        WHERE split.timestamp > ? AND split.timestamp <= ?
                    ),
                    count(*) FILTER (WHERE split.timestamp <= ?),
                    sum(feature.is_laundering) FILTER (WHERE split.timestamp <= ?),
                    count(*) FILTER (WHERE split.timestamp > ? AND split.timestamp <= ?),
                    sum(feature.is_laundering) FILTER (
                        WHERE split.timestamp > ? AND split.timestamp <= ?
                    )
                FROM read_parquet(?) split
                INNER JOIN read_parquet(?) feature USING (transaction_id)
                WHERE split.partition = 'train' AND split.timestamp <= ?
                """,
                [
                    train_end,
                    train_end,
                    train_end,
                    validation_end,
                    train_end,
                    validation_end,
                    train_end,
                    train_end,
                    train_end,
                    validation_end,
                    train_end,
                    validation_end,
                    split_path,
                    feature_path,
                    validation_end,
                ],
            ).fetchone()
            expected = (
                _iso_timestamp(row["train_minimum_timestamp"], name="fold train minimum"),
                train_end,
                _iso_timestamp(row["validation_minimum_timestamp"], name="fold validation minimum"),
                validation_end,
                int(row["train_rows"]),
                int(row["train_positive_labels"]),
                int(row["validation_rows"]),
                int(row["validation_positive_labels"]),
            )
            if tuple(recomputed) != expected:
                raise RefinementVerificationError(
                    f"Frozen-input fold recount differs for fold {row['fold']}: "
                    f"saved={expected!r}, recomputed={tuple(recomputed)!r}"
                )
            if fold_manifest_dir is not None:
                saved_manifest = fold_manifest_dir / f"fold_{int(row['fold'])}.parquet"
                if not saved_manifest.is_file():
                    raise RefinementVerificationError(
                        f"Persisted split manifest is missing for fold {row['fold']}"
                    )
                membership = connection.execute(
                    """
                    WITH expected AS (
                        SELECT
                            transaction_id,
                            timestamp,
                            CASE
                                WHEN timestamp <= ? THEN 'train'
                                ELSE 'validation'
                            END AS partition
                        FROM read_parquet(?)
                        WHERE partition = 'train' AND timestamp <= ?
                    ),
                    saved AS (
                        SELECT transaction_id, timestamp, partition
                        FROM read_parquet(?)
                    )
                    SELECT
                        count(*) FILTER (WHERE expected.transaction_id IS NULL),
                        count(*) FILTER (WHERE saved.transaction_id IS NULL),
                        count(*) FILTER (
                            WHERE expected.transaction_id IS NOT NULL
                              AND saved.transaction_id IS NOT NULL
                              AND (
                                  expected.timestamp IS DISTINCT FROM saved.timestamp
                                  OR expected.partition IS DISTINCT FROM saved.partition
                              )
                        ),
                        count(saved.transaction_id),
                        count(DISTINCT saved.transaction_id)
                    FROM expected
                    FULL OUTER JOIN saved USING (transaction_id)
                    """,
                    [
                        train_end,
                        split_path,
                        validation_end,
                        str(saved_manifest.resolve()),
                    ],
                ).fetchone()
                expected_saved_rows = int(row["train_rows"]) + int(row["validation_rows"])
                expected_membership = (0, 0, 0, expected_saved_rows, expected_saved_rows)
                if tuple(int(value) for value in membership) != expected_membership:
                    raise RefinementVerificationError(
                        f"Persisted fold membership differs for fold {row['fold']}: "
                        f"{tuple(membership)!r}"
                    )
    finally:
        connection.close()
    return {
        "status": "PASS",
        "folds_recounted": len(rows),
        "saved_fold_manifests_reconciled": (0 if fold_manifest_dir is None else len(rows)),
        "query_partition": "train",
        "test_rows_queried": False,
    }


def _contract(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise RefinementVerificationError("Preprocessing manifest has no feature contract")
    return contract


def validate_feature_family_manifests(
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconcile saved A/B/C contracts with the reviewed source allow-lists."""

    if not set(PRIMARY_FEATURE_FAMILIES).issubset(manifests):
        missing = sorted(set(PRIMARY_FEATURE_FAMILIES).difference(manifests))
        raise RefinementVerificationError(f"Primary feature-family manifests missing: {missing}")
    predictor_sets: dict[str, set[str]] = {}
    feature_counts: dict[str, int] = {}
    for family in PRIMARY_FEATURE_FAMILIES:
        manifest = manifests[family]
        expected = FEATURE_FAMILY_CONTRACTS[family].to_dict()
        actual = {key: list(value) for key, value in _contract(manifest).items()}
        if actual != expected:
            raise RefinementVerificationError(f"Saved feature contract differs for {family}")
        if manifest.get("fit_scope") != "train_only":
            raise RefinementVerificationError(f"{family} preprocessing was not train-fit only")
        names = manifest.get("transformed_feature_names")
        count = manifest.get("transformed_feature_count")
        if (
            not isinstance(names, list)
            or int(count or -1) != len(names)
            or len(set(names)) != len(names)
        ):
            raise RefinementVerificationError(
                f"{family} transformed feature names/count are inconsistent"
            )
        predictor_sets[family] = set(FEATURE_FAMILY_CONTRACTS[family].predictor_columns)
        feature_counts[family] = len(names)
    family_a, family_b, family_c = (predictor_sets[name] for name in PRIMARY_FEATURE_FAMILIES)
    if not family_a < family_b < family_c:
        raise RefinementVerificationError("A/B/C source predictor sets are not strictly nested")
    if family_c - family_b != set(GRAPH_HISTORY_FEATURES):
        raise RefinementVerificationError("C minus B is not exactly the reviewed graph family")
    return {
        "status": "PASS",
        "families": list(PRIMARY_FEATURE_FAMILIES),
        "strictly_nested": True,
        "c_minus_b": list(GRAPH_HISTORY_FEATURES),
        "transformed_feature_counts": feature_counts,
        "train_fit_only": True,
    }


def _required_column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise RefinementVerificationError(f"Ablation table is missing one of columns {names}")


def _verify_test_named_columns_are_empty(frame: pd.DataFrame, *, context: str) -> int:
    """Allow explicit closed-policy columns, but reject populated test results."""

    checked = 0
    for column in frame.columns:
        if "test" not in column.lower():
            continue
        checked += 1
        for value in frame[column].tolist():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            if isinstance(value, (bool, np.bool_)) and not bool(value):
                continue
            if isinstance(value, str) and value.strip().lower() in {
                "",
                "false",
                "none",
                "null",
                "nan",
            }:
                continue
            raise RefinementVerificationError(
                f"{context} contains populated test-derived field {column!r}"
            )
    return checked


def validate_ablation_comparability(
    frame: pd.DataFrame,
    *,
    expected_split_sha256: str,
    expected_random_seed: int,
) -> dict[str, Any]:
    """Ensure A/B/C isolate features while model, split, and protocol stay fixed."""

    required = {"feature_family", "model_family", "candidate_id", "evaluation_partition"}
    missing = required - set(frame.columns)
    if missing:
        raise RefinementVerificationError(f"Ablation table columns missing: {sorted(missing)}")
    _verify_test_named_columns_are_empty(frame, context="Ablation table")
    if frame["feature_family"].duplicated().any():
        raise RefinementVerificationError("Ablation table has duplicate feature-family rows")
    indexed = frame.set_index("feature_family", drop=False)
    missing_families = set(PRIMARY_FEATURE_FAMILIES).difference(indexed.index)
    if missing_families:
        raise RefinementVerificationError(
            f"Ablation table lacks primary families: {sorted(missing_families)}"
        )
    primary = indexed.loc[list(PRIMARY_FEATURE_FAMILIES)]
    if set(primary["evaluation_partition"].astype(str)) != {"validation"}:
        raise RefinementVerificationError("Ablation evaluation is not validation-only")

    parameter_column = _required_column(
        primary, "effective_parameters_sha256", "effective_parameters_digest"
    )
    split_column = _required_column(primary, "split_manifest_sha256", "frozen_split_sha256")
    required_equal = [
        "model_family",
        "candidate_id",
        parameter_column,
        "random_seed",
        split_column,
    ]
    optional_protocol = [
        column
        for column in ("evaluation_protocol_sha256", "ranking_score_type", "top_k_values")
        if column in primary.columns
    ]
    for column in (*required_equal, *optional_protocol):
        if primary[column].astype(str).nunique() != 1:
            raise RefinementVerificationError(
                f"A/B/C ablation changes non-feature factor {column!r}"
            )
    if str(primary.iloc[0][split_column]) != expected_split_sha256:
        raise RefinementVerificationError("Ablation split digest differs from frozen split")
    if int(primary.iloc[0]["random_seed"]) != int(expected_random_seed):
        raise RefinementVerificationError("Ablation random seed differs from run configuration")
    metric_column = _required_column(primary, "average_precision", "pr_auc_average_precision")
    if not np.isfinite(pd.to_numeric(primary[metric_column], errors="coerce")).all():
        raise RefinementVerificationError("Ablation PR-AUC values are not finite")
    return {
        "status": "PASS",
        "evaluation_partition": "validation",
        "primary_families": list(PRIMARY_FEATURE_FAMILIES),
        "same_model_family": True,
        "same_candidate": True,
        "same_effective_parameters": True,
        "same_frozen_split": True,
        "same_random_seed": True,
        "same_evaluation_protocol": True,
        "metric_column": metric_column,
    }


def validate_graph_value_conclusion(
    payload: Mapping[str, Any],
    ablation: pd.DataFrame,
) -> dict[str, Any]:
    """Recompute B-to-C graph delta and reject causal or test-based claims."""

    if payload.get("evaluation_partition") != "validation":
        raise RefinementVerificationError("Graph-value conclusion is not validation-only")
    _require_bool(payload, "test_metrics_used", False)
    comparability = payload.get("comparability", payload)
    if not isinstance(comparability, Mapping):
        raise RefinementVerificationError("Graph-value comparability evidence is absent")
    for key in (
        "same_model_family",
        "same_candidate",
        "same_effective_parameters",
        "same_frozen_split",
        "same_evaluation_protocol",
    ):
        _require_bool(comparability, key, True)

    indexed = ablation.set_index("feature_family")
    metric = "average_precision"
    if metric not in indexed.columns:
        metric = "pr_auc_average_precision"
    baseline = float(indexed.loc[TRANSACTION_TEMPORAL_HISTORY_FAMILY, metric])
    enhanced = float(indexed.loc[TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY, metric])
    saved_baseline = payload.get("baseline_average_precision", payload.get("b_average_precision"))
    saved_enhanced = payload.get("graph_average_precision", payload.get("c_average_precision"))
    saved_delta = payload.get(
        "absolute_average_precision_delta", payload.get("delta_average_precision")
    )
    _assert_close(saved_baseline, baseline, name="graph conclusion baseline PR-AUC")
    _assert_close(saved_enhanced, enhanced, name="graph conclusion enhanced PR-AUC")
    _assert_close(saved_delta, enhanced - baseline, name="graph conclusion PR-AUC delta")
    interpretation = str(payload.get("interpretation", "")).lower()
    if any(term in interpretation for term in ("causes", "caused", "causal proof")):
        raise RefinementVerificationError("Ablation importance is incorrectly framed as causal")
    return {
        "status": "PASS",
        "baseline_average_precision": baseline,
        "graph_average_precision": enhanced,
        "absolute_average_precision_delta": enhanced - baseline,
        "causal_claim_made": False,
    }


def validate_ablation_model_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
    ablation: pd.DataFrame,
) -> dict[str, Any]:
    """Cross-check same-model evidence against each persisted fitted model record."""

    missing = set(ALL_ABLATION_FEATURE_FAMILIES).difference(metadata)
    if missing:
        raise RefinementVerificationError(
            f"Ablation model metadata is missing families: {sorted(missing)}"
        )
    indexed = ablation.set_index("feature_family")
    common: dict[str, str] = {}
    comparison_fields = (
        "model_family",
        "candidate_id",
        "effective_parameters_sha256",
        "random_seed",
        "split_manifest_sha256",
        "ranking_score_type",
        "configured_top_k",
    )
    for family in ALL_ABLATION_FEATURE_FAMILIES:
        record = metadata[family]
        if record.get("feature_family") != family:
            raise RefinementVerificationError(f"Model metadata feature family differs for {family}")
        if record.get("evaluation_partition") != "validation":
            raise RefinementVerificationError(f"Ablation model {family} is not validation-only")
        _require_bool(record, "test_predictions_generated", False)
        if record.get("test_metrics") is not None:
            raise RefinementVerificationError(f"Ablation model {family} contains test metrics")
        effective = record.get("effective_parameters")
        if not isinstance(effective, Mapping):
            raise RefinementVerificationError(f"Effective parameters are missing for {family}")
        digest = _stable_digest(effective)
        if digest != record.get("effective_parameters_sha256"):
            raise RefinementVerificationError(f"Effective parameter digest differs for {family}")
        row = indexed.loc[family]
        for column in (
            "model_family",
            "candidate_id",
            "effective_parameters_sha256",
            "random_seed",
            "split_manifest_sha256",
        ):
            if str(record.get(column)) != str(row[column]):
                raise RefinementVerificationError(
                    f"Ablation table and model metadata differ for {family}: {column}"
                )
        if int(record.get("transformed_feature_count", -1)) != int(
            row["transformed_feature_count"]
        ):
            raise RefinementVerificationError(
                f"Ablation table and model metadata differ for {family}: transformed_feature_count"
            )
        for field in comparison_fields:
            serialized = json.dumps(record.get(field), sort_keys=True, default=str)
            if field in common and common[field] != serialized:
                raise RefinementVerificationError(
                    f"A/B/C model metadata changes non-feature factor {field!r}"
                )
            common[field] = serialized
    return {
        "status": "PASS",
        "families": list(ALL_ABLATION_FEATURE_FAMILIES),
        "same_fitted_model_family_candidate_parameters_seed_split_and_protocol": True,
        "evaluation_partition": "validation",
        "test_metrics_used": False,
    }


def verify_ablation_metrics_from_predictions(
    ablation: pd.DataFrame,
    recomputed_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconcile every ablation row with independently recomputed saved scores."""

    score_key_by_family = {
        TRANSACTION_ONLY_FAMILY: "ablation_transaction_only",
        TRANSACTION_TEMPORAL_HISTORY_FAMILY: "refined_lightgbm",
        TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY: ("ablation_transaction_temporal_history_graph"),
        "transaction_temporal_history_graph_novel3": (
            "ablation_transaction_temporal_history_graph_novel3"
        ),
    }
    indexed = ablation.set_index("feature_family")
    if set(indexed.index) != set(score_key_by_family):
        raise RefinementVerificationError(
            "Ablation table rows differ from A/B/C plus novel-three sensitivity"
        )
    checked: dict[str, Any] = {}
    for family, score_key in score_key_by_family.items():
        values = recomputed_scores.get(score_key)
        if not isinstance(values, Mapping):
            raise RefinementVerificationError(
                f"Recomputed validation scores are missing for ablation family {family}"
            )
        row = indexed.loc[family]
        _assert_close(
            row["average_precision"],
            values["average_precision"],
            name=f"{family} ablation average precision",
        )
        _assert_close(
            row["roc_auc"],
            values["roc_auc"],
            name=f"{family} ablation ROC-AUC",
        )
        checked[family] = {
            "score_artifact": score_key,
            "average_precision": float(values["average_precision"]),
            "roc_auc": float(values["roc_auc"]),
        }
    return {
        "status": "PASS",
        "rows_recomputed": len(checked),
        "validation_only": True,
        "families": checked,
        "test_metrics_used": False,
    }


def verify_core_model_artifacts(
    run_dir: Path,
    selected_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Deserialize refined core models and reconcile selection/convergence metadata."""

    expected_classes = {
        "logistic_regression": "LogisticRegression",
        "random_forest": "RandomForestClassifier",
        "lightgbm": "LGBMClassifier",
    }
    evidence: dict[str, Any] = {}
    for model_name, expected_class in expected_classes.items():
        model_dir = run_dir / "models" / model_name
        metadata = _load_object(model_dir / "metadata.json")
        selected = selected_candidates.get(model_name)
        if not isinstance(selected, Mapping):
            raise RefinementVerificationError(f"Selected candidate missing for {model_name}")
        if metadata.get("candidate_id") != selected.get("candidate_id"):
            raise RefinementVerificationError(
                f"Outer model does not use the inner-CV winner for {model_name}"
            )
        if metadata.get("selection_source") != "inner_temporal_validation_folds":
            raise RefinementVerificationError(f"Outer {model_name} selection source differs")
        if metadata.get("feature_family") != TRANSACTION_TEMPORAL_HISTORY_FAMILY:
            raise RefinementVerificationError(f"Core {model_name} does not use feature family B")
        _require_bool(metadata, "test_predictions_generated", False)
        if metadata.get("test_metrics") is not None:
            raise RefinementVerificationError(f"Core {model_name} contains test metrics")
        parameters = metadata.get("effective_parameters")
        if not isinstance(parameters, Mapping):
            raise RefinementVerificationError(f"Effective parameters missing for {model_name}")
        if _stable_digest(parameters) != metadata.get("effective_parameters_sha256"):
            raise RefinementVerificationError(
                f"Effective parameter digest differs for {model_name}"
            )
        estimator = joblib.load(model_dir / "model.joblib")
        if type(estimator).__name__ != expected_class:
            raise RefinementVerificationError(
                f"Saved {model_name} class differs: {type(estimator).__name__}"
            )
        model_evidence: dict[str, Any] = {
            "candidate_id": metadata["candidate_id"],
            "implementation_class": expected_class,
            "effective_parameters_sha256": metadata["effective_parameters_sha256"],
        }
        if model_name == "logistic_regression":
            convergence = metadata.get("convergence")
            if not isinstance(convergence, Mapping) or convergence.get("converged") is not True:
                raise RefinementVerificationError("Refined Logistic Regression did not converge")
            observed = int(np.asarray(estimator.n_iter_, dtype=np.int64).max())
            configured = int(estimator.max_iter)
            if observed >= configured:
                raise RefinementVerificationError(
                    "Serialized Logistic Regression reached its max_iter boundary"
                )
            if observed != int(convergence.get("observed_n_iter", -1)):
                raise RefinementVerificationError(
                    "Serialized Logistic Regression iterations differ from metadata"
                )
            if configured != int(convergence.get("configured_max_iter", -1)):
                raise RefinementVerificationError(
                    "Serialized Logistic Regression max_iter differs from metadata"
                )
            if convergence.get("convergence_warnings") != []:
                raise RefinementVerificationError(
                    "Refined Logistic Regression records convergence warnings"
                )
            model_evidence.update(
                {
                    "converged": True,
                    "observed_n_iter": observed,
                    "configured_max_iter": configured,
                    "stopped_before_max_iter": True,
                }
            )
        evidence[model_name] = model_evidence
    return {
        "status": "PASS",
        "models_deserialized": evidence,
        "selection_source": "inner_temporal_validation_folds",
        "test_metrics_used": False,
    }


def verify_duplicate_graph_evidence(
    saved: Mapping[str, Any],
    *,
    feature_table: Path,
    split_table: Path,
    expected_rows: int,
) -> dict[str, Any]:
    """Recompute graph/history alias evidence over train+validation, never test."""

    expected_pairs = {
        "sender_prior_fan_out_degree": "sender_previous_unique_counterparties",
        "receiver_prior_fan_in_degree": "receiver_previous_unique_counterparties",
    }
    for graph_feature, history_feature in expected_pairs.items():
        if saved.get(graph_feature) != history_feature:
            raise RefinementVerificationError(
                f"Saved duplicate-feature mapping differs for {graph_feature}"
            )
    if saved.get("partitions_checked") != ["train", "validation"]:
        raise RefinementVerificationError("Duplicate graph evidence is not train+validation-only")
    _require_bool(saved, "test_rows_checked", False)

    connection = duckdb.connect()
    try:
        row = connection.execute(
            """
            SELECT
                count(*),
                count(*) FILTER (
                    WHERE sender_prior_fan_out_degree
                          IS DISTINCT FROM sender_previous_unique_counterparties
                ),
                count(*) FILTER (
                    WHERE receiver_prior_fan_in_degree
                          IS DISTINCT FROM receiver_previous_unique_counterparties
                )
            FROM read_parquet(?) feature
            INNER JOIN (
                SELECT transaction_id
                FROM read_parquet(?)
                WHERE partition IN ('train', 'validation')
            ) permitted USING (transaction_id)
            """,
            [str(feature_table.resolve()), str(split_table.resolve())],
        ).fetchone()
    finally:
        connection.close()
    recomputed = {
        "rows_checked": int(row[0]),
        "sender_mismatch_rows": int(row[1]),
        "receiver_mismatch_rows": int(row[2]),
        "mismatch_rows_full_train_plus_validation": int(row[1]) + int(row[2]),
    }
    if recomputed["rows_checked"] != expected_rows:
        raise RefinementVerificationError("Duplicate graph recount row count differs")
    for key, value in recomputed.items():
        if int(saved.get(key, -1)) != value:
            raise RefinementVerificationError(f"Duplicate graph evidence differs at {key}")
    return {
        "status": "PASS",
        **recomputed,
        "partitions_checked": ["train", "validation"],
        "test_rows_checked": False,
    }


def validate_tuning_artifacts(
    trials: pd.DataFrame,
    selected: Mapping[str, Any],
    *,
    expected_folds: int,
    configured_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Recompute bounded candidate selection from inner temporal folds only."""

    _verify_test_named_columns_are_empty(trials, context="Tuning trials")
    if "partition" not in trials or set(trials["partition"].astype(str)) != {"inner_validation"}:
        raise RefinementVerificationError("Tuning trials are not inner-validation-only")
    if "feature_family" not in trials or set(trials["feature_family"].astype(str)) != {
        TRANSACTION_TEMPORAL_HISTORY_FAMILY
    }:
        raise RefinementVerificationError("Tuning did not use frozen feature family B")
    expected_candidate_ids: dict[str, set[str]] = {
        family: {str(candidate["candidate_id"]) for candidate in candidates}
        for family, candidates in configured_candidates.items()
    }
    if set(trials["model_family"].astype(str)) != set(expected_candidate_ids):
        raise RefinementVerificationError("Executed tuning model families differ from config")
    for family, candidates in expected_candidate_ids.items():
        observed = set(
            trials.loc[trials["model_family"].astype(str) == family, "candidate_id"].astype(str)
        )
        if observed != candidates:
            raise RefinementVerificationError(
                f"Executed candidates differ from config for {family}: {sorted(observed)}"
            )
    if set(pd.to_numeric(trials["fold"], errors="raise").astype(int)) != set(
        range(1, expected_folds + 1)
    ):
        raise RefinementVerificationError("Tuning fold identifiers are incomplete")
    if not np.isfinite(pd.to_numeric(trials["average_precision"], errors="coerce")).all():
        raise RefinementVerificationError("Tuning PR-AUC contains non-finite values")

    _, recomputed = summarize_and_select_candidates(trials, expected_folds=expected_folds)
    saved = selected.get("selected_candidates", selected)
    if not isinstance(saved, Mapping):
        raise RefinementVerificationError("Selected-candidate artifact is invalid")
    for family, result in recomputed.items():
        candidate = saved.get(family)
        if not isinstance(candidate, Mapping):
            raise RefinementVerificationError(f"Selected candidate missing for {family}")
        if candidate.get("candidate_id") != result["candidate_id"]:
            raise RefinementVerificationError(
                f"Selected candidate differs from inner-CV recomputation for {family}"
            )
        if candidate.get("selection_partition") != "inner_temporal_validation_folds":
            raise RefinementVerificationError(f"{family} candidate was not selected on inner CV")
        _require_bool(candidate, "test_metrics_used", False)
    return {
        "status": "PASS",
        "folds": expected_folds,
        "trial_rows": len(trials),
        "candidate_count": sum(len(values) for values in expected_candidate_ids.values()),
        "selection_partition": "inner_temporal_validation_folds",
        "selected_candidates": {
            family: result["candidate_id"] for family, result in recomputed.items()
        },
    }


def validate_threshold_artifact(
    payload: Mapping[str, Any],
    *,
    fpr_ceiling: float,
    alert_budget: int,
) -> dict[str, Any]:
    """Validate a whole-tie-group, validation-only operating-point artifact."""

    if payload.get("partition") != "validation" or payload.get("validation_only") is not True:
        raise RefinementVerificationError("Threshold optimization is not validation-only")
    _require_bool(payload, "test_partition_used", False)
    definition = str(payload.get("candidate_definition", "")).lower()
    if "equal-score group" not in definition:
        raise RefinementVerificationError("Threshold candidates do not preserve score ties")
    summaries = payload.get("summaries")
    if not isinstance(summaries, Mapping):
        raise RefinementVerificationError("Threshold summaries are missing")
    primary = summaries.get("predeclared_joint_primary_constraint")
    if not isinstance(primary, Mapping):
        raise RefinementVerificationError("Predeclared primary threshold is missing")
    constraints = primary.get("constraints")
    if not isinstance(constraints, Mapping):
        raise RefinementVerificationError("Primary threshold constraints are missing")
    _assert_close(
        constraints.get("false_positive_rate_at_most"),
        fpr_ceiling,
        name="primary threshold FPR ceiling",
    )
    if int(constraints.get("alert_count_at_most", -1)) != int(alert_budget):
        raise RefinementVerificationError("Primary threshold alert budget differs from config")
    point = primary.get("operating_point")
    if primary.get("status") == "selected":
        if not isinstance(point, Mapping):
            raise RefinementVerificationError("Selected primary threshold has no operating point")
        if int(point["alert_count"]) > int(alert_budget):
            raise RefinementVerificationError("Selected threshold exceeds alert budget")
        if float(point["false_positive_rate"]) > float(fpr_ceiling) + 1e-15:
            raise RefinementVerificationError("Selected threshold exceeds FPR ceiling")
    return {
        "status": "PASS",
        "partition": "validation",
        "whole_equal_score_groups": True,
        "primary_selection_status": primary.get("status"),
        "test_partition_used": False,
    }


def recompute_threshold_artifact(
    payload: Mapping[str, Any],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    fpr_ceiling: float,
    alert_budget: int,
) -> None:
    """Recompute the saved threshold summaries from validation labels and scores."""

    recomputed = optimize_validation_thresholds(
        labels,
        scores,
        partition="validation",
        fpr_ceiling=fpr_ceiling,
        alert_budget=alert_budget,
        primary_fpr_ceiling=fpr_ceiling,
        primary_alert_budget=alert_budget,
        include_frontier="frontier" in payload,
    )
    for key in (
        "partition",
        "validation_only",
        "test_partition_used",
        "candidate_definition",
        "frontier_point_count_including_no_alert_sentinel",
        "summaries",
    ):
        if payload.get(key) != recomputed.get(key):
            raise RefinementVerificationError(
                f"Saved threshold artifact differs from recomputation at {key}"
            )


def validate_threshold_analysis(
    payload: Mapping[str, Any],
    *,
    fpr_ceiling: float,
    alert_budget: int,
) -> dict[str, Any]:
    """Validate the run-level wrapper around per-model threshold evidence."""

    if payload.get("partition") != "validation":
        raise RefinementVerificationError("Run-level threshold analysis is not validation-only")
    _require_bool(payload, "test_metrics_used", False)
    if payload.get("score_type") != "raw_ranking_score":
        raise RefinementVerificationError("Thresholds were not optimized on raw ranking scores")
    if payload.get("tie_policy") != "include_complete_equal_score_group":
        raise RefinementVerificationError("Threshold wrapper does not preserve complete ties")
    models = payload.get("models")
    if not isinstance(models, Mapping) or not models:
        raise RefinementVerificationError("Run-level threshold analysis contains no models")
    results = {
        str(name): validate_threshold_artifact(
            model,
            fpr_ceiling=fpr_ceiling,
            alert_budget=alert_budget,
        )
        for name, model in models.items()
        if isinstance(model, Mapping)
    }
    if len(results) != len(models):
        raise RefinementVerificationError("Run-level threshold model payload is invalid")
    return {
        "status": "PASS",
        "partition": "validation",
        "score_type": "raw_ranking_score",
        "models": results,
        "test_metrics_used": False,
    }


def _threshold_for_metrics(payload: Mapping[str, Any], scores: np.ndarray) -> float:
    summaries = payload.get("summaries")
    if not isinstance(summaries, Mapping):
        raise RefinementVerificationError("Threshold summaries are missing")
    primary = summaries.get("predeclared_joint_primary_constraint")
    if not isinstance(primary, Mapping):
        raise RefinementVerificationError("Primary threshold is missing")
    point = primary.get("operating_point")
    if isinstance(point, Mapping) and point.get("threshold") is not None:
        return float(point["threshold"])
    return float(np.nextafter(np.max(scores), np.inf))


def verify_validation_predictions(
    prediction_path: Path,
    prediction_manifest: Mapping[str, Any],
    *,
    feature_table: Path,
    split_table: Path,
    expected_rows: int,
    expected_positive_labels: int,
    top_k_values: Sequence[int],
    run_dir: Path,
    fpr_ceiling: float,
    alert_budget: int,
) -> dict[str, Any]:
    """Verify outer-validation membership and recompute refined metrics/thresholds."""

    if prediction_manifest.get("partition") != "validation":
        raise RefinementVerificationError("Prediction manifest is not validation-only")
    _require_bool(prediction_manifest, "test_predictions_included", False)
    score_map = prediction_manifest.get("model_score_columns")
    if not isinstance(score_map, Mapping) or not score_map:
        raise RefinementVerificationError("Prediction score-column mapping is missing")
    if set(score_map) != set(_PREDICTION_SCORE_ARTIFACTS):
        raise RefinementVerificationError(
            "Prediction score artifacts differ from the three refined and three ablation models"
        )
    if int(prediction_manifest.get("rows", -1)) != expected_rows:
        raise RefinementVerificationError("Prediction manifest row count differs")
    if int(prediction_manifest.get("unique_transaction_ids", -1)) != expected_rows:
        raise RefinementVerificationError("Prediction manifest unique-ID count differs")
    if int(prediction_manifest.get("positive_labels", -1)) != expected_positive_labels:
        raise RefinementVerificationError("Prediction manifest positive count differs")

    prediction_sql = str(prediction_path.resolve())
    feature_sql = str(feature_table.resolve())
    split_sql = str(split_table.resolve())
    connection = duckdb.connect()
    try:
        columns = {
            str(row[0])
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [prediction_sql]
            ).fetchall()
        }
        required = {"transaction_id", "source_row_number", "is_laundering"}
        if not required.issubset(columns):
            raise RefinementVerificationError(
                f"Validation prediction columns are missing: {sorted(required - columns)}"
            )
        aggregate = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT transaction_id),
                count(DISTINCT source_row_number),
                sum(is_laundering),
                min(is_laundering),
                max(is_laundering)
            FROM read_parquet(?)
            """,
            [prediction_sql],
        ).fetchone()
        expected_aggregate = (
            expected_rows,
            expected_rows,
            expected_rows,
            expected_positive_labels,
            0,
            1,
        )
        if tuple(int(value) for value in aggregate) != expected_aggregate:
            raise RefinementVerificationError(
                "Validation prediction identity/label counts differ from frozen metadata"
            )
        membership = dict(
            connection.execute(
                """
                SELECT split.partition, count(*)
                FROM read_parquet(?) prediction
                INNER JOIN read_parquet(?) split USING (transaction_id)
                GROUP BY split.partition
                ORDER BY split.partition
                """,
                [prediction_sql, split_sql],
            ).fetchall()
        )
        if membership != {"validation": expected_rows}:
            raise RefinementVerificationError(
                f"Prediction identities are not exactly outer validation: {membership}"
            )
        frozen_rows = connection.execute(
            """
            WITH validation AS (
                SELECT transaction_id
                FROM read_parquet(?)
                WHERE partition = 'validation'
            )
            SELECT
                count(*),
                count(*) FILTER (
                    WHERE prediction.source_row_number
                              IS DISTINCT FROM feature.source_row_number
                       OR prediction.is_laundering
                              IS DISTINCT FROM feature.is_laundering
                )
            FROM validation
            INNER JOIN read_parquet(?) feature USING (transaction_id)
            INNER JOIN read_parquet(?) prediction USING (transaction_id)
            """,
            [split_sql, feature_sql, prediction_sql],
        ).fetchone()
        if tuple(int(value) for value in frozen_rows) != (expected_rows, 0):
            raise RefinementVerificationError(
                "Prediction identity/labels differ from frozen validation features"
            )

        base = connection.execute(
            """
            SELECT source_row_number, is_laundering
            FROM read_parquet(?) ORDER BY source_row_number
            """,
            [prediction_sql],
        ).fetchnumpy()
        source_rows = np.asarray(base["source_row_number"], dtype=np.int64)
        labels = np.asarray(base["is_laundering"], dtype=np.int8)
        recomputed: dict[str, Any] = {}
        all_recomputed: dict[str, Any] = {}
        for score_key, (artifact_key, feature_family) in _PREDICTION_SCORE_ARTIFACTS.items():
            column_payload = score_map.get(score_key)
            if not isinstance(column_payload, Mapping):
                raise RefinementVerificationError(f"Prediction mapping lacks model {score_key}")
            raw_column = str(column_payload.get("raw_score", ""))
            probability_column = str(column_payload.get("probability", ""))
            if {raw_column, probability_column}.difference(columns):
                raise RefinementVerificationError(
                    f"Prediction score columns are missing for {score_key}"
                )
            values = connection.execute(
                f'SELECT "{raw_column}", "{probability_column}" '
                "FROM read_parquet(?) ORDER BY source_row_number",
                [prediction_sql],
            ).fetchnumpy()
            raw = np.asarray(values[raw_column], dtype=np.float64)
            probability = np.asarray(values[probability_column], dtype=np.float64)
            if not np.isfinite(raw).all() or not np.isfinite(probability).all():
                raise RefinementVerificationError(f"Non-finite saved scores for {score_key}")
            if np.any((probability < 0.0) | (probability > 1.0)):
                raise RefinementVerificationError(
                    f"Saved probabilities are invalid for {score_key}"
                )

            model_dir = run_dir / "models" / artifact_key
            thresholds = _load_object(model_dir / "thresholds.json")
            validate_threshold_artifact(
                thresholds,
                fpr_ceiling=fpr_ceiling,
                alert_budget=alert_budget,
            )
            recompute_threshold_artifact(
                thresholds,
                labels,
                raw,
                fpr_ceiling=fpr_ceiling,
                alert_budget=alert_budget,
            )
            metrics = evaluate_binary_predictions(
                labels,
                raw,
                source_rows,
                threshold=_threshold_for_metrics(thresholds, raw),
                top_k_values=top_k_values,
            )
            saved_metrics = _load_object(model_dir / "validation_metrics.json")
            if metrics != saved_metrics:
                raise RefinementVerificationError(
                    f"Saved validation metrics differ from score recomputation for {score_key}"
                )
            saturation = analyze_validation_score_saturation(
                labels,
                probabilities=probability,
                raw_scores=raw,
                source_row_number=source_rows,
                top_k_values=top_k_values,
                partition="validation",
            )
            saturation["raw_average_precision"] = compute_average_precision(labels, raw)
            saturation["probability_average_precision"] = compute_average_precision(
                labels, probability
            )
            saved_saturation = _load_object(model_dir / "saturation.json")
            if saturation != saved_saturation:
                raise RefinementVerificationError(
                    f"Saved saturation evidence differs from score recomputation for {score_key}"
                )
            metadata = _load_object(model_dir / "metadata.json")
            if metadata.get("ranking_metrics") != metrics:
                raise RefinementVerificationError(
                    f"Model metadata metrics differ from recomputation for {score_key}"
                )
            if metadata.get("threshold_optimization") != thresholds:
                raise RefinementVerificationError(
                    f"Model metadata thresholds differ for {score_key}"
                )
            if metadata.get("saturation") != saturation:
                raise RefinementVerificationError(
                    f"Model metadata saturation differs for {score_key}"
                )
            if metadata.get("feature_family") != feature_family:
                raise RefinementVerificationError(
                    f"Model metadata feature family differs for {score_key}"
                )
            if metadata.get("evaluation_partition") != "validation":
                raise RefinementVerificationError(f"Model {score_key} is not validation-only")
            _require_bool(metadata, "test_predictions_generated", False)
            if metadata.get("test_metrics") is not None:
                raise RefinementVerificationError(f"Model {score_key} contains test metrics")
            model_metrics = {
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
            }
            all_recomputed[score_key] = {
                "artifact_key": artifact_key,
                "feature_family": feature_family,
                **model_metrics,
            }
            if score_key.startswith("refined_"):
                recomputed[score_key.removeprefix("refined_")] = model_metrics
    finally:
        connection.close()
    return {
        "status": "PASS",
        "rows": expected_rows,
        "positive_labels": expected_positive_labels,
        "frozen_split_membership": {"validation": expected_rows},
        "models_recomputed": recomputed,
        "all_score_artifacts_recomputed": all_recomputed,
        "frozen_validation_identity_and_labels_verified": True,
        "saturation_artifacts_recomputed": len(all_recomputed),
        "test_rows_present": False,
    }


def verify_validation_champion(
    payload: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reselect the refined transaction champion using validation AP only."""

    results = {
        name: {"ranking_metrics": {"average_precision": values["average_precision"]}}
        for name, values in metrics.items()
    }
    recomputed = select_outer_validation_champion(results, partition="validation")
    if payload != recomputed:
        raise RefinementVerificationError(
            "Saved refined champion differs from validation-only recomputation"
        )
    return recomputed


def _verify_saved_inventory(run_dir: Path, manifest: Mapping[str, Any]) -> int:
    inventory = manifest.get("artifacts")
    if not isinstance(inventory, list):
        raise RefinementVerificationError("Run manifest has no artifact inventory")
    if int(manifest.get("artifact_count_excluding_manifest", -1)) != len(inventory):
        raise RefinementVerificationError("Run manifest artifact count is inconsistent")
    for item in inventory:
        if not isinstance(item, Mapping):
            raise RefinementVerificationError("Run manifest inventory entry is invalid")
        path = run_dir / str(item["path"])
        if not path.is_file():
            raise RefinementVerificationError(f"Manifest artifact is missing: {path}")
        if path.stat().st_size != int(item["size_bytes"]) or sha256_file(path) != item["sha256"]:
            raise RefinementVerificationError(f"Manifest fingerprint differs: {path}")
    current = {
        item["path"]
        for item in build_artifact_inventory(run_dir, exclude_paths=[run_dir / "run_manifest.json"])
    }
    saved = {str(item["path"]) for item in inventory}
    extra = current - saved
    if extra - {"verification_report.json", "quality_report.json"}:
        raise RefinementVerificationError(f"Unmanifested artifacts found: {sorted(extra)}")
    if saved - current:
        raise RefinementVerificationError(f"Manifest entries are absent: {sorted(saved - current)}")
    return len(inventory)


def _verify_frozen_provenance(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict:
    frozen = manifest.get("provenance", {}).get("frozen_inputs", {})
    expected_paths = {
        "transaction_features.parquet": get_path(config, "feature_table"),
        "split_manifest.parquet": get_path(config, "split_table"),
        "split_metadata.json": get_path(config, "split_metadata"),
    }
    if not isinstance(frozen, Mapping):
        raise RefinementVerificationError("Frozen-input provenance is missing")
    verified: dict[str, str] = {}
    for name, path in expected_paths.items():
        item = frozen.get(name)
        if not isinstance(item, Mapping):
            raise RefinementVerificationError(f"Frozen provenance is missing {name}")
        digest = sha256_file(path)
        if digest != item.get("sha256") or path.stat().st_size != int(item.get("size_bytes", -1)):
            raise RefinementVerificationError(f"Frozen input fingerprint differs for {name}")
        configured_digest = config["baseline"]["frozen_upstream"].get(
            {
                "transaction_features.parquet": "feature_table_sha256",
                "split_manifest.parquet": "split_table_sha256",
                "split_metadata.json": "split_metadata_sha256",
            }[name]
        )
        if configured_digest != digest:
            raise RefinementVerificationError(f"Configured frozen digest differs for {name}")
        verified[name] = digest
    return verified


def verify_input_provenance(
    run_dir: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile the explicit Sprint 1/2 input chain with current immutable files."""

    saved = _load_object(run_dir / "input_provenance.json")
    expected_files = {
        "feature_table": get_path(config, "feature_table"),
        "split_table": get_path(config, "split_table"),
        "split_metadata": get_path(config, "split_metadata"),
        "sprint2_manifest": get_path(config, "sprint2_run_dir") / "run_manifest.json",
    }
    verified: dict[str, Any] = {}
    for name, path in expected_files.items():
        actual = file_fingerprint(path)
        if saved.get(name) != actual:
            raise RefinementVerificationError(f"Explicit input provenance differs for {name}")
        verified[name] = actual
    sprint2_manifest = _load_object(expected_files["sprint2_manifest"])
    if sprint2_manifest.get("status") != "PASS":
        raise RefinementVerificationError("Provenance references a non-PASS Sprint 2 run")
    checkpoint = saved.get("sprint2_checkpoint")
    if not isinstance(checkpoint, str) or re.fullmatch(r"[0-9a-f]{40}", checkpoint) is None:
        raise RefinementVerificationError("Sprint 2 checkpoint is not a full Git SHA")
    if manifest.get("sprint2_checkpoint") != checkpoint:
        raise RefinementVerificationError("Manifest Sprint 2 checkpoint differs from provenance")
    if manifest.get("sprint2_manifest") != verified["sprint2_manifest"]:
        raise RefinementVerificationError("Manifest Sprint 2 fingerprint differs from provenance")
    return {
        "status": "PASS",
        "sprint2_checkpoint": checkpoint,
        "inputs_verified": verified,
        "sprint2_status": "PASS",
    }


def _preprocessor_manifests(run_dir: Path, expected_train_rows: int) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    base = FEATURE_FAMILY_CONTRACTS[TRANSACTION_TEMPORAL_HISTORY_FAMILY]
    duplicate_graph_features = set(GRAPH_HISTORY_FEATURES).difference(NOVEL_GRAPH_FEATURES)
    expected_contracts = {
        **{family: FEATURE_FAMILY_CONTRACTS[family] for family in PRIMARY_FEATURE_FAMILIES},
        NOVEL_GRAPH_SENSITIVITY_FAMILY: FeatureContract(
            numeric_features=(*base.numeric_features, *NOVEL_GRAPH_FEATURES),
            low_cardinality_categories=base.low_cardinality_categories,
            bank_frequency_categories=base.bank_frequency_categories,
            missing_indicator_features=base.missing_indicator_features,
            forbidden_predictors=(
                *ALWAYS_FORBIDDEN_PREDICTORS,
                *sorted(duplicate_graph_features),
            ),
        ),
    }
    for family in ALL_ABLATION_FEATURE_FAMILIES:
        directory = run_dir / "preprocessing" / "outer_train" / family
        manifest = _load_object(directory / "manifest.json")
        state = FittedPreprocessor.load_state(directory / "fitted_state.json")
        if state.state_sha256 != manifest.get("state_sha256"):
            raise RefinementVerificationError(f"Fitted-state digest differs for {family}")
        if state.contract.to_dict() != dict(_contract(manifest)):
            raise RefinementVerificationError(f"Fitted-state contract differs for {family}")
        if state.contract != expected_contracts[family]:
            raise RefinementVerificationError(f"Reviewed feature contract differs for {family}")
        if state.fitted_train_rows != expected_train_rows:
            raise RefinementVerificationError(f"{family} was not fitted on full outer train")
        manifests[family] = manifest
    evidence = validate_feature_family_manifests(manifests)
    sensitivity = manifests[NOVEL_GRAPH_SENSITIVITY_FAMILY]
    sensitivity_names = sensitivity.get("transformed_feature_names")
    sensitivity_count = sensitivity.get("transformed_feature_count")
    if (
        sensitivity.get("fit_scope") != "train_only"
        or not isinstance(sensitivity_names, list)
        or int(sensitivity_count or -1) != len(sensitivity_names)
        or len(set(sensitivity_names)) != len(sensitivity_names)
    ):
        raise RefinementVerificationError(
            "Novel-three sensitivity preprocessing names/count/scope are inconsistent"
        )
    evidence["sensitivity_family"] = NOVEL_GRAPH_SENSITIVITY_FAMILY
    evidence["sensitivity_transformed_feature_count"] = len(sensitivity_names)
    evidence["sensitivity_contract_verified"] = True
    return evidence


def _configured_candidates(config: Mapping[str, Any]) -> dict[str, Sequence[Mapping[str, Any]]]:
    tuning = config["sprint3"]["tuning"]
    return {
        family: tuning[family] for family in ("logistic_regression", "random_forest", "lightgbm")
    }


def _selected_payload(path: Path) -> dict[str, Any]:
    return _load_object(path)


def _validate_acceptance(payload: Mapping[str, Any]) -> int:
    if not payload:
        raise RefinementVerificationError("Acceptance checklist is empty")
    failures = [key for key, value in payload.items() if value is not True]
    if failures:
        raise RefinementVerificationError(f"Acceptance checklist is not PASS: {sorted(failures)}")
    return len(payload)


def verify_sprint3_run(config_path: str | Path = "configs/refinement.yaml") -> dict[str, Any]:
    """Verify the complete saved Sprint 3 run without opening final-test features."""

    config = load_config(config_path)
    run_dir = get_path(config, "run_dir")
    manifest = _load_object(run_dir / "run_manifest.json")
    if "Sprint 3" not in str(manifest.get("sprint")):
        raise RefinementVerificationError("Run manifest is not a Sprint 3 manifest")
    if manifest.get("status") not in {"PASS", "FAIL_QUALITY_CHECKS"}:
        raise RefinementVerificationError("Sprint 3 run has no verifiable completed status")
    sprint3_config = config["sprint3"]
    if sprint3_config.get("full_data") is not True or sprint3_config.get("sampled") is not False:
        raise RefinementVerificationError("Sprint 3 config is not an unsampled full-data run")

    inventory_count = _verify_saved_inventory(run_dir, manifest)
    input_provenance = verify_input_provenance(run_dir, config, manifest)
    frozen = _verify_frozen_provenance(config, manifest)
    policy = _load_object(run_dir / "final_test_policy.json")
    artifact_paths = [str(item["path"]) for item in manifest["artifacts"]]
    test_evidence = validate_closed_test_policy(
        policy,
        manifest.get("data_access_audit", {}),
        artifact_paths,
    )

    folds_payload = _load_object(run_dir / "temporal_cv" / "folds.json")
    prevalence = _load_object(run_dir / "prevalence.json")
    split_prevalence = prevalence.get("split_prevalence", prevalence)
    if not isinstance(split_prevalence, Mapping) or not isinstance(
        split_prevalence.get("train"), Mapping
    ):
        raise RefinementVerificationError("Frozen train prevalence is missing")
    fold_structure = validate_temporal_folds(
        folds_payload,
        outer_train=split_prevalence["train"],
    )
    fold_recount = verify_folds_against_frozen_inputs(
        folds_payload,
        feature_table=get_path(config, "feature_table"),
        split_table=get_path(config, "split_table"),
        fold_manifest_dir=run_dir / "temporal_cv" / "splits",
    )

    trials = pd.read_csv(run_dir / "temporal_cv" / "trials.csv")
    selected_candidates = _selected_payload(run_dir / "temporal_cv" / "selected_candidates.json")
    tuning = validate_tuning_artifacts(
        trials,
        selected_candidates,
        expected_folds=len(_fold_rows(folds_payload)),
        configured_candidates=_configured_candidates(config),
    )
    features = _preprocessor_manifests(
        run_dir,
        int(split_prevalence["train"]["rows"]),
    )

    ablation = pd.read_csv(run_dir / "ablation" / "feature_family_ablation.csv")
    comparability = validate_ablation_comparability(
        ablation,
        expected_split_sha256=frozen["split_manifest.parquet"],
        expected_random_seed=int(config["project"]["random_seed"]),
    )
    graph_conclusion = _load_object(run_dir / "ablation" / "graph_value_conclusion.json")
    graph_value = validate_graph_value_conclusion(graph_conclusion, ablation)
    ablation_metadata = validate_ablation_model_metadata(
        {
            TRANSACTION_ONLY_FAMILY: _load_object(
                run_dir / "models" / "ablation_transaction_only" / "metadata.json"
            ),
            TRANSACTION_TEMPORAL_HISTORY_FAMILY: _load_object(
                run_dir / "models" / "lightgbm" / "metadata.json"
            ),
            TRANSACTION_TEMPORAL_HISTORY_GRAPH_FAMILY: _load_object(
                run_dir / "models" / "ablation_transaction_temporal_history_graph" / "metadata.json"
            ),
            NOVEL_GRAPH_SENSITIVITY_FAMILY: _load_object(
                run_dir
                / "models"
                / "ablation_transaction_temporal_history_graph_novel3"
                / "metadata.json"
            ),
        },
        ablation,
    )
    duplicate_payload = graph_conclusion.get("duplicate_graph_feature_evidence")
    if not isinstance(duplicate_payload, Mapping):
        raise RefinementVerificationError("Graph conclusion lacks duplicate-feature evidence")
    duplicate_evidence = verify_duplicate_graph_evidence(
        duplicate_payload,
        feature_table=get_path(config, "feature_table"),
        split_table=get_path(config, "split_table"),
        expected_rows=(
            int(split_prevalence["train"]["rows"]) + int(split_prevalence["validation"]["rows"])
        ),
    )

    threshold_config = sprint3_config["threshold_optimization"]
    threshold_analysis_payload = _load_object(run_dir / "threshold" / "analysis.json")
    threshold_wrapper = validate_threshold_analysis(
        threshold_analysis_payload,
        fpr_ceiling=float(threshold_config["fpr_ceiling"]),
        alert_budget=int(threshold_config["alert_budget"]),
    )
    threshold_results: dict[str, Any] = {}
    model_root = run_dir / "models"
    for model_directory in sorted(path for path in model_root.iterdir() if path.is_dir()):
        threshold_path = model_directory / "thresholds.json"
        if threshold_path.is_file():
            threshold_results[model_directory.name] = validate_threshold_artifact(
                _load_object(threshold_path),
                fpr_ceiling=float(threshold_config["fpr_ceiling"]),
                alert_budget=int(threshold_config["alert_budget"]),
            )
    if not threshold_results:
        raise RefinementVerificationError("No model threshold artifacts were found")
    expected_threshold_artifacts = {
        artifact_key for artifact_key, _ in _PREDICTION_SCORE_ARTIFACTS.values()
    }
    if set(threshold_results) != expected_threshold_artifacts:
        raise RefinementVerificationError(
            "Threshold artifact directories differ from saved prediction model set"
        )
    wrapped_models = threshold_analysis_payload.get("models")
    if not isinstance(wrapped_models, Mapping) or set(wrapped_models) != (
        expected_threshold_artifacts
    ):
        raise RefinementVerificationError("Threshold wrapper model set differs")
    for model_name, wrapped in wrapped_models.items():
        if wrapped != _load_object(model_root / model_name / "thresholds.json"):
            raise RefinementVerificationError(
                f"Threshold wrapper differs from model artifact for {model_name}"
            )
    expected_feature_families = {
        artifact_key: feature_family
        for artifact_key, feature_family in _PREDICTION_SCORE_ARTIFACTS.values()
    }
    if threshold_analysis_payload.get("model_feature_families") != expected_feature_families:
        raise RefinementVerificationError("Threshold wrapper feature-family mapping differs")

    validation_rows = int(split_prevalence["validation"]["rows"])
    validation_positives = int(split_prevalence["validation"]["positive_labels"])
    prediction_evidence = verify_validation_predictions(
        run_dir / "validation_predictions.parquet",
        _load_object(run_dir / "validation_predictions_manifest.json"),
        feature_table=get_path(config, "feature_table"),
        split_table=get_path(config, "split_table"),
        expected_rows=validation_rows,
        expected_positive_labels=validation_positives,
        top_k_values=[int(value) for value in sprint3_config["top_k"]],
        run_dir=run_dir,
        fpr_ceiling=float(threshold_config["fpr_ceiling"]),
        alert_budget=int(threshold_config["alert_budget"]),
    )
    ablation_metrics = verify_ablation_metrics_from_predictions(
        ablation,
        prediction_evidence["all_score_artifacts_recomputed"],
    )
    core_models = verify_core_model_artifacts(run_dir, selected_candidates)
    champion_payload = _load_object(run_dir / "refined_transaction_champion.json")
    champion = verify_validation_champion(
        champion_payload,
        prediction_evidence["models_recomputed"],
    )
    if manifest.get("champion") != champion:
        raise RefinementVerificationError("Run-manifest champion differs from verified artifact")

    acceptance = _load_object(run_dir / "acceptance_checklist.json")
    acceptance_count = _validate_acceptance(acceptance)
    if not isinstance(manifest.get("acceptance"), Mapping):
        raise RefinementVerificationError("Run manifest acceptance evidence is missing")
    _validate_acceptance(manifest["acceptance"])

    report = {
        "status": "PASS",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "method": (
            "SHA-256 inventory and frozen-input verification; train-only temporal-fold "
            "recount; inner-CV candidate reselection; fitted-state A/B/C contract checks; "
            "same-model graph ablation reconciliation; validation-only whole-tie threshold "
            "checks; explicit closed final-test policy"
        ),
        "saved_inventory_entries_verified": inventory_count,
        "input_provenance": input_provenance,
        "frozen_inputs_verified": frozen,
        "final_test_policy": test_evidence,
        "temporal_fold_structure": fold_structure,
        "temporal_fold_recount": fold_recount,
        "tuning": tuning,
        "feature_families": features,
        "ablation_comparability": comparability,
        "ablation_model_metadata": ablation_metadata,
        "ablation_metrics_recomputed": ablation_metrics,
        "graph_value": graph_value,
        "duplicate_graph_evidence": duplicate_evidence,
        "validation_predictions": prediction_evidence,
        "core_models": core_models,
        "champion_recomputed": champion,
        "threshold_analysis": threshold_wrapper,
        "threshold_models_verified": threshold_results,
        "acceptance_items_verified": acceptance_count,
        "final_test_opened_by_verifier": False,
    }
    atomic_write_json(report, run_dir / "verification_report.json")
    manifest["post_run_verification"] = report
    refreshed = write_run_manifest(manifest, run_dir)
    return {
        **report,
        "refreshed_artifact_count_excluding_manifest": refreshed[
            "artifact_count_excluding_manifest"
        ],
    }


__all__ = [
    "ALL_ABLATION_FEATURE_FAMILIES",
    "PRIMARY_FEATURE_FAMILIES",
    "RefinementVerificationError",
    "recompute_threshold_artifact",
    "validate_ablation_comparability",
    "validate_ablation_model_metadata",
    "validate_closed_test_policy",
    "validate_feature_family_manifests",
    "validate_graph_value_conclusion",
    "validate_temporal_folds",
    "validate_threshold_artifact",
    "validate_threshold_analysis",
    "validate_tuning_artifacts",
    "verify_ablation_metrics_from_predictions",
    "verify_core_model_artifacts",
    "verify_duplicate_graph_evidence",
    "verify_folds_against_frozen_inputs",
    "verify_input_provenance",
    "verify_validation_champion",
    "verify_validation_predictions",
    "verify_sprint3_run",
]
