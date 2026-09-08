"""Load and validate saved Sprint 4 and Sprint 5 product artifacts.

The application deliberately has no model-training dependency. It accepts either a
single Sprint 4 ``dashboard_bundle.json`` or the documented collection of product files.
Sprint 5 final-test evidence is loaded into a separate object so it cannot influence the
validation-leader or investigation-queue logic.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactLoadError(RuntimeError):
    """Raised when saved artifacts are absent or violate the dashboard contract."""


@dataclass(frozen=True)
class DashboardArtifacts:
    """Validated, presentation-ready validation and optional final-test artifacts."""

    root: Path
    summary: dict[str, Any]
    queue: pd.DataFrame
    cases: dict[str, dict[str, Any]]
    model_comparison: pd.DataFrame
    pr_curves: pd.DataFrame
    top_k: pd.DataFrame
    ablation: pd.DataFrame
    provenance: dict[str, str]
    final_evaluation: FinalEvaluationArtifacts | None = None


@dataclass(frozen=True)
class FinalEvaluationArtifacts:
    """Strictly validated, presentation-only Sprint 5 final-test artifacts."""

    summary: dict[str, Any]
    model_comparison: pd.DataFrame
    pr_curves: pd.DataFrame
    top_k: pd.DataFrame
    frozen_champion: str
    provenance: dict[str, str]


_BUNDLE_CANDIDATES = ("dashboard_bundle.json", "product/dashboard_bundle.json")
_QUEUE_CANDIDATES = (
    "product/investigation_queue.csv",
    "product/investigation_queue.json",
    "cases/investigation_queue.csv",
    "cases/case_queue.csv",
    "investigation_queue.csv",
)
_CASE_CANDIDATES = ("product/cases.json", "cases/cases.json", "cases.json")
_COMPARISON_CANDIDATES = (
    "product/model_comparison.csv",
    "comparison/model_comparison.csv",
    "model_comparison.csv",
    "graphsage_model_comparison.csv",
)
_PR_CURVE_CANDIDATES = (
    "product/pr_curves.csv",
    "comparison/pr_curves.csv",
    "pr_curves.csv",
)
_TOP_K_CANDIDATES = (
    "product/top_k_metrics.csv",
    "comparison/top_k_metrics.csv",
    "top_k_metrics.csv",
)
_ABLATION_CANDIDATES = (
    "product/feature_family_ablation.csv",
    "comparison/feature_family_ablation.csv",
    "feature_family_ablation.csv",
)
_SUMMARY_CANDIDATES = ("product/dashboard_summary.json", "dashboard_summary.json")
_VALIDATION_REFERENCE_CANDIDATES = (
    "product/validation_artifact_reference.json",
    "product/sprint4_artifact_reference.json",
    "validation_artifact_reference.json",
    "sprint4_artifact_reference.json",
)
_FINAL_PATHS = {
    "model_comparison": "product/final_model_comparison.csv",
    "summary": "product/final_test_summary.json",
    "pr_curves": "product/final_pr_curves.csv",
    "top_k": "product/final_top_k_metrics.csv",
}
_FROZEN_FINAL_CHAMPION = "graph_enhanced_lightgbm"


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError(f"Could not read JSON artifact {path}: {exc}") from exc


def _first_existing(root: Path, candidates: tuple[str, ...]) -> Path | None:
    return next(
        (root / candidate for candidate in candidates if (root / candidate).is_file()), None
    )


def _missing_message(kind: str, root: Path, candidates: tuple[str, ...]) -> str:
    expected = ", ".join(str(root / value) for value in candidates)
    return f"Missing required {kind} artifact. Expected one of: {expected}"


def _frame_from_value(value: Any, *, label: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        if "records" in value and isinstance(value["records"], list):
            return pd.DataFrame(value["records"])
        return pd.DataFrame([value])
    raise ArtifactLoadError(f"{label} must be a JSON list/object, not {type(value).__name__}.")


def _read_frame(path: Path, *, label: str) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return _frame_from_value(_read_json(path), label=label)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ArtifactLoadError(f"Could not read {label} artifact {path}: {exc}") from exc


def _rename_first(frame: pd.DataFrame, canonical: str, aliases: tuple[str, ...]) -> None:
    if canonical in frame.columns:
        return
    found = next((alias for alias in aliases if alias in frame.columns), None)
    if found is not None:
        frame.rename(columns={found: canonical}, inplace=True)


def _normalise_queue(frame: pd.DataFrame) -> pd.DataFrame:
    queue = frame.copy()
    aliases = {
        "case_id": ("id",),
        "priority": ("priority_band", "band"),
        "risk_score": ("priority_score", "score", "model_score"),
        "account_count": ("accounts", "n_accounts"),
        "transaction_count": ("transactions", "n_transactions"),
        "total_flow": ("total_amount", "flow_amount"),
        "major_pattern": ("pattern", "primary_pattern"),
        "evidence_count": ("observed_evidence_count", "n_evidence"),
    }
    for canonical, options in aliases.items():
        _rename_first(queue, canonical, options)
    if "case_id" not in queue:
        raise ArtifactLoadError("Investigation queue must contain a non-empty 'case_id' column.")
    queue["case_id"] = queue["case_id"].astype("string")
    if queue["case_id"].isna().any() or (queue["case_id"].str.strip() == "").any():
        raise ArtifactLoadError("Investigation queue contains an empty case_id.")
    if queue["case_id"].duplicated().any():
        raise ArtifactLoadError("Investigation queue case_id values must be unique.")
    if "risk_score" in queue:
        queue["risk_score"] = pd.to_numeric(queue["risk_score"], errors="coerce")
        if queue["risk_score"].isna().any():
            raise ArtifactLoadError("Investigation queue risk_score values must be numeric.")
    return queue


def _extract_cases(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        raise ArtifactLoadError("Cases artifact must be a JSON list or object.")
    if isinstance(value.get("cases"), list):
        return value["cases"]
    if "case_id" in value:
        return [value]
    if value and all(isinstance(item, dict) for item in value.values()):
        return [dict(item, case_id=item.get("case_id", key)) for key, item in value.items()]
    raise ArtifactLoadError("Cases artifact does not contain a 'cases' list or case objects.")


def _load_case_directory(root: Path) -> tuple[list[dict[str, Any]], Path] | None:
    case_dir = root / "cases"
    if not case_dir.is_dir():
        return None
    paths = sorted(
        path
        for path in case_dir.glob("*.json")
        if path.name not in {"investigation_queue.json", "cases.json"}
    )
    if not paths:
        return None
    cases: list[dict[str, Any]] = []
    for path in paths:
        cases.extend(_extract_cases(_read_json(path)))
    return cases, case_dir


def _normalise_cases(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, original in enumerate(records):
        if not isinstance(original, dict):
            raise ArtifactLoadError(f"Case record {index} must be a JSON object.")
        case = dict(original)
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            raise ArtifactLoadError(f"Case record {index} has no case_id.")
        if case_id in result:
            raise ArtifactLoadError(f"Duplicate case_id in cases artifact: {case_id}")

        observed = case.get("observed_evidence")
        if observed is None:
            combined = case.get("evidence", [])
            observed = [
                item
                for item in combined
                if isinstance(item, dict) and item.get("basis", "observed") == "observed"
            ]
        if not isinstance(observed, list):
            raise ArtifactLoadError(f"Case {case_id} observed_evidence must be a list.")
        for evidence_index, item in enumerate(observed):
            if not isinstance(item, dict) or not str(item.get("statement", "")).strip():
                raise ArtifactLoadError(
                    f"Case {case_id} observed evidence {evidence_index} needs a concrete statement."
                )
        case["observed_evidence"] = observed
        model_evidence = case.get("model_evidence", {})
        if model_evidence is None:
            model_evidence = {}
        if not isinstance(model_evidence, dict):
            raise ArtifactLoadError(f"Case {case_id} model_evidence must be an object.")
        case["model_evidence"] = model_evidence
        result[case_id] = case
    if not result:
        raise ArtifactLoadError("Cases artifact contains no cases.")
    return result


def _normalise_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    comparison = frame.copy()
    aliases = {
        "model": ("model_name", "name"),
        "version": ("stage", "model_version"),
        "pr_auc": ("average_precision", "validation_pr_auc"),
        "roc_auc": ("validation_roc_auc",),
        "recall_at_k": ("recall_at_1000", "recall_at_budget"),
        "precision_at_k": ("precision_at_1000", "precision_at_budget"),
        "fpr": ("false_positive_rate",),
        "alerts": ("alert_count", "alert_volume"),
    }
    for canonical, options in aliases.items():
        _rename_first(comparison, canonical, options)
    missing = sorted({"model", "pr_auc"} - set(comparison.columns))
    if missing:
        raise ArtifactLoadError(f"Model comparison is missing required columns: {missing}")
    comparison["model"] = comparison["model"].astype("string")
    for column in ("pr_auc", "roc_auc", "recall_at_k", "precision_at_k", "fpr", "alerts"):
        if column in comparison:
            comparison[column] = pd.to_numeric(comparison[column], errors="coerce")
    if comparison["pr_auc"].isna().any():
        raise ArtifactLoadError("Model comparison pr_auc values must be numeric.")
    if "evaluation_partition" in comparison:
        partitions = set(comparison["evaluation_partition"].dropna().astype(str).str.lower())
        if "test" in partitions:
            raise ArtifactLoadError(
                "Model comparison contains test metrics; Sprint 4 dashboard must remain "
                "validation-only."
            )
    if "test_metrics_used" in comparison:
        opened = comparison["test_metrics_used"].map(
            lambda value: value is True or str(value).strip().lower() == "true"
        )
        if opened.any():
            raise ArtifactLoadError(
                "Model comparison reports test_metrics_used=true; final test must remain unopened."
            )
    return comparison


def _normalise_pr_curves(frame: pd.DataFrame) -> pd.DataFrame:
    curves = frame.copy()
    _rename_first(curves, "model", ("model_name", "name"))
    _rename_first(curves, "precision", ("pr_precision",))
    _rename_first(curves, "recall", ("pr_recall",))
    if curves.empty:
        return curves
    missing = sorted({"model", "precision", "recall"} - set(curves.columns))
    if missing:
        raise ArtifactLoadError(f"PR-curve artifact is missing required columns: {missing}")
    return curves


def _normalise_top_k(frame: pd.DataFrame) -> pd.DataFrame:
    top_k = frame.copy()
    _rename_first(top_k, "model", ("model_name", "name"))
    _rename_first(top_k, "k", ("requested_k", "effective_k"))
    _rename_first(top_k, "recall_at_k", ("recall",))
    _rename_first(top_k, "precision_at_k", ("precision",))
    if top_k.empty:
        return top_k
    missing = sorted({"model", "k", "recall_at_k", "precision_at_k"} - set(top_k.columns))
    if missing:
        raise ArtifactLoadError(f"Top-K artifact is missing required columns: {missing}")
    return top_k


def _normalise_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    ablation = frame.copy()
    _rename_first(ablation, "feature_family", ("family", "name"))
    _rename_first(ablation, "pr_auc", ("average_precision",))
    if ablation.empty:
        return ablation
    missing = sorted({"feature_family", "pr_auc"} - set(ablation.columns))
    if missing:
        raise ArtifactLoadError(f"Ablation artifact is missing required columns: {missing}")
    return ablation


def _canonical_model_name(value: Any) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _strict_boolean_series(series: pd.Series, *, label: str) -> pd.Series:
    tokens = series.map(lambda value: str(value).strip().lower())
    invalid = ~tokens.isin({"true", "false"})
    if invalid.any():
        raise ArtifactLoadError(f"{label} values must be explicit booleans.")
    return tokens.eq("true")


def _summary_lookup(summary: dict[str, Any], *names: str) -> Any:
    containers = [summary]
    containers.extend(
        value
        for key in ("protocol", "final_test", "test", "freeze_contract")
        if isinstance((value := summary.get(key)), dict)
    )
    for container in containers:
        for name in names:
            if name in container:
                return container[name]
    return None


def _require_summary_flag(
    summary: dict[str, Any], *, names: tuple[str, ...], expected: bool
) -> None:
    value = _summary_lookup(summary, *names)
    label = names[0]
    if value is None:
        raise ArtifactLoadError(f"Final-test summary is missing required contract flag '{label}'.")
    if not isinstance(value, bool) or value is not expected:
        raise ArtifactLoadError(f"Final-test contract flag '{label}' must be {expected}.")


def _finite_number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactLoadError(f"{label} must be numeric and finite.") from exc
    if not math.isfinite(number):
        raise ArtifactLoadError(f"{label} must be numeric and finite.")
    return number


def _nonnegative_integer(value: Any, *, label: str, positive: bool = False) -> int:
    number = _finite_number(value, label=label)
    if not number.is_integer() or number < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise ArtifactLoadError(f"{label} must be a {qualifier} integer.")
    return int(number)


def _normalise_final_comparison(frame: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    comparison = frame.copy()
    aliases = {
        "model": ("model_name", "name"),
        "version": ("stage", "model_version"),
        "pr_auc": ("average_precision", "test_pr_auc"),
        "roc_auc": ("test_roc_auc",),
        "precision": ("test_precision",),
        "recall": ("test_recall",),
        "f1": ("test_f1",),
        "fpr": ("false_positive_rate", "test_fpr"),
        "recall_at_k": ("recall_at_1000", "recall_at_budget"),
        "precision_at_k": ("precision_at_1000", "precision_at_budget"),
        "alerts": ("alert_count", "alert_volume"),
        "row_count": ("test_row_count",),
        "positive_count": ("test_positive_count",),
        "evaluation_partition": ("partition",),
        "is_frozen_champion": (
            "is_champion",
            "frozen_champion",
            "champion_frozen_before_test",
        ),
    }
    for canonical, options in aliases.items():
        _rename_first(comparison, canonical, options)
    if "is_frozen_champion" not in comparison and "role" in comparison:
        roles = comparison["role"].astype("string").str.strip().str.lower()
        comparison["is_frozen_champion"] = roles.isin({"frozen_champion", "final_champion"})

    required = {
        "model",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "alerts",
        "row_count",
        "positive_count",
        "evaluation_partition",
        "is_frozen_champion",
    }
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ArtifactLoadError(f"Final model comparison is missing required columns: {missing}")
    if comparison.empty:
        raise ArtifactLoadError("Final model comparison contains no model rows.")

    comparison["model"] = comparison["model"].astype("string").str.strip()
    if comparison["model"].isna().any() or comparison["model"].eq("").any():
        raise ArtifactLoadError("Final model comparison contains an empty model name.")
    canonical_models = comparison["model"].map(_canonical_model_name)
    if canonical_models.duplicated().any():
        duplicates = comparison.loc[canonical_models.duplicated(keep=False), "model"].tolist()
        raise ArtifactLoadError(f"Final model comparison contains duplicate models: {duplicates}")

    partitions = comparison["evaluation_partition"].astype("string").str.strip().str.lower()
    if partitions.isna().any() or set(partitions) != {"test"}:
        raise ArtifactLoadError(
            "Final model comparison evaluation_partition must be exactly 'test'."
        )

    bounded_metrics = (
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "fpr",
    )
    optional_bounded_metrics = tuple(
        column for column in ("recall_at_k", "precision_at_k") if column in comparison
    )
    for column in (*bounded_metrics, *optional_bounded_metrics):
        numeric = pd.to_numeric(comparison[column], errors="coerce")
        if numeric.isna().any() or (~numeric.map(math.isfinite)).any():
            raise ArtifactLoadError(f"Final model comparison {column} values must be finite.")
        if ((numeric < 0.0) | (numeric > 1.0)).any():
            raise ArtifactLoadError(f"Final model comparison {column} values must be in [0, 1].")
        comparison[column] = numeric.astype(float)

    for row_index in comparison.index:
        row_count = _nonnegative_integer(
            comparison.at[row_index, "row_count"],
            label=f"Final model comparison row_count at row {row_index}",
            positive=True,
        )
        positive_count = _nonnegative_integer(
            comparison.at[row_index, "positive_count"],
            label=f"Final model comparison positive_count at row {row_index}",
        )
        alerts = _nonnegative_integer(
            comparison.at[row_index, "alerts"],
            label=f"Final model comparison alerts at row {row_index}",
        )
        if positive_count > row_count or alerts > row_count:
            raise ArtifactLoadError(
                "Final model comparison counts may not exceed the test row_count."
            )
        comparison.at[row_index, "row_count"] = row_count
        comparison.at[row_index, "positive_count"] = positive_count
        comparison.at[row_index, "alerts"] = alerts

        if "negative_count" in comparison:
            negative_count = _nonnegative_integer(
                comparison.at[row_index, "negative_count"],
                label=f"Final model comparison negative_count at row {row_index}",
            )
            if negative_count != row_count - positive_count:
                raise ArtifactLoadError(
                    "Final model comparison negative_count disagrees with row/positive counts."
                )

        confusion_columns = ("true_positive", "false_positive", "false_negative", "true_negative")
        present_confusion = [column for column in confusion_columns if column in comparison]
        if present_confusion and len(present_confusion) != len(confusion_columns):
            raise ArtifactLoadError(
                "Final model comparison must provide all four confusion-matrix counts together."
            )
        if present_confusion:
            counts = {
                column: _nonnegative_integer(
                    comparison.at[row_index, column],
                    label=f"Final model comparison {column} at row {row_index}",
                )
                for column in confusion_columns
            }
            if (
                counts["true_positive"] + counts["false_negative"] != positive_count
                or counts["true_negative"] + counts["false_positive"] != row_count - positive_count
                or counts["true_positive"] + counts["false_positive"] != alerts
            ):
                raise ArtifactLoadError(
                    "Final model comparison confusion counts are internally inconsistent."
                )
            expected_precision = counts["true_positive"] / alerts if alerts else 0.0
            expected_recall = counts["true_positive"] / positive_count if positive_count else 0.0
            negative_count = row_count - positive_count
            expected_fpr = counts["false_positive"] / negative_count if negative_count else 0.0
            expected_f1 = (
                2.0 * expected_precision * expected_recall / (expected_precision + expected_recall)
                if expected_precision + expected_recall > 0.0
                else 0.0
            )
            for metric, expected in (
                ("precision", expected_precision),
                ("recall", expected_recall),
                ("f1", expected_f1),
                ("fpr", expected_fpr),
            ):
                if not math.isclose(
                    float(comparison.at[row_index, metric]),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ArtifactLoadError(
                        f"Final model comparison {metric} disagrees with confusion counts."
                    )

    champion_flags = _strict_boolean_series(
        comparison["is_frozen_champion"], label="Final frozen-champion flag"
    )
    if int(champion_flags.sum()) != 1:
        raise ArtifactLoadError("Final model comparison must mark exactly one frozen champion.")
    frozen_model = _canonical_model_name(comparison.loc[champion_flags, "model"].iloc[0])
    summary_model = _canonical_model_name(
        _summary_lookup(summary, "frozen_champion_model", "frozen_final_model")
    )
    if frozen_model != _FROZEN_FINAL_CHAMPION or summary_model != _FROZEN_FINAL_CHAMPION:
        raise ArtifactLoadError(
            "The pre-frozen final champion must be graph_enhanced_lightgbm in both summary "
            "and comparison artifacts."
        )
    comparison["is_frozen_champion"] = champion_flags
    forbidden_test_uses = (
        "model_selection_used_test",
        "threshold_tuned_on_test",
        "feature_selection_used_test",
        "retrained_after_test",
    )
    missing_flags = sorted(set(forbidden_test_uses) - set(comparison.columns))
    if missing_flags:
        raise ArtifactLoadError(
            f"Final model comparison is missing no-test-tuning flags: {missing_flags}"
        )
    for column in forbidden_test_uses:
        values = _strict_boolean_series(
            comparison[column], label=f"Final model comparison {column}"
        )
        if values.any():
            raise ArtifactLoadError(f"Final model comparison reports forbidden {column}=true.")
    return comparison


def _validate_final_summary(summary: dict[str, Any]) -> dict[str, Any]:
    partition = str(_summary_lookup(summary, "evaluation_partition", "partition") or "").lower()
    if partition != "test":
        raise ArtifactLoadError("Final-test summary evaluation partition must be exactly 'test'.")
    _require_summary_flag(summary, names=("final_test_opened",), expected=True)
    _require_summary_flag(
        summary,
        names=("one_shot_final_evaluation", "one_shot_test_access"),
        expected=True,
    )
    _require_summary_flag(
        summary,
        names=("champion_frozen_before_test", "model_frozen_before_test"),
        expected=True,
    )
    _require_summary_flag(
        summary,
        names=("test_used_for_model_selection", "model_selection_used_test"),
        expected=False,
    )
    _require_summary_flag(
        summary,
        names=("tuning_after_test", "post_test_tuning_performed"),
        expected=False,
    )

    access_count = _nonnegative_integer(
        _summary_lookup(summary, "final_test_access_count", "test_access_count"),
        label="Final-test access count",
    )
    if access_count != 1:
        raise ArtifactLoadError("Final-test access count must be exactly 1.")
    row_count = _nonnegative_integer(
        _summary_lookup(summary, "test_row_count", "row_count"),
        label="Final-test row_count",
        positive=True,
    )
    positive_count = _nonnegative_integer(
        _summary_lookup(summary, "test_positive_count", "positive_count"),
        label="Final-test positive_count",
    )
    if positive_count > row_count:
        raise ArtifactLoadError("Final-test positive_count may not exceed row_count.")
    positive_rate = _finite_number(
        _summary_lookup(summary, "test_positive_rate", "positive_rate"),
        label="Final-test positive_rate",
    )
    validation_rate = _finite_number(
        _summary_lookup(summary, "validation_positive_rate"),
        label="Validation positive_rate",
    )
    if not 0.0 <= positive_rate <= 1.0 or not 0.0 <= validation_rate <= 1.0:
        raise ArtifactLoadError("Validation and final-test positive rates must be in [0, 1].")
    if not math.isclose(positive_rate, positive_count / row_count, rel_tol=0.0, abs_tol=1e-12):
        raise ArtifactLoadError(
            "Final-test positive_rate does not match positive_count / row_count."
        )
    return dict(summary)


def _normalise_partitioned_curves(
    frame: pd.DataFrame,
    *,
    label: str,
    comparison_models: set[str],
) -> pd.DataFrame:
    curves = frame.copy()
    _rename_first(curves, "model", ("model_name", "name"))
    _rename_first(curves, "precision", ("pr_precision",))
    _rename_first(curves, "recall", ("pr_recall",))
    _rename_first(curves, "evaluation_partition", ("partition",))
    missing = sorted({"model", "precision", "recall", "evaluation_partition"} - set(curves.columns))
    if missing:
        raise ArtifactLoadError(f"{label} is missing required columns: {missing}")
    if curves.empty:
        raise ArtifactLoadError(f"{label} contains no rows.")
    partitions = curves["evaluation_partition"].astype("string").str.lower()
    if partitions.isna().any() or set(partitions) != {"test"}:
        raise ArtifactLoadError(f"{label} evaluation_partition must be exactly 'test'.")
    curve_models = curves["model"].map(_canonical_model_name)
    if set(curve_models) != comparison_models:
        raise ArtifactLoadError(f"{label} must contain exactly the final comparison model set.")
    for column in ("precision", "recall"):
        values = pd.to_numeric(curves[column], errors="coerce")
        if values.isna().any() or (~values.map(math.isfinite)).any():
            raise ArtifactLoadError(f"{label} {column} values must be finite.")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ArtifactLoadError(f"{label} {column} values must be in [0, 1].")
        curves[column] = values.astype(float)
    return curves


def _normalise_partitioned_top_k(
    frame: pd.DataFrame,
    *,
    comparison_models: set[str],
    test_row_count: int,
) -> pd.DataFrame:
    top_k = frame.copy()
    _rename_first(top_k, "model", ("model_name", "name"))
    _rename_first(top_k, "k", ("requested_k", "effective_k"))
    _rename_first(top_k, "recall_at_k", ("recall",))
    _rename_first(top_k, "precision_at_k", ("precision",))
    _rename_first(top_k, "evaluation_partition", ("partition",))
    required = {"model", "k", "recall_at_k", "precision_at_k", "evaluation_partition"}
    missing = sorted(required - set(top_k.columns))
    if missing:
        raise ArtifactLoadError(f"Final Top-K artifact is missing required columns: {missing}")
    if top_k.empty:
        raise ArtifactLoadError("Final Top-K artifact contains no rows.")
    partitions = top_k["evaluation_partition"].astype("string").str.lower()
    if partitions.isna().any() or set(partitions) != {"test"}:
        raise ArtifactLoadError("Final Top-K artifact evaluation_partition must be exactly 'test'.")
    top_k_models = top_k["model"].map(_canonical_model_name)
    if set(top_k_models) != comparison_models:
        raise ArtifactLoadError(
            "Final Top-K artifact must contain exactly the final comparison model set."
        )
    for column in ("recall_at_k", "precision_at_k"):
        values = pd.to_numeric(top_k[column], errors="coerce")
        if values.isna().any() or (~values.map(math.isfinite)).any():
            raise ArtifactLoadError(f"Final Top-K {column} values must be finite.")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ArtifactLoadError(f"Final Top-K {column} values must be in [0, 1].")
        top_k[column] = values.astype(float)
    converted_k = []
    for row_index, value in top_k["k"].items():
        k = _nonnegative_integer(value, label=f"Final Top-K k at row {row_index}", positive=True)
        if k > test_row_count:
            raise ArtifactLoadError("Final Top-K k may not exceed the final-test row_count.")
        converted_k.append(k)
    top_k["k"] = converted_k
    duplicate_keys = pd.DataFrame({"model": top_k_models, "k": top_k["k"]})
    if duplicate_keys.duplicated().any():
        raise ArtifactLoadError("Final Top-K artifact contains duplicate model/K rows.")
    k_sets = {
        frozenset(group["k"].astype(int))
        for _, group in top_k.assign(_canonical_model=top_k_models).groupby(
            "_canonical_model", sort=False
        )
    }
    if len(k_sets) != 1:
        raise ArtifactLoadError("Every final model must use the identical Top-K budgets.")
    return top_k


def _attach_top_k_display(comparison: pd.DataFrame, top_k: pd.DataFrame) -> pd.DataFrame:
    result = comparison.copy()
    ranked = top_k.assign(_canonical_model=top_k["model"].map(_canonical_model_name))
    display = (
        ranked.sort_values("k", kind="mergesort").groupby("_canonical_model", sort=False).tail(1)
    )
    for metric in ("recall_at_k", "precision_at_k"):
        lookup = display.set_index("_canonical_model")[metric]
        derived = result["model"].map(_canonical_model_name).map(lookup)
        if metric in result:
            observed = pd.to_numeric(result[metric], errors="coerce")
            if not all(
                math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(observed, derived, strict=True)
            ):
                raise ArtifactLoadError(
                    f"Final comparison {metric} disagrees with maximum-K artifact rows."
                )
        result[metric] = derived.astype(float)
    result["top_k_for_display"] = (
        result["model"].map(_canonical_model_name).map(display.set_index("_canonical_model")["k"])
    )
    return result


def _load_final_evaluation(root: Path) -> FinalEvaluationArtifacts | None:
    paths = {name: root / relative for name, relative in _FINAL_PATHS.items()}
    present = {name for name, path in paths.items() if path.is_file()}
    if not present:
        return None
    missing = sorted(set(paths) - present)
    if missing:
        expected = ", ".join(str(paths[name]) for name in missing)
        raise ArtifactLoadError(
            f"Sprint 5 final-test payload is incomplete; missing required artifacts: {expected}"
        )

    raw_summary = _read_json(paths["summary"])
    if not isinstance(raw_summary, dict):
        raise ArtifactLoadError("Final-test summary must be a JSON object.")
    summary = _validate_final_summary(raw_summary)
    comparison = _normalise_final_comparison(
        _read_frame(paths["model_comparison"], label="final model comparison"), summary
    )
    test_row_count = _nonnegative_integer(
        _summary_lookup(summary, "test_row_count", "row_count"),
        label="Final-test row_count",
        positive=True,
    )
    test_positive_count = _nonnegative_integer(
        _summary_lookup(summary, "test_positive_count", "positive_count"),
        label="Final-test positive_count",
    )
    if not comparison["row_count"].map(int).eq(test_row_count).all():
        raise ArtifactLoadError("Final comparison row_count disagrees with final-test summary.")
    if not comparison["positive_count"].map(int).eq(test_positive_count).all():
        raise ArtifactLoadError("Final comparison positive_count disagrees with summary.")
    comparison_models = set(comparison["model"].map(_canonical_model_name))
    pr_curves = _normalise_partitioned_curves(
        _read_frame(paths["pr_curves"], label="final PR-curve artifact"),
        label="Final PR-curve artifact",
        comparison_models=comparison_models,
    )
    top_k = _normalise_partitioned_top_k(
        _read_frame(paths["top_k"], label="final Top-K artifact"),
        comparison_models=comparison_models,
        test_row_count=test_row_count,
    )
    comparison = _attach_top_k_display(comparison, top_k)
    return FinalEvaluationArtifacts(
        summary=summary,
        model_comparison=comparison,
        pr_curves=pr_curves,
        top_k=top_k,
        frozen_champion=_FROZEN_FINAL_CHAMPION,
        provenance={name: str(path) for name, path in paths.items()},
    )


def _has_validation_payload(root: Path) -> bool:
    return _first_existing(root, _BUNDLE_CANDIDATES) is not None or (
        _first_existing(root, _QUEUE_CANDIDATES) is not None
        and _first_existing(root, _COMPARISON_CANDIDATES) is not None
    )


def _resolve_validation_root(root: Path) -> Path:
    if _has_validation_payload(root):
        return root

    reference_path = _first_existing(root, _VALIDATION_REFERENCE_CANDIDATES)
    candidates: list[Path] = []
    if reference_path is not None:
        reference = _read_json(reference_path)
        if not isinstance(reference, dict):
            raise ArtifactLoadError(
                f"Validation artifact reference must be an object: {reference_path}"
            )
        path_keys = (
            "validation_artifact_root",
            "sprint4_artifact_root",
            "artifact_root",
            "path",
        )
        raw_path = next(
            (
                reference[key]
                for key in path_keys
                if isinstance(reference.get(key), str) and reference[key].strip()
            ),
            None,
        )
        if raw_path is None:
            raise ArtifactLoadError(
                "Validation artifact reference needs validation_artifact_root, "
                "sprint4_artifact_root, artifact_root, or path."
            )
        referenced = Path(raw_path).expanduser()
        if not referenced.is_absolute():
            referenced = root / referenced
        candidates.append(referenced.resolve())

    configured = os.environ.get("ARGUS_SPRINT4_ARTIFACT_DIR")
    if configured:
        candidates.append(Path(configured).expanduser().resolve())
    candidates.extend(
        path.resolve()
        for path in (root / "validation", root / "sprint4", root / "product" / "validation")
    )
    for candidate in candidates:
        if candidate.is_dir() and _has_validation_payload(candidate):
            return candidate

    if not any((root / relative).is_file() for relative in _FINAL_PATHS.values()):
        # Preserve the detailed Sprint 4 missing-artifact diagnostics for a legacy root.
        return root

    expected = ", ".join(str(root / name) for name in _VALIDATION_REFERENCE_CANDIDATES)
    raise ArtifactLoadError(
        "Sprint 5 artifacts do not include the validation-derived Sprint 4 product payload. "
        f"Provide a copied payload or a configured reference at one of: {expected}"
    )


def _optional_frame(
    root: Path,
    candidates: tuple[str, ...],
    *,
    label: str,
    bundle_value: Any = None,
) -> tuple[pd.DataFrame, str | None]:
    if bundle_value is not None:
        return _frame_from_value(bundle_value, label=label), "dashboard_bundle.json"
    path = _first_existing(root, candidates)
    if path is None:
        return pd.DataFrame(), None
    return _read_frame(path, label=label), str(path)


def _high_priority_count(queue: pd.DataFrame) -> int:
    if "priority" not in queue:
        return 0
    high_labels = {"high", "elevated", "elevated_review_priority", "high_priority"}
    return int(queue["priority"].astype(str).str.lower().isin(high_labels).sum())


def _derive_summary(
    supplied: dict[str, Any], queue: pd.DataFrame, comparison: pd.DataFrame
) -> dict[str, Any]:
    summary = dict(supplied)
    summary.setdefault("flagged_transactions", int(len(queue)))
    summary.setdefault("high_priority_cases", _high_priority_count(queue))
    if "transactions_analyzed" not in summary and "validation_row_count" in comparison:
        values = pd.to_numeric(comparison["validation_row_count"], errors="coerce").dropna()
        if not values.empty:
            summary["transactions_analyzed"] = int(values.max())
    return summary


def load_dashboard_artifacts(root: str | Path) -> DashboardArtifacts:
    """Load the Network Investigator without fitting or importing any model."""

    presentation_root = Path(root).expanduser().resolve()
    if not presentation_root.is_dir():
        raise ArtifactLoadError(
            f"Dashboard artifact directory does not exist: {presentation_root}. "
            "Run the offline Sprint 4 artifact pipeline before starting Streamlit."
        )
    artifact_root = _resolve_validation_root(presentation_root)

    bundle_path = _first_existing(artifact_root, _BUNDLE_CANDIDATES)
    bundle: dict[str, Any] = {}
    if bundle_path is not None:
        loaded = _read_json(bundle_path)
        if not isinstance(loaded, dict):
            raise ArtifactLoadError(f"Dashboard bundle must be a JSON object: {bundle_path}")
        bundle = loaded

    provenance: dict[str, str] = {}
    if bundle_path is not None:
        provenance["bundle"] = str(bundle_path)

    queue_value = bundle.get("investigation_queue", bundle.get("queue"))
    queue_frame, queue_source = _optional_frame(
        artifact_root,
        _QUEUE_CANDIDATES,
        label="investigation queue",
        bundle_value=queue_value,
    )
    if queue_frame.empty:
        raise ArtifactLoadError(
            _missing_message("investigation queue", artifact_root, _QUEUE_CANDIDATES)
        )
    queue = _normalise_queue(queue_frame)
    provenance["queue"] = queue_source or ""

    case_value = bundle.get("cases")
    cases_source: str | None = None
    if case_value is not None:
        case_records = _extract_cases(case_value)
        cases_source = "dashboard_bundle.json"
    else:
        case_path = _first_existing(artifact_root, _CASE_CANDIDATES)
        if case_path is not None:
            case_records = _extract_cases(_read_json(case_path))
            cases_source = str(case_path)
        else:
            directory_result = _load_case_directory(artifact_root)
            if directory_result is None:
                raise ArtifactLoadError(_missing_message("cases", artifact_root, _CASE_CANDIDATES))
            case_records, case_dir = directory_result
            cases_source = str(case_dir)
    cases = _normalise_cases(case_records)
    provenance["cases"] = cases_source

    unknown_case_ids = sorted(set(queue["case_id"].astype(str)) - set(cases))
    if unknown_case_ids:
        preview = ", ".join(unknown_case_ids[:5])
        raise ArtifactLoadError(f"Queue references case IDs absent from cases artifact: {preview}")

    comparison_value = bundle.get("model_comparison")
    comparison_frame, comparison_source = _optional_frame(
        artifact_root,
        _COMPARISON_CANDIDATES,
        label="model comparison",
        bundle_value=comparison_value,
    )
    if comparison_frame.empty:
        raise ArtifactLoadError(
            _missing_message("model comparison", artifact_root, _COMPARISON_CANDIDATES)
        )
    comparison = _normalise_comparison(comparison_frame)
    provenance["model_comparison"] = comparison_source or ""

    pr_curves_raw, pr_source = _optional_frame(
        artifact_root,
        _PR_CURVE_CANDIDATES,
        label="PR curves",
        bundle_value=bundle.get("pr_curves"),
    )
    top_k_raw, top_k_source = _optional_frame(
        artifact_root,
        _TOP_K_CANDIDATES,
        label="Top-K metrics",
        bundle_value=bundle.get("top_k_metrics", bundle.get("top_k")),
    )
    ablation_raw, ablation_source = _optional_frame(
        artifact_root,
        _ABLATION_CANDIDATES,
        label="feature-family ablation",
        bundle_value=bundle.get("feature_family_ablation", bundle.get("ablation")),
    )
    if pr_source:
        provenance["pr_curves"] = pr_source
    if top_k_source:
        provenance["top_k"] = top_k_source
    if ablation_source:
        provenance["ablation"] = ablation_source

    summary_value = bundle.get("summary", {})
    if not isinstance(summary_value, dict):
        raise ArtifactLoadError("Dashboard bundle summary must be a JSON object.")
    summary_path = _first_existing(artifact_root, _SUMMARY_CANDIDATES)
    if not summary_value and summary_path is not None:
        summary_value = _read_json(summary_path)
        if not isinstance(summary_value, dict):
            raise ArtifactLoadError(f"Dashboard summary must be a JSON object: {summary_path}")
        provenance["summary"] = str(summary_path)
    if summary_value.get("final_test_opened") is True:
        raise ArtifactLoadError(
            "Dashboard summary says final_test_opened=true; Sprint 4 must be validation-only."
        )
    summary = _derive_summary(summary_value, queue, comparison)
    final_evaluation = _load_final_evaluation(presentation_root)
    provenance["validation_artifact_root"] = str(artifact_root)

    return DashboardArtifacts(
        root=presentation_root,
        summary=summary,
        queue=queue,
        cases=cases,
        model_comparison=comparison,
        pr_curves=_normalise_pr_curves(pr_curves_raw),
        top_k=_normalise_top_k(top_k_raw),
        ablation=_normalise_ablation(ablation_raw),
        provenance=provenance,
        final_evaluation=final_evaluation,
    )
