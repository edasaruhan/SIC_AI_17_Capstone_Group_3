"""Load and validate saved Sprint 4 product artifacts.

The application deliberately has no model-training dependency. It accepts either a
single ``dashboard_bundle.json`` or the documented collection of CSV/JSON files under
``artifacts/sprint4/product``. The normalisation layer keeps the UI stable when an
experiment artifact uses the repository's longer metric names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactLoadError(RuntimeError):
    """Raised when saved artifacts are absent or violate the dashboard contract."""


@dataclass(frozen=True)
class DashboardArtifacts:
    """Validated, presentation-ready Sprint 4 artifacts."""

    root: Path
    summary: dict[str, Any]
    queue: pd.DataFrame
    cases: dict[str, dict[str, Any]]
    model_comparison: pd.DataFrame
    pr_curves: pd.DataFrame
    top_k: pd.DataFrame
    ablation: pd.DataFrame
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

    artifact_root = Path(root).expanduser().resolve()
    if not artifact_root.is_dir():
        raise ArtifactLoadError(
            f"Sprint 4 artifact directory does not exist: {artifact_root}. "
            "Run the offline Sprint 4 artifact pipeline before starting Streamlit."
        )

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

    return DashboardArtifacts(
        root=artifact_root,
        summary=summary,
        queue=queue,
        cases=cases,
        model_comparison=comparison,
        pr_curves=_normalise_pr_curves(pr_curves_raw),
        top_k=_normalise_top_k(top_k_raw),
        ablation=_normalise_ablation(ablation_raw),
        provenance=provenance,
    )
