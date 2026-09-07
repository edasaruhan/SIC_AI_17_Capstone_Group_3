"""Evidence-driven Markdown reporting for ARGUS Sprint 3.

The renderer deliberately consumes an already-persistable, manifest-like
mapping.  It does not calculate model metrics or infer acceptance outcomes, so
every numeric value in the human-readable report remains traceable to an
executable artifact produced by the refinement pipeline.

The public payload contract is intentionally small and JSON-compatible::

    {
        "status": "PASS | FAIL | BLOCKED",
        "selection": {
            "champion_model": "...",
            "selection_partition": "validation",
            "primary_metric": "average_precision",
            "final_test_used": False,
        },
        "scope": {"sprint4_started": False},
        "baseline_vs_refined": [{...metric row...}],
        "temporal_cv": [{...fold result row...}],
        "ablation": [{...feature-family result row...}],
        "threshold_analysis": [{...selected operating point row...}],
        "saturation": [{...score diagnostic row...}],
        "prevalence": {"train": {...}, "validation": {...}, "test": {...}},
        "runtime_seconds": {"stage": seconds},
        "artifact_paths": ["path", {"path": "path", "sha256": "..."}],
        "acceptance": [{"criterion": "...", "status": "PASS", "evidence": "..."}],
    }

Metric rows may put metrics directly on the row, below ``metrics``, or below
``validation_metrics``.  This tolerance lets the report consume both compact
comparison artifacts and detailed run-manifest records without duplicating
their values by hand.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class Sprint3ReportError(ValueError):
    """Raised when a Sprint 3 report payload violates a scope invariant."""


_MISSING = object()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: object) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[Mapping[str, Any]] = []
        for key in sorted(value, key=str):
            item = value[key]
            if isinstance(item, Mapping):
                rows.append({"model": str(key), **item})
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _metric(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Find a metric in common manifest containers without deriving a value."""

    containers = [
        row,
        _mapping(row.get("metrics")),
        _mapping(row.get("validation_metrics")),
        _mapping(row.get("ranking_metrics")),
        _mapping(row.get("operating_point")),
        _mapping(_mapping(row.get("selection")).get("operating_point")),
    ]
    threshold_metrics = _mapping(_mapping(row.get("validation_metrics")).get("threshold_metrics"))
    if threshold_metrics:
        containers.append(threshold_metrics)
    ranking_threshold_metrics = _mapping(
        _mapping(row.get("ranking_metrics")).get("threshold_metrics")
    )
    if ranking_threshold_metrics:
        containers.append(ranking_threshold_metrics)
    for container in containers:
        value = _first(container, *keys, default=_MISSING)
        if value is not _MISSING:
            return value
    return default


def _format_number(value: object, *, digits: int = 8) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _escape_cell(value)
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric:.{digits}f}"


def _format_integer(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return _escape_cell(value)


def _format_seconds(value: object) -> str:
    return "N/A" if value is None else f"{float(value):,.3f}"


def _escape_cell(value: object) -> str:
    if value is None:
        return "N/A"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _optional_bool_flag(value: object, *, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise Sprint3ReportError(f"{name} must be true, false, or absent")
    return value


def _scope_evidence(payload: Mapping[str, Any]) -> tuple[bool | None, bool | None]:
    """Extract test/Sprint-4 flags from compact or full run manifests."""

    selection = _mapping(payload.get("selection"))
    champion = _mapping(payload.get("champion"))
    scope = _mapping(payload.get("scope"))
    acceptance = _mapping(payload.get("acceptance"))
    final_test_value = _first(
        selection,
        "final_test_used",
        "test_metrics_used",
        default=_first(
            champion,
            "final_test_used",
            "test_metrics_used",
            default=_first(payload, "final_test_used", "test_metrics_used", default=None),
        ),
    )
    sprint4_value = _first(
        scope,
        "sprint4_started",
        default=_first(payload, "sprint4_started", default=None),
    )
    if sprint4_value is None and isinstance(acceptance.get("sprint4_not_started"), bool):
        sprint4_value = not acceptance["sprint4_not_started"]
    if sprint4_value is None and isinstance(acceptance.get("sprint4_graphsage_not_started"), bool):
        sprint4_value = not acceptance["sprint4_graphsage_not_started"]
    return (
        _optional_bool_flag(final_test_value, name="final-test usage evidence"),
        _optional_bool_flag(sprint4_value, name="Sprint 4 scope evidence"),
    )


def _top_k_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        metrics = _mapping(row.get("validation_metrics"))
        ranking_metrics = _mapping(row.get("ranking_metrics"))
        top_k = _first(
            row,
            "top_k_metrics",
            default=_first(
                metrics,
                "top_k_metrics",
                default=_first(ranking_metrics, "top_k_metrics", default=[]),
            ),
        )
        for item in _sequence_of_mappings(top_k):
            flattened.append(
                {
                    "stage": _first(row, "stage", "comparison", default="refined"),
                    "model": _first(row, "model", "candidate", "candidate_id", default="N/A"),
                    **item,
                }
            )
    return flattened


def _append_metric_table(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    lines.extend(
        [
            "| Stage | Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {stage} | {model} | {ap} | {roc} | {precision} | {recall} | {f1} | "
            "{fpr} | {alerts} |".format(
                stage=_escape_cell(_first(row, "stage", "comparison", default="N/A")),
                model=_escape_cell(
                    _first(row, "model", "candidate", "candidate_id", default="N/A")
                ),
                ap=_format_number(
                    _metric(row, "average_precision", "pr_auc_average_precision", "pr_auc")
                ),
                roc=_format_number(_metric(row, "roc_auc", "roc_auc_secondary")),
                precision=_format_number(_metric(row, "precision")),
                recall=_format_number(_metric(row, "recall")),
                f1=_format_number(_metric(row, "f1")),
                fpr=_format_number(_metric(row, "false_positive_rate", "fpr")),
                alerts=_format_integer(_metric(row, "alert_count", "alert_volume")),
            )
        )


def _append_top_k_table(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    top_k_rows = _top_k_rows(rows)
    if not top_k_rows:
        lines.append("No Top-K records were supplied.")
        return
    lines.extend(
        [
            "| Stage | Model | K | Precision@K | Recall@K | TP deterministic | "
            "TP expected | TP min-max | Material cutoff tie |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in top_k_rows:
        minimum = _first(item, "minimum_true_positives_at_k", default=None)
        maximum = _first(item, "maximum_true_positives_at_k", default=None)
        interval = (
            "N/A"
            if minimum is None or maximum is None
            else f"{_format_integer(minimum)}-{_format_integer(maximum)}"
        )
        lines.append(
            "| {stage} | {model} | {k} | {precision} | {recall} | {tp} | {expected} | "
            "{interval} | {tie} |".format(
                stage=_escape_cell(item["stage"]),
                model=_escape_cell(item["model"]),
                k=_format_integer(_first(item, "requested_k", "k")),
                precision=_format_number(
                    _first(item, "deterministic_precision_at_k", "precision_at_k")
                ),
                recall=_format_number(_first(item, "deterministic_recall_at_k", "recall_at_k")),
                tp=_format_integer(
                    _first(item, "deterministic_true_positives_at_k", "true_positives")
                ),
                expected=_format_number(
                    _first(item, "expected_true_positives_at_k", default=None), digits=3
                ),
                interval=interval,
                tie=_escape_cell(_first(item, "tie_break_material_at_cutoff", default="N/A")),
            )
        )


def _append_temporal_cv(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        lines.append("No temporal-CV records were supplied.")
        return
    lines.extend(
        [
            "| Candidate | Model family | Fold | Train rows | Validation rows | PR-AUC (AP) | "
            "ROC-AUC | Eligible | Fit seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {candidate} | {family} | {fold} | {train} | {validation} | {ap} | {roc} | "
            "{eligible} | {runtime} |".format(
                candidate=_escape_cell(
                    _first(row, "candidate_id", "candidate", "model", default="N/A")
                ),
                family=_escape_cell(_first(row, "model_family", "family", default="N/A")),
                fold=_escape_cell(_first(row, "fold", "fold_id", default="N/A")),
                train=_format_integer(_first(row, "train_rows", "train_row_count")),
                validation=_format_integer(_first(row, "validation_rows", "validation_row_count")),
                ap=_format_number(
                    _metric(
                        row,
                        "average_precision",
                        "mean_average_precision",
                        "pr_auc_average_precision",
                        "pr_auc",
                    )
                ),
                roc=_format_number(_metric(row, "roc_auc", "roc_auc_secondary")),
                eligible=_escape_cell(_first(row, "eligible", "selection_eligible", default="N/A")),
                runtime=_format_seconds(
                    _first(
                        row,
                        "fit_seconds",
                        default=_mapping(row.get("runtime_seconds")).get("fit"),
                    )
                ),
            )
        )


def _append_fold_boundaries(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        lines.append("No temporal-fold boundary records were supplied.")
        return
    lines.extend(
        [
            "| Fold | Train timestamp range | Train rows | Train positives | "
            "Validation timestamp range | Validation rows | Validation positives |",
            "| ---: | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {fold} | `{train_min}` - `{train_max}` | {train_rows} | {train_pos} | "
            "`{val_min}` - `{val_max}` | {val_rows} | {val_pos} |".format(
                fold=_escape_cell(_first(row, "fold", "fold_id", default="N/A")),
                train_min=_escape_cell(
                    _first(row, "train_minimum_timestamp", "train_min_timestamp", default="N/A")
                ),
                train_max=_escape_cell(
                    _first(row, "train_maximum_timestamp", "train_max_timestamp", default="N/A")
                ),
                train_rows=_format_integer(_first(row, "train_rows", "train_row_count")),
                train_pos=_format_integer(
                    _first(row, "train_positive_labels", "train_positive_count")
                ),
                val_min=_escape_cell(
                    _first(
                        row,
                        "validation_minimum_timestamp",
                        "validation_min_timestamp",
                        default="N/A",
                    )
                ),
                val_max=_escape_cell(
                    _first(
                        row,
                        "validation_maximum_timestamp",
                        "validation_max_timestamp",
                        default="N/A",
                    )
                ),
                val_rows=_format_integer(_first(row, "validation_rows", "validation_row_count")),
                val_pos=_format_integer(
                    _first(row, "validation_positive_labels", "validation_positive_count")
                ),
            )
        )


def _append_selected_candidates(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        lines.append("No selected-candidate records were supplied.")
        return
    lines.extend(
        [
            "| Model family | Selected candidate | Metric | Value | Selection partition |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {family} | {candidate} | {metric} | {value} | {partition} |".format(
                family=_escape_cell(_first(row, "model_family", "family", "model", default="N/A")),
                candidate=_escape_cell(_first(row, "candidate_id", "candidate", default="N/A")),
                metric=_escape_cell(
                    _first(row, "selection_metric", default="mean_average_precision")
                ),
                value=_format_number(
                    _first(row, "selection_metric_value", "mean_average_precision")
                ),
                partition=_escape_cell(_first(row, "selection_partition", default="N/A")),
            )
        )


def _append_ablation(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        lines.append("No feature-family ablation records were supplied.")
        return
    lines.extend(
        [
            "| Feature family | Scope | Model family | Parameter digest | Features | "
            "PR-AUC (AP) | ROC-AUC | Delta vs B |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {family} | {scope} | {model} | `{digest}` | {features} | {ap} | {roc} | "
            "{delta} |".format(
                family=_escape_cell(_first(row, "feature_family", "family", default="N/A")),
                scope=_escape_cell(_first(row, "scope", "ablation_label", default="primary")),
                model=_escape_cell(_first(row, "model_family", "model", default="N/A")),
                digest=_escape_cell(
                    _first(
                        row,
                        "parameter_digest",
                        "effective_parameters_sha256",
                        "params_sha256",
                        default="N/A",
                    )
                ),
                features=_format_integer(_first(row, "feature_count", "transformed_feature_count")),
                ap=_format_number(
                    _metric(row, "average_precision", "pr_auc_average_precision", "pr_auc")
                ),
                roc=_format_number(_metric(row, "roc_auc", "roc_auc_secondary")),
                delta=_format_number(_first(row, "delta_vs_b", "average_precision_delta_vs_b")),
            )
        )


def _append_thresholds(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        lines.append("No selected validation operating points were supplied.")
        return
    lines.extend(
        [
            "| Model | Rule | Status | Threshold | Precision | Recall | F1 | FPR | Alerts |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        summaries = _mapping(row.get("summaries"))
        selection = _mapping(row.get("selection")) or _mapping(
            summaries.get("predeclared_joint_primary_constraint")
        )
        operating_point = _mapping(
            _first(row, "operating_point", default=selection.get("operating_point", {}))
        )
        merged = {**row, **operating_point}
        lines.append(
            "| {model} | {rule} | {status} | {threshold} | {precision} | {recall} | "
            "{f1} | {fpr} | {alerts} |".format(
                model=_escape_cell(_first(row, "model", "candidate", default="N/A")),
                rule=_escape_cell(
                    _first(
                        row,
                        "rule",
                        "selection_rule",
                        default=_first(
                            selection,
                            "rule",
                            "objective",
                            default="predeclared_joint_primary_constraint",
                        ),
                    )
                ),
                status=_escape_cell(_first(row, "status", default=selection.get("status", "N/A"))),
                threshold=_format_number(_metric(merged, "threshold"), digits=12),
                precision=_format_number(_metric(merged, "precision")),
                recall=_format_number(_metric(merged, "recall")),
                f1=_format_number(_metric(merged, "f1")),
                fpr=_format_number(_metric(merged, "false_positive_rate", "fpr")),
                alerts=_format_integer(_metric(merged, "alert_count", "alert_volume")),
            )
        )


def _flatten_threshold_rows(value: object) -> list[dict[str, Any]]:
    """Expand every saved validation threshold summary for every model."""

    wrapper = _mapping(value)
    models = _mapping(wrapper.get("models"))
    if not models:
        return [dict(row) for row in _sequence_of_mappings(wrapper.get("rows", value))]
    rows: list[dict[str, Any]] = []
    for model in sorted(models, key=str):
        artifact = _mapping(models[model])
        summaries = _mapping(artifact.get("summaries"))
        if not summaries:
            rows.append({"model": str(model), **artifact})
            continue
        for rule in sorted(summaries, key=str):
            selection = _mapping(summaries[rule])
            rows.append(
                {
                    "model": str(model),
                    "selection_rule": str(rule),
                    "selection": selection,
                    "status": selection.get("status"),
                    "operating_point": selection.get("operating_point"),
                }
            )
    return rows


def _append_saturation(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        lines.append("No saturation/tie diagnostic records were supplied.")
        return
    lines.extend(
        [
            "| Stage | Model | Raw unique | Probability unique | Exact p=0 | Exact p=1 | "
            "Raw AP | Probability AP | Material Top-K ties |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        raw = _mapping(_first(row, "raw", "raw_score", "raw_scores", default={}))
        probability = _mapping(
            _first(row, "probability", "probabilities", "probability_scores", default={})
        )
        material_ties = _first(
            row,
            "material_top_k_ties",
            "material_tie_count",
            default=None,
        )
        if material_ties is None:
            diagnostic_top_k = _sequence_of_mappings(raw.get("top_k", probability.get("top_k", [])))
            material_ties = sum(
                bool(item.get("tie_break_material_at_cutoff")) for item in diagnostic_top_k
            )
        lines.append(
            "| {stage} | {model} | {raw_unique} | {prob_unique} | {zero} | {one} | "
            "{raw_ap} | {prob_ap} | {ties} |".format(
                stage=_escape_cell(_first(row, "stage", default="N/A")),
                model=_escape_cell(_first(row, "model", "candidate", default="N/A")),
                raw_unique=_format_integer(
                    _first(row, "raw_unique_scores", default=raw.get("unique_score_count"))
                ),
                prob_unique=_format_integer(
                    _first(
                        row,
                        "probability_unique_scores",
                        default=probability.get("unique_score_count"),
                    )
                ),
                zero=_format_integer(
                    _first(
                        row,
                        "exact_probability_zero_count",
                        default=_first(
                            probability,
                            "exact_zero_count",
                            "exact_probability_zero_count",
                        ),
                    )
                ),
                one=_format_integer(
                    _first(
                        row,
                        "exact_probability_one_count",
                        default=_first(
                            probability,
                            "exact_one_count",
                            "exact_probability_one_count",
                        ),
                    )
                ),
                raw_ap=_format_number(_first(row, "raw_average_precision", "raw_pr_auc")),
                prob_ap=_format_number(
                    _first(row, "probability_average_precision", "probability_pr_auc")
                ),
                ties=_escape_cell(material_ties),
            )
        )
        finding = _first(row, "finding", "conclusion", "cause", default=None)
        if finding:
            lines.append(f"<!-- finding: {_escape_cell(finding)} -->")


def _acceptance_rows(value: object) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        rows = []
        for criterion in sorted(value, key=str):
            record = value[criterion]
            if isinstance(record, Mapping):
                rows.append(
                    {
                        "criterion": criterion,
                        "status": _first(record, "status", default="UNKNOWN"),
                        "evidence": _first(record, "evidence", "summary", default=""),
                    }
                )
            elif isinstance(record, bool):
                rows.append(
                    {"criterion": criterion, "status": "PASS" if record else "FAIL", "evidence": ""}
                )
            else:
                rows.append({"criterion": criterion, "status": record, "evidence": ""})
        return rows
    return [
        {
            "criterion": _first(item, "criterion", "check", "name", default="N/A"),
            "status": _first(item, "status", default="UNKNOWN"),
            "evidence": _first(item, "evidence", "summary", default=""),
        }
        for item in _sequence_of_mappings(value)
    ]


def _atomic_write_markdown(lines: Sequence[str], destination: str | Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def render_sprint3_markdown(
    payload: Mapping[str, Any],
    destination: str | Path,
) -> Path:
    """Render an auditable Sprint 3 status report from machine-readable results.

    The renderer accepts both the compact contract documented in the module
    docstring and the full run-manifest shape used by the pipeline.  If an
    explicit final-test or Sprint-4 flag is true, rendering aborts; absent
    optional sections are rendered as not recorded instead of being invented.
    """

    if not isinstance(payload, Mapping):
        raise Sprint3ReportError("payload must be a mapping")

    selection = _mapping(payload.get("selection")) or _mapping(payload.get("champion"))
    final_test_used, sprint4_started = _scope_evidence(payload)
    if final_test_used is True:
        raise Sprint3ReportError("Sprint 3 report cannot be rendered after final-test use")
    if sprint4_started is True:
        raise Sprint3ReportError("Sprint 3 report cannot claim completion after Sprint 4 started")

    comparison_rows = _sequence_of_mappings(
        _first(payload, "baseline_vs_refined", "models", default=[])
    )
    temporal_cv = payload.get("temporal_cv", [])
    temporal_cv_mapping = _mapping(temporal_cv)
    fold_rows = _sequence_of_mappings(temporal_cv_mapping.get("folds", []))
    cv_rows = _sequence_of_mappings(
        _first(
            temporal_cv_mapping,
            "trials",
            "fold_results",
            "candidate_trials",
            "candidate_summary",
            default=temporal_cv,
        )
    )
    selected_candidate_rows = _sequence_of_mappings(
        temporal_cv_mapping.get("selected_candidates", [])
    )
    ablation = payload.get("ablation", [])
    ablation_mapping = _mapping(ablation)
    ablation_rows = _sequence_of_mappings(ablation_mapping.get("rows", ablation))
    threshold_analysis = payload.get("threshold_analysis", [])
    threshold_rows = _flatten_threshold_rows(threshold_analysis)
    saturation = payload.get("saturation", [])
    saturation_mapping = _mapping(saturation)
    saturation_rows = _sequence_of_mappings(
        saturation_mapping.get("models", saturation_mapping.get("rows", saturation))
    )

    champion = _first(
        selection,
        "champion_model",
        "refined_champion",
        "transaction_baseline_champion",
        default="N/A",
    )
    headline_graph = _mapping(
        _first(
            payload,
            "graph_value_conclusion",
            default=ablation_mapping.get("graph_value_conclusion", {}),
        )
    )
    status = _first(payload, "status", "sprint3_status", default="UNKNOWN")
    lines = [
        "# ARGUS AI - Sprint 3 Refinement and Graph-Value Status",
        "",
        "This report is generated solely from executable, machine-readable Sprint 3 "
        "artifacts. No model result is entered or estimated by the report renderer.",
        "",
        "## Outcome",
        "",
        f"- Sprint 3 status: **{_escape_cell(status)}**",
        f"- Refined Transaction Baseline Champion: `{_escape_cell(champion)}`",
        "- Champion/model selection partition: "
        f"`{_escape_cell(_first(selection, 'selection_partition', default='N/A'))}`",
        "- Primary selection metric: "
        f"`{_escape_cell(_first(selection, 'primary_metric', 'selection_metric', default='N/A'))}`",
        "- Final test fitting, transformation, inference, tuning, and selection: "
        + ("**NOT USED**" if final_test_used is False else "**NOT RECORDED**"),
        "- Sprint 4 / GraphSAGE: "
        + ("**NOT STARTED**" if sprint4_started is False else "**NOT RECORDED**"),
        "- Accuracy: intentionally not used as a primary metric",
    ]
    if headline_graph:
        graph_delta = _first(
            headline_graph,
            "average_precision_delta",
            "average_precision_delta_c_minus_b",
            "c_minus_b_average_precision",
            "absolute_average_precision_delta",
            "delta_vs_b",
        )
        same_protocol = _first(
            headline_graph,
            "same_protocol_verified",
            "same_parameter_digest",
            default=all(
                headline_graph.get(key) is True
                for key in (
                    "same_model_family",
                    "same_candidate_id",
                    "same_effective_parameters",
                    "same_random_seed",
                    "same_frozen_split",
                    "same_validation_protocol",
                )
            ),
        )
        graph_comparison = _first(
            headline_graph,
            "comparison",
            "headline_comparison",
            default="C vs B",
        )
        lines.extend(
            [
                f"- Headline graph-value comparison: `{_escape_cell(graph_comparison)}`",
                f"- Headline graph-value AP delta: `{_format_number(graph_delta)}`",
                f"- Same-family/parameter protocol verified: `{_escape_cell(same_protocol)}`",
            ]
        )

    lines.extend(["", "## Baseline vs refined validation results", ""])
    if comparison_rows:
        _append_metric_table(lines, comparison_rows)
    else:
        lines.append("No baseline/refined comparison records were supplied.")
    lines.extend(["", "### Validation Top-K and tie bounds", ""])
    _append_top_k_table(lines, comparison_rows)

    lines.extend(
        [
            "",
            "## Expanding-window temporal cross-validation",
            "",
            "Temporal folds remain inside the frozen outer-training partition. Outer validation "
            "is not used to select hyperparameters, and the final test remains unopened.",
            "",
        ]
    )
    if fold_rows:
        lines.extend(["### Fold boundaries", ""])
        _append_fold_boundaries(lines, fold_rows)
        lines.extend(["", "### Candidate results", ""])
    _append_temporal_cv(lines, cv_rows)
    if selected_candidate_rows:
        lines.extend(["", "### Selected candidates", ""])
        _append_selected_candidates(lines, selected_candidate_rows)

    lines.extend(
        [
            "",
            "## Feature-family ablation and graph value",
            "",
            "Primary A/B/C graph-value claims require the same model family, selected "
            "hyperparameters, seed, chronological split, preprocessing discipline, and "
            "evaluation code; only the permitted feature family changes.",
            "",
        ]
    )
    _append_ablation(lines, ablation_rows)

    lines.extend(
        [
            "",
            "## Validation-only threshold analysis",
            "",
            "Thresholds are selected only from complete validation score groups using the "
            "recorded alert-budget and FPR/recall trade-off. No test-set threshold feedback is "
            "permitted.",
            "",
        ]
    )
    _append_thresholds(lines, threshold_rows)

    lines.extend(
        [
            "",
            "## Saturation and tie investigation",
            "",
            "Raw ranking scores and transformed probabilities are reported separately. If a "
            "Top-K cutoff intersects an equal-score group, deterministic values use the "
            "declared immutable-row tie break and the expected/minimum/maximum bounds describe "
            "the unresolved ordering within that group.",
            "",
        ]
    )
    _append_saturation(lines, saturation_rows)
    findings = [
        _first(row, "finding", "conclusion", "cause", default=None) for row in saturation_rows
    ]
    for finding in findings:
        if finding:
            lines.append(f"- {_escape_cell(finding)}")
    root_cause = _mapping(saturation_mapping.get("root_cause_conclusion"))
    if root_cause:
        lines.extend(
            [
                "",
                "### Executable root-cause conclusion",
                "",
                "- Saved-score/model reproduction verified: "
                f"`{_escape_cell(root_cause.get('serialization_reproduction_verified'))}`",
                "- Serialization error observed: "
                f"`{_escape_cell(root_cause.get('serialization_error'))}`",
                "- Immediate probability-tie mechanism: "
                f"{_escape_cell(root_cause.get('immediate_tie_mechanism'))}",
                "- Logistic Regression upstream finding: "
                f"{_escape_cell(root_cause.get('logistic_upstream_cause'))}",
                "- LightGBM upstream finding: "
                f"{_escape_cell(root_cause.get('lightgbm_upstream_cause'))}",
                f"- Refinement controls: {_escape_cell(root_cause.get('refinement_controls'))}",
                f"- Ranking policy: {_escape_cell(root_cause.get('ranking_policy'))}",
            ]
        )

    prevalence = _mapping(_first(payload, "prevalence", "split_prevalence", default={}))
    lines.extend(
        [
            "",
            "## Frozen chronological prevalence",
            "",
            "| Partition | Rows | Positives | Positive rate | Timestamp range |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for partition in ("train", "validation", "test"):
        item = _mapping(prevalence.get(partition))
        minimum = _first(item, "minimum_timestamp", "min_timestamp", default="N/A")
        maximum = _first(item, "maximum_timestamp", "max_timestamp", default="N/A")
        lines.append(
            f"| {partition} | {_format_integer(_first(item, 'rows', 'row_count'))} | "
            f"{_format_integer(_first(item, 'positive_labels', 'positive_count'))} | "
            f"{_format_number(_first(item, 'positive_rate'), digits=10)} | "
            f"`{_escape_cell(minimum)}` - `{_escape_cell(maximum)}` |"
        )
    lines.extend(
        [
            "",
            "The chronological prevalence shift is preserved. Precision and alert volume are "
            "prevalence-sensitive, so a validation operating point must not be assumed to "
            "transfer unchanged to the later test period.",
        ]
    )
    if final_test_used is False:
        lines.append(
            "Test counts above are metadata-only; Sprint 3 did not load test features, labels, "
            "scores, or predictions."
        )
    else:
        lines.append("Final-test non-use evidence was not recorded in the supplied payload.")
    lines.extend(["", "## Runtime", ""])
    runtimes = _mapping(payload.get("runtime_seconds_by_stage"))
    total_runtime = payload.get("runtime_seconds")
    if isinstance(total_runtime, (int, float)):
        lines.append(f"- `total`: {_format_seconds(total_runtime)} seconds")
    if runtimes:
        for stage in sorted(runtimes, key=str):
            lines.append(f"- `{_escape_cell(stage)}`: {_format_seconds(runtimes[stage])} seconds")
    else:
        lines.append("No runtime records were supplied.")

    lines.extend(["", "## Artifacts", ""])
    artifacts = _first(payload, "artifact_paths", "core_artifact_paths", default=[])
    artifact_items: list[object]
    if isinstance(artifacts, Mapping):
        artifact_items = [artifacts[key] for key in sorted(artifacts, key=str)]
    elif isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes, bytearray)):
        artifact_items = list(artifacts)
    else:
        artifact_items = []
    if artifact_items:
        for artifact in artifact_items:
            if isinstance(artifact, Mapping):
                path = _first(artifact, "path", "relative_path", default="N/A")
                sha256 = _first(artifact, "sha256", default=None)
                suffix = "" if sha256 is None else f" (sha256 `{_escape_cell(sha256)}`)"
                lines.append(f"- `{_escape_cell(path)}`{suffix}")
            else:
                lines.append(f"- `{_escape_cell(artifact)}`")
    else:
        lines.append("No artifact paths were supplied.")

    lines.extend(["", "## Acceptance checklist", ""])
    acceptance = _acceptance_rows(payload.get("acceptance", []))
    if acceptance:
        for item in acceptance:
            evidence = f" - {_escape_cell(item['evidence'])}" if item["evidence"] else ""
            lines.append(
                f"- **{_escape_cell(item['status'])}** - "
                f"{_escape_cell(item['criterion'])}{evidence}"
            )
    else:
        lines.append("- **UNKNOWN** - No acceptance records were supplied.")

    quality = _mapping(_first(payload, "quality", "quality_verification", default={}))
    if quality:
        pytest_evidence = _first(
            quality,
            "pytest_passed",
            "pytest_summary",
            default="N/A",
        )
        artifact_evidence = _first(
            quality,
            "artifact_verification_status",
            default="N/A",
        )
        lines.extend(
            [
                "",
                "## Automated quality evidence",
                "",
                f"- Status: **{_escape_cell(_first(quality, 'status', default='UNKNOWN'))}**",
                f"- Pytest passed: {_escape_cell(pytest_evidence)}",
                f"- Artifact verification: {_escape_cell(artifact_evidence)}",
                "- Ruff lint/format: inspect machine-readable `quality_report.json` checks",
            ]
        )

    lines.extend(["", "## Scope stop", ""])
    if sprint4_started is False:
        lines.append(
            "Sprint 3 stops here. Sprint 4 / GraphSAGE training, tuning, inference, and "
            "evaluation were not started."
        )
    else:
        lines.append("Sprint 4 / GraphSAGE non-start evidence was not recorded in the payload.")
    return _atomic_write_markdown(lines, destination)


__all__ = ["Sprint3ReportError", "render_sprint3_markdown"]
