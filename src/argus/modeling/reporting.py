"""Generated comparison tables, figures, and Sprint 2 evidence report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from argus.modeling.artifacts import atomic_write_csv

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PR_CURVE_MAX_PLOT_POINTS = 10_000


def _deterministic_plot_indices(size: int, max_points: int) -> np.ndarray:
    """Return ordered, endpoint-preserving indices for visualization only."""

    if size <= max_points:
        return np.arange(size, dtype=np.int64)
    return np.unique(np.linspace(0, size - 1, num=max_points, dtype=np.int64))


def comparison_context_from_run(
    *,
    provenance: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    configuration: Mapping[str, Any],
    libraries: Mapping[str, Any],
    feature_scope: str,
) -> dict[str, Any]:
    """Build constant provenance columns embedded in every comparison row."""

    frozen = provenance["frozen_inputs"]
    raw_sources = provenance.get("raw_sources", {})
    transactions = raw_sources.get("transactions", {})
    return {
        "dataset_name": provenance["dataset"],
        "raw_transaction_sha256": transactions.get("sha256"),
        "feature_table_sha256": frozen["transaction_features.parquet"]["sha256"],
        "split_manifest_sha256": frozen["split_manifest.parquet"]["sha256"],
        "split_metadata_sha256": frozen["split_metadata.json"]["sha256"],
        "feature_scope": feature_scope,
        "preprocessing_state_sha256": feature_manifest["state_sha256"],
        "random_seed": configuration["random_seed"],
        "final_test_used": False,
        "library_versions": dict(libraries),
    }


def comparison_frame(
    model_results: Mapping[str, Mapping[str, Any]],
    *,
    context: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Flatten per-model validation evidence into a deterministic result table."""

    rows: list[dict[str, Any]] = []
    shared = dict(context or {})
    library_versions = shared.pop("library_versions", {})
    for model_name in sorted(model_results):
        result = model_results[model_name]
        metrics = result["validation_metrics"]
        threshold = metrics["threshold_metrics"]
        row: dict[str, Any] = {
            "model": model_name,
            "model_implementation": result.get("implementation"),
            "model_library_version": (
                library_versions.get("lightgbm")
                if model_name == "lightgbm"
                else library_versions.get("scikit_learn")
            ),
            **shared,
            "evaluation_partition": "validation",
            "row_count": metrics["row_count"],
            "positive_count": metrics["positive_count"],
            "positive_rate": metrics["positive_rate"],
            "pr_auc_average_precision": metrics["average_precision"],
            "roc_auc_secondary": metrics["roc_auc"],
            "decision_threshold": threshold["threshold"],
            "threshold_basis": result["threshold_basis"],
            "precision": threshold["precision"],
            "recall": threshold["recall"],
            "f1": threshold["f1"],
            "false_positive_rate": threshold["false_positive_rate"],
            "alert_count": threshold["alert_count"],
            "alert_rate": threshold["alert_rate"],
            "true_positive": threshold["true_positive"],
            "false_positive": threshold["false_positive"],
            "false_negative": threshold["false_negative"],
            "true_negative": threshold["true_negative"],
            "fit_seconds": result["runtime_seconds"]["fit"],
            "validation_predict_seconds": result["runtime_seconds"]["validation_predict"],
        }
        for top_k in metrics["top_k_metrics"]:
            requested_k = int(top_k["requested_k"])
            row[f"precision_at_{requested_k}"] = top_k["precision_at_k"]
            row[f"recall_at_{requested_k}"] = top_k["recall_at_k"]
            row[f"true_positives_at_{requested_k}"] = top_k["true_positives"]
            row[f"cutoff_score_at_{requested_k}"] = top_k["cutoff_score"]
            row[f"cutoff_tie_group_size_at_{requested_k}"] = top_k["cutoff_tie_group_size"]
            row[f"selected_from_cutoff_tie_group_at_{requested_k}"] = top_k[
                "selected_from_cutoff_tie_group"
            ]
            row[f"tie_break_material_at_{requested_k}"] = top_k["tie_break_material_at_cutoff"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["pr_auc_average_precision", "model"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def write_comparison_table(
    model_results: Mapping[str, Mapping[str, Any]],
    destination: str | Path,
    *,
    context: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate and atomically save the machine-readable comparison CSV."""

    frame = comparison_frame(model_results, context=context)
    atomic_write_csv(frame, destination, float_format="%.12g")
    return frame


def plot_metric_comparison(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Plot validation AP and secondary ROC-AUC from the generated table."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(frame))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(
        positions - width / 2,
        frame["pr_auc_average_precision"],
        width,
        label="PR-AUC (average precision)",
    )
    axis.bar(
        positions + width / 2,
        frame["roc_auc_secondary"],
        width,
        label="ROC-AUC (secondary)",
    )
    axis.set_xticks(positions, frame["model"], rotation=15, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Validation metric")
    axis.set_title("ARGUS Sprint 2 transaction-baseline comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    temporary = output.with_suffix(output.suffix + ".tmp")
    figure.savefig(temporary, dpi=160, format=output.suffix.lstrip("."))
    plt.close(figure)
    temporary.replace(output)
    return output


def plot_precision_recall_curves(
    labels: np.ndarray,
    scores_by_model: Mapping[str, np.ndarray],
    destination: str | Path,
) -> Path:
    """Plot exact validation precision-recall curves from frozen score vectors."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    for model_name in sorted(scores_by_model):
        precision, recall, _ = precision_recall_curve(labels, scores_by_model[model_name])
        plot_indices = _deterministic_plot_indices(
            len(precision),
            PR_CURVE_MAX_PLOT_POINTS,
        )
        axis.plot(
            recall[plot_indices],
            precision[plot_indices],
            linewidth=1.3,
            label=model_name,
        )
    prevalence = float(np.mean(labels))
    axis.axhline(
        prevalence,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"validation prevalence ({prevalence:.6%})",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Validation precision-recall curves (metrics exact; display decimated)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.text(
        0.5,
        0.01,
        "Display only: endpoint-preserving deterministic decimation to <=10,000 points/model; "
        "numeric metrics use every row.",
        ha="center",
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    temporary = output.with_suffix(output.suffix + ".tmp")
    figure.savefig(temporary, dpi=160, format=output.suffix.lstrip("."))
    plt.close(figure)
    temporary.replace(output)
    return output


def _format_optional(value: object, digits: int = 8) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_sprint2_markdown(
    *,
    model_results: Mapping[str, Mapping[str, Any]],
    champion: Mapping[str, Any],
    prevalence: Mapping[str, Mapping[str, Any]],
    runtime_seconds: Mapping[str, float],
    artifact_paths: Sequence[str],
    destination: str | Path,
    verification: Mapping[str, Any] | None = None,
    total_runtime_seconds: float | None = None,
) -> Path:
    """Render results from machine-readable objects without hand-entered metrics."""

    frame = comparison_frame(model_results)
    columns = [
        "model",
        "pr_auc_average_precision",
        "roc_auc_secondary",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "alert_count",
    ]
    lines = [
        "# ARGUS AI — Sprint 2 Validation Status",
        "",
        "This report is generated from executable Sprint 2 artifacts. It contains no "
        "manually entered model metric.",
        "",
        "## Outcome",
        "",
        f"- Transaction Baseline Champion: `{champion['champion_model']}`",
        "- Selection evidence: validation average precision only",
        "- Final test model inference/evaluation: **NOT USED**",
        "- Threshold: fixed configuration value; not optimized in Sprint 2",
        "- Accuracy: intentionally not used as a primary metric",
        "- PR-curve figure: deterministic endpoint-preserving display decimation to "
        f"at most {PR_CURVE_MAX_PLOT_POINTS:,} points/model; numerical metrics use all rows",
        "",
        "## Validation comparison",
        "",
        "| Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame[columns].iterrows():
        lines.append(
            f"| {row['model']} | {_format_optional(row['pr_auc_average_precision'])} | "
            f"{_format_optional(row['roc_auc_secondary'])} | "
            f"{_format_optional(row['precision'])} | {_format_optional(row['recall'])} | "
            f"{_format_optional(row['f1'])} | "
            f"{_format_optional(row['false_positive_rate'])} | "
            f"{int(row['alert_count']):,} |"
        )

    k_values = sorted(
        {
            int(item["requested_k"])
            for result in model_results.values()
            for item in result["validation_metrics"]["top_k_metrics"]
        }
    )
    lines.extend(["", "## Validation Top-K", ""])
    header = (
        "| Model | "
        + " | ".join(value for k in k_values for value in (f"Precision@{k}", f"Recall@{k}"))
        + " |"
    )
    separator = "| --- | " + " | ".join("---:" for _ in range(len(k_values) * 2)) + " |"
    lines.extend([header, separator])
    for model_name in sorted(model_results):
        metrics_by_k = {
            int(item["requested_k"]): item
            for item in model_results[model_name]["validation_metrics"]["top_k_metrics"]
        }
        values = [
            value
            for k in k_values
            for value in (
                _format_optional(metrics_by_k[k]["precision_at_k"]),
                _format_optional(metrics_by_k[k]["recall_at_k"]),
            )
        ]
        lines.append(f"| {model_name} | " + " | ".join(values) + " |")

    material_ties = [
        (model_name, item)
        for model_name, result in sorted(model_results.items())
        for item in result["validation_metrics"]["top_k_metrics"]
        if item["tie_break_material_at_cutoff"]
    ]
    if material_ties:
        lines.extend(
            [
                "",
                "### Top-K cutoff tie diagnostics",
                "",
                "Rows are ranked by descending score and then ascending immutable "
                "`source_row_number`. When a cutoff intersects an equal-score group, "
                "Precision@K and Recall@K are deterministic but depend on that declared "
                "secondary ordering; they do not measure discrimination within the tied "
                "group.",
                "",
                "| Model | K | Cutoff score | Tied rows at cutoff | Rows selected from tie |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_name, item in material_ties:
            lines.append(
                f"| {model_name} | {int(item['requested_k'])} | "
                f"{_format_optional(item['cutoff_score'], digits=12)} | "
                f"{int(item['cutoff_tie_group_size']):,} | "
                f"{int(item['selected_from_cutoff_tie_group']):,} |"
            )

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
        item = prevalence[partition]
        lines.append(
            f"| {partition} | {int(item['rows']):,} | {int(item['positive_labels']):,} | "
            f"{float(item['positive_rate']):.9%} | `{item['minimum_timestamp']}`–"
            f"`{item['maximum_timestamp']}` |"
        )
    lines.extend(
        [
            "",
            "Validation prevalence differs from train prevalence, and test prevalence is "
            "higher again. Precision and threshold alert volume are prevalence-sensitive; "
            "therefore validation precision/alert volume must not be assumed to transfer "
            "unchanged to the later test period. The frozen splits were not rebalanced.",
            "",
            "The test row count, timestamp range, and positive count above are pre-existing "
            "Sprint 1 split metadata. Sprint 2 did not load test feature rows, labels, or "
            "predictions.",
            "",
            "## Runtime",
            "",
        ]
    )
    if total_runtime_seconds is not None:
        lines.append(f"- `total_core_pipeline`: {float(total_runtime_seconds):.3f} seconds")
    for stage, seconds in runtime_seconds.items():
        lines.append(f"- `{stage}`: {float(seconds):.3f} seconds")
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{path}`" for path in artifact_paths)
    lines.extend(
        [
            "",
            "## Acceptance checklist",
            "",
            "- PASS — Logistic Regression baseline executed on frozen train/validation data.",
            "- PASS — Random Forest baseline executed on the identical matrix and partitions.",
            "- PASS — LightGBM baseline executed on the identical matrix and partitions.",
            "- PASS — Comparison uses PR-AUC (non-interpolated average precision) as primary.",
            "- PASS — ROC-AUC, fixed-threshold metrics, FPR, alert volume, and "
            "top-K metrics saved.",
            "- PASS — Champion selected exclusively from validation average precision.",
            "- PASS — Final test was not used for fitting, tuning, selection, or inference.",
            "- PASS — Graph-history features were excluded from transaction baselines.",
            "- PASS — Learned preprocessing state was fit from train only.",
            "- NOT IN SPRINT 2 — Hyperparameter tuning/refinement and graph-value experiment.",
        ]
    )
    if verification is None:
        lines.append("- PENDING — Post-run complete pytest/quality verification.")
    else:
        lines.append(f"- {verification['status']} — {verification['summary']}")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(output)
    return output
