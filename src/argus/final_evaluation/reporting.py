"""Artifact-driven final comparison and documentation publishing."""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from argus.config import get_path, load_config
from argus.final_evaluation.contract import load_json_object
from argus.final_evaluation.verify import verify_final_evaluation

_README_START = "<!-- ARGUS_FINAL_EVALUATION_START -->"
_README_END = "<!-- ARGUS_FINAL_EVALUATION_END -->"
_MODEL_CARD_START = "<!-- ARGUS_FINAL_MODEL_CARD_START -->"
_MODEL_CARD_END = "<!-- ARGUS_FINAL_MODEL_CARD_END -->"


class FinalDocumentationError(RuntimeError):
    """Raised when verified final artifacts cannot produce consistent documentation."""


def _metric(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.8f}"


def _model_label(name: str) -> str:
    return {
        "graph_enhanced_lightgbm": "Graph-enhanced LightGBM",
        "refined_transaction_lightgbm": "Refined transaction LightGBM",
        "graphsage_edge_classifier": "GraphSAGE edge classifier",
    }.get(name, name)


def _top_k_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["requested_k"]))
        if key in result:
            raise FinalDocumentationError(f"Duplicate Top-K row: {key}")
        result[key] = row
    return result


def render_final_comparison_markdown(
    comparison: Sequence[Mapping[str, Any]],
    top_k_rows: Sequence[Mapping[str, Any]],
    prevalence_shift: Mapping[str, Any],
    *,
    quality: Mapping[str, Any] | None = None,
    artifact_prefix: str = "../../artifacts/sprint5",
) -> str:
    """Render final evidence only from parsed machine-readable values."""

    if len(comparison) != 3 or {str(row["model"]) for row in comparison} != {
        "graph_enhanced_lightgbm",
        "refined_transaction_lightgbm",
        "graphsage_edge_classifier",
    }:
        raise FinalDocumentationError("Final comparison must contain the three frozen models")
    top_k = _top_k_lookup(top_k_rows)
    lines = [
        "# ARGUS AI — Final Model Comparison",
        "",
        "This table is generated from the persisted one-shot final-test artifacts. The model "
        "order follows pre-test role and name; it is not a test-driven ranking or selection.",
        "",
        "| Role | Frozen model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        role = "Frozen champion" if row["role"] == "frozen_champion" else "Comparator"
        lines.append(
            "| {role} | {model} | {ap} | {roc} | {precision} | {recall} | {f1} | "
            "{fpr} | {alerts:,} |".format(
                role=role,
                model=_model_label(str(row["model"])),
                ap=_metric(row["average_precision"]),
                roc=_metric(row["roc_auc"]),
                precision=_metric(row["precision"]),
                recall=_metric(row["recall"]),
                f1=_metric(row["f1"]),
                fpr=_metric(row["false_positive_rate"]),
                alerts=int(row["alert_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Operational Top-K metrics",
            "",
            "| Model | K | Precision@K | Recall@K | True positives |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison:
        name = str(row["model"])
        for k in (100, 500, 1000):
            item = top_k[(name, k)]
            lines.append(
                f"| {_model_label(name)} | {k:,} | {_metric(item['precision_at_k'])} | "
                f"{_metric(item['recall_at_k'])} | {int(item['true_positives']):,} |"
            )
    validation = prevalence_shift["validation"]
    test = prevalence_shift["test"]
    lines.extend(
        [
            "",
            "## Prevalence shift",
            "",
            f"- Validation: {int(validation['positives']):,} / {int(validation['rows']):,} "
            f"({_metric(validation['positive_rate'])}).",
            f"- Final test: {int(test['positives']):,} / {int(test['rows']):,} "
            f"({_metric(test['positive_rate'])}).",
            "- Test/validation positive-rate ratio: "
            f"{float(prevalence_shift['test_to_validation_positive_rate_ratio']):.8f}.",
            f"- Interpretation: {prevalence_shift['interpretation']}",
            "",
            "## Frozen protocol statement",
            "",
            "`graph_enhanced_lightgbm` was selected by validation PR-AUC and frozen before "
            "final-test access. Test results did not change the model, feature family, "
            "hyperparameters, or validation-selected threshold. No post-test tuning or "
            "retraining was performed.",
            "",
            "## Machine-readable evidence",
            "",
            "The paths below are generated locally and intentionally excluded from Git. They "
            "become available after the frozen artifact package is restored locally.",
            "",
            f"- Final metrics: `{artifact_prefix}/final_metrics.json`",
            f"- Final comparison: `{artifact_prefix}/final_model_comparison.json`",
            f"- Final Top-K metrics: `{artifact_prefix}/final_top_k_metrics.json`",
            f"- Saved final predictions: `{artifact_prefix}/final_test_predictions.parquet`",
            f"- Prevalence analysis: `{artifact_prefix}/prevalence_shift.json`",
            f"- Test identity/leakage audit: `{artifact_prefix}/test_identity_audit.json`",
            f"- Immutable run manifest: `{artifact_prefix}/run_manifest.json`",
            f"- Read-only verification: `{artifact_prefix}/verification_report.json`",
            f"- Quality report: `{artifact_prefix}/quality_report.json`",
            "- Streamlit — Executive Dashboard: "
            f"`{artifact_prefix}/screenshots/executive_dashboard.png`",
            "- Streamlit — Investigation Queue: "
            f"`{artifact_prefix}/screenshots/investigation_queue.png`",
            "- Streamlit — Case Investigator: "
            f"`{artifact_prefix}/screenshots/case_investigator.png`",
            f"- Streamlit — Model Comparison: `{artifact_prefix}/screenshots/model_comparison.png`",
        ]
    )
    if quality:
        lines.extend(
            [
                "",
                "## Final quality",
                "",
                f"- Status: **{quality.get('status', 'UNKNOWN')}**",
                f"- Pytest: {quality.get('pytest_passed', 'UNKNOWN')} passed",
                f"- Saved-artifact verification: "
                f"{quality.get('artifact_verification_status', 'UNKNOWN')}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _replace_block(text: str, start: str, end: str, block: str, *, after: str) -> str:
    payload = f"{start}\n{block.rstrip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), flags=re.DOTALL)
    if pattern.search(text):
        return pattern.sub(payload, text, count=1)
    index = text.find(after)
    if index < 0:
        raise FinalDocumentationError(f"Documentation insertion anchor is absent: {after}")
    insertion = index + len(after)
    return text[:insertion] + "\n\n" + payload + text[insertion:]


def _atomic_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_quality(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "quality_report.json"
    if not path.is_file():
        return None
    return load_json_object(path, "final quality report")


def publish_final_documentation(
    config_path: str | Path = "configs/final_evaluation.yaml",
) -> dict[str, Any]:
    """Verify saved final artifacts and publish README/Model Card result blocks."""

    verification = verify_final_evaluation(config_path)
    config = load_config(config_path)
    project_root = Path(config["_meta"]["project_root"])
    run_dir = get_path(config, "run_dir")
    comparison_payload = load_json_object(run_dir / "final_model_comparison.json", "comparison")
    top_k_payload = load_json_object(run_dir / "final_top_k_metrics.json", "Top-K metrics")
    shift = load_json_object(run_dir / "prevalence_shift.json", "prevalence shift")
    quality = _read_quality(run_dir)
    comparison = comparison_payload["models"]
    top_k_rows = top_k_payload["rows"]

    generated = render_final_comparison_markdown(
        comparison,
        top_k_rows,
        shift,
        quality=quality,
        artifact_prefix="../../artifacts/sprint5",
    )
    comparison_path = project_root / "reports" / "generated" / "FINAL_MODEL_COMPARISON.md"
    _atomic_markdown(comparison_path, generated)

    readme_table = render_final_comparison_markdown(
        comparison,
        top_k_rows,
        shift,
        quality=quality,
        artifact_prefix="artifacts/sprint5",
    )
    readme_block = "## Sprint 5 one-shot final evaluation\n\n" + "\n".join(
        readme_table.splitlines()[4:]
    )
    readme_path = project_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = _replace_block(
        readme,
        _README_START,
        _README_END,
        readme_block,
        after="## Current status",
    )
    _atomic_markdown(readme_path, readme)

    model_table = render_final_comparison_markdown(
        comparison,
        top_k_rows,
        shift,
        quality=quality,
        artifact_prefix="../artifacts/sprint5",
    )
    model_block = "## One-shot final-test evidence\n\n" + "\n".join(model_table.splitlines()[4:])
    model_card_path = project_root / "docs" / "MODEL_CARD.md"
    model_card = model_card_path.read_text(encoding="utf-8")
    model_card = _replace_block(
        model_card,
        _MODEL_CARD_START,
        _MODEL_CARD_END,
        model_block,
        after="## Current state",
    )
    _atomic_markdown(model_card_path, model_card)
    return {
        "status": "PASS",
        "verification_status": verification["status"],
        "generated_comparison": str(comparison_path),
        "readme": str(readme_path),
        "model_card": str(model_card_path),
        "source_artifacts": [
            str(run_dir / "final_model_comparison.json"),
            str(run_dir / "final_top_k_metrics.json"),
            str(run_dir / "prevalence_shift.json"),
        ],
        "manual_metric_entry": False,
    }


__all__ = [
    "FinalDocumentationError",
    "publish_final_documentation",
    "render_final_comparison_markdown",
]
