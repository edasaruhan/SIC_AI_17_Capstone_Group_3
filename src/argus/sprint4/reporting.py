"""Render Sprint 4 evidence exclusively from executable manifest values."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _metric(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.8f}"


def render_sprint4_report(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Write a human-readable ledger without accepting manually supplied prose results."""

    comparison = payload["model_comparison"]
    sampling = payload["sampling"]
    cases = payload["product"]["cases"]
    quality = payload.get("quality", {})
    lines = [
        "# ARGUS AI - Sprint 4 GraphSAGE and Product Layer Status",
        "",
        "This report is rendered only from executable Sprint 4 artifacts. No metric, case, "
        "or explanation is entered manually.",
        "",
        "## Outcome",
        "",
        f"- Sprint 4 status: **{payload['status']}**",
        "- GraphSAGE target: transaction/edge label; unsupported account-level label: "
        f"`{payload['target_design']['unsupported_account_label_created']}`",
        f"- GraphSAGE training scope: `{payload['experiment_scope']}`",
        f"- Outer validation evaluation: **FULL {int(comparison[0]['row_count']):,} rows**",
        "- Final test fitting, transformation, graph construction, inference, tuning, and "
        "evaluation: **NOT USED**",
        "- Final-test opening: **STOPPED BEFORE FINAL EVALUATION**",
        "",
        "## Fair full-validation model comparison",
        "",
        "| Model | PR-AUC (AP) | ROC-AUC | Precision@1K | Recall@1K | "
        "Threshold precision | Threshold recall | F1 | FPR | Alerts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            "| {model} | {ap} | {roc} | {precision_at_k} | {recall_at_k} | "
            "{precision} | {recall} | {f1} | {fpr} | {alerts:,} |".format(
                model=row["model"],
                ap=_metric(row["average_precision"]),
                roc=_metric(row["roc_auc"]),
                precision_at_k=_metric(row["precision_at_k"]),
                recall_at_k=_metric(row["recall_at_k"]),
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
            "All three score vectors use the identical frozen outer-validation rows and the "
            "same 5,000-alert / 1% FPR validation-only operating-point rule. The two LightGBM "
            "rows are immutable Sprint 3 references; GraphSAGE uses its saved raw logits.",
            "",
            "## Explicit scalable-subset disclosure",
            "",
            f"- Train-prefix message graph: {sampling['training_context']['sampled_rows']:,} / "
            f"{sampling['training_context']['population_rows']:,} edges "
            f"({_metric(sampling['training_context']['coverage_fraction'])}).",
            f"- Validation-inference historical graph: "
            f"{sampling['inference_context']['sampled_rows']:,} / "
            f"{sampling['inference_context']['population_rows']:,} train edges "
            f"({_metric(sampling['inference_context']['coverage_fraction'])}).",
            f"- Supervised train sample: {sampling['supervised_training']['sampled_rows']:,} "
            f"transactions, including all "
            f"{sampling['supervised_training']['sampled_positives']:,} positives after the "
            "message-graph cutoff and a deterministic negative sample.",
            "- Context sampling is target-agnostic and ordered by stable MD5 transaction-ID "
            "digest. This is not described as full-graph GraphSAGE training.",
            "- Node structural features use directed degrees, not cross-currency monetary "
            "sums; no FX conversion table is available.",
            "",
            "## Graph construction and leakage boundary",
            "",
            f"- Training message graph maximum timestamp: "
            f"`{sampling['training_context']['maximum_timestamp']}`",
            f"- Supervised edge minimum timestamp: "
            f"`{sampling['supervised_training']['minimum_timestamp']}`",
            "- Outer-validation embeddings use only sampled outer-train edges; no validation "
            "edge enters message passing.",
            "- Directed endpoints and repeated transactions are retained; repeated edges act "
            "as frequency weight in neighbor means.",
            "",
            "## Cases, evidence, and explanations",
            "",
            f"- Saved cases: {cases['count']}",
            f"- Minimum observed evidence per case: {cases['minimum_observed_evidence']}",
            f"- Deterministic no-LLM notes: {cases['deterministic_fallback_notes']}",
            "- Observed evidence and model evidence are stored separately.",
            "- LightGBM explanations use native `pred_contrib` TreeSHAP with an additivity "
            "check. GraphSAGE explanations are labeled local gradient-times-input sensitivity, not "
            "SHAP or causal attribution.",
            "",
            "## Streamlit application",
            "",
            "- Screens: Executive Dashboard, Investigation Queue, Case Investigator, Model "
            "Comparison.",
            "- Page-load training: **DISABLED**; the app reads saved Sprint 4 artifacts only.",
            "- Optional LLM unavailable path: deterministic fallback is complete and tested.",
            "- GraphSAGE sigmoid values are uncalibrated ranking scores because training uses "
            "sampled negatives and positive weighting; they are not event probabilities.",
            "",
            "## Runtime",
            "",
        ]
    )
    for name, seconds in sorted(payload["runtime_seconds_by_stage"].items()):
        lines.append(f"- `{name}`: {float(seconds):,.3f} seconds")
    lines.append(f"- `total`: {float(payload['runtime_seconds']):,.3f} seconds")
    lines.extend(["", "## Acceptance checklist", ""])
    for name, passed in sorted(payload["acceptance"].items()):
        lines.append(f"- **{'PASS' if passed else 'FAIL'}** - {name}")
    if quality:
        lines.extend(
            [
                "",
                "## Automated quality evidence",
                "",
                f"- Status: **{quality.get('status', 'PENDING')}**",
                f"- Pytest passed: {quality.get('pytest_passed', 'PENDING')}",
                "- Artifact verification: "
                f"{quality.get('artifact_verification_status', 'PENDING')}",
            ]
        )
        for name, result in sorted(quality.get("checks", {}).items()):
            lines.append(f"- `{name}`: **{result.get('status', 'PENDING')}**")
    lines.extend(
        [
            "",
            "## Stop boundary",
            "",
            "Sprint 4 stops before final-test opening. Test features, labels, predictions, and "
            "metrics remain sealed.",
            "",
        ]
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = ["render_sprint4_report"]
