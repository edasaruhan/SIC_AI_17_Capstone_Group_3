from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from argus.app.artifacts import ArtifactLoadError, load_dashboard_artifacts


def _case(case_id: str = "ARG-0001") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "transaction_id": "TX-1",
        "status": "pending_human_review",
        "priority": {"basis": "model", "band": "high", "threshold_exceeded": True},
        "observed_evidence": [
            {
                "basis": "observed",
                "evidence_id": f"EV-{index}",
                "kind": "fan_out",
                "statement": f"Observed fact {index}",
                "source_fields": ["sender_prior_fan_out_degree"],
                "scope": "strict-prior history",
            }
            for index in range(1, 4)
        ],
        "model_evidence": {
            "basis": "model",
            "model_name": "graphsage_edge_classifier",
            "score_name": "probability",
            "score": 0.91,
            "threshold": 0.8,
            "threshold_exceeded": True,
            "rank": 1,
            "feature_contributions": [{"feature": "sender_embedding_0", "contribution": 0.13}],
        },
        "guardrails": {
            "human_review_required": True,
            "automatic_action_authorized": False,
        },
    }


def _write_required_artifacts(root: Path, *, test_metrics_used: bool = False) -> None:
    product = root / "product"
    product.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case_id": "ARG-0001",
                "priority_band": "high",
                "priority_score": 0.91,
                "accounts": 4,
                "transactions": 7,
                "total_amount": 1250.0,
                "pattern": "fan-out",
                "observed_evidence_count": 3,
            }
        ]
    ).to_csv(product / "investigation_queue.csv", index=False)
    (product / "cases.json").write_text(
        json.dumps({"schema_version": "1.0", "cases": [_case()]}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "model_name": "graph-enhanced-lightgbm",
                "stage": "Graph-enhanced",
                "average_precision": 0.47,
                "false_positive_rate": 0.006,
                "recall_at_1000": 0.4,
                "precision_at_1000": 0.3,
                "alert_count": 5000,
                "evaluation_partition": "validation",
                "test_metrics_used": test_metrics_used,
                "validation_row_count": 761749,
            }
        ]
    ).to_csv(product / "model_comparison.csv", index=False)


def test_loads_separate_artifacts_and_normalises_repository_aliases(tmp_path: Path) -> None:
    _write_required_artifacts(tmp_path)
    product = tmp_path / "product"
    pd.DataFrame([{"model_name": "GraphSAGE", "pr_recall": 0.5, "pr_precision": 0.2}]).to_csv(
        product / "pr_curves.csv", index=False
    )
    pd.DataFrame(
        [{"model_name": "GraphSAGE", "requested_k": 100, "recall": 0.3, "precision": 0.2}]
    ).to_csv(product / "top_k_metrics.csv", index=False)
    pd.DataFrame([{"family": "graph", "average_precision": 0.47}]).to_csv(
        product / "feature_family_ablation.csv", index=False
    )

    artifacts = load_dashboard_artifacts(tmp_path)

    assert artifacts.queue.loc[0, "risk_score"] == pytest.approx(0.91)
    assert artifacts.queue.loc[0, "account_count"] == 4
    assert artifacts.model_comparison.loc[0, "pr_auc"] == pytest.approx(0.47)
    assert artifacts.pr_curves.columns.tolist() == ["model", "recall", "precision"]
    assert artifacts.top_k.loc[0, "k"] == 100
    assert artifacts.ablation.loc[0, "feature_family"] == "graph"
    assert artifacts.summary == {
        "flagged_transactions": 1,
        "high_priority_cases": 1,
        "transactions_analyzed": 761749,
    }
    assert len(artifacts.cases["ARG-0001"]["observed_evidence"]) == 3


def test_loads_single_dashboard_bundle(tmp_path: Path) -> None:
    bundle = {
        "summary": {"transactions_analyzed": 100, "final_test_opened": False},
        "queue": [{"case_id": "ARG-0001", "risk_score": 0.91}],
        "cases": [_case()],
        "model_comparison": [
            {
                "model": "GraphSAGE",
                "pr_auc": 0.2,
                "evaluation_partition": "validation",
                "test_metrics_used": False,
            }
        ],
    }
    (tmp_path / "dashboard_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")

    artifacts = load_dashboard_artifacts(tmp_path)

    assert artifacts.summary["transactions_analyzed"] == 100
    assert artifacts.provenance["bundle"].endswith("dashboard_bundle.json")


def test_missing_artifacts_report_expected_paths(tmp_path: Path) -> None:
    with pytest.raises(
        ArtifactLoadError, match="Missing required investigation queue artifact"
    ) as exc:
        load_dashboard_artifacts(tmp_path)

    assert "product" in str(exc.value)
    assert "investigation_queue.csv" in str(exc.value)


def test_nonexistent_root_explains_offline_pipeline_requirement(tmp_path: Path) -> None:
    with pytest.raises(ArtifactLoadError, match="offline product artifact pipeline"):
        load_dashboard_artifacts(tmp_path / "absent")


def test_queue_may_not_reference_an_absent_case(tmp_path: Path) -> None:
    _write_required_artifacts(tmp_path)
    cases_path = tmp_path / "product" / "cases.json"
    cases_path.write_text(json.dumps({"cases": [_case("ARG-9999")]}), encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match="ARG-0001"):
        load_dashboard_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("partition", "test_metrics_used", "expected"),
    [("test", False, "final-test metrics"), ("validation", True, "test_metrics_used=true")],
)
def test_sprint4_loader_rejects_opened_test_metrics(
    tmp_path: Path, partition: str, test_metrics_used: bool, expected: str
) -> None:
    _write_required_artifacts(tmp_path, test_metrics_used=test_metrics_used)
    comparison_path = tmp_path / "product" / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    comparison["evaluation_partition"] = partition
    comparison.to_csv(comparison_path, index=False)

    with pytest.raises(ArtifactLoadError, match=expected):
        load_dashboard_artifacts(tmp_path)


def test_case_evidence_requires_a_concrete_statement(tmp_path: Path) -> None:
    _write_required_artifacts(tmp_path)
    case = _case()
    case["observed_evidence"][0]["statement"] = ""  # type: ignore[index]
    (tmp_path / "product" / "cases.json").write_text(
        json.dumps({"cases": [case]}), encoding="utf-8"
    )

    with pytest.raises(ArtifactLoadError, match="needs a concrete statement"):
        load_dashboard_artifacts(tmp_path)
