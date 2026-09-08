from __future__ import annotations

from argus.final_evaluation.reporting import render_final_comparison_markdown


def _comparison() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (name, role) in enumerate(
        (
            ("graph_enhanced_lightgbm", "frozen_champion"),
            ("refined_transaction_lightgbm", "comparator"),
            ("graphsage_edge_classifier", "comparator"),
        )
    ):
        rows.append(
            {
                "model": name,
                "role": role,
                "average_precision": 0.4 - index * 0.1,
                "roc_auc": 0.9 - index * 0.1,
                "precision": 0.3 - index * 0.05,
                "recall": 0.6 - index * 0.1,
                "f1": 0.4 - index * 0.05,
                "false_positive_rate": 0.01 + index * 0.01,
                "alert_count": 5 + index,
            }
        )
    return rows


def _top_k() -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "requested_k": k,
            "precision_at_k": 0.1,
            "recall_at_k": 0.2,
            "true_positives": 1,
        }
        for model in (
            "graph_enhanced_lightgbm",
            "refined_transaction_lightgbm",
            "graphsage_edge_classifier",
        )
        for k in (100, 500, 1000)
    ]


def test_final_comparison_report_is_artifact_driven_and_labels_frozen_role() -> None:
    rendered = render_final_comparison_markdown(
        _comparison(),
        _top_k(),
        {
            "validation": {"rows": 10, "positives": 1, "positive_rate": 0.1},
            "test": {"rows": 10, "positives": 2, "positive_rate": 0.2},
            "test_to_validation_positive_rate_ratio": 2.0,
            "interpretation": "Synthetic fixture shift.",
        },
        quality={
            "status": "PASS",
            "pytest_passed": 123,
            "artifact_verification_status": "PASS",
        },
    )
    assert "Frozen champion | Graph-enhanced LightGBM | 0.40000000" in rendered
    assert "one-shot final-test artifacts" in rendered
    assert "not a test-driven ranking or selection" in rendered
    assert "Test/validation positive-rate ratio: 2.00000000" in rendered
    assert "123 passed" in rendered
    assert "final_test_predictions.parquet" in rendered


def test_final_comparison_report_rejects_incomplete_model_set() -> None:
    comparison = _comparison()[:-1]
    try:
        render_final_comparison_markdown(
            comparison,
            _top_k(),
            {
                "validation": {"rows": 1, "positives": 1, "positive_rate": 1.0},
                "test": {"rows": 1, "positives": 1, "positive_rate": 1.0},
                "test_to_validation_positive_rate_ratio": 1.0,
                "interpretation": "fixture",
            },
        )
    except RuntimeError as exc:
        assert "three frozen models" in str(exc)
    else:
        raise AssertionError("Incomplete final comparison was accepted")
