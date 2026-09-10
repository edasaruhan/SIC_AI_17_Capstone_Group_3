from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from argus.app.artifacts import ArtifactLoadError, load_dashboard_artifacts
from argus.app.dashboard import _comparison_focus


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _login(app: AppTest) -> AppTest:
    _button(app, "Corporate Login").click()
    app.run(timeout=20)
    app.text_input[0].set_value("analyst@bank.example")
    app.text_input[1].set_value("prototype-access")
    _button(app, "Sign in").click()
    return app.run(timeout=20)


def _case() -> dict[str, object]:
    return {
        "case_id": "ARG-0001",
        "transaction_id": "TX-1",
        "observed_evidence": [
            {"kind": "observed", "statement": f"Observed evidence {index}"} for index in range(1, 4)
        ],
        "model_evidence": {"model_name": "graphsage_edge_classifier", "score": 0.8},
    }


def _write_validation_product(root: Path) -> None:
    product = root / "product"
    product.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case_id": "ARG-0001",
                "priority_band": "high",
                "priority_score": 0.8,
                "observed_evidence_count": 3,
            }
        ]
    ).to_csv(product / "investigation_queue.csv", index=False)
    (product / "cases.json").write_text(json.dumps({"cases": [_case()]}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "model_name": "graph_enhanced_lightgbm",
                "average_precision": 0.47,
                "recall_at_1000": 0.50,
                "precision_at_1000": 0.38,
                "false_positive_rate": 0.006,
                "alert_count": 50,
                "evaluation_partition": "validation",
                "test_metrics_used": False,
                "is_champion": True,
                "validation_row_count": 100,
            },
            {
                "model_name": "graphsage_edge_classifier",
                "average_precision": 0.01,
                "recall_at_1000": 0.03,
                "precision_at_1000": 0.02,
                "false_positive_rate": 0.007,
                "alert_count": 40,
                "evaluation_partition": "validation",
                "test_metrics_used": False,
                "is_champion": False,
                "validation_row_count": 100,
            },
        ]
    ).to_csv(product / "model_comparison.csv", index=False)
    (product / "dashboard_summary.json").write_text(
        json.dumps(
            {
                "partition": "validation",
                "final_test_opened": False,
                "transactions_analyzed": 100,
            }
        ),
        encoding="utf-8",
    )


def _final_rows() -> list[dict[str, object]]:
    rows = []
    for model, pr_auc, champion in (
        ("graph_enhanced_lightgbm", 0.41, True),
        ("refined_transaction_lightgbm", 0.99, False),
        ("graphsage_edge_classifier", 0.02, False),
    ):
        rows.append(
            {
                "model": model,
                "version": "frozen comparator",
                "pr_auc": pr_auc,
                "roc_auc": 0.8,
                "precision": 0.25,
                "recall": 0.5,
                "f1": 1 / 3,
                "fpr": 0.375,
                "recall_at_k": 0.5,
                "precision_at_k": 0.2,
                "alerts": 4,
                "row_count": 10,
                "positive_count": 2,
                "true_positive": 1,
                "false_positive": 3,
                "false_negative": 1,
                "true_negative": 5,
                "evaluation_partition": "test",
                "is_frozen_champion": champion,
                "model_selection_used_test": False,
                "threshold_tuned_on_test": False,
                "feature_selection_used_test": False,
                "retrained_after_test": False,
            }
        )
    return rows


def _final_summary() -> dict[str, object]:
    return {
        "evaluation_partition": "test",
        "final_test_opened": True,
        "one_shot_final_evaluation": True,
        "champion_frozen_before_test": True,
        "test_used_for_model_selection": False,
        "tuning_after_test": False,
        "final_test_access_count": 1,
        "frozen_champion_model": "graph_enhanced_lightgbm",
        "test_row_count": 10,
        "test_positive_count": 2,
        "test_positive_rate": 0.2,
        "validation_positive_rate": 0.1,
    }


def _write_final_product(root: Path) -> None:
    product = root / "product"
    product.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_final_rows()).to_csv(product / "final_model_comparison.csv", index=False)
    (product / "final_test_summary.json").write_text(json.dumps(_final_summary()), encoding="utf-8")
    curves = []
    top_k = []
    for row in _final_rows():
        model = str(row["model"])
        curves.extend(
            [
                {
                    "model": model,
                    "recall": 0.0,
                    "precision": 1.0,
                    "evaluation_partition": "test",
                },
                {
                    "model": model,
                    "recall": 1.0,
                    "precision": 0.2,
                    "evaluation_partition": "test",
                },
            ]
        )
        top_k.append(
            {
                "model": model,
                "k": 5,
                "recall_at_k": 0.5,
                "precision_at_k": 0.2,
                "evaluation_partition": "test",
            }
        )
    pd.DataFrame(curves).to_csv(product / "final_pr_curves.csv", index=False)
    pd.DataFrame(top_k).to_csv(product / "final_top_k_metrics.csv", index=False)


@pytest.fixture
def sprint5_root(tmp_path: Path) -> Path:
    sprint4 = tmp_path / "sprint4"
    sprint5 = tmp_path / "sprint5"
    _write_validation_product(sprint4)
    _write_final_product(sprint5)
    reference = sprint5 / "product" / "validation_artifact_reference.json"
    reference.write_text(json.dumps({"validation_artifact_root": "../sprint4"}), encoding="utf-8")
    return sprint5


def test_sprint5_loader_keeps_validation_product_and_final_payload_separate(
    sprint5_root: Path,
) -> None:
    artifacts = load_dashboard_artifacts(sprint5_root)

    assert artifacts.root == sprint5_root.resolve()
    assert artifacts.provenance["validation_artifact_root"].endswith("sprint4")
    assert artifacts.queue["case_id"].tolist() == ["ARG-0001"]
    assert artifacts.model_comparison["evaluation_partition"].eq("validation").all()
    assert artifacts.final_evaluation is not None
    assert artifacts.final_evaluation.model_comparison["evaluation_partition"].eq("test").all()
    assert artifacts.final_evaluation.frozen_champion == "graph_enhanced_lightgbm"


def test_final_rows_never_feed_validation_leader_logic(sprint5_root: Path) -> None:
    artifacts = load_dashboard_artifacts(sprint5_root)
    assert artifacts.final_evaluation is not None
    assert artifacts.final_evaluation.model_comparison["pr_auc"].max() == pytest.approx(0.99)

    focus = _comparison_focus(artifacts)

    assert focus["model"] == "graph_enhanced_lightgbm"
    assert focus["pr_auc"] == pytest.approx(0.47)


def test_loader_accepts_pipeline_fields_and_derives_display_top_k(sprint5_root: Path) -> None:
    path = sprint5_root / "product" / "final_model_comparison.csv"
    frame = pd.read_csv(path)
    frame["role"] = frame["is_frozen_champion"].map(
        lambda selected: "frozen_champion" if selected else "comparator"
    )
    frame["champion_frozen_before_test"] = frame.pop("is_frozen_champion")
    frame["false_positive_rate"] = frame.pop("fpr")
    frame["alert_count"] = frame.pop("alerts")
    frame.drop(columns=["recall_at_k", "precision_at_k"], inplace=True)
    frame.to_csv(path, index=False)

    artifacts = load_dashboard_artifacts(sprint5_root)

    assert artifacts.final_evaluation is not None
    comparison = artifacts.final_evaluation.model_comparison
    assert comparison["is_frozen_champion"].sum() == 1
    assert comparison["recall_at_k"].eq(0.5).all()
    assert comparison["precision_at_k"].eq(0.2).all()
    assert comparison["top_k_for_display"].eq(5).all()


@pytest.mark.parametrize(
    ("artifact_name", "replacement", "expected"),
    [
        ("final_test_summary.json", {"evaluation_partition": "validation"}, "partition"),
        ("final_model_comparison.csv", {"evaluation_partition": "validation"}, "partition"),
        ("final_pr_curves.csv", {"evaluation_partition": "validation"}, "partition"),
        ("final_top_k_metrics.csv", {"evaluation_partition": "validation"}, "partition"),
    ],
)
def test_rejects_wrong_partition_in_every_final_artifact(
    sprint5_root: Path,
    artifact_name: str,
    replacement: dict[str, object],
    expected: str,
) -> None:
    path = sprint5_root / "product" / artifact_name
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(replacement)
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        frame = pd.read_csv(path)
        for column, value in replacement.items():
            frame[column] = value
        frame.to_csv(path, index=False)

    with pytest.raises(ArtifactLoadError, match=expected):
        load_dashboard_artifacts(sprint5_root)


def test_rejects_duplicate_final_models(sprint5_root: Path) -> None:
    path = sprint5_root / "product" / "final_model_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "model"] = "graph-enhanced-lightgbm"
    frame.to_csv(path, index=False)

    with pytest.raises(ArtifactLoadError, match="duplicate models"):
        load_dashboard_artifacts(sprint5_root)


@pytest.mark.parametrize(
    ("artifact_name", "column", "value", "expected"),
    [
        ("final_model_comparison.csv", "pr_auc", float("nan"), "finite"),
        ("final_model_comparison.csv", "fpr", 1.01, r"\[0, 1\]"),
        ("final_pr_curves.csv", "precision", float("inf"), "finite"),
        ("final_top_k_metrics.csv", "recall_at_k", -0.01, r"\[0, 1\]"),
    ],
)
def test_rejects_nonfinite_and_out_of_range_final_metrics(
    sprint5_root: Path,
    artifact_name: str,
    column: str,
    value: float,
    expected: str,
) -> None:
    path = sprint5_root / "product" / artifact_name
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)

    with pytest.raises(ArtifactLoadError, match=expected):
        load_dashboard_artifacts(sprint5_root)


@pytest.mark.parametrize(
    ("artifact_name", "column", "value", "expected"),
    [
        ("final_model_comparison.csv", "alerts", 11, "counts"),
        ("final_model_comparison.csv", "row_count", 9, "inconsistent|disagrees"),
        ("final_top_k_metrics.csv", "k", 11, "may not exceed"),
    ],
)
def test_rejects_impossible_or_inconsistent_final_counts(
    sprint5_root: Path,
    artifact_name: str,
    column: str,
    value: int,
    expected: str,
) -> None:
    path = sprint5_root / "product" / artifact_name
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)

    with pytest.raises(ArtifactLoadError, match=expected):
        load_dashboard_artifacts(sprint5_root)


def test_rejects_inconsistent_prevalence_count(sprint5_root: Path) -> None:
    path = sprint5_root / "product" / "final_test_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["test_positive_rate"] = 0.3
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match="does not match"):
        load_dashboard_artifacts(sprint5_root)


def test_rejects_metric_that_disagrees_with_confusion_counts(sprint5_root: Path) -> None:
    path = sprint5_root / "product" / "final_model_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "precision"] = 0.5
    frame.to_csv(path, index=False)

    with pytest.raises(ArtifactLoadError, match="precision disagrees"):
        load_dashboard_artifacts(sprint5_root)


@pytest.mark.parametrize(
    ("flag", "invalid_value", "expected"),
    [
        ("final_test_opened", False, "final_test_opened"),
        ("one_shot_final_evaluation", False, "one_shot_final_evaluation"),
        ("champion_frozen_before_test", False, "champion_frozen_before_test"),
        ("test_used_for_model_selection", True, "test_used_for_model_selection"),
        ("tuning_after_test", True, "tuning_after_test"),
        ("final_test_access_count", 2, "exactly 1"),
    ],
)
def test_rejects_broken_final_protocol_contract_flags(
    sprint5_root: Path,
    flag: str,
    invalid_value: object,
    expected: str,
) -> None:
    path = sprint5_root / "product" / "final_test_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary[flag] = invalid_value
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match=expected):
        load_dashboard_artifacts(sprint5_root)


def test_rejects_model_row_that_reports_test_informed_tuning(sprint5_root: Path) -> None:
    path = sprint5_root / "product" / "final_model_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "threshold_tuned_on_test"] = True
    frame.to_csv(path, index=False)

    with pytest.raises(ArtifactLoadError, match="forbidden threshold_tuned_on_test=true"):
        load_dashboard_artifacts(sprint5_root)


def test_streamlit_modules_have_presentation_only_argus_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sources = [project_root / "app.py"]
    sources.extend(sorted((project_root / "src" / "argus" / "app").glob("*.py")))
    imported_argus_modules: set[str] = set()
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_argus_modules.update(
                    alias.name for alias in node.names if alias.name.startswith("argus")
                )
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("argus"):
                imported_argus_modules.add(node.module or "")

    assert imported_argus_modules
    assert all(name.startswith("argus.app") for name in imported_argus_modules)


def test_streamlit_labels_final_test_as_frozen_reporting_only(
    sprint5_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARGUS_ARTIFACT_DIR", str(sprint5_root))
    monkeypatch.delenv("ARGUS_SPRINT4_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_EMAIL", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_PASSWORD", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not app.exception
    app = _login(app)
    assert not app.exception
    assert app.title[0].value == "Overview"
    assert app.sidebar.radio[0].options == [
        "Overview",
        "Investigations",
        "Case Investigator",
        "Model Evidence",
    ]
    case_provenance = "Graph-enhanced LightGBM is the primary ranking model"
    assert not any(case_provenance in item.value for item in app.info)

    app.sidebar.radio[0].set_value("Investigations")
    app.run(timeout=20)
    assert not app.exception
    investigations_copy = " ".join(str(item.value) for item in app.markdown)
    assert "Research case set" in investigations_copy
    assert "Graph-enhanced LightGBM remains the primary model" in investigations_copy
    assert not any(case_provenance in item.value for item in app.info)
    assert all("uncalibrated_ranking_score" not in str(item.value) for item in app.dataframe)

    app.sidebar.radio[0].set_value("Case Investigator")
    app.run(timeout=20)
    assert not app.exception
    case_copy = " ".join(str(item.value) for group in (app.markdown, app.caption) for item in group)
    assert "Research case selection" in case_copy
    assert "Selected by GraphSAGE research comparator." in case_copy
    assert "Graph-enhanced LightGBM remains the primary model." in case_copy
    assert not any(case_provenance in item.value for item in app.info)

    app.sidebar.radio[0].set_value("Model Evidence")
    app.run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Model Evidence"
    assert any("Frozen primary model" in item.value for item in app.markdown)
    assert any("Graph-enhanced LightGBM" in item.value for item in app.markdown)
    assert any("final test was used once" in item.value.lower() for item in app.warning)
