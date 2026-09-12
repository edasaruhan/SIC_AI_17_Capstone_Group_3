from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from argus.app.dashboard import _metric_value, configured_artifact_root
from argus.app.portal import _selected_dataframe_rows


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _login(app: AppTest) -> AppTest:
    _button(app, "Corporate Login").click()
    app.run(timeout=20)
    app.text_input[0].set_value("analyst@bank.example")
    app.text_input[1].set_value("prototype-access")
    _button(app, "Sign in").click()
    return app.run(timeout=20)


def _write_bundle(root: Path) -> None:
    case = {
        "case_id": "ARG-0001",
        "transaction_id": "TX-1",
        "status": "pending_human_review",
        "priority": {"band": "high", "basis": "model"},
        "sender_id": "BANK-A::A1",
        "receiver_id": "BANK-B::B1",
        "total_flow": 100.0,
        "observed_evidence": [
            {"kind": "observed_fact", "statement": f"Observed evidence {index}"}
            for index in range(1, 4)
        ],
        "model_evidence": {
            "model_name": "GraphSAGE",
            "score": 0.9,
            "threshold": 0.7,
            "rank": 1,
            "feature_contributions": [{"feature": "amount", "contribution": 0.2}],
        },
        "analyst_note": {
            "text": "Deterministic evidence-only note. Human review is required.",
            "mode": "deterministic_template",
            "fallback_reason": "LLM not configured",
        },
        "transactions": [
            {
                "transaction_id": "TX-1",
                "from_node_id": "BANK-A::A1",
                "to_node_id": "BANK-B::B1",
                "timestamp": "2022-09-01T00:00:00Z",
                "amount": 100.0,
                "suspicious": True,
            }
        ],
    }
    bundle = {
        "summary": {"transactions_analyzed": 10, "final_test_opened": False},
        "queue": [
            {
                "case_id": "ARG-0001",
                "priority": "high",
                "risk_score": 0.9,
                "account_count": 2,
                "transaction_count": 1,
                "total_flow": 100.0,
                "major_pattern": "directed transfer",
                "evidence_count": 3,
            }
        ],
        "cases": [case],
        "model_comparison": [
            {
                "model": "GraphSAGE",
                "version": "GraphSAGE subset",
                "pr_auc": 0.2,
                "recall_at_k": 0.3,
                "precision_at_k": 0.1,
                "fpr": 0.01,
                "alerts": 5,
                "evaluation_partition": "validation",
                "test_metrics_used": False,
            }
        ],
        "pr_curves": [
            {"model": "GraphSAGE", "recall": 0.0, "precision": 1.0},
            {"model": "GraphSAGE", "recall": 1.0, "precision": 0.01},
        ],
        "top_k": [{"model": "GraphSAGE", "k": 5, "recall_at_k": 0.3, "precision_at_k": 0.1}],
        "ablation": [{"feature_family": "embeddings + transaction", "pr_auc": 0.2}],
    }
    (root / "dashboard_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")


def test_streamlit_all_required_screens_render_from_saved_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    _write_bundle(tmp_path)
    monkeypatch.setenv("ARGUS_SPRINT4_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("ARGUS_DEMO_EMAIL", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_PASSWORD", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not app.exception
    assert _button(app, "Corporate Login")

    app = _login(app)
    assert not app.exception
    assert app.title[0].value == "Overview"
    portal_markup = " ".join(str(item.value) for item in app.markdown)
    assert 'class="argus-skip-link" href="#argus-main-content"' in portal_markup
    assert 'id="argus-main-content"' in portal_markup
    assert all(metric.label != "GraphSAGE queue alerts" for metric in app.metric)
    for label in ("Open cases", "In review", "Prior context", "Analyst actions"):
        assert label in portal_markup

    for page in ("Investigations", "Case Investigator", "Model Evidence"):
        app.sidebar.radio[0].set_value(page)
        app.run(timeout=20)
        assert not app.exception
        assert app.title[0].value == page

    app.sidebar.radio[0].set_value("Investigations")
    app.run(timeout=20)
    _button(app, "Open case").click()
    app.run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Case Investigator"
    assert app.sidebar.radio[0].value == "Case Investigator"

    _button(app, "Back to investigations").click()
    app.run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Investigations"
    assert app.sidebar.radio[0].value == "Investigations"

    _button(app, "Log out").click()
    app.run(timeout=20)
    assert not app.exception
    assert _button(app, "Corporate Login")


def test_streamlit_missing_artifacts_shows_actionable_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_SPRINT4_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("ARGUS_DEMO_EMAIL", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_PASSWORD", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not app.exception
    assert _button(app, "Corporate Login")

    app = _login(app)

    assert not app.exception
    assert app.title[0].value == "Investigation workspace unavailable"
    assert app.error[0].value == (
        "The saved investigation package could not be loaded in this environment."
    )
    visible_text = " ".join(
        str(element.value)
        for group in (app.error, app.info, app.markdown, app.caption)
        for element in group
    )
    assert str(tmp_path) not in visible_text
    assert "investigation_queue.csv" not in visible_text


def test_clean_clone_uses_tracked_synthetic_demo(monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("ARGUS_SPRINT4_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_EMAIL", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_PASSWORD", raising=False)

    root = configured_artifact_root()
    assert root.parts[-2:] == ("demo", "artifacts")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = _login(AppTest.from_file(str(app_path)).run(timeout=20))

    assert not app.exception
    assert app.title[0].value == "Overview"
    overview = " ".join(str(item.value) for item in app.markdown)
    assert "Illustrative synthetic demo" in overview

    app.sidebar.radio[0].set_value("Investigations")
    app.run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Investigations"
    assert any("Illustrative demo case set" in item.value for item in app.markdown)

    app.sidebar.radio[0].set_value("Case Investigator")
    app.run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Case Investigator"
    case_copy = " ".join(str(item.value) for item in app.markdown)
    assert "Demo priority signal" in case_copy
    assert "Not a probability of wrongdoing" in case_copy


def test_portal_guides_review_workflow_and_explains_metrics(monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("ARGUS_SPRINT4_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_EMAIL", raising=False)
    monkeypatch.delenv("ARGUS_DEMO_PASSWORD", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = _login(AppTest.from_file(str(app_path)).run(timeout=20))

    overview = " ".join(str(item.value) for item in app.markdown)
    assert "NEXT BEST ACTION" in overview
    assert "Awaiting an analyst decision" in overview
    _button(app, "Review next case").click()
    app.run(timeout=20)
    assert app.title[0].value == "Case Investigator"
    assert app.selectbox[0].value == "ARG-DEMO-001"

    app.sidebar.radio[0].set_value("Investigations")
    app.run(timeout=20)
    investigations = " ".join(str(item.value) for item in app.markdown)
    assert 'aria-label="Investigation workflow"' in investigations
    assert "Choose the case that needs attention." in investigations
    assert "SELECTED CASE" in investigations
    assert [control.value for control in app.multiselect] == [[], [], []]

    app.sidebar.radio[0].set_value("Case Investigator")
    app.run(timeout=20)
    case_copy = " ".join(str(item.value) for item in app.markdown)
    assert "CURRENT TASK" in case_copy
    assert "Why the case entered the worklist" in case_copy
    assert "DECISION & DOCUMENTATION" in case_copy
    _button(app, "Start review").click()
    app.run(timeout=20)
    updated_case_copy = " ".join(str(item.value) for item in app.markdown)
    assert "Review status: In review" in updated_case_copy
    assert any(button.label == "Escalate" for button in app.button)

    app.sidebar.radio[0].set_value("Model Evidence")
    app.run(timeout=20)
    model_copy = " ".join(str(item.value) for item in app.markdown)
    assert "Performance in plain language" in model_copy
    assert "Higher is better; compare models on the same data" in model_copy
    assert "Alert usefulness" in model_copy
    for label in (
        "Primary model",
        "Ranking quality",
        "Positive cases found",
        "Useful reviewed alerts",
    ):
        assert label in model_copy


def test_executive_metrics_do_not_mix_queue_and_validation_leader_models() -> None:
    summary = {
        "validation_pr_auc": 0.9,
        "validation_recall_at_k": 0.8,
        "validation_fpr": 0.1,
        "alert_volume": 111,
        "flagged_transactions": 222,
    }
    validation_leader = pd.Series({"pr_auc": 0.7, "recall_at_k": 0.6, "fpr": 0.05, "alerts": 333})

    assert _metric_value(summary, validation_leader, "pr_auc") == 0.7
    assert _metric_value(summary, validation_leader, "recall_at_k") == 0.6
    assert _metric_value(summary, validation_leader, "fpr") == 0.05
    assert _metric_value(summary, validation_leader, "alerts") == 333


def test_dataframe_selection_rows_are_read_defensively() -> None:
    assert _selected_dataframe_rows({"selection": {"rows": [2]}}) == (2,)
    assert _selected_dataframe_rows({"selection": {"rows": [-1, "bad", 1]}}) == (1,)
    assert _selected_dataframe_rows({"selection": {"rows": None}}) == ()
    assert _selected_dataframe_rows({}) == ()
