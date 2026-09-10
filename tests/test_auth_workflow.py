from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from argus.app.auth import (
    ANALYST_EMAIL_KEY,
    AUTH_MODE_KEY,
    AUTHENTICATED_KEY,
    ROUTE_KEY,
    SELECTED_CASE_KEY,
    corporate_email_error,
    initialize_auth_session,
    is_authenticated,
    normalize_email,
    password_error,
    sign_in,
    sign_out,
    validate_credentials,
)
from argus.app.workflow import (
    ADD_NOTE,
    CLOSE_CASE,
    ESCALATE,
    MARK_FALSE_POSITIVE,
    START_REVIEW,
    WORKFLOW_SESSION_KEY,
    apply_case_action,
    case_with_workflow,
    case_workflow,
    clear_workflow_session,
    initialize_workflow_session,
    status_label,
)


@pytest.mark.parametrize(
    "email",
    ["", "analyst", "analyst@", "@bank.example", "analyst@gmail.com", "a b@bank.example"],
)
def test_corporate_email_validation_rejects_missing_malformed_or_personal_email(email: str) -> None:
    assert corporate_email_error(email) is not None


def test_corporate_email_validation_normalizes_a_work_address() -> None:
    assert normalize_email("  Analyst@Bank.Example ") == "analyst@bank.example"
    assert corporate_email_error("analyst@bank.example") is None


def test_password_validation_checks_prototype_form_completeness() -> None:
    assert password_error("") == "Enter your password."
    assert password_error("short") is not None
    assert password_error("long-enough") is None


def test_unconfigured_demo_auth_accepts_complete_form_without_claiming_real_auth() -> None:
    result = validate_credentials("analyst@bank.example", "long-enough", environment={})

    assert result.accepted is True
    assert result.mode == "prototype_form_demo"
    assert result.email == "analyst@bank.example"
    assert "No production identity provider" in result.message


@pytest.mark.parametrize(
    ("environment", "email", "password"),
    [
        ({"ARGUS_DEMO_EMAIL": "analyst@bank.example"}, "analyst@bank.example", "password1"),
        ({"ARGUS_DEMO_PASSWORD": "password1"}, "analyst@bank.example", "password1"),
        (
            {"ARGUS_DEMO_EMAIL": "", "ARGUS_DEMO_PASSWORD": "password1"},
            "analyst@bank.example",
            "password1",
        ),
    ],
)
def test_partial_environment_credential_configuration_fails_closed(
    environment: dict[str, str], email: str, password: str
) -> None:
    result = validate_credentials(email, password, environment=environment)

    assert result.accepted is False
    assert result.mode == "configuration_error"
    assert "ARGUS_DEMO" not in result.message


def test_configured_demo_credentials_require_an_exact_password_match() -> None:
    environment = {
        "ARGUS_DEMO_EMAIL": "analyst@bank.example",
        "ARGUS_DEMO_PASSWORD": "environment-only-password",
    }

    rejected = validate_credentials(
        "analyst@bank.example", "different-password", environment=environment
    )
    accepted = validate_credentials(
        " Analyst@Bank.Example ", "environment-only-password", environment=environment
    )

    assert rejected.accepted is False
    assert rejected.mode == "configured_demo"
    assert accepted.accepted is True
    assert accepted.mode == "configured_demo"


def test_sign_in_persists_no_password_and_logout_resets_portal_workflow() -> None:
    state: dict[str, object] = {}
    initialize_auth_session(state)
    assert is_authenticated(state) is False

    result = sign_in(state, "analyst@bank.example", "long-enough", environment={})
    state[SELECTED_CASE_KEY] = "ARGUS-1"
    apply_case_action(state, "ARGUS-1", START_REVIEW)

    assert result.accepted is True
    assert state[AUTHENTICATED_KEY] is True
    assert state[ANALYST_EMAIL_KEY] == "analyst@bank.example"
    assert state[AUTH_MODE_KEY] == "prototype_form_demo"
    assert state[ROUTE_KEY] == "portal"
    assert all("password" not in key.casefold() for key in state)

    sign_out(state)

    assert is_authenticated(state) is False
    assert state[ROUTE_KEY] == "public"
    assert ANALYST_EMAIL_KEY not in state
    assert AUTH_MODE_KEY not in state
    assert SELECTED_CASE_KEY not in state
    assert WORKFLOW_SESSION_KEY not in state


def test_failed_sign_in_clears_stale_authenticated_identity() -> None:
    state: dict[str, object] = {
        AUTHENTICATED_KEY: True,
        ANALYST_EMAIL_KEY: "old@bank.example",
        AUTH_MODE_KEY: "configured_demo",
    }
    environment = {
        "ARGUS_DEMO_EMAIL": "new@bank.example",
        "ARGUS_DEMO_PASSWORD": "environment-only-password",
    }

    result = sign_in(state, "new@bank.example", "incorrect", environment=environment)

    assert result.accepted is False
    assert state[AUTHENTICATED_KEY] is False
    assert state[ANALYST_EMAIL_KEY] is None
    assert state[AUTH_MODE_KEY] is None
    assert state[ROUTE_KEY] == "login"


def test_session_initializers_preserve_existing_values_and_clear_is_idempotent() -> None:
    state: dict[str, object] = {AUTHENTICATED_KEY: True, WORKFLOW_SESSION_KEY: {"A": {}}}

    initialize_auth_session(state)
    initialize_workflow_session(state)
    assert state[AUTHENTICATED_KEY] is True
    assert state[WORKFLOW_SESSION_KEY] == {"A": {}}

    clear_workflow_session(state)
    clear_workflow_session(state)
    assert WORKFLOW_SESSION_KEY not in state


def test_start_review_and_note_are_session_only_and_source_case_remains_unchanged() -> None:
    source_case = {
        "case_id": "ARGUS-1",
        "status": "pending_human_review",
        "observed_evidence": [{"statement": "Saved fact"}],
    }
    original = deepcopy(source_case)
    state: dict[str, object] = {}
    now = datetime(2026, 9, 10, 9, 30, tzinfo=UTC)

    started = apply_case_action(
        state,
        "ARGUS-1",
        START_REVIEW,
        source_case=source_case,
        actor="analyst@bank.example",
        now=now,
    )
    blank_note = apply_case_action(
        state,
        "ARGUS-1",
        ADD_NOTE,
        source_case=source_case,
        note="   ",
        now=now,
    )
    note = apply_case_action(
        state,
        "ARGUS-1",
        ADD_NOTE,
        source_case=source_case,
        note="Verify the receiving account context.",
        actor="analyst@bank.example",
        now=now,
    )
    overlay = case_workflow(state, "ARGUS-1", source_case=source_case)

    assert started.accepted is True
    assert started.status == "in_review"
    assert blank_note.accepted is False
    assert blank_note.message == "Enter a note before saving."
    assert note.accepted is True
    assert source_case == original
    assert overlay["status"] == "in_review"
    assert overlay["notes"] == [
        {
            "text": "Verify the receiving account context.",
            "timestamp": "2026-09-10T09:30:00+00:00",
            "analyst": "analyst@bank.example",
            "storage": "session_only",
        }
    ]
    assert [event["label"] for event in overlay["activity"]] == [
        "Review started",
        "Note added",
    ]


@pytest.mark.parametrize(
    ("action", "expected_status", "expected_label"),
    [
        (ESCALATE, "escalated", "Case escalated"),
        (MARK_FALSE_POSITIVE, "false_positive", "Marked false positive"),
        (CLOSE_CASE, "closed", "Case closed"),
    ],
)
def test_sensitive_actions_require_confirmation_before_any_state_change(
    action: str, expected_status: str, expected_label: str
) -> None:
    state: dict[str, object] = {}
    initial = case_workflow(state, "ARGUS-1")

    pending = apply_case_action(state, "ARGUS-1", action)

    assert pending.accepted is False
    assert pending.requires_confirmation is True
    assert case_workflow(state, "ARGUS-1") == initial

    completed = apply_case_action(
        state,
        "ARGUS-1",
        action,
        confirmed=True,
        now=datetime(2026, 9, 10, 10, 0, tzinfo=UTC),
    )
    overlay = case_workflow(state, "ARGUS-1")

    assert completed.accepted is True
    assert completed.requires_confirmation is False
    assert completed.status == expected_status
    assert overlay["status"] == expected_status
    assert overlay["activity"][-1]["label"] == expected_label


def test_display_composition_returns_defensive_copies() -> None:
    source = {"case_id": "ARGUS-1", "status": "pending_human_review", "network": {"nodes": []}}
    state: dict[str, object] = {}
    apply_case_action(state, "ARGUS-1", START_REVIEW, source_case=source)
    overlay = case_workflow(state, "ARGUS-1", source_case=source)

    displayed = case_with_workflow(source, overlay)
    displayed["network"]["nodes"].append({"id": "changed"})
    displayed["session_activity"].append({"label": "changed"})

    assert source["network"]["nodes"] == []
    assert len(case_workflow(state, "ARGUS-1")["activity"]) == 1
    assert status_label(displayed["status"]) == "In review"
    assert status_label("future_status") == "Future Status"


def test_workflow_rejects_unknown_actions_and_empty_case_ids() -> None:
    with pytest.raises(ValueError, match="Unsupported workflow action"):
        apply_case_action({}, "ARGUS-1", "delete_case")
    with pytest.raises(ValueError, match="case_id"):
        apply_case_action({}, "", START_REVIEW)
