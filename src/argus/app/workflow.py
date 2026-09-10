"""Session-only analyst workflow state for saved ARGUS case examples."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

WORKFLOW_SESSION_KEY = "argus_case_workflow"

START_REVIEW = "start_review"
ADD_NOTE = "add_note"
ESCALATE = "escalate"
MARK_FALSE_POSITIVE = "mark_false_positive"
CLOSE_CASE = "close_case"

_CONFIRMATION_ACTIONS = frozenset({ESCALATE, MARK_FALSE_POSITIVE, CLOSE_CASE})
_ACTION_STATUS = {
    START_REVIEW: "in_review",
    ESCALATE: "escalated",
    MARK_FALSE_POSITIVE: "false_positive",
    CLOSE_CASE: "closed",
}
_ACTION_LABEL = {
    START_REVIEW: "Review started",
    ADD_NOTE: "Note added",
    ESCALATE: "Case escalated",
    MARK_FALSE_POSITIVE: "Marked false positive",
    CLOSE_CASE: "Case closed",
}
_STATUS_LABELS = {
    "pending_human_review": "Pending review",
    "pending_review": "Pending review",
    "in_review": "In review",
    "escalated": "Escalated",
    "false_positive": "False positive",
    "closed": "Closed",
}


@dataclass(frozen=True)
class WorkflowResult:
    """Outcome of one requested analyst workflow action."""

    accepted: bool
    requires_confirmation: bool
    status: str
    message: str


def _case_id(value: object) -> str:
    case_id = str(value or "").strip()
    if not case_id:
        raise ValueError("case_id must not be empty")
    return case_id


def _timestamp(value: datetime | None) -> str:
    moment = value or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _actor(value: object) -> str:
    return str(value or "").strip() or "Demo analyst"


def _initial_status(source_case: Mapping[str, Any] | None) -> str:
    if source_case is None:
        return "pending_review"
    candidate = str(source_case.get("status", "pending_review")).strip().casefold()
    return candidate if candidate in _STATUS_LABELS else "pending_review"


def initialize_workflow_session(state: MutableMapping[str, Any]) -> None:
    """Install an empty workflow container without touching any source artifacts."""

    current = state.get(WORKFLOW_SESSION_KEY)
    if not isinstance(current, dict):
        state[WORKFLOW_SESSION_KEY] = {}


def clear_workflow_session(state: MutableMapping[str, Any]) -> None:
    """Discard all transient case actions for the current demo session."""

    state.pop(WORKFLOW_SESSION_KEY, None)


def _stored_overlay(
    state: MutableMapping[str, Any],
    case_id: object,
    source_case: Mapping[str, Any] | None,
) -> dict[str, Any]:
    initialize_workflow_session(state)
    identifier = _case_id(case_id)
    store = state[WORKFLOW_SESSION_KEY]
    if identifier not in store:
        store[identifier] = {
            "status": _initial_status(source_case),
            "notes": [],
            "activity": [],
        }
    return store[identifier]


def case_workflow(
    state: MutableMapping[str, Any],
    case_id: object,
    *,
    source_case: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a defensive copy of a case's transient workflow overlay."""

    return deepcopy(_stored_overlay(state, case_id, source_case))


def case_with_workflow(
    source_case: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose a display copy while keeping the loaded source case immutable."""

    result = deepcopy(dict(source_case))
    result["status"] = str(overlay.get("status", result.get("status", "pending_review")))
    result["session_notes"] = deepcopy(list(overlay.get("notes", [])))
    result["session_activity"] = deepcopy(list(overlay.get("activity", [])))
    return result


def status_label(status: object) -> str:
    """Return a readable status while remaining safe for unknown future values."""

    canonical = str(status or "pending_review").strip().casefold()
    if canonical in _STATUS_LABELS:
        return _STATUS_LABELS[canonical]
    return canonical.replace("_", " ").strip().title() or "Pending review"


def apply_case_action(
    state: MutableMapping[str, Any],
    case_id: object,
    action: str,
    *,
    source_case: Mapping[str, Any] | None = None,
    note: object | None = None,
    confirmed: bool = False,
    actor: object = "Demo analyst",
    now: datetime | None = None,
) -> WorkflowResult:
    """Apply a demo workflow action to session state, never to a saved case artifact."""

    identifier = _case_id(case_id)
    if action not in _ACTION_LABEL:
        raise ValueError(f"Unsupported workflow action: {action}")

    overlay = _stored_overlay(state, identifier, source_case)
    current_status = str(overlay["status"])
    if action in _CONFIRMATION_ACTIONS and not confirmed:
        return WorkflowResult(
            accepted=False,
            requires_confirmation=True,
            status=current_status,
            message=f"Confirm to continue: {_ACTION_LABEL[action]}.",
        )

    note_text = str(note or "").strip()
    if action == ADD_NOTE and not note_text:
        return WorkflowResult(
            accepted=False,
            requires_confirmation=False,
            status=current_status,
            message="Enter a note before saving.",
        )

    event_time = _timestamp(now)
    analyst = _actor(actor)
    if action in _ACTION_STATUS:
        overlay["status"] = _ACTION_STATUS[action]
    if action == ADD_NOTE:
        overlay["notes"].append(
            {
                "text": note_text,
                "timestamp": event_time,
                "analyst": analyst,
                "storage": "session_only",
            }
        )
    overlay["activity"].append(
        {
            "action": action,
            "label": _ACTION_LABEL[action],
            "timestamp": event_time,
            "analyst": analyst,
            "storage": "session_only",
        }
    )
    return WorkflowResult(
        accepted=True,
        requires_confirmation=False,
        status=str(overlay["status"]),
        message=f"{_ACTION_LABEL[action]}. This change is stored for this session only.",
    )
