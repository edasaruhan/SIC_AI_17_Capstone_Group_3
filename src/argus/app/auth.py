"""Prototype authentication helpers for the Streamlit product experience.

The module deliberately keeps authentication state in the supplied session mapping. It does
not persist passwords, call an identity provider, or claim production authentication.
"""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from argus.app.workflow import clear_workflow_session

DEMO_EMAIL_ENV = "ARGUS_DEMO_EMAIL"
DEMO_PASSWORD_ENV = "ARGUS_DEMO_PASSWORD"
MINIMUM_DEMO_PASSWORD_LENGTH = 8

AUTHENTICATED_KEY = "argus_authenticated"
ANALYST_EMAIL_KEY = "argus_analyst_email"
AUTH_MODE_KEY = "argus_auth_mode"
ROUTE_KEY = "argus_route"
SELECTED_CASE_KEY = "argus_selected_case_id"

_EMAIL_PATTERN = re.compile(r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+$", re.I)
_PERSONAL_EMAIL_DOMAINS = frozenset(
    {
        "aol.com",
        "gmail.com",
        "googlemail.com",
        "hotmail.com",
        "icloud.com",
        "live.com",
        "outlook.com",
        "proton.me",
        "protonmail.com",
        "yahoo.com",
        "yandex.com",
    }
)


@dataclass(frozen=True)
class AuthResult:
    """Result returned by the pure demo-credential validator."""

    accepted: bool
    mode: str
    message: str
    email: str | None = None


def normalize_email(value: object) -> str:
    """Return the normalized representation used for demo login identity."""

    return str(value or "").strip().casefold()


def corporate_email_error(value: object) -> str | None:
    """Return a user-facing validation error, or ``None`` for a work email."""

    email = normalize_email(value)
    if not email:
        return "Enter your corporate email address."
    if len(email) > 254 or _EMAIL_PATTERN.fullmatch(email) is None:
        return "Enter a valid corporate email address."
    domain = email.rsplit("@", 1)[1]
    if domain in _PERSONAL_EMAIL_DOMAINS:
        return "Use your company or bank email address."
    return None


def password_error(value: object) -> str | None:
    """Validate only prototype form completeness, never password strength."""

    password = str(value or "")
    if not password:
        return "Enter your password."
    if len(password) < MINIMUM_DEMO_PASSWORD_LENGTH:
        return f"Password must contain at least {MINIMUM_DEMO_PASSWORD_LENGTH} characters."
    return None


def _configured_credentials(environment: Mapping[str, str]) -> tuple[str | None, str | None]:
    configured_email = environment.get(DEMO_EMAIL_ENV)
    configured_password = environment.get(DEMO_PASSWORD_ENV)
    email = configured_email.strip() if configured_email is not None else None
    password = configured_password if configured_password is not None else None
    return email or None, password or None


def validate_credentials(
    email: object,
    password: object,
    *,
    environment: Mapping[str, str] | None = None,
) -> AuthResult:
    """Validate optional environment credentials or a clearly limited prototype fallback.

    When both demo environment variables exist, credentials must match them. When neither
    exists, a corporate-looking email and a complete password field grant prototype-only
    access. A partial environment configuration fails closed.
    """

    values = os.environ if environment is None else environment
    configured_email, configured_password = _configured_credentials(values)
    has_email = configured_email is not None
    has_password = configured_password is not None
    normalized_email = normalize_email(email)
    supplied_password = str(password or "")

    if has_email != has_password:
        return AuthResult(
            accepted=False,
            mode="configuration_error",
            message="Demo access is temporarily unavailable.",
        )

    email_problem = corporate_email_error(normalized_email)
    if email_problem is not None:
        return AuthResult(False, "validation_error", email_problem)
    password_problem = password_error(supplied_password)
    if password_problem is not None:
        return AuthResult(False, "validation_error", password_problem)

    if has_email and has_password:
        configured_email_problem = corporate_email_error(configured_email)
        if configured_email_problem is not None:
            return AuthResult(
                accepted=False,
                mode="configuration_error",
                message="Demo access is temporarily unavailable.",
            )
        email_matches = hmac.compare_digest(normalized_email, normalize_email(configured_email))
        password_matches = hmac.compare_digest(supplied_password, configured_password)
        if not (email_matches and password_matches):
            return AuthResult(
                accepted=False,
                mode="configured_demo",
                message="The email or password is incorrect.",
            )
        return AuthResult(
            accepted=True,
            mode="configured_demo",
            message="Demo session started.",
            email=normalized_email,
        )

    return AuthResult(
        accepted=True,
        mode="prototype_form_demo",
        message="Prototype session started. No production identity provider is connected.",
        email=normalized_email,
    )


def initialize_auth_session(state: MutableMapping[str, Any]) -> None:
    """Install authentication defaults without replacing an existing session."""

    state.setdefault(AUTHENTICATED_KEY, False)
    state.setdefault(ANALYST_EMAIL_KEY, None)
    state.setdefault(AUTH_MODE_KEY, None)
    state.setdefault(ROUTE_KEY, "public")


def sign_in(
    state: MutableMapping[str, Any],
    email: object,
    password: object,
    *,
    environment: Mapping[str, str] | None = None,
) -> AuthResult:
    """Validate credentials and update only non-secret session identity fields."""

    initialize_auth_session(state)
    result = validate_credentials(email, password, environment=environment)
    if result.accepted:
        state[AUTHENTICATED_KEY] = True
        state[ANALYST_EMAIL_KEY] = result.email
        state[AUTH_MODE_KEY] = result.mode
        state[ROUTE_KEY] = "portal"
    else:
        state[AUTHENTICATED_KEY] = False
        state[ANALYST_EMAIL_KEY] = None
        state[AUTH_MODE_KEY] = None
        state[ROUTE_KEY] = "login"
    return result


def sign_out(state: MutableMapping[str, Any]) -> None:
    """End the demo session and discard all session-only investigation work."""

    state[AUTHENTICATED_KEY] = False
    state.pop(ANALYST_EMAIL_KEY, None)
    state.pop(AUTH_MODE_KEY, None)
    state.pop(SELECTED_CASE_KEY, None)
    clear_workflow_session(state)
    state[ROUTE_KEY] = "public"


def is_authenticated(state: Mapping[str, Any]) -> bool:
    """Return whether the supplied session represents an active demo login."""

    return state.get(AUTHENTICATED_KEY) is True
