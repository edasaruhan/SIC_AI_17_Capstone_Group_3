"""Evidence-grounded analyst notes with a deterministic no-LLM path."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

from argus.product.schemas import AnalystNote, EvidenceBundle

_PROHIBITED_OUTPUT_PATTERNS = (
    r"\bguilty\b",
    r"\bcriminal\b",
    r"\bfraudster\b",
    r"\bmoney[ -]launderer\b",
    r"\bdefinit(?:e|ely) fraudulent\b",
    r"\bfreez\w*\b.{0,30}\baccount\b",
    r"\b(?:block|close|seize)\w*\b.{0,30}\baccount\b",
    r"\baccount\b.{0,30}\b(?:freez|block|close|seize)\w*\b",
    r"\bsuçlu\b",
    r"\bkara para akl(?:ıyor|amıştır)\b",
    r"\bhesab\w*\b.{0,30}\b(?:dondur|bloke|kapat)\w*\b",
)
_REVIEW_MARKERS = (
    "human review",
    "analyst review",
    "trained analyst",
    "analist incelemesi",
    "uzman incelemesi",
    "insan incelemesi",
)
_UNCERTAINTY_MARKERS = (
    "does not establish",
    "does not prove",
    "requires verification",
    "uncertain",
    "belirsiz",
    "kanıtlamaz",
    "doğrulama gerektir",
)


class EvidenceNoteGenerator(Protocol):
    """Optional adapter that receives evidence only, never raw tables or credentials."""

    def summarize_evidence(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return ``summary`` plus at least three cited observed evidence IDs."""


def _deterministic_text(bundle: EvidenceBundle) -> str:
    observations = " ".join(fact.statement for fact in bundle.observed_evidence[:3])
    model = bundle.model_evidence
    threshold_context = ""
    if model.threshold is not None:
        relation = "met or exceeded" if model.threshold_exceeded else "did not meet"
        threshold_context = (
            f" It {relation} the saved validation-selected threshold of {model.threshold:.8g}."
        )
    rank_context = f" Saved queue rank: {model.rank}." if model.rank is not None else ""
    return (
        f"Transaction {bundle.transaction_id} was prioritized by {model.model_name} with "
        f"{model.score_name}={model.score:.8g}.{threshold_context}{rank_context} "
        f"Observed context: {observations} The score and any feature attribution are model "
        "evidence only and do not establish wrongdoing. The saved case context may be "
        "incomplete and requires verification through trained analyst review. No automatic "
        "account action is authorized."
    )


def _safe_llm_draft(bundle: EvidenceBundle, result: object) -> str | None:
    if not isinstance(result, Mapping):
        return None
    summary = result.get("summary")
    citations = result.get("cited_observed_evidence_ids")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 2_500:
        return None
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        return None
    if len(set(citations)) < 3:
        return None
    valid_ids = {fact.evidence_id for fact in bundle.observed_evidence}
    if any(item not in valid_ids for item in citations):
        return None

    normalized = " ".join(summary.split())
    lowered = normalized.casefold()
    if any(
        re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _PROHIBITED_OUTPUT_PATTERNS
    ):
        return None
    # The adapter must itself state uncertainty and human review; we append a
    # fixed guardrail as defense in depth but do not use that suffix to make an
    # otherwise overconfident draft pass validation.
    if not any(marker in lowered for marker in _REVIEW_MARKERS):
        return None
    if not any(marker in lowered for marker in _UNCERTAINTY_MARKERS):
        return None
    cited = ", ".join(dict.fromkeys(citations))
    return (
        f"{normalized} Cited observed evidence: {cited}. The note is a review aid only; "
        "no automatic account action is authorized."
    )


def generate_analyst_note(
    bundle: EvidenceBundle,
    *,
    llm: EvidenceNoteGenerator | None = None,
) -> AnalystNote:
    """Generate a grounded note and safely fall back on any optional-LLM failure.

    The optional adapter sees only the already validated, JSON-safe evidence
    bundle. It must return a structured draft with citations to at least three
    observed evidence IDs. Invalid, unsafe, empty, or exceptional output is
    discarded without exposing exception messages (which might contain secrets).
    """

    fallback = _deterministic_text(bundle)
    if llm is None:
        return AnalystNote(
            text=fallback,
            mode="deterministic_fallback",
            fallback_reason="llm_not_configured",
        )

    try:
        result = llm.summarize_evidence(bundle.to_dict())
    except Exception as exc:  # optional external clients must never break case creation
        return AnalystNote(
            text=fallback,
            mode="deterministic_fallback",
            fallback_reason=f"llm_error_{type(exc).__name__}",
        )
    safe_text = _safe_llm_draft(bundle, result)
    if safe_text is None:
        return AnalystNote(
            text=fallback,
            mode="deterministic_fallback",
            fallback_reason="unsafe_or_invalid_llm_output",
        )
    return AnalystNote(text=safe_text, mode="optional_llm")


__all__ = ["EvidenceNoteGenerator", "generate_analyst_note"]
