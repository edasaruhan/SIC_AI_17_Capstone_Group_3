"""Human-review product primitives for ARGUS investigation artifacts.

The product layer consumes saved model and graph outputs. It does not train a
model, infer an account-level label, or authorize an operational action.
"""

from argus.product.cases import (
    build_case_from_frames,
    build_investigation_case,
    build_investigation_cases,
)
from argus.product.evidence import build_evidence_bundle
from argus.product.narrative import EvidenceNoteGenerator, generate_analyst_note
from argus.product.schemas import (
    AnalystNote,
    EvidenceBundle,
    InvestigationCase,
    ModelEvidence,
    ObservedEvidenceFact,
    ProductSchemaError,
    validate_case_payload,
    validate_evidence_payload,
)

__all__ = [
    "AnalystNote",
    "EvidenceBundle",
    "EvidenceNoteGenerator",
    "InvestigationCase",
    "ModelEvidence",
    "ObservedEvidenceFact",
    "ProductSchemaError",
    "build_case_from_frames",
    "build_evidence_bundle",
    "build_investigation_case",
    "build_investigation_cases",
    "generate_analyst_note",
    "validate_case_payload",
    "validate_evidence_payload",
]
