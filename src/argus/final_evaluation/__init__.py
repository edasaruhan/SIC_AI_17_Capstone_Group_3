"""One-shot, frozen-specification final evaluation for ARGUS AI.

The public API is loaded lazily.  In particular, importing the saved-artifact
verifier or quality layer must not import estimator or inference code as a side
effect.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FinalEvaluationError",
    "FinalEvaluationRunResult",
    "run_final_evaluation",
    "verify_final_evaluation",
]


def __getattr__(name: str) -> Any:
    """Resolve public symbols without weakening the post-run import boundary."""

    if name in {
        "FinalEvaluationError",
        "FinalEvaluationRunResult",
        "run_final_evaluation",
    }:
        from argus.final_evaluation import pipeline

        return getattr(pipeline, name)
    if name == "verify_final_evaluation":
        from argus.final_evaluation.verify import verify_final_evaluation

        return verify_final_evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
