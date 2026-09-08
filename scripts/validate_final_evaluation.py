"""Run the read-only final verification and complete repository quality checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.final_evaluation.quality import run_final_evaluation_quality_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_evaluation.yaml"),
    )
    arguments = parser.parse_args()
    result = run_final_evaluation_quality_checks(arguments.config)
    print(f"Final-evaluation quality status: {result['status']}")
    print(f"Pytest passed: {result['pytest_passed']}")
    print(f"Artifact verification: {result['final_artifact_verification_status']}")
    print(f"Run manifest unchanged: {result['run_manifest_unchanged']}")
    print(f"Quality report: {result['quality_report_path']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
