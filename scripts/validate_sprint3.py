"""Run final Sprint 3 artifact verification and repository quality checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.modeling.refinement_quality import run_sprint3_quality_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/refinement.yaml"))
    arguments = parser.parse_args()
    result = run_sprint3_quality_checks(arguments.config)
    print(f"Sprint 3 quality status: {result['status']}")
    print(f"Pytest passed: {result['pytest_passed']}")
    print(f"Artifact count: {result['artifact_count_excluding_manifest']}")
    print(f"Report: {result['report_path']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
