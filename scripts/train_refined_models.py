"""Run the full-data Sprint 3 refinement and graph-value experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.modeling.refinement import run_sprint3_refinement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refinement.yaml"),
        help="Sprint 3 refinement YAML configuration",
    )
    arguments = parser.parse_args()
    result = run_sprint3_refinement(arguments.config)
    print(f"Sprint 3 status: {result.manifest['status']}")
    print(f"Champion: {result.manifest['champion']['champion_model']}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Report: {result.report_path}")
    return 0 if result.manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
