"""Run the frozen, non-retryable ARGUS final-test evaluation exactly once."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.final_evaluation.pipeline import run_final_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_evaluation.yaml"),
        help="Frozen Sprint 5 final-evaluation YAML configuration",
    )
    arguments = parser.parse_args()
    result = run_final_evaluation(arguments.config)
    print(f"Final evaluation status: {result.manifest['status']}")
    print(f"Frozen champion: {result.manifest['frozen_champion']}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Report: {result.report_path}")
    print("The exclusive final-test receipt remains in place; reruns are not permitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
