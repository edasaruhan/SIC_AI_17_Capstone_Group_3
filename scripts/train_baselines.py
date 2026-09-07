"""Run the frozen full-data Sprint 2 transaction baseline experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.modeling.baseline import run_sprint2_baselines


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Sprint 2 baseline YAML configuration",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    result = run_sprint2_baselines(arguments.config)
    print(f"Sprint 2 baseline status: {result.manifest['status']}")
    print(f"Champion: {result.manifest['champion']['champion_model']}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
