"""Publish README, Model Card, and comparison table from verified final artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.final_evaluation.reporting import publish_final_documentation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final_evaluation.yaml"))
    arguments = parser.parse_args()
    result = publish_final_documentation(arguments.config)
    print(f"Final documentation status: {result['status']}")
    print(f"Comparison: {result['generated_comparison']}")
    print(f"README: {result['readme']}")
    print(f"Model Card: {result['model_card']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
