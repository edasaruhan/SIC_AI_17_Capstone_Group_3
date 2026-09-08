"""Publish read-only verification of the persisted final-evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.final_evaluation.verify import publish_final_verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_evaluation.yaml"),
    )
    arguments = parser.parse_args()
    result = publish_final_verification(arguments.config)
    print(f"Final-evaluation artifact verification: {result['status']}")
    print(f"Saved final-test rows checked: {result['predictions']['rows']}")
    print(f"Frozen champion: {result['metrics']['frozen_champion']}")
    print(f"Run-manifest SHA-256: {result['run_manifest_sha256']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
