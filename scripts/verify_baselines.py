"""Independently verify persisted Sprint 2 validation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.modeling.verify import verify_sprint2_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    arguments = parser.parse_args()
    report = verify_sprint2_run(arguments.config)
    print(f"Sprint 2 artifact verification: {report['status']}")
    print(f"Validation rows independently checked: {report['validation_predictions']['rows']:,}")
    print(f"Refreshed artifact count: {report['refreshed_artifact_count_excluding_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
