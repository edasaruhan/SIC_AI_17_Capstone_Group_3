"""Independently verify persisted Sprint 4 artifacts without model fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.sprint4.verify import verify_sprint4_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sprint4.yaml"))
    arguments = parser.parse_args()
    result = verify_sprint4_run(arguments.config)
    print(f"Sprint 4 artifact verification: {result['status']}")
    print(f"Validation rows independently checked: {result['predictions']['rows']}")
    print(f"Cases independently checked: {result['cases']['case_count']}")
    print(
        "Refreshed artifact count: "
        f"{result.get('refreshed_artifact_count_excluding_manifest', 'unchanged')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
