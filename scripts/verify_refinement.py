"""Independently verify persisted Sprint 3 refinement artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.modeling.refinement_verify import verify_sprint3_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/refinement.yaml"))
    arguments = parser.parse_args()
    result = verify_sprint3_run(arguments.config)
    print(f"Sprint 3 artifact verification: {result['status']}")
    print(
        "Frozen temporal folds independently recounted: "
        f"{result['temporal_fold_recount']['folds_recounted']}"
    )
    print(f"Refreshed artifact count: {result['refreshed_artifact_count_excluding_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
