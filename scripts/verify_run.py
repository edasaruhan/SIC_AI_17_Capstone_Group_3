"""Verify a saved ARGUS Sprint 1 run without changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from argus.verify import verify_sprint1_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/quick"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_sprint1_run(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
