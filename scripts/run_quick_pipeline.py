"""Run the complete ARGUS Sprint 1 pipeline (quick config by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.pipeline import run_sprint1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--accounts-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Label inputs as a locally generated test fixture, never as IBM results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_sprint1_pipeline(
        args.config,
        transaction_path=args.data_path,
        accounts_path=args.accounts_path,
        output_dir=args.output_dir,
        fixture=args.fixture,
    )
    print(f"Sprint 1 pipeline: {result.manifest['status']}")
    print(f"Loaded rows: {result.manifest['configuration']['loaded_rows']}")
    print(f"Artifacts: {result.manifest['artifact_count_excluding_manifest'] + 1}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
