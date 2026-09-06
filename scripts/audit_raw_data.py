"""Create a full-file, streaming audit of supplied HI-Small CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.raw_audit import audit_hi_small_files, write_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transactions",
        type=Path,
        default=Path("data/raw/HI-Small_Trans.csv"),
    )
    parser.add_argument(
        "--accounts",
        type=Path,
        default=Path("data/raw/HI-Small_accounts.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/full/raw_data_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_hi_small_files(args.transactions, args.accounts)
    output = write_audit(report, args.output)
    print(f"Full raw-data audit written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
