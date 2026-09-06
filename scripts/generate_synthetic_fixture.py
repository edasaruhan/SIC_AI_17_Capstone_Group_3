"""Generate an explicitly synthetic IBM-shaped fixture for CI and smoke tests.

This command never writes to the real HI-Small filenames by default and never
claims that its output is empirical IBM data.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

TRANSACTION_HEADER = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

ACCOUNT_HEADER = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]


def generate_fixture(
    transactions_output: Path,
    accounts_output: Path,
    *,
    rows: int = 120,
) -> None:
    """Write deterministic synthetic transactions and companion accounts."""

    if rows < 12:
        raise ValueError("rows must be at least 12 so chronological splits are meaningful")
    transactions_output.parent.mkdir(parents=True, exist_ok=True)
    accounts_output.parent.mkdir(parents=True, exist_ok=True)

    accounts: set[tuple[str, str]] = set()
    start = datetime(2024, 1, 1, 0, 0)
    formats = ("ACH", "Wire", "Cheque", "Credit Card")
    currencies = ("US Dollar", "Euro", "UK Pound")

    with transactions_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TRANSACTION_HEADER)
        for index in range(rows):
            sender_bank = f"{(index % 4) + 1:03d}"
            receiver_bank = f"{((index * 3 + 1) % 4) + 1:03d}"
            sender_account = f"A{index % 17:08X}"
            receiver_account = f"B{(index * 5) % 19:08X}"
            timestamp = start + timedelta(minutes=index // 3)
            amount = round(5.0 + ((index * 37) % 1500) / 10.0, 2)
            currency = currencies[index % len(currencies)]
            receiving_currency = currency if index % 5 else currencies[(index + 1) % 3]
            received = amount if receiving_currency == currency else round(amount * 0.91, 2)
            label = 1 if index in {47, 94} else 0
            writer.writerow(
                [
                    timestamp.strftime("%Y/%m/%d %H:%M"),
                    sender_bank,
                    sender_account,
                    receiver_bank,
                    receiver_account,
                    f"{received:.2f}",
                    receiving_currency,
                    f"{amount:.2f}",
                    currency,
                    formats[index % len(formats)],
                    label,
                ]
            )
            accounts.add((str(int(sender_bank)), sender_account))
            accounts.add((str(int(receiver_bank)), receiver_account))

    with accounts_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(ACCOUNT_HEADER)
        for index, (bank, account) in enumerate(sorted(accounts), start=1):
            writer.writerow(
                [
                    f"Synthetic Bank {bank}",
                    bank,
                    account,
                    f"E{index:08X}",
                    f"Synthetic Entity {index}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transactions-output",
        type=Path,
        default=Path("data/raw/synthetic_HI-Small_Trans.csv"),
    )
    parser.add_argument(
        "--accounts-output",
        type=Path,
        default=Path("data/raw/synthetic_HI-Small_accounts.csv"),
    )
    parser.add_argument("--rows", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_fixture(args.transactions_output, args.accounts_output, rows=args.rows)
    print(f"Synthetic transaction fixture: {args.transactions_output}")
    print(f"Synthetic accounts fixture: {args.accounts_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
