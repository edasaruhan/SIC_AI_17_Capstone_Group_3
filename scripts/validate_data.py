"""Validate HI-Small using quick pandas checks or the streaming full-file audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.config import get_path, load_config
from argus.data.accounts import load_ibm_accounts, validate_account_references
from argus.data.load import load_ibm_aml
from argus.data.preprocess import preprocess_transactions
from argus.data.validate import validate_raw_schema, validate_transactions
from argus.pipeline import _write_json
from argus.raw_audit import audit_hi_small_files, write_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    transactions_path = get_path(config, "raw_data")
    accounts_path = get_path(config, "raw_accounts")
    run_dir = get_path(config, "run_dir")
    sampling = config["data"]["sampling"]
    max_rows = sampling.get("max_rows") if sampling.get("enabled") else None

    if max_rows is None:
        report = audit_hi_small_files(
            transactions_path,
            accounts_path,
            dataset_label="IBM AML-Data HI-Small",
            dataset_is_synthetic=True,
            locally_generated_fixture=False,
        )
        destination = write_audit(report, run_dir / "raw_data_audit.json")
        print(f"Full source validation: PASS ({destination})")
        return 0

    raw = load_ibm_aml(
        transactions_path,
        sample_size=int(max_rows),
        random_seed=int(config["project"]["random_seed"]),
    )
    raw_report = validate_raw_schema(raw)
    raw_report.raise_for_errors()
    canonical = preprocess_transactions(raw, column_mapping=config["data"]["column_mapping"])
    canonical_report = validate_transactions(canonical)
    canonical_report.raise_for_errors()
    accounts = load_ibm_accounts(accounts_path)
    reference_report = validate_account_references(canonical, accounts)
    if not reference_report.valid:
        raise ValueError("Quick validation found unmatched account references")
    destination = run_dir / "validation_report.json"
    _write_json(
        {
            "status": "PASS",
            "scope": f"chronological prefix of at most {max_rows} transactions",
            "raw": raw_report.to_dict(),
            "canonical": canonical_report.to_dict(),
            "accounts": reference_report.to_dict(),
        },
        destination,
    )
    print(f"Quick data validation: PASS ({destination})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
