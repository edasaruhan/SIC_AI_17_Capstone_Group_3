from __future__ import annotations

import json
from pathlib import Path

from argus.pipeline import run_sprint1_pipeline
from argus.verify import verify_sprint1_run
from scripts.generate_synthetic_fixture import generate_fixture


def test_quick_pipeline_completes_end_to_end_on_deterministic_fixture(tmp_path) -> None:
    transactions = tmp_path / "synthetic_transactions.csv"
    accounts = tmp_path / "synthetic_accounts.csv"
    output = tmp_path / "artifacts"
    generate_fixture(transactions, accounts, rows=120)
    config = Path(__file__).resolve().parents[1] / "configs" / "quick.yaml"

    result = run_sprint1_pipeline(
        config,
        transaction_path=transactions,
        accounts_path=accounts,
        output_dir=output,
        fixture=True,
    )

    assert result.manifest["status"] == "PASS"
    assert result.manifest["provenance"]["locally_generated_fixture"] is True
    assert result.manifest["features"]["strictly_prior_timestamp_batches"] is True
    assert result.manifest["split"]["strict_boundaries_verified"] is True
    assert (output / "tables" / "transactions_clean.csv").is_file()
    assert (output / "tables" / "transaction_features.csv").is_file()
    assert (output / "eda" / "dataset_profile.json").is_file()
    saved = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["artifact_count_excluding_manifest"] == len(saved["artifacts"])
    verification = verify_sprint1_run(output)
    assert verification["status"] == "PASS"
    assert verification["verified_transaction_rows"] == 120
