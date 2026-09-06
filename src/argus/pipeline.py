"""End-to-end orchestration for the ARGUS Sprint 1 data proof."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from argus.config import get_path, load_config
from argus.data.accounts import load_ibm_accounts, validate_account_references
from argus.data.load import load_ibm_aml
from argus.data.preprocess import preprocess_transactions
from argus.data.split import chronological_split
from argus.data.validate import validate_raw_schema, validate_transactions
from argus.eda import generate_eda_artifacts
from argus.features.graph import add_graph_history_features
from argus.features.temporal import add_history_features, add_time_features
from argus.features.transaction import add_transaction_features


@dataclass(frozen=True)
class PipelineRunResult:
    """Successful pipeline location and its saved manifest."""

    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _artifact_inventory(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.resolve() == manifest_path.resolve() or path.name == ".gitkeep":
            continue
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def _resolved_run_inputs(
    config_path: str | Path,
    *,
    transaction_path: str | Path | None,
    accounts_path: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    config = load_config(config_path)
    transaction_source = (
        Path(transaction_path).expanduser().resolve()
        if transaction_path is not None
        else get_path(config, "raw_data")
    )
    account_source = (
        Path(accounts_path).expanduser().resolve()
        if accounts_path is not None
        else get_path(config, "raw_accounts")
    )
    run_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else get_path(config, "run_dir")
    )
    return config, transaction_source, account_source, run_dir


def run_sprint1_pipeline(
    config_path: str | Path = "configs/quick.yaml",
    *,
    transaction_path: str | Path | None = None,
    accounts_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    fixture: bool = False,
) -> PipelineRunResult:
    """Run loading through EDA/features/split and save all evidence artifacts."""

    started = time.perf_counter()
    started_at = datetime.now(UTC)
    config, transaction_source, account_source, run_dir = _resolved_run_inputs(
        config_path,
        transaction_path=transaction_path,
        accounts_path=accounts_path,
        output_dir=output_dir,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = run_dir / "tables"
    metadata_dir = run_dir / "metadata"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    project = config["project"]
    data_config = config["data"]
    sampling = data_config.get("sampling", {})
    sample_size = sampling.get("max_rows") if sampling.get("enabled", False) else None
    random_seed = int(project["random_seed"])

    raw = load_ibm_aml(
        transaction_source,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    raw_report = validate_raw_schema(raw)
    raw_report.raise_for_errors()
    canonical = preprocess_transactions(
        raw,
        column_mapping=dict(data_config["column_mapping"]),
        drop_exact_duplicates=False,
    )
    canonical_report = validate_transactions(canonical)
    canonical_report.raise_for_errors()

    accounts = load_ibm_accounts(account_source)
    account_report = validate_account_references(canonical, accounts)
    if not account_report.valid:
        raise ValueError(
            "Canonical transaction nodes do not all exist in HI-Small_accounts.csv: "
            f"{account_report.unmatched_transaction_nodes} unmatched unique node(s)"
        )

    featured = add_transaction_features(canonical)
    featured = add_time_features(featured)
    featured = add_history_features(
        featured,
        windows=tuple(config["features"]["history_windows"]),
    )
    featured = add_graph_history_features(featured)

    split_config = config["split"]
    split = chronological_split(
        featured,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
        train_end=split_config.get("train_end"),
        validation_end=split_config.get("validation_end"),
        metadata_path=metadata_dir / "split_metadata.json",
    )

    canonical.to_csv(tables_dir / "transactions_clean.csv", index=False)
    featured.to_csv(tables_dir / "transaction_features.csv", index=False)
    canonical[
        [
            "transaction_id",
            "timestamp",
            "from_node_id",
            "to_node_id",
            "amount_paid",
            "amount_received",
            "is_laundering",
        ]
    ].to_csv(tables_dir / "graph_edges.csv", index=False)
    split_manifest = pd.concat(
        [
            part[["transaction_id", "timestamp"]].assign(partition=name)
            for name, part in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
        ],
        ignore_index=True,
    ).sort_values(["timestamp", "transaction_id"], kind="mergesort")
    split_manifest.to_csv(tables_dir / "split_manifest.csv", index=False)

    _write_json(raw_report.to_dict(), run_dir / "raw_validation_report.json")
    _write_json(canonical_report.to_dict(), run_dir / "canonical_validation_report.json")
    _write_json(account_report.to_dict(), run_dir / "account_reference_report.json")
    _write_json(config, run_dir / "resolved_config.json")

    source_provenance = {
        "dataset": ("locally generated IBM-shaped fixture" if fixture else "IBM AML-Data HI-Small"),
        "dataset_is_synthetic": True,
        "locally_generated_fixture": fixture,
        "analysis_scope": (
            f"chronological prefix of at most {sample_size} rows"
            if sample_size is not None
            else "full transaction file"
        ),
        "transactions": _source_fingerprint(transaction_source),
        "accounts": _source_fingerprint(account_source),
    }
    eda_config = config.get("eda", {})
    eda_manifest = generate_eda_artifacts(
        canonical,
        run_dir / "eda",
        random_seed=random_seed,
        graph_max_edges=eda_config.get("graph_max_edges", 50_000),
        top_n=int(eda_config.get("top_n", 20)),
        provenance=source_provenance,
    )

    transaction_feature_columns = [
        "log_amount_paid",
        "log_amount_received",
        "same_bank",
        "currency_match",
        "amount_difference_same_currency",
        "amount_ratio_same_currency",
    ]
    time_columns = [
        "hour",
        "day_of_week",
        "is_weekend",
        "sender_seconds_since_previous",
        "receiver_seconds_since_previous",
    ]
    history_columns = [
        column
        for column in featured
        if column.startswith(
            (
                "sender_previous_",
                "receiver_previous_",
                "sender_burst_",
                "receiver_burst_",
                "sender_rolling_",
                "receiver_rolling_",
            )
        )
    ]
    graph_columns = [
        column
        for column in featured
        if "prior_fan_" in column or column == "pair_previous_transfer_count"
    ]
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "status": "PASS",
        "sprint": "Sprint 1 - Repository Foundation + Data Proof",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "provenance": source_provenance,
        "configuration": {
            "file": Path(config_path).name,
            "mode": project.get("mode"),
            "random_seed": random_seed,
            "loaded_rows": len(raw),
            "sample_size_requested": sample_size,
            "sampling_method": raw.attrs.get("sampling_method"),
        },
        "validation": {
            "raw": raw_report.to_dict(),
            "canonical": canonical_report.to_dict(),
            "accounts": account_report.to_dict(),
        },
        "preprocessing": {
            "canonical_rows": len(canonical),
            "exact_duplicate_rows_observed": canonical.attrs.get(
                "exact_duplicate_rows_observed", 0
            ),
            "exact_duplicate_policy": canonical.attrs.get("exact_duplicate_policy"),
            "composite_node_delimiter": "::",
            "bank_id_normalization": "strip leading zeroes; all-zero becomes 0",
        },
        "split": split.metadata,
        "features": {
            "rows": len(featured),
            "columns": len(featured.columns),
            "transaction": transaction_feature_columns,
            "time": time_columns,
            "history": history_columns,
            "graph_history": graph_columns,
            "strictly_prior_timestamp_batches": True,
        },
        "eda": {
            "status": eda_manifest["status"],
            "artifact_count_excluding_manifest": eda_manifest["artifact_count_excluding_manifest"],
        },
        "models": "NOT_IN_SPRINT_1",
        "artifacts": [],
    }
    manifest["artifacts"] = _artifact_inventory(run_dir, manifest_path)
    manifest["artifact_count_excluding_manifest"] = len(manifest["artifacts"])
    _write_json(manifest, manifest_path)
    return PipelineRunResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)
