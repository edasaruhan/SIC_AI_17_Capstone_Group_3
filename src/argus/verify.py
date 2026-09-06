"""Read-only verification of a completed Sprint 1 run and its artifact hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class RunVerificationError(AssertionError):
    """Raised when saved artifacts disagree with their run manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sprint1_run(run_dir: str | Path) -> dict[str, Any]:
    """Verify hashes, row identities, split boundaries, and required EDA files."""

    root = Path(run_dir)
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise RunVerificationError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RunVerificationError("Run manifest does not declare PASS")
    if manifest.get("models") != "NOT_IN_SPRINT_1":
        raise RunVerificationError("Unexpected modeling output in Sprint 1 manifest")

    artifact_entries = manifest.get("artifacts", [])
    if len(artifact_entries) != manifest.get("artifact_count_excluding_manifest"):
        raise RunVerificationError("Artifact count disagrees with manifest inventory")
    for entry in artifact_entries:
        path = root / entry["path"]
        if not path.is_file():
            raise RunVerificationError(f"Missing inventoried artifact: {entry['path']}")
        if path.stat().st_size != entry["size_bytes"]:
            raise RunVerificationError(f"Size mismatch for artifact: {entry['path']}")
        if _sha256(path) != entry["sha256"]:
            raise RunVerificationError(f"SHA-256 mismatch for artifact: {entry['path']}")

    clean = pd.read_csv(root / "tables" / "transactions_clean.csv", usecols=["transaction_id"])
    features = pd.read_csv(root / "tables" / "transaction_features.csv", usecols=["transaction_id"])
    split_manifest = pd.read_csv(root / "tables" / "split_manifest.csv")
    if not clean["transaction_id"].is_unique or not features["transaction_id"].is_unique:
        raise RunVerificationError("Transaction IDs must remain unique")
    if set(clean["transaction_id"]) != set(features["transaction_id"]):
        raise RunVerificationError("Clean and feature tables contain different transactions")
    if set(clean["transaction_id"]) != set(split_manifest["transaction_id"]):
        raise RunVerificationError("Split manifest does not cover exactly the clean transactions")
    if split_manifest["transaction_id"].duplicated().any():
        raise RunVerificationError("A transaction occurs in multiple split rows")

    split_manifest["timestamp"] = pd.to_datetime(split_manifest["timestamp"], errors="raise")
    groups = {
        name: split_manifest.loc[split_manifest["partition"].eq(name), "timestamp"]
        for name in ("train", "validation", "test")
    }
    if any(series.empty for series in groups.values()):
        raise RunVerificationError("A chronological partition is empty")
    if not groups["train"].max() < groups["validation"].min():
        raise RunVerificationError("Train/validation timestamps overlap")
    if not groups["validation"].max() < groups["test"].min():
        raise RunVerificationError("Validation/test timestamps overlap")

    required_figures = {
        "amount_distribution.png",
        "bank_activity.png",
        "class_distribution.png",
        "connected_components.png",
        "currency_distribution.png",
        "in_out_degree_distribution.png",
        "laundering_activity_over_time.png",
        "log_amount_distribution.png",
        "payment_format_distribution.png",
        "repeated_pair_activity.png",
        "suspicious_local_subgraph.png",
        "transaction_activity_over_time.png",
        "unique_counterparties.png",
    }
    figures_dir = root / "eda" / "figures"
    actual_figures = {path.name for path in figures_dir.glob("*.png") if path.stat().st_size}
    missing_figures = sorted(required_figures - actual_figures)
    if missing_figures:
        raise RunVerificationError(f"Missing required EDA figures: {missing_figures}")

    return {
        "status": "PASS",
        "verified_artifact_hashes": len(artifact_entries),
        "verified_transaction_rows": len(clean),
        "verified_split_rows": len(split_manifest),
        "verified_eda_figures": len(required_figures),
        "strict_chronology": True,
        "no_row_overlap": True,
    }
