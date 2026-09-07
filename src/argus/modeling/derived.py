"""Rebuild derived Sprint 2 metrics/reports from frozen validation predictions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from argus.config import get_path, load_config
from argus.modeling.artifacts import atomic_write_json, write_run_manifest
from argus.modeling.metrics import evaluate_binary_predictions
from argus.modeling.reporting import (
    PR_CURVE_MAX_PLOT_POINTS,
    comparison_context_from_run,
    plot_metric_comparison,
    plot_precision_recall_curves,
    write_comparison_table,
)
from argus.modeling.selection import select_transaction_baseline_champion


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def refresh_sprint2_derived_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
) -> dict[str, Any]:
    """Recompute validation metrics and comparisons without fitting any model."""

    config = load_config(config_path)
    baseline = config["baseline"]
    run_dir = get_path(config, "run_dir")
    print("[derived] Loading saved validation predictions", flush=True)
    manifest = _load_json(run_dir / "run_manifest.json")
    if manifest.get("status") not in {"PASS", "FAIL_QUALITY_CHECKS"}:
        raise RuntimeError("No completed Sprint 2 core run is available to refresh")

    connection = duckdb.connect()
    try:
        prediction = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY source_row_number",
            [str((run_dir / "validation_predictions.parquet").resolve())],
        ).fetch_df()
    finally:
        connection.close()
    labels = prediction["is_laundering"].to_numpy(dtype=np.int8)
    source_rows = prediction["source_row_number"].to_numpy(dtype=np.int64)
    expected = manifest["split_prevalence"]["validation"]
    if len(prediction) != int(expected["rows"]) or int(labels.sum()) != int(
        expected["positive_labels"]
    ):
        raise RuntimeError("Saved validation predictions do not match frozen split metadata")

    scores_by_model: dict[str, np.ndarray] = {}
    model_results: dict[str, dict[str, Any]] = {}
    for model_name in ("logistic_regression", "random_forest", "lightgbm"):
        print(f"[derived] Recomputing validation metrics: {model_name}", flush=True)
        score_column = f"score_{model_name}"
        scores = prediction[score_column].to_numpy(dtype=np.float64)
        scores_by_model[model_name] = scores
        metrics = evaluate_binary_predictions(
            labels,
            scores,
            source_rows,
            threshold=float(baseline["decision_threshold"]),
            top_k_values=baseline["top_k"],
        )
        model_dir = run_dir / "models" / model_name
        metadata = _load_json(model_dir / "metadata.json")
        metadata["validation_metrics"] = metrics
        metadata["metric_schema_version"] = 2
        metadata["top_k_cutoff_tie_diagnostics"] = True
        atomic_write_json(metrics, model_dir / "validation_metrics.json")
        atomic_write_json(metadata, model_dir / "metadata.json")
        model_results[model_name] = metadata

    print("[derived] Rebuilding comparison tables and validation figures", flush=True)
    champion = select_transaction_baseline_champion(
        {
            name: float(result["validation_metrics"]["average_precision"])
            for name, result in model_results.items()
        },
        partition="validation",
        test_metrics=None,
    )
    atomic_write_json(champion, run_dir / "transaction_baseline_champion.json")

    upstream = _load_json(get_path(config, "upstream_manifest"))
    manifest["provenance"]["raw_sources"] = {
        "transactions": upstream["provenance"]["transactions"],
        "accounts": upstream["provenance"]["accounts"],
    }
    comparison_context = comparison_context_from_run(
        provenance=manifest["provenance"],
        feature_manifest=manifest["feature_contract"],
        configuration=manifest["configuration"],
        libraries=manifest["libraries"],
        feature_scope=str(baseline["feature_scope"]),
    )
    comparison_payload = {
        "evaluation_partition": "validation",
        "primary_metric": "average_precision",
        "accuracy_is_primary": False,
        "provenance": comparison_context,
        "model_results": model_results,
    }
    atomic_write_json(comparison_payload, run_dir / "model_comparison.json")
    comparison = write_comparison_table(
        model_results,
        run_dir / "model_comparison.csv",
        context=comparison_context,
    )
    plot_metric_comparison(comparison, run_dir / "figures" / "metric_comparison.png")
    plot_precision_recall_curves(
        labels,
        scores_by_model,
        run_dir / "figures" / "validation_precision_recall_curves.png",
    )

    manifest["models"] = model_results
    manifest["champion"] = champion
    manifest["derived_artifacts"] = {
        "status": "PASS",
        "refreshed_at_utc": datetime.now(UTC).isoformat(),
        "source": "saved_validation_predictions.parquet",
        "model_refit_performed": False,
        "test_inference_performed": False,
        "metric_schema_version": 2,
        "top_k_cutoff_tie_diagnostics": True,
        "comparison_provenance_embedded": True,
        "precision_recall_figure": {
            "numeric_metrics_use_all_validation_rows": True,
            "display_only_decimation": "deterministic_even_index_with_endpoints",
            "maximum_points_per_model": PR_CURVE_MAX_PLOT_POINTS,
        },
    }
    print("[derived] Refreshing run-manifest inventory", flush=True)
    refreshed_manifest = write_run_manifest(manifest, run_dir)
    print("[derived] Derived artifact refresh PASS", flush=True)
    return {
        "status": "PASS",
        "model_refit_performed": False,
        "validation_rows": len(prediction),
        "champion": champion["champion_model"],
        "artifact_count_excluding_manifest": refreshed_manifest[
            "artifact_count_excluding_manifest"
        ],
    }
