"""Out-of-core Sprint 1 EDA over a canonical Parquet transaction table.

Full-population tables are aggregated by DuckDB and only aggregate result sets
are materialized in pandas.  NetworkX plots are intentionally generated from a
small, deterministic transaction-edge sample and are labeled as descriptive of
that sample only.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from argus.eda import REQUIRED_COLUMNS, generate_eda_artifacts


class FullEDAError(ValueError):
    """Raised when out-of-core EDA cannot be produced truthfully."""


FULL_REQUIRED_COLUMNS = REQUIRED_COLUMNS | {"transaction_id"}
SAMPLE_ORDER_SQL = "hash(transaction_id), transaction_id"
SAMPLE_ID_DIGEST_ALGORITHM = "sha256_length_prefixed_utf8_in_sample_order"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sample_id_sha256(identifiers: pd.Series) -> str:
    digest = hashlib.sha256()
    for identifier in identifiers.astype(str):
        encoded = identifier.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _artifact_record(path: Path, root: Path, scope: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "scope": scope,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sql_string(value: str | Path) -> str:
    """Return a single-quoted DuckDB string literal for a trusted local path."""

    return "'" + str(value).replace("'", "''") + "'"


def _query_frame(connection: Any, sql: str) -> pd.DataFrame:
    return connection.execute(sql).fetchdf()


def _write_query_csv(connection: Any, sql: str, path: Path) -> int:
    """Write an aggregate query result; never fetch the broad source table."""

    frame = _query_frame(connection, sql)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return len(frame)


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FullEDAError(f"{name} must be a positive integer")
    return value


def _sampling_rationale(
    *,
    population_rows: int,
    sample_size: int,
    graph_max_edges: int | None,
) -> dict[str, Any]:
    networkx_edges = sample_size if graph_max_edges is None else min(sample_size, graph_max_edges)
    return {
        "constraint": (
            "Full-population pandas and NetworkX materialization is memory-bounded at "
            "HI-Small scale."
        ),
        "full_exact_policy": (
            "All feasible tabular aggregates are computed by DuckDB from every population row."
        ),
        "sampled_operations": [
            "plotting",
            "NetworkX component analysis",
            "NetworkX local-subgraph analysis",
        ],
        "selection_unit": "transaction_edge",
        "selection_method": "duckdb_stable_hash_order_first_n",
        "selection_order_sql": SAMPLE_ORDER_SQL,
        "selection_independent_of_target": True,
        "population_rows": population_rows,
        "descriptive_sample_rows": sample_size,
        "networkx_graph_edges": networkx_edges,
        "interpretation": (
            "Sampled graph statistics are descriptive of the selected edges and are not "
            "full-population graph estimates."
        ),
    }


def _create_source_view(connection: Any, view: str, parquet_path: Path) -> None:
    connection.execute(
        f"CREATE TEMP VIEW {view} AS "
        f"SELECT * FROM read_parquet({_sql_string(parquet_path.as_posix())})"
    )


def _validate_source(connection: Any, view: str, sample_size: int) -> int:
    columns = {str(row[0]) for row in connection.execute(f"DESCRIBE {view}").fetchall()}
    missing = sorted(FULL_REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise FullEDAError(f"Canonical Parquet is missing required EDA columns: {missing}")

    population_rows, null_transaction_ids, invalid_targets = connection.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT,
            COUNT(*) FILTER (WHERE transaction_id IS NULL)::BIGINT,
            COUNT(*) FILTER (
                WHERE is_laundering IS NULL
                   OR CAST(is_laundering AS BIGINT) NOT IN (0, 1)
            )::BIGINT
        FROM {view}
        """
    ).fetchone()
    population_rows = int(population_rows)
    if population_rows == 0:
        raise FullEDAError("Canonical Parquet contains no transactions")
    if int(null_transaction_ids):
        raise FullEDAError("Canonical Parquet contains null transaction_id values")
    if int(invalid_targets):
        raise FullEDAError("Canonical Parquet contains is_laundering values outside 0/1")
    if sample_size > population_rows:
        raise FullEDAError(
            f"sample_size={sample_size} exceeds population_rows={population_rows}; "
            "an exact-size sample cannot be selected"
        )
    return population_rows


def _write_amount_summary(connection: Any, view: str, path: Path) -> None:
    aggregates = _query_frame(
        connection,
        f"""
        SELECT
            COUNT(amount_paid)::BIGINT AS amount_paid_count,
            AVG(CAST(amount_paid AS DOUBLE)) AS amount_paid_mean,
            STDDEV_SAMP(CAST(amount_paid AS DOUBLE)) AS amount_paid_std,
            MIN(CAST(amount_paid AS DOUBLE)) AS amount_paid_min,
            QUANTILE_CONT(CAST(amount_paid AS DOUBLE), 0.50) AS amount_paid_p50,
            QUANTILE_CONT(CAST(amount_paid AS DOUBLE), 0.90) AS amount_paid_p90,
            QUANTILE_CONT(CAST(amount_paid AS DOUBLE), 0.95) AS amount_paid_p95,
            QUANTILE_CONT(CAST(amount_paid AS DOUBLE), 0.99) AS amount_paid_p99,
            MAX(CAST(amount_paid AS DOUBLE)) AS amount_paid_max,
            COUNT(amount_received)::BIGINT AS amount_received_count,
            AVG(CAST(amount_received AS DOUBLE)) AS amount_received_mean,
            STDDEV_SAMP(CAST(amount_received AS DOUBLE)) AS amount_received_std,
            MIN(CAST(amount_received AS DOUBLE)) AS amount_received_min,
            QUANTILE_CONT(CAST(amount_received AS DOUBLE), 0.50) AS amount_received_p50,
            QUANTILE_CONT(CAST(amount_received AS DOUBLE), 0.90) AS amount_received_p90,
            QUANTILE_CONT(CAST(amount_received AS DOUBLE), 0.95) AS amount_received_p95,
            QUANTILE_CONT(CAST(amount_received AS DOUBLE), 0.99) AS amount_received_p99,
            MAX(CAST(amount_received AS DOUBLE)) AS amount_received_max
        FROM {view}
        """,
    ).iloc[0]
    statistic_suffixes = (
        ("count", "count"),
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("50%", "p50"),
        ("90%", "p90"),
        ("95%", "p95"),
        ("99%", "p99"),
        ("max", "max"),
    )
    rows = [
        {
            "statistic": statistic,
            "amount_paid": aggregates[f"amount_paid_{suffix}"],
            "amount_received": aggregates[f"amount_received_{suffix}"],
        }
        for statistic, suffix in statistic_suffixes
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _generate_full_exact_tables(
    connection: Any,
    *,
    view: str,
    pair_table: str,
    degree_table: str,
    output_dir: Path,
    population_rows: int,
    top_n: int,
    canonical_fingerprint: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    class_path = tables_dir / "class_distribution.csv"
    _write_query_csv(
        connection,
        f"""
        WITH desired_labels(is_laundering) AS (VALUES (0), (1)),
        observed AS (
            SELECT CAST(is_laundering AS BIGINT) AS is_laundering, COUNT(*)::BIGINT AS count
            FROM {view}
            GROUP BY 1
        )
        SELECT
            desired_labels.is_laundering,
            COALESCE(observed.count, 0)::BIGINT AS count,
            COALESCE(observed.count, 0)::DOUBLE / {population_rows}::DOUBLE AS rate
        FROM desired_labels
        LEFT JOIN observed USING (is_laundering)
        ORDER BY is_laundering
        """,
        class_path,
    )
    artifacts.append(class_path)

    laundering_path = tables_dir / "laundering_rate.csv"
    class_table = pd.read_csv(class_path, usecols=["is_laundering", "rate"])
    class_table.to_csv(laundering_path, index=False)
    artifacts.append(laundering_path)

    amount_path = tables_dir / "amount_summary.csv"
    _write_amount_summary(connection, view, amount_path)
    artifacts.append(amount_path)

    activity_path = tables_dir / "activity_over_time.csv"
    _write_query_csv(
        connection,
        f"""
        SELECT
            CAST("timestamp" AS DATE) AS timestamp_bin,
            COUNT(*)::BIGINT AS transactions,
            SUM(CAST(is_laundering AS BIGINT))::BIGINT AS laundering_transactions
        FROM {view}
        GROUP BY 1
        ORDER BY 1
        """,
        activity_path,
    )
    artifacts.append(activity_path)

    payment_path = tables_dir / "payment_format_distribution.csv"
    _write_query_csv(
        connection,
        f"""
        SELECT CAST(payment_format AS VARCHAR) AS payment_format, COUNT(*)::BIGINT AS count
        FROM {view}
        GROUP BY 1
        ORDER BY count DESC, payment_format
        """,
        payment_path,
    )
    artifacts.append(payment_path)

    currency_path = tables_dir / "currency_distribution.csv"
    _write_query_csv(
        connection,
        f"""
        WITH currency_counts AS (
            SELECT
                CAST(payment_currency AS VARCHAR) AS currency,
                'payment' AS role,
                COUNT(*)::BIGINT AS count
            FROM {view}
            GROUP BY 1
            UNION ALL
            SELECT
                CAST(receiving_currency AS VARCHAR) AS currency,
                'receiving' AS role,
                COUNT(*)::BIGINT AS count
            FROM {view}
            GROUP BY 1
        )
        SELECT currency, count, role
        FROM currency_counts
        ORDER BY CASE role WHEN 'payment' THEN 0 ELSE 1 END, count DESC, currency
        """,
        currency_path,
    )
    artifacts.append(currency_path)

    bank_path = tables_dir / "bank_activity.csv"
    _write_query_csv(
        connection,
        f"""
        WITH bank_counts AS (
            SELECT
                CAST(from_bank AS VARCHAR) AS bank_id,
                'sender' AS role,
                COUNT(*)::BIGINT AS count
            FROM {view}
            GROUP BY 1
            UNION ALL
            SELECT
                CAST(to_bank AS VARCHAR) AS bank_id,
                'receiver' AS role,
                COUNT(*)::BIGINT AS count
            FROM {view}
            GROUP BY 1
        )
        SELECT bank_id, count, role
        FROM bank_counts
        ORDER BY CASE role WHEN 'sender' THEN 0 ELSE 1 END, count DESC, bank_id
        """,
        bank_path,
    )
    artifacts.append(bank_path)

    connection.execute(
        f"""
        CREATE TEMP TABLE {pair_table} AS
        SELECT
            CAST(from_node_id AS VARCHAR) AS from_node_id,
            CAST(to_node_id AS VARCHAR) AS to_node_id,
            COUNT(*)::BIGINT AS transfer_count,
            SUM(CAST(amount_paid AS DOUBLE)) AS total_amount_paid,
            SUM(CAST(is_laundering AS BIGINT))::BIGINT AS laundering_labels
        FROM {view}
        GROUP BY 1, 2
        """
    )
    repeated_path = tables_dir / "repeated_sender_receiver_pairs.csv"
    _write_query_csv(
        connection,
        f"""
        SELECT
            from_node_id,
            to_node_id,
            transfer_count,
            total_amount_paid,
            laundering_labels
        FROM {pair_table}
        WHERE transfer_count > 1
        ORDER BY transfer_count DESC, total_amount_paid DESC, from_node_id, to_node_id
        LIMIT {top_n}
        """,
        repeated_path,
    )
    artifacts.append(repeated_path)

    connection.execute(
        f"""
        CREATE TEMP TABLE {degree_table} AS
        WITH outgoing AS (
            SELECT
                CAST(from_node_id AS VARCHAR) AS node_id,
                COUNT(*)::BIGINT AS out_edge_count,
                COUNT(DISTINCT CAST(to_node_id AS VARCHAR))::BIGINT AS unique_successors
            FROM {view}
            GROUP BY 1
        ),
        incoming AS (
            SELECT
                CAST(to_node_id AS VARCHAR) AS node_id,
                COUNT(*)::BIGINT AS in_edge_count,
                COUNT(DISTINCT CAST(from_node_id AS VARCHAR))::BIGINT AS unique_predecessors
            FROM {view}
            GROUP BY 1
        )
        SELECT
            COALESCE(outgoing.node_id, incoming.node_id) AS node_id,
            COALESCE(incoming.in_edge_count, 0)::BIGINT AS in_edge_count,
            COALESCE(outgoing.out_edge_count, 0)::BIGINT AS out_edge_count,
            COALESCE(incoming.unique_predecessors, 0)::BIGINT AS unique_predecessors,
            COALESCE(outgoing.unique_successors, 0)::BIGINT AS unique_successors
        FROM outgoing
        FULL OUTER JOIN incoming USING (node_id)
        """
    )
    degree_path = tables_dir / "node_degree_counterparties.csv"
    _write_query_csv(
        connection,
        f"""
        SELECT
            node_id,
            in_edge_count,
            out_edge_count,
            unique_predecessors,
            unique_successors
        FROM {degree_table}
        ORDER BY node_id
        """,
        degree_path,
    )
    artifacts.append(degree_path)

    dataset_row = _query_frame(
        connection,
        f"""
        SELECT
            COUNT(*)::BIGINT AS rows,
            MIN("timestamp") AS timestamp_minimum,
            MAX("timestamp") AS timestamp_maximum,
            SUM(CAST(is_laundering AS BIGINT))::BIGINT AS positive_labels,
            AVG(CAST(is_laundering AS DOUBLE)) AS laundering_rate,
            COUNT(DISTINCT CAST(from_node_id AS VARCHAR))::BIGINT AS unique_sender_nodes,
            COUNT(DISTINCT CAST(to_node_id AS VARCHAR))::BIGINT AS unique_receiver_nodes,
            COUNT(DISTINCT CAST(payment_format AS VARCHAR))::BIGINT AS payment_formats,
            COUNT(DISTINCT CAST(payment_currency AS VARCHAR))::BIGINT AS payment_currencies,
            COUNT(DISTINCT CAST(receiving_currency AS VARCHAR))::BIGINT AS receiving_currencies,
            COUNT(*) FILTER (WHERE from_node_id = to_node_id)::BIGINT AS self_loop_edges
        FROM {view}
        """,
    ).iloc[0]
    pair_row = _query_frame(
        connection,
        f"""
        SELECT
            COUNT(*)::BIGINT AS unique_directed_pairs,
            COUNT(*) FILTER (WHERE transfer_count > 1)::BIGINT AS repeated_directed_pairs
        FROM {pair_table}
        """,
    ).iloc[0]
    degree_row = _query_frame(
        connection,
        f"""
        SELECT
            COUNT(*)::BIGINT AS nodes,
            MAX(in_edge_count)::BIGINT AS maximum_in_degree,
            MAX(out_edge_count)::BIGINT AS maximum_out_degree
        FROM {degree_table}
        """,
    ).iloc[0]

    unique_pairs = int(pair_row["unique_directed_pairs"])
    node_count = int(degree_row["nodes"])
    counts_payload: dict[str, Any] = {
        "scope": "full_exact",
        "provenance": {
            **dict(provenance),
            "canonical_parquet": dict(canonical_fingerprint),
        },
        "dataset": {
            "rows": int(dataset_row["rows"]),
            "timestamp_minimum": pd.Timestamp(dataset_row["timestamp_minimum"]).isoformat(),
            "timestamp_maximum": pd.Timestamp(dataset_row["timestamp_maximum"]).isoformat(),
            "positive_labels": int(dataset_row["positive_labels"]),
            "laundering_rate": float(dataset_row["laundering_rate"]),
            "unique_sender_nodes": int(dataset_row["unique_sender_nodes"]),
            "unique_receiver_nodes": int(dataset_row["unique_receiver_nodes"]),
            "unique_nodes": node_count,
            "unique_directed_pairs": unique_pairs,
            "repeated_directed_pairs": int(pair_row["repeated_directed_pairs"]),
            "payment_formats": int(dataset_row["payment_formats"]),
            "payment_currencies": int(dataset_row["payment_currencies"]),
            "receiving_currencies": int(dataset_row["receiving_currencies"]),
        },
        "graph_counts": {
            "directed": True,
            "multigraph": True,
            "edges": population_rows,
            "nodes": node_count,
            "unique_directed_pairs": unique_pairs,
            "repeated_edge_occurrences_after_first": population_rows - unique_pairs,
            "self_loop_edges": int(dataset_row["self_loop_edges"]),
            "maximum_in_degree": int(degree_row["maximum_in_degree"]),
            "maximum_out_degree": int(degree_row["maximum_out_degree"]),
            "mean_in_degree": population_rows / node_count,
            "mean_out_degree": population_rows / node_count,
            "component_metrics_computed": False,
            "component_metrics_reason": (
                "NetworkX component materialization is intentionally limited to the "
                "separately labeled deterministic edge sample."
            ),
        },
        "limitations": [
            "Amounts in different currencies are not converted or compared directly.",
            "Full-exact graph counts do not include weak or strong connected components.",
            "A positive source label is evidence for human review, not a guilt finding.",
        ],
    }
    counts_path = output_dir / "dataset_graph_counts.json"
    _write_json(counts_payload, counts_path)
    artifacts.append(counts_path)
    return counts_payload, artifacts


def _select_sample(
    connection: Any,
    *,
    view: str,
    sample_size: int,
) -> pd.DataFrame:
    columns = (
        "transaction_id",
        '"timestamp"',
        "from_bank",
        "to_bank",
        "from_node_id",
        "to_node_id",
        "amount_paid",
        "amount_received",
        "payment_format",
        "payment_currency",
        "receiving_currency",
        "is_laundering",
    )
    sample = _query_frame(
        connection,
        f"""
        SELECT {", ".join(columns)}
        FROM {view}
        ORDER BY {SAMPLE_ORDER_SQL}
        LIMIT {sample_size}
        """,
    )
    if len(sample) != sample_size:
        raise FullEDAError(
            f"Stable hash query returned {len(sample)} row(s); expected exactly {sample_size}"
        )
    return sample


def generate_full_eda_artifacts(
    connection: Any,
    canonical_parquet_path: str | Path,
    output_dir: str | Path,
    *,
    sample_size: int = 50_000,
    graph_max_edges: int | None = 50_000,
    top_n: int = 20,
    random_seed: int = 42,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate full-exact tables and separately labeled sampled graph plots.

    Args:
        connection: Open DuckDB connection used for out-of-core SQL aggregation.
        canonical_parquet_path: Canonical Sprint 1 transaction Parquet file.
        output_dir: Root directory for full and sampled EDA artifacts.
        sample_size: Exact number of transactions selected for descriptive plots.
        graph_max_edges: Maximum sampled transactions materialized as a NetworkX graph.
        top_n: Maximum number of exact repeated-pair rows to emit.
        random_seed: Seed for the capped NetworkX subsample and graph layout.
        provenance: Optional upstream dataset/run provenance copied into manifests.

    Sampling membership is deliberately independent of physical Parquet row
    order: DuckDB selects the first ``sample_size`` rows after
    ``ORDER BY hash(transaction_id), transaction_id``.  DuckDB's version is
    recorded because hash stability is version scoped.
    """

    sample_size = _validate_positive_integer(sample_size, "sample_size")
    if graph_max_edges is not None:
        graph_max_edges = _validate_positive_integer(graph_max_edges, "graph_max_edges")
    top_n = _validate_positive_integer(top_n, "top_n")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise FullEDAError("random_seed must be a non-negative integer")

    source = Path(canonical_parquet_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Canonical Parquet file not found: {source}")
    root = Path(output_dir).expanduser().resolve()
    full_exact_dir = root / "full_exact"
    sample_dir = root / "sample_descriptive"
    full_exact_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    provided_provenance = dict(provenance or {})
    started = time.perf_counter()
    started_at = datetime.now(UTC)
    stage_started = time.perf_counter()
    canonical_fingerprint = _file_fingerprint(source)
    runtimes: dict[str, float] = {
        "canonical_parquet_fingerprint_seconds": time.perf_counter() - stage_started
    }
    duckdb_version = str(connection.execute("SELECT version()").fetchone()[0])

    token = uuid.uuid4().hex
    view = f"_argus_full_eda_source_{token}"
    pair_table = f"_argus_full_eda_pairs_{token}"
    degree_table = f"_argus_full_eda_degrees_{token}"
    try:
        _create_source_view(connection, view, source)
        stage_started = time.perf_counter()
        population_rows = _validate_source(connection, view, sample_size)
        runtimes["source_validation_seconds"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        counts_payload, exact_paths = _generate_full_exact_tables(
            connection,
            view=view,
            pair_table=pair_table,
            degree_table=degree_table,
            output_dir=full_exact_dir,
            population_rows=population_rows,
            top_n=top_n,
            canonical_fingerprint=canonical_fingerprint,
            provenance=provided_provenance,
        )
        runtimes["full_exact_aggregation_seconds"] = time.perf_counter() - stage_started

        # These high-cardinality aggregates can occupy substantial DuckDB memory
        # or spill space.  Their CSV/JSON evidence is complete, so release them
        # before materializing the descriptive pandas/NetworkX sample.
        connection.execute(f"DROP TABLE IF EXISTS {degree_table}")
        connection.execute(f"DROP TABLE IF EXISTS {pair_table}")

        exact_records = [_artifact_record(path, root, "full_exact") for path in sorted(exact_paths)]
        full_exact_manifest_path = full_exact_dir / "full_exact_manifest.json"
        full_exact_manifest: dict[str, Any] = {
            "status": "PASS",
            "scope": "full_exact",
            "generated_by": "argus.full_eda.generate_full_eda_artifacts",
            "population_rows": population_rows,
            "duckdb_version": duckdb_version,
            "canonical_parquet": canonical_fingerprint,
            "artifact_count_excluding_manifest": len(exact_records),
            "artifacts": exact_records,
            "quantile_method": "DuckDB quantile_cont",
        }
        _write_json(full_exact_manifest, full_exact_manifest_path)

        stage_started = time.perf_counter()
        sample = _select_sample(
            connection,
            view=view,
            sample_size=sample_size,
        )
        sample_digest = _sample_id_sha256(sample["transaction_id"])
        sampling_rationale = _sampling_rationale(
            population_rows=population_rows,
            sample_size=sample_size,
            graph_max_edges=graph_max_edges,
        )
        sample_ids_path = sample_dir / "sample_transaction_ids.csv"
        pd.DataFrame(
            {
                "selection_rank": range(1, sample_size + 1),
                "transaction_id": sample["transaction_id"].astype(str),
            }
        ).to_csv(sample_ids_path, index=False)
        sample_provenance = {
            **provided_provenance,
            "artifact_scope": "sample_descriptive_only",
            "canonical_parquet": canonical_fingerprint,
            "population_rows": population_rows,
            "sample_rows": sample_size,
            "sampling_unit": "transaction_edge",
            "sampling_method": "duckdb_stable_hash_order_first_n",
            "sampling_order_sql": SAMPLE_ORDER_SQL,
            "sample_transaction_id_sha256": sample_digest,
            "sample_transaction_id_sha256_algorithm": SAMPLE_ID_DIGEST_ALGORITHM,
            "sample_fraction_of_population": sample_size / population_rows,
            "duckdb_version": duckdb_version,
            "networkx_graph_max_edges": graph_max_edges,
            "networkx_graph_edges": (
                sample_size if graph_max_edges is None else min(sample_size, graph_max_edges)
            ),
            "random_seed_for_networkx_subsample_and_layout": random_seed,
            "sampling_rationale": sampling_rationale,
            "limitations": [
                "Plots and NetworkX statistics describe only the selected edge sample.",
                "Sampled degree and component values are not full-population graph estimates.",
            ],
        }
        sampled_eda_manifest = generate_eda_artifacts(
            sample,
            sample_dir,
            random_seed=random_seed,
            graph_max_edges=graph_max_edges,
            top_n=top_n,
            provenance=sample_provenance,
        )
        runtimes["sample_selection_and_descriptive_eda_seconds"] = (
            time.perf_counter() - stage_started
        )

        sampled_records = [
            _artifact_record(
                sample_dir / entry["path"],
                root,
                "sample_descriptive_only",
            )
            for entry in sampled_eda_manifest["artifacts"]
        ]
        sampled_manifest_path = sample_dir / "eda_manifest.json"
        sampled_records.append(
            _artifact_record(sampled_manifest_path, root, "sample_descriptive_only")
        )
        exact_records.append(_artifact_record(full_exact_manifest_path, root, "full_exact"))

        finished_at = datetime.now(UTC)
        total_runtime = time.perf_counter() - started
        runtimes["total_seconds"] = total_runtime
        manifest_path = root / "full_eda_manifest.json"
        manifest: dict[str, Any] = {
            "status": "PASS",
            "sprint": "Sprint 1 - Repository Foundation + Data Proof",
            "generated_by": "argus.full_eda.generate_full_eda_artifacts",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "runtime_seconds": total_runtime,
            "runtime_by_stage_seconds": runtimes,
            "duckdb_version": duckdb_version,
            "canonical_parquet": canonical_fingerprint,
            "population_rows": population_rows,
            "provenance": provided_provenance,
            "sampling_rationale": sampling_rationale,
            "scopes": {
                "full_exact": {
                    "label": "full_exact",
                    "rows": population_rows,
                    "artifact_count_including_scope_manifest": len(exact_records),
                    "dataset_graph_counts": counts_payload,
                },
                "sample_descriptive_only": {
                    "label": "sample_descriptive_only",
                    "population_rows": population_rows,
                    "requested_rows": sample_size,
                    "actual_rows": len(sample),
                    "sampling_unit": "transaction_edge",
                    "sampling_method": "duckdb_stable_hash_order_first_n",
                    "sampling_order_sql": SAMPLE_ORDER_SQL,
                    "duckdb_version": duckdb_version,
                    "sample_transaction_id_sha256": sample_digest,
                    "sample_transaction_id_sha256_algorithm": SAMPLE_ID_DIGEST_ALGORITHM,
                    "sample_fraction_of_population": sample_size / population_rows,
                    "networkx_graph_max_edges": graph_max_edges,
                    "networkx_graph_edges": (
                        sample_size
                        if graph_max_edges is None
                        else min(sample_size, graph_max_edges)
                    ),
                    "networkx_graph_fraction_of_population": (
                        sample_size
                        if graph_max_edges is None
                        else min(sample_size, graph_max_edges)
                    )
                    / population_rows,
                    "random_seed_for_networkx_subsample_and_layout": random_seed,
                    "sampling_rationale": sampling_rationale,
                    "artifact_count_including_scope_manifest": len(sampled_records),
                    "limitations": [
                        "All plots and NetworkX values in sample_descriptive/ describe only "
                        "the deterministic edge sample.",
                        "They must not be reported as full-population degree or component facts.",
                    ],
                },
            },
            "artifact_count_excluding_manifest": len(exact_records) + len(sampled_records),
            "artifacts": sorted(
                [*exact_records, *sampled_records],
                key=lambda entry: entry["path"],
            ),
            "models": "NOT_IN_SPRINT_1",
        }
        _write_json(manifest, manifest_path)
        return manifest
    finally:
        for object_type, name in (
            ("TABLE", degree_table),
            ("TABLE", pair_table),
            ("VIEW", view),
        ):
            try:
                connection.execute(f"DROP {object_type} IF EXISTS {name}")
            except Exception:
                # Preserve the original failure; these are connection-local temp objects.
                pass
