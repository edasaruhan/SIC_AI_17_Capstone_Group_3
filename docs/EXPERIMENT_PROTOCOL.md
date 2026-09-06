# ARGUS AI Experiment Protocol

**Protocol version:** 1.0  
**Active execution scope:** Sprint 1 data proof  
**Pipeline/test results:** `PASS` for quick, full out-of-core, and full raw-audit paths

This document defines how ARGUS experiments become comparable and scientifically
defensible. It is a protocol, not a results report. No metric in this document is
an observed model result.

## 1. Reproducibility unit

The reproducibility unit is one run with:

- a source-data fingerprint;
- a resolved YAML configuration;
- a deterministic seed;
- a declared sampling scope;
- a code revision or source-tree fingerprint where available;
- start/end timestamps and runtime;
- split boundaries and row counts;
- an inventory of generated artifacts;
- a terminal status and error details if incomplete.

The quick and full runs write inventories to `artifacts/quick/run_manifest.json`
and `artifacts/full/run_manifest.json`. Generated artifacts and local datasets are
Git-ignored, so manifests and commands must make them reproducible rather than
implying they are committed.

## 2. Fixed source snapshot

The current full-file reference is:

| Input | SHA-256 | Rows |
| --- | --- | ---: |
| `data/raw/HI-Small_Trans.csv` | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` | 5,078,345 |
| `data/raw/HI-Small_accounts.csv` | `786808526e33cfc441212dd6fccda7edfc24172149bed59c6ef59b186836b014` | 518,581 |

The row counts and hashes above were obtained through an independent full-file
audit and reproduced by `artifacts/full/raw_data_audit.json`. A different hash is a
different dataset version and must not be merged into the same result series
without explicit documentation.

## 3. Run modes

### Quick mode

- Config: `configs/quick.yaml`
- Seed: `42`
- Sampling enabled; maximum 10,000 source rows
- Purpose: development, CI, leakage tests, and fast artifact-contract checks
- Output root: `artifacts/quick/`
- Interpretation: sampled engineering evidence only
- Verified result: PASS; 10,000 rows, 52 columns, 39 generated artifacts, and
  11.197340699996857 seconds runtime

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_quick_pipeline.py
```

### Full mode

- Config: `configs/full.yaml`
- Seed: `42`
- Sampling disabled
- Purpose: exact full HI-Small preprocessing, splitting, feature engineering, and
  feasible tabular profiling
- Output root: `artifacts/full/`
- Engine: DuckDB 1.5.5, `2GB`, one thread, external spill enabled
- Verified result: PASS; 5,078,345 rows, 52 columns, 304.93406899999536 seconds
- Interpretation: exact full-data tables are distinct from sampled plots/NetworkX

Commands:

```powershell
.\.venv\Scripts\python.exe scripts/audit_raw_data.py
.\.venv\Scripts\python.exe scripts/run_full_pipeline.py
```

The streaming raw audit and out-of-core full pipeline both passed. Canonical,
feature, split, and graph-edge outputs each contain 5,078,345 unique transaction
IDs. The first `1GB`/two-thread attempt exhausted memory in the six-way feature
export join after completing the history stages. The successful retry used `2GB`
and one thread, materialized the feature join before sorted export, and cleaned its
temporary work files after success.

The full-run stage runtimes, in seconds, are preserved in the manifest:

| Stage | Seconds |
| --- | ---: |
| Source fingerprints | 0.4863478999977815 |
| DuckDB initialization | 0.018459499995515216 |
| Single-threaded ingestion | 13.602533099998254 |
| Raw validation | 7.242892599999323 |
| Canonical external sort/export | 23.96039380000002 |
| Full account-reference check | 4.587355700001353 |
| Chronological split | 0.2312817999991239 |
| Sender history | 36.51647480000247 |
| Receiver history | 29.85599199999706 |
| Directed graph-history features | 33.679646700002195 |
| Feature join materialization | 66.20344630000181 |
| Materialized-feature verification | 6.057654500000353 |
| Sorted feature export | 32.58873909999966 |
| Split and graph exports | 8.66468559999339 |
| Full and sampled EDA | 26.223705899996276 |
| SQL output verification | 13.752454000001308 |
| Work-file cleanup | 0.08349149999412475 |
| Artifact hash inventory | 1.0316739000045345 |

## 4. Deterministic ingestion and preprocessing

1. Validate the transaction and accounts headers before renaming.
2. Read identifiers as strings; do not let CSV inference destroy identity.
3. Preserve the physical source-row identity.
4. Parse timestamps and numeric amounts with explicit failure behavior.
5. Validate target values, blanks, non-finite/negative amounts, and exact duplicate
   rows.
6. Normalize bank IDs consistently across both files.
7. Construct `from_node_id` and `to_node_id` as bank-account composites.
8. Verify transaction endpoints against the accounts table.
9. Use a stable deterministic order for tied timestamps.
10. Record sampling and duplicate policy in the run manifest.

Repeated sender→receiver transfers are not automatically duplicates and must remain
available to history and graph features. Exact source-row duplicates are reported
separately; no silent row-count change is allowed.

## 5. Primary chronological split

The base configuration requests:

```text
train      70%
validation 15%
test       15%
```

The splitter must operate on chronological timestamp groups, not shuffled rows.
When a target fraction intersects a timestamp group, the entire group is assigned
to one partition. Realized fractions may therefore differ slightly from configured
fractions.

Required invariants for non-empty partitions:

```text
max(train.timestamp) < min(validation.timestamp)
max(validation.timestamp) < min(test.timestamp)
```

Additional invariants:

- each source transaction belongs to exactly one partition;
- no transaction identity overlaps partitions;
- concatenated partition row counts equal the eligible canonical input count;
- repeated executions with the same input/config produce the same assignment;
- target labels do not influence boundaries;
- timestamp ties never cross a boundary.

Quick evidence was saved at `artifacts/quick/metadata/split_metadata.json` and
`artifacts/quick/tables/split_manifest.csv`:

| Partition | Rows | Timestamp minimum–maximum | Positives |
| --- | ---: | --- | ---: |
| Train | 7,011 | `2022-09-01 00:00`–`00:20` | 0 |
| Validation | 1,657 | `2022-09-01 00:21`–`00:25` | 1 |
| Test | 1,332 | `2022-09-01 00:26`–`00:29` | 0 |

The artifact records `strict_boundaries_verified=true`,
`timestamp_groups_kept_intact=true`, and
`no_transaction_overlap_verified=true`. Saved-run verification independently
confirmed strict chronology and no transaction-ID overlap.

Full evidence is saved at `artifacts/full/metadata/split_metadata.json` and
`artifacts/full/tables/split_manifest.parquet`:

| Partition | Rows | Timestamp minimum–maximum | Positives |
| --- | ---: | --- | ---: |
| Train | 3,554,957 | `2022-09-01 00:00`–`2022-09-07 14:55` | 2,856 |
| Validation | 761,749 | `2022-09-07 14:56`–`2022-09-09 03:16` | 760 |
| Test | 761,639 | `2022-09-09 03:17`–`2022-09-18 16:18` | 1,561 |

The full verification report confirms strict chronology, timestamp-group
integrity, 5,078,345 split rows, and no row overlap.

## 6. Leakage-safe feature timing

### Current-row features

Amount logs, bank equality, currency equality, and same-currency amount comparisons
may use the current row because they are observed at transaction time.

### Historical features

For an event at `t`, every count, volume, counterparty, degree, burst, and repeated-
pair feature uses only events with timestamp `< t`.

Events that share `t` are processed as one temporal batch. None may observe another
event from that batch, regardless of source-row order.

Required automated proofs include:

- same-timestamp events receive identical eligible prior state for an entity;
- appending future events does not change earlier feature rows;
- first events have zero counts/volumes and null time gaps where appropriate;
- sender→receiver direction is preserved;
- repeated pair count changes only after a prior timestamp;
- labels are never inputs to feature generation.

## 7. Sprint 1 EDA protocol

EDA must call reusable code or consume generated artifacts. It must not embed
manually typed plot values. At minimum, Sprint 1 should generate observed evidence
for:

- class count and laundering rate;
- amount and log-amount distributions;
- transaction and labeled activity over time;
- payment-format and currency distributions;
- sender/receiver bank activity;
- repeated directed pairs and unique counterparties;
- manageable directed-degree/component summaries;
- labeled local examples only where the data and sampling scope support them.

Every figure/table must state run mode and row scope. Empty or unavailable analyses
must be reported, not replaced with decorative data. The quick run generated all
required EDA families from 10,000 observed IBM rows: 13 PNG files under
`artifacts/quick/eda/figures/`, 12 CSV tables under
`artifacts/quick/eda/tables/`, plus dataset, graph, suspicious-subgraph, and EDA
manifests. This remains sampled EDA, not full-source EDA.

The full run uses two scopes recorded in
`artifacts/full/eda/full_eda_manifest.json`:

- `full_exact`: DuckDB computes feasible tabular aggregates from all 5,078,345
  rows, including exact class/rate, amount summary, daily activity, format,
  currency, bank, repeated-pair, node degree/counterparty, and dataset/graph-count
  outputs;
- `sample_descriptive_only`: plotting and NetworkX component/local-subgraph work
  use exactly 100,000 target-independent transaction edges selected by
  `ORDER BY hash(transaction_id), transaction_id`. NetworkX is capped at 50,000
  edges.

The sample transaction-ID SHA-256 is
`e81cee70a8c52d088f5c7db21ba61bbabe2bf82fb32a5be8c73aca854400ded5`.
Sampled graph statistics are descriptive and are not population estimates.

## 8. Artifact verification

Verified full-run core paths include:

```text
artifacts/full/run_manifest.json
artifacts/full/verification_report.json
artifacts/full/metadata/split_metadata.json
artifacts/full/tables/transactions_clean.parquet
artifacts/full/tables/transaction_features.parquet
artifacts/full/tables/split_manifest.parquet
artifacts/full/tables/graph_edges.parquet
artifacts/full/eda/full_eda_manifest.json
artifacts/full/eda/full_exact/
artifacts/full/eda/sample_descriptive/
```

Verified quick-run core paths:

```text
artifacts/quick/run_manifest.json
artifacts/quick/metadata/split_metadata.json
artifacts/quick/tables/transactions_clean.csv
artifacts/quick/tables/transaction_features.csv
artifacts/quick/tables/split_manifest.csv
artifacts/quick/account_reference_report.json
artifacts/quick/raw_validation_report.json
artifacts/quick/canonical_validation_report.json
artifacts/quick/resolved_config.json
artifacts/quick/tables/graph_edges.csv
artifacts/quick/eda/dataset_profile.json
artifacts/quick/eda/eda_manifest.json
artifacts/quick/eda/figures/
artifacts/quick/eda/tables/
```

A run is successful only if:

1. the command exits zero;
2. the manifest says it completed;
3. all declared artifacts exist and are readable;
4. recorded row counts reconcile across stages;
5. split invariants pass;
6. figures/tables identify quick versus full scope;
7. no metric or plot is copied from documentation placeholders.

All seven conditions passed for their declared scopes. The quick manifest hashes
38 payloads; the manifest itself makes 39 generated quick artifacts.
`verify_run.py` verified all 38 quick hashes,
10,000 feature rows, 10,000 split rows, strict chronology, disjoint transaction
IDs, and 13 readable PNG figures. The pre-existing `.gitkeep` is not counted.

The full manifest hashes 54 payloads; with the manifest itself, the published full
evidence set has 55 output files (excluding `.gitkeep`). This inventory includes
`raw_data_audit.json`, which
predates the successful preprocessing retry. `verification_report.json` reconciles
all four 5,078,345-row Parquet
outputs using DuckDB counts, distinct counts, `EXCEPT` identity checks, schema
checks, and split aggregates without Python identifier sets.

## 9. Test execution evidence

The canonical test command is:

```powershell
.\.venv\Scripts\pytest.exe -q
```

The fresh final run passed 34 tests in 25.36 seconds under CPython 3.12.10.
Ruff lint, Ruff format verification, and `pip check` also pass; exact commands and
resolved intermediate failures are preserved in the final Sprint status report.

## 10. Recommended Sprint 2 model protocol

This section is a recommendation only. No Sprint 2 implementation or result is
claimed.

### Models

Train three transaction-level baselines on identical partitions and feature scope:

1. Logistic Regression;
2. Random Forest;
3. LightGBM or XGBoost, chosen once with dependency/runtime justification.

Any encoding, imputation, scaling, feature selection, or class weighting with
learned state must be fit on training data only. Do not oversample before splitting.

### Selection and test discipline

- Train on `train`.
- Compare/tune on `validation`.
- Name a Transaction Baseline Champion using validation evidence only.
- Optimize an operational threshold using validation only.
- Freeze preprocessing, features, model, and threshold before final test use.
- Do not repeatedly inspect test results or retune after seeing them.

Final untouched-test evaluation belongs to the later final-test phase, not Sprint 2
selection.

### Recommended metrics

Accuracy is not a primary metric because the audited positive rate is only
0.101942660453%. Save, at minimum:

- PR-AUC, with implementation explicitly identified (for example average
  precision rather than an ambiguous label);
- ROC-AUC as secondary context;
- precision, recall, F1, and false-positive rate at the selected threshold;
- alert count/volume;
- Recall@K and Precision@K for configured values such as 100, 500, and 1,000;
- confusion matrix;
- runtime and model/config metadata.

Top-K ranking must use descending score with a deterministic tie-breaker. If `K`
exceeds partition size, the behavior must be explicit. Threshold and K metrics must
be computed on the same frozen predictions, not mixed across runs.

### Comparison artifacts

Sprint 2 should generate machine-readable JSON/CSV and plots from code. It should
not populate Markdown with hand-copied numbers. The comparison table must identify
dataset hash, split metadata, feature family, model version, threshold basis, and
whether results are validation or test.

## 11. Human review

Metrics measure ranking/classification behavior on synthetic labels. They do not
establish guilt or justify adverse action. Any later alert or case must show
observed evidence separately from model contribution, state uncertainty, and remain
subject to trained human review.
