# ARGUS AI — Sprint 1 Final Engineering Status

**Report date:** 2026-09-07  
**Authorized scope:** Sprint 1 — Repository Foundation + Data Proof only  
**Overall status:** **PASS**  
**Acceptance criteria:** **14/14 PASS**  
**IBM data blocker:** **RESOLVED**  
**Sprint 2 work:** **NOT STARTED**

The deterministic quick and full Sprint 1 pipelines, saved-run verification,
complete test suite, quality checks, and full streaming source audit all passed.
No model metric, baseline champion, GraphSAGE result, or graph-value conclusion is
claimed.

## 1. Authority and instruction separation

### Explicit user request

The user directed the work to:

1. treat `ARGUS_CODEX_MASTER_PROMPT.md` as the main technical specification;
2. inspect the existing workspace without damaging existing ARGUS material;
3. work only through Sprint 1 and not proceed to Sprint 2;
4. implement, run, test, fix, and continue until Sprint 1 criteria passed as far as
   evidence allowed;
5. use the supplied `HI-Small_Trans.csv` and `HI-Small_accounts.csv` without
   inventing metrics or graphics;
6. build the project in the desktop folder named `ARGUS PROJECET`;
7. report changes, commands, tests, artifacts, limitations, and final status.

### Attached documents

The user-designated master prompt supplied the technical requirements and fourteen
Sprint 1 acceptance criteria. The other six academic documents were supporting
course context, not executable instructions. They were not allowed to override the
user request or master prompt. All seven documents were preserved unchanged under
`reports/existing_coursework/` with SHA-256 values in `SHA256SUMS.txt`.

The two CSV files are data inputs, not instruction-bearing documents. All nine
copied inputs—seven documents and two CSVs—match their corresponding Downloads
files byte-for-byte by SHA-256.

## 2. Repository isolation and preservation

The initial working directory, `DATATON`, was an unrelated GridUp Git repository.
Read-only inspection found:

- local `main` equal to `origin/main`;
- existing modifications in `pyproject.toml`, `requirements.txt`, and
  `src/gridup/audit.py`;
- approximately 2,564 untracked paths totaling approximately 13.4 GiB.

The existing GridUp unittest suite was run read-only and passed 92/92. That result
belongs to GridUp and is explicitly excluded from ARGUS acceptance evidence. All
DATATON/GridUp files belonged to the user and were left untouched.

ARGUS was created as a separate Git repository at:

```text
C:\Users\MONSTER\OneDrive - Istanbul Kultur Universitesi\Masaüstü\ARGUS PROJECET
```

`git init -b main` completed there. Raw CSVs, generated artifacts, `.venv`, caches,
and packaging outputs are ignored. Documentation placeholders and
`artifacts/README.md` remain trackable.

## 3. Final verification summary

| Verification | Final result | Evidence |
| --- | --- | --- |
| Sprint 1 acceptance | **PASS** | 14/14 criteria below |
| Editable development install | **PASS** | Approved rerun completed |
| Dependency consistency | **PASS** | `No broken requirements found.` |
| Unit/integration suite | **PASS** | 34 passed in 25.36 seconds |
| Ruff lint | **PASS** | `All checks passed!` |
| Ruff formatting | **PASS** | Format verification completed |
| Quick pipeline | **PASS** | 10,000 rows, 52 columns, 11.197340699996857 seconds |
| Saved quick-run verification | **PASS** | 38 hashes, 13 figures, 10,000 feature/split rows |
| Strict split chronology | **PASS** | Strict boundaries and no ID overlap |
| Quick EDA | **PASS** | 13 PNG, 12 CSV, generated JSON/manifests |
| Full streaming raw audit | **PASS** | `artifacts/full/raw_data_audit.json` |
| Full preprocessing/features/split | **PASS** | 5,078,345 rows, 52 columns, unsampled |
| Full EDA | **PASS** | Full-exact tables plus labeled 100,000-row descriptive sample |
| Full pipeline runtime | **PASS** | 304.93406899999536 seconds using DuckDB 1.5.5, `2GB`, one thread |
| Saved full-run verification | **PASS** | Four 5,078,345-row outputs; strict chronology and identical IDs |
| Sprint 2 models/metrics | **NOT IN SCOPE** | No results claimed |

## 4. Master-prompt acceptance criteria

| # | Criterion | Status | Executable evidence |
| ---: | --- | --- | --- |
| 1 | Existing ARGUS documents are preserved | **PASS** | Seven files retained under `reports/existing_coursework/`; checksums match Downloads |
| 2 | Target package structure exists | **PASS** | Separate `main` Git repository with config, package, scripts, tests, docs, notebook, reports, data, and artifact areas |
| 3 | Dependency management works | **PASS** | Editable dev install completed; `pip check` reports no broken requirements |
| 4 | IBM AML HI-Small loads from a documented local path | **PASS** | Real files loaded from `data/raw/`; quick run and full audit use verified hashes |
| 5 | Schema validation works | **PASS** | Full raw/canonical reports validate all 5,078,345 transactions; schema tests pass |
| 6 | Preprocessing works | **PASS** | `transactions_clean.parquet` has 5,078,345 stable canonical rows, normalized composite nodes, and nine exact duplicates preserved |
| 7 | Chronological train/validation/test split works | **PASS** | 3,554,957/761,749/761,639 full rows; intact timestamp groups and strict boundaries |
| 8 | Leakage tests exist | **PASS** | Tests cover same-timestamp batching, strictly-prior state, future-append invariance, split overlap, and direction; final suite passes |
| 9 | EDA generates real artifacts | **PASS** | Exact all-row tables and clearly scoped 100,000-row plots/50,000-edge NetworkX evidence were generated |
| 10 | Transaction/time/history features are generated | **PASS** | `transaction_features.parquet` has 5,078,345 rows and 52 total columns, including all configured feature families |
| 11 | Initial fan-in/fan-out graph features are generated | **PASS** | Five directed, strictly-prior graph-history columns plus full `graph_edges.parquet`; graph tests pass |
| 12 | `pytest` passes | **PASS** | Fresh `.venv\Scripts\pytest.exe -q`: 34 passed in 25.36 seconds |
| 13 | `python scripts/run_quick_pipeline.py` completes | **PASS** | Manifest status PASS; runtime 11.197340699996857 seconds; 39 generated artifacts including manifest |
| 14 | README explains exact reproduction | **PASS** | Paths, setup, commands, outputs, quick/full distinction, limitations, and responsible-use language synchronized with final artifacts |

All fourteen items have direct code, test, or artifact evidence. Sprint 1 is not
being marked PASS merely because an expected filename exists.

## 5. Exact commands and outcomes

Commands below were run from the `ARGUS PROJECET` repository unless noted.

| Purpose | Exact invocation | Final outcome |
| --- | --- | --- |
| Initialize isolated repository | `git init -b main` | PASS; separate `main` repository created |
| Install editable dev environment | `.venv\Scripts\python.exe -m pip install -e ".[dev]"` | PASS after approved network-capable rerun |
| Final tests | `.venv\Scripts\pytest.exe -q` | PASS; 34 passed in 25.36s |
| Lint | `.venv\Scripts\python.exe -m ruff check --no-cache src scripts tests` | PASS; `All checks passed!` |
| Format verification | `.venv\Scripts\python.exe -m ruff format --no-cache --check src scripts tests` | PASS |
| Dependency check | `.venv\Scripts\python.exe -m pip check` | PASS; no broken requirements |
| Quick vertical pipeline | `.venv\Scripts\python.exe scripts\run_quick_pipeline.py` | PASS; 10,000 rows, 39 artifacts |
| Verify saved run | `.venv\Scripts\python.exe scripts\verify_run.py` | PASS; 38 hashes, 13 figures, strict chronology/no overlap |
| Full streaming data audit | `.venv\Scripts\python.exe scripts\audit_raw_data.py` | PASS; full JSON audit written |
| Full Sprint 1 pipeline | `.venv\Scripts\python.exe scripts\run_full_pipeline.py` | PASS; 5,078,345 rows, 52 columns, 304.93406899999536s |

The quick pipeline internally executed loading, validation, preprocessing, account
reference checks, splitting, EDA, transaction/time/history features, directed
graph-history features, graph edge export, and manifest generation. Standalone
scripts remain available but their presence is not counted as separate execution
evidence unless listed above.

## 6. Resolved intermediate failures

Failures encountered during implementation are retained here rather than hidden by
the final green run.

| Stage | Intermediate observation | Diagnosis and correction | Final evidence |
| --- | --- | --- | --- |
| Dependency installation | Initial pip attempt could not reach dependencies inside the restricted sandbox | Requested the required approval and reran installation with network access | Editable install PASS; `pip check` PASS |
| Full-audit artifact write | First audit attempt was denied when writing to the user-requested desktop repository outside the initial sandbox root | Reran the same audit with approved filesystem access | `artifacts/full/raw_data_audit.json` written and inspected |
| First full feature run | DuckDB at `1GB` with two threads completed history tables, then exhausted memory in the six-way feature export join | Materialized the feature join before sorted export, reduced processing to one thread, raised the bounded limit to `2GB`, and released large temporary tables before pandas/NetworkX work | Optimized retry PASS in 304.93406899999536s; work files cleaned |
| Focused feature/leakage tests | Early focused subset reported 1 failed and 5 passed | The failure was an incorrect test expectation, not evidence to preserve as a result; corrected the expectation to the intended contract and reran | Fresh full suite: 34 passed |
| Direct test collection | Initial `.venv\Scripts\pytest.exe -q` reported 3 import collection errors | Added the repository’s `src` and root paths through `tests/conftest.py` for deterministic local collection | Fresh final command: 34 passed in 25.36s |

The exact early focused-subset command was not reliably retained, so it is not
invented here. Attempt history is kept distinct from final acceptance evidence.

## 7. Quick pipeline evidence

### 7.1 Run identity

| Field | Verified value |
| --- | --- |
| Config | `configs/quick.yaml` extending `configs/base.yaml` |
| Python | CPython 3.12.10 |
| Seed | 42 |
| Sampling | Chronological prefix, maximum 10,000 rows |
| Actual rows | 10,000 |
| Timestamp scope | `2022-09-01 00:00`–`2022-09-01 00:29` |
| Labels | 9,999 label `0`; 1 label `1` |
| Canonical rows | 10,000 |
| Feature rows/columns | 10,000 / 52 |
| Exact duplicates in quick prefix | 0; preserve-all policy remains active |
| Unique quick transaction nodes | 8,330 |
| Unmatched sender/receiver rows | 0 / 0 |
| Runtime | 11.197340699996857 seconds |
| Manifest status | PASS |

The quick prefix is deterministic and history-safe, but not statistically
representative of the whole source.

### 7.2 Split evidence

| Partition | Rows | Timestamp minimum | Timestamp maximum | Unique timestamps | Positive labels |
| --- | ---: | --- | --- | ---: | ---: |
| Train | 7,011 | `2022-09-01 00:00` | `2022-09-01 00:20` | 21 | 0 |
| Validation | 1,657 | `2022-09-01 00:21` | `2022-09-01 00:25` | 5 | 1 |
| Test | 1,332 | `2022-09-01 00:26` | `2022-09-01 00:29` | 4 | 0 |

`split_metadata.json` records:

- `strategy = chronological`;
- `timestamp_groups_kept_intact = true`;
- `strict_boundaries_verified = true`;
- `no_transaction_overlap_verified = true`;
- requested fractions 0.70 / 0.15 / 0.15, snapped to timestamp groups.

`verify_run.py` confirmed 10,000 split rows, 10,000 transaction-feature rows,
strict chronology, and no transaction-ID overlap. The absence of positive examples
in quick train/test means this split is suitable for smoke/leakage validation, not
model evaluation.

### 7.3 Feature evidence

`artifacts/quick/tables/transaction_features.csv` contains 18 canonical/provenance
columns plus 34 engineered columns:

- 6 transaction-local fields;
- 5 calendar/previous-event fields;
- 18 strictly-prior account-history fields across `1h`, `24h`, and `7d` windows;
- 5 directed strictly-prior graph-history fields.

The graph-history fields are:

```text
sender_prior_fan_out_degree
sender_prior_fan_in_degree
receiver_prior_fan_out_degree
receiver_prior_fan_in_degree
pair_previous_transfer_count
```

Events sharing timestamp `t` see only state from timestamps strictly before `t`.

## 8. Full pipeline evidence

### 8.1 Run identity and resource contract

| Field | Manifest value |
| --- | --- |
| Config | `configs/full.yaml` |
| Engine | DuckDB 1.5.5 |
| Memory limit | `2GB` |
| Ingestion / processing threads | 1 / 1 |
| External spill | Enabled; maximum temporary directory size `100GB` |
| Parquet compression | Zstandard |
| Transactions / accounts loaded | 5,078,345 / 518,581 |
| Canonical rows | 5,078,345 |
| Feature rows / columns | 5,078,345 / 52 |
| Feature engineering sampled | `false` |
| Strictly-prior timestamp batches | `true` |
| Runtime | 304.93406899999536 seconds |
| Manifest status | PASS |

The raw file was not assumed chronological. Canonical rows were externally sorted
by `timestamp, source_row_number`. Nine exact duplicate rows were observed and
preserved. The six transaction-local, five time, eighteen history, and five
directed graph-history features were produced for every transaction.

### 8.2 Full chronological split

| Partition | Rows | Timestamp minimum | Timestamp maximum | Unique timestamps | Positive labels |
| --- | ---: | --- | --- | ---: | ---: |
| Train | 3,554,957 | `2022-09-01 00:00` | `2022-09-07 14:55` | 9,536 | 2,856 |
| Validation | 761,749 | `2022-09-07 14:56` | `2022-09-09 03:16` | 2,181 | 760 |
| Test | 761,639 | `2022-09-09 03:17` | `2022-09-18 16:18` | 3,301 | 1,561 |

The full split metadata records strict boundaries, intact timestamp groups, and no
transaction overlap. `verification_report.json` reconciles 5,078,345 rows and
5,078,345 unique IDs in each canonical, feature, split, and graph-edge output with
DuckDB counts, distinct counts, and bidirectional `EXCEPT` checks.

### 8.3 Runtime by stage

| Stage | Seconds |
| --- | ---: |
| Source fingerprints | 0.4863478999977815 |
| DuckDB initialization | 0.018459499995515216 |
| Single-threaded ingestion | 13.602533099998254 |
| Raw validation | 7.242892599999323 |
| Canonical external sort and export | 23.96039380000002 |
| Full account-reference check | 4.587355700001353 |
| Chronological split | 0.2312817999991239 |
| Sender history features | 36.51647480000247 |
| Receiver history features | 29.85599199999706 |
| Directed graph-history features | 33.679646700002195 |
| Feature join materialization | 66.20344630000181 |
| Materialized-feature verification | 6.057654500000353 |
| Sorted feature export | 32.58873909999966 |
| Split and graph exports | 8.66468559999339 |
| Full and sampled EDA | 26.223705899996276 |
| SQL output verification | 13.752454000001308 |
| Work-file cleanup | 0.08349149999412475 |
| Artifact hash inventory | 1.0316739000045345 |

These stage timers are direct manifest values. The overall wall-clock runtime also
includes orchestration outside the measured stage contexts.

### 8.4 Full-exact and sampled EDA

`artifacts/full/eda/full_eda_manifest.json` separates two scopes:

- **`full_exact`:** ten payloads plus its scope manifest. DuckDB used every one of
  the 5,078,345 rows for class/rate, amount, daily activity, payment-format,
  currency, bank, repeated-pair, node-degree/counterparty, and dataset/graph-count
  evidence.
- **`sample_descriptive_only`:** twenty-nine payloads plus its scope manifest.
  Exactly 100,000 transaction edges were selected independently of target with
  stable `ORDER BY hash(transaction_id), transaction_id`. Plots use this scope;
  NetworkX component/local-subgraph work is capped at 50,000 edges.

The 100,000 sampled IDs have SHA-256
`e81cee70a8c52d088f5c7db21ba61bbabe2bf82fb32a5be8c73aca854400ded5`.
The sample contains 109 positive labels. Sampled degrees and components are
descriptive of selected edges and are not estimates of the full graph. Full-exact
graph counts record 515,088 nodes, 5,078,345 edges, 1,015,736 unique directed
pairs, and 4,062,609 repeated edge occurrences after first; full weak/strong
component metrics were intentionally not materialized.

### 8.5 Resource-limited first attempt and successful retry

The first actual full attempt used a DuckDB `1GB` limit with two threads. It
completed the history computations but exhausted memory in the six-way feature
export join. The optimized retry:

- materialized the six-way feature join before its sorted Parquet export;
- used one thread to reduce per-thread buffers;
- used a bounded `2GB` DuckDB memory limit with external spill;
- dropped materialized feature and high-cardinality EDA tables when no longer
  needed;
- completed successfully and removed its temporary database/spill directory.

This failed attempt is not counted as a successful run or mixed into the final
runtime; it explains the evidence-backed resource settings used by the PASS retry.

## 9. Generated artifact inventory

The quick command generated **39 artifacts**: one run manifest and 38 payloads
listed with SHA-256 and byte size inside the manifest. The directory also contains
a pre-existing `.gitkeep`; it is not a generated artifact.

### 9.1 Quick core artifacts

```text
artifacts/quick/run_manifest.json
artifacts/quick/account_reference_report.json
artifacts/quick/raw_validation_report.json
artifacts/quick/canonical_validation_report.json
artifacts/quick/resolved_config.json
artifacts/quick/metadata/split_metadata.json
artifacts/quick/tables/transactions_clean.csv
artifacts/quick/tables/transaction_features.csv
artifacts/quick/tables/split_manifest.csv
artifacts/quick/tables/graph_edges.csv
```

### 9.2 Quick EDA metadata/evidence

```text
artifacts/quick/eda/eda_manifest.json
artifacts/quick/eda/dataset_profile.json
artifacts/quick/eda/graph_summary.json
artifacts/quick/eda/suspicious_subgraph_evidence.json
```

### 9.3 Quick EDA figures — 13 PNG

```text
artifacts/quick/eda/figures/amount_distribution.png
artifacts/quick/eda/figures/bank_activity.png
artifacts/quick/eda/figures/class_distribution.png
artifacts/quick/eda/figures/connected_components.png
artifacts/quick/eda/figures/currency_distribution.png
artifacts/quick/eda/figures/in_out_degree_distribution.png
artifacts/quick/eda/figures/laundering_activity_over_time.png
artifacts/quick/eda/figures/log_amount_distribution.png
artifacts/quick/eda/figures/payment_format_distribution.png
artifacts/quick/eda/figures/repeated_pair_activity.png
artifacts/quick/eda/figures/suspicious_local_subgraph.png
artifacts/quick/eda/figures/transaction_activity_over_time.png
artifacts/quick/eda/figures/unique_counterparties.png
```

The final `class_distribution.png` was visually inspected and renders correctly
with a logarithmic y-axis. `suspicious_local_subgraph.png` was also visually
inspected and renders a valid data-derived graph. No visual claim is made here for
the other eleven beyond successful file/hash validation.

### 9.4 Quick EDA tables — 12 CSV

```text
artifacts/quick/eda/tables/activity_over_time.csv
artifacts/quick/eda/tables/amount_summary.csv
artifacts/quick/eda/tables/bank_activity.csv
artifacts/quick/eda/tables/class_distribution.csv
artifacts/quick/eda/tables/connected_components.csv
artifacts/quick/eda/tables/currency_distribution.csv
artifacts/quick/eda/tables/degree_distribution.csv
artifacts/quick/eda/tables/laundering_rate.csv
artifacts/quick/eda/tables/payment_format_distribution.csv
artifacts/quick/eda/tables/repeated_sender_receiver_pairs.csv
artifacts/quick/eda/tables/suspicious_local_subgraph_edges.csv
artifacts/quick/eda/tables/unique_counterparties.csv
```

### 9.5 Full-run artifacts

```text
artifacts/full/run_manifest.json
artifacts/full/raw_data_audit.json
artifacts/full/verification_report.json
artifacts/full/raw_validation_report.json
artifacts/full/raw_accounts_validation_report.json
artifacts/full/canonical_validation_report.json
artifacts/full/account_reference_report.json
artifacts/full/resolved_config.json
artifacts/full/metadata/split_metadata.json
artifacts/full/tables/transactions_clean.parquet
artifacts/full/tables/transaction_features.parquet
artifacts/full/tables/split_manifest.parquet
artifacts/full/tables/graph_edges.parquet
artifacts/full/eda/full_eda_manifest.json
artifacts/full/eda/full_exact/full_exact_manifest.json
artifacts/full/eda/full_exact/dataset_graph_counts.json
artifacts/full/eda/full_exact/tables/                 9 exact CSV tables
artifacts/full/eda/sample_descriptive/eda_manifest.json
artifacts/full/eda/sample_descriptive/dataset_profile.json
artifacts/full/eda/sample_descriptive/figures/        13 PNG files
artifacts/full/eda/sample_descriptive/tables/         12 CSV tables
```

The full manifest inventories 54 payloads; including `run_manifest.json`, the
published evidence set contains 55 output files (excluding `.gitkeep`).
`raw_data_audit.json` predates the successful
preprocessing retry, so this is a directory inventory rather than a claim that the
retry alone created all 55 files. Every payload has a recorded SHA-256 and byte
size. The four primary
Parquet outputs occupy 200,098,094, 507,016,511, 15,395,565, and 122,104,300 bytes,
respectively, in the order listed above.

## 10. Full streaming IBM HI-Small audit

The project-generated audit scanned all source records and matched the earlier
independent read-only audit.

| File | Exact bytes | Data rows | SHA-256 |
| --- | ---: | ---: | --- |
| `HI-Small_Trans.csv` | 475,664,283 | 5,078,345 | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| `HI-Small_accounts.csv` | 34,053,187 | 518,581 | `786808526e33cfc441212dd6fccda7edfc24172149bed59c6ef59b186836b014` |

Verified transaction facts:

- label `0`: 5,073,168;
- label `1`: 5,177 (rate `0.0010194266045335635`, or 0.101942660453%);
- timestamp range: `2022-09-01 00:00`–`2022-09-18 16:18`;
- invalid timestamps: 0;
- exact duplicate rows after first: 9 in 9 groups;
- both amount columns have zero invalid, negative, non-finite, or zero values;
- each amount column ranges from `0.000001` to `1,046,302,363,293.48`;
- unique canonical transaction nodes: 515,088.

Verified accounts/cross-file facts:

- 518,581 unique composite nodes;
- 518,573 unique account-number strings and 8 collision groups;
- 30,470 unique normalized banks and 166,207 entity IDs;
- all 515,088 transaction nodes matched accounts after bank-ID canonicalization;
- unmatched transaction nodes: 0;
- account nodes unused as transaction endpoints: 3,493;
- exact duplicate account rows: 0.

Raw string joins are invalid because transaction bank IDs are zero-prefixed while
accounts IDs are not. Canonical identity is
`normalized_bank_id::UPPERCASE_ACCOUNT_ID`. Exact transaction duplicates are
reported and preserved by default rather than silently discarded.

## 11. Files created or changed in the ARGUS repository

### Repository and configuration

```text
.env.example
.gitignore
AGENTS.md
LICENSE
Makefile
pyproject.toml
requirements.txt
configs/base.yaml
configs/quick.yaml
configs/full.yaml
```

### Reusable implementation

```text
src/argus/config.py
src/argus/full_eda.py
src/argus/full_pipeline.py
src/argus/pipeline.py
src/argus/raw_audit.py
src/argus/verify.py
src/argus/data/accounts.py
src/argus/data/load.py
src/argus/data/preprocess.py
src/argus/data/split.py
src/argus/data/validate.py
src/argus/features/transaction.py
src/argus/features/temporal.py
src/argus/features/graph.py
src/argus/graph/build.py
```

Package `__init__.py` files were added for the corresponding modules.

### Command-line scripts

```text
scripts/audit_raw_data.py
scripts/build_graph_features.py
scripts/generate_synthetic_fixture.py
scripts/run_eda.py
scripts/run_full_pipeline.py
scripts/run_quick_pipeline.py
scripts/validate_data.py
scripts/verify_run.py
```

### Tests

```text
tests/conftest.py
tests/test_accounts.py
tests/test_config.py
tests/test_eda.py
tests/test_full_eda.py
tests/test_full_pipeline.py
tests/test_graph_build.py
tests/test_graph_features.py
tests/test_no_leakage.py
tests/test_pipeline_smoke.py
tests/test_preprocessing.py
tests/test_raw_audit.py
tests/test_schema.py
tests/test_synthetic_fixture.py
tests/test_temporal_features.py
tests/test_temporal_split.py
tests/test_transaction_features.py
```

### Documentation, notebook, and reports

```text
README.md
data/README.md
artifacts/README.md
reports/README.md
reports/generated/SPRINT_1_STATUS.md
docs/PROJECT_SPEC.md
docs/DATA_DICTIONARY.md
docs/EXPERIMENT_PROTOCOL.md
docs/ROADMAP.md
docs/DECISIONS.md
docs/MODEL_CARD.md
notebooks/01_eda.ipynb
```

The seven existing academic inputs and two raw CSVs were copied without content
changes. Raw/interim/processed directories, artifact locations, reports, and
notebook/package structure were organized around them. The unrelated DATATON/GridUp
repository was not edited.

## 12. Limitations and unresolved future work

- The quick run is only the first 10,000 chronologically ordered transactions and
  covers 30 minutes. It is deterministic/history-safe, not representative.
- Quick train/test contain no positive labels. They cannot support model selection,
  threshold selection, or performance claims.
- Full exact tabular EDA covers every row, but full-population NetworkX weak/strong
  components and local-subgraph layouts are intentionally not materialized. The
  100,000-row plot sample and 50,000-edge graph subset are descriptive only.
- The successful full resource settings are evidence for this machine/run, not a
  guarantee that `2GB` and one thread are optimal on other hardware.
- IBM HI-Small is synthetic; transferability to any institution is untested.
- Nine exact full-source transaction duplicates are retained by default; any later
  deduplication must be a versioned, measured decision.
- Raw data, `.venv`, and generated artifacts are intentionally Git-ignored and must
  be recreated locally.
- No Logistic Regression, Random Forest, boosting model, PR-AUC, Recall@K,
  Precision@K, F1, FPR, alert count, GraphSAGE, SHAP, case-builder, or Streamlit
  result exists in Sprint 1.

None of these is an unresolved Sprint 1 acceptance blocker. They bound what the
successful Sprint 1 evidence can support.

## 13. Responsible-use interpretation

The source labels are synthetic transaction-level ground truth. Quick EDA and the
full run's sampled descriptive local subgraph are data-derived evidence requiring
human review, not findings about real people or authorization for adverse action.

Future model output must use language such as “suspicious network candidate,”
“elevated investigation priority,” and “unusual transfer pattern.” Observed facts
must remain separate from model contributions, uncertainty must be stated, and a
trained human must make any decision.

## 14. Recommended Sprint 2 — not executed

Only in a separately authorized Sprint 2 should the project:

1. freeze the verified full-data feature/split contract before modeling;
2. train Logistic Regression, Random Forest, and one justified boosting family on
   identical chronological partitions;
3. fit learned preprocessing on train only;
4. select the Transaction Baseline Champion and threshold on validation only;
5. generate PR-AUC, Recall@K, Precision@K, F1, FPR, alert-volume, confusion-matrix,
   and secondary ROC-AUC artifacts from code;
6. keep the final test partition untouched during selection.

GraphSAGE and product-layer work remain later than Sprint 2. This recommendation is
not a claim that model work has started.

## Final decision

**Sprint 1: PASS — 14/14 acceptance criteria satisfied with executable evidence.**

The data-access blocker is resolved, intermediate failures were corrected and
rerun, generated artifacts were verified, existing user work was preserved, and no
Sprint 2 result was fabricated or implemented.
