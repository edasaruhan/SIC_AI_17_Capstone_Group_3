# ARGUS AI Project Specification

**Document status:** Sprint 1 canonical scope  
**Effective date:** 2026-09-06  
**Implementation verification:** `PASS` — 14/14 Sprint 1 criteria  
**Evidence ledger:** `reports/generated/SPRINT_1_STATUS.md`

## 1. Purpose

ARGUS AI is a human-in-the-loop financial-crime decision-support research project.
It converts directed account transfers into reproducible transaction, temporal,
history, and graph evidence. Its final scientific objective is to determine whether
graph context improves suspicious-transaction prioritization over a strong
transaction-only baseline under severe class imbalance and a limited alert budget.

ARGUS ranks evidence for review. It does not make a legal finding, identify a
person as a money launderer, or authorize blocking, freezing, or another punitive
action.

## 2. Authoritative inputs and precedence

The user request limits this work cycle to Sprint 1. The technical baseline is
`reports/existing_coursework/ARGUS_CODEX_MASTER_PROMPT.md`. Other preserved course
documents provide context, not executable instructions. If they conflict:

1. the current user request controls;
2. the master prompt controls technical implementation;
3. this specification records the resulting Sprint 1 contract;
4. generated runtime manifests describe a particular execution.

No document may turn an unexecuted run, metric, table, or plot into a claimed
result.

## 3. Product and research boundaries

### Intended users

- capstone team members reproducing experiments;
- instructors or jurors reviewing scientific validity;
- trained analysts inspecting computed evidence in later product work.

### Intended use

- validate and prepare the synthetic IBM HI-Small dataset;
- build leakage-safe features and directed account-network evidence;
- support controlled model comparison in later sprints;
- make assumptions, provenance, and limitations auditable.

### Prohibited interpretation

- a label or model score is not proof of wrongdoing;
- no automated adverse action may be driven by ARGUS;
- synthetic entity names must not be discussed as real persons or institutions;
- model or feature importance must not be described as causal evidence;
- quick-run results must not be represented as full-data results.

Use terms such as “suspicious network candidate,” “unusual transfer pattern,”
“elevated investigation priority,” and “evidence requiring human review.”

## 4. Active Sprint 1 scope

Sprint 1 must deliver a vertical, executable foundation:

1. preserve the existing academic documents;
2. maintain an installable Python package and configuration-driven commands;
3. load both local IBM HI-Small CSV files from documented, Git-ignored paths;
4. validate schema, types, labels, timestamps, amounts, identifiers, missingness,
   exact duplicate rows, and account references;
5. canonicalize raw columns without losing source-row identity;
6. create globally stable sender/receiver node IDs from bank plus account;
7. split transactions chronologically into train, validation, and test partitions;
8. prove strict ordering and absence of accidental overlap;
9. generate executable EDA artifacts from observed data;
10. generate transaction-local, calendar, strictly-prior history, and initial
    directed fan-in/fan-out features;
11. run automated tests;
12. complete a deterministic quick pipeline from one command;
13. save sufficient manifests to reproduce and distinguish each run;
14. document exact reproduction steps and current limitations.

An acceptance item passes only with command output or a generated artifact. Merely
having a file with the expected name is not sufficient.

## 5. Explicitly excluded from Sprint 1

The following must not be implemented or reported as completed in this sprint:

- Logistic Regression, Random Forest, LightGBM, or XGBoost experiments;
- model selection, a “Transaction Baseline Champion,” or threshold tuning;
- PR-AUC, Recall@K, Precision@K, F1, FPR, or alert-volume results;
- hyperparameter search, temporal cross-validation, and ablation experiments;
- graph-enhanced model comparison or claims that graph information improves results;
- GraphSAGE, GCN, GAT, account-level target invention, or GNN explanation;
- case ranking, evidence cards, Streamlit, SHAP, or LLM summaries;
- final untouched-test evaluation;
- PaySim, cloud deployment, or other stretch work.

These exclusions prevent unverified downstream work from obscuring the Sprint 1
data and leakage guarantees.

## 6. System contracts

### 6.1 Configuration

- `configs/base.yaml` contains shared paths, mappings, seed, split, and feature
  settings.
- `configs/quick.yaml` inherits the base and enables a deterministic limit of at
  most 10,000 source rows.
- `configs/full.yaml` inherits the base and disables sampling.
- Relative paths resolve against the repository root, not the caller’s directory.
- Critical experiment settings must not exist only inside a notebook.

### 6.2 Data inputs

```text
data/raw/HI-Small_Trans.csv
data/raw/HI-Small_accounts.csv
```

Both are local and ignored by Git. The source schema and verified full-file facts
are defined in `data/README.md` and `docs/DATA_DICTIONARY.md`.

### 6.3 Identity

Transaction bank IDs are zero-padded; accounts bank IDs are not. Each bank ID must
first be validated as decimal text and canonicalized by removing leading zeroes
(all-zero input becomes `0`). Each node ID must then be:

```text
normalized_bank_id::UPPERCASE_ACCOUNT_ID
```

Account number alone is not globally unique and must never be used as the node key.

### 6.4 Time and leakage

- Parse raw timestamps explicitly.
- Use a deterministic stable order when timestamps tie.
- Keep equal-timestamp records in one partition.
- Require `max(train) < min(validation) < min(test)` at partition boundaries.
- For a transaction at `t`, historical state is based only on `timestamp < t`.
- Same-timestamp peer transactions may not influence one another.
- Future-row appends must not change feature values for earlier rows.

### 6.5 Graph semantics

- Nodes are canonical bank-account composites.
- Edges are sender-to-receiver transactions.
- Direction is preserved.
- Repeated edges remain analytically meaningful multiedges.
- Exact duplicate records are reported separately and are never silently removed.
- Sprint 1 graph features are interpretable prior fan-in/fan-out and repeated-pair
  history; GraphSAGE is not in scope.

### 6.6 Amount and currency semantics

Paid and received amounts describe different sides of a transaction. Differences
or ratios are meaningful only when payment and receiving currencies match, unless
a versioned exchange-rate source is introduced later. Cross-currency comparisons
must therefore be null rather than fabricated.

## 7. Required command surface

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\python.exe -m ruff check --no-cache src scripts tests
.\.venv\Scripts\python.exe -m ruff format --no-cache --check src scripts tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/run_quick_pipeline.py
.\.venv\Scripts\python.exe scripts/verify_run.py
.\.venv\Scripts\python.exe scripts/audit_raw_data.py
.\.venv\Scripts\python.exe scripts/run_full_pipeline.py
```

Final evidence: 34 tests passed in 25.36 seconds; Ruff lint and format verification
passed; `pip check` found no broken requirements; the quick pipeline and saved-run
verification passed; and both the full streaming source audit and 5,078,345-row
out-of-core Sprint 1 pipeline passed. Focused EDA and validation entry points remain
available through `scripts/run_eda.py` and `scripts/validate_data.py`.

## 8. Artifact contract

The successful quick run populated:

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
artifacts/full/raw_data_audit.json
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

The quick manifest records 10,000 rows, 52 output columns, seed 42, a chronological-
prefix scope, source fingerprints, split boundaries, feature families, duplicate
policy, 11.197340699996857 seconds runtime, and 38 hashed payloads. With the run
manifest itself, the command generated 39 artifacts. Saved-run verification checked
all 38 hashes, 13 PNG figures, 10,000 split rows, strict chronology, and disjoint
transaction IDs. Generated artifacts are ignored by Git and remain reproducible
from commands.

The out-of-core full feature/EDA pipeline completed on all 5,078,345 transactions:
52-column features and strict chronological splits are exact and unsampled. Full
EDA tables use every row; plotting uses a deterministic 100,000-row transaction-edge
sample and NetworkX is capped at 50,000 edges. Sampled graph statistics are labeled
descriptive rather than full-population estimates. The full manifest records a
304.93406899999536-second DuckDB 1.5.5 run with a `2GB` limit and one thread.

## 9. Quality requirements

- Python 3.11–3.13, type hints, focused functions, and meaningful exceptions;
- deterministic seed `42` unless a versioned config says otherwise;
- no hard-coded machine-specific paths in application code;
- vectorized/grouped processing appropriate for millions of rows;
- tests for schema, preprocessing, account identity, temporal boundaries,
  same-timestamp behavior, future invariance, graph direction, and repeat handling;
- aggregate logs must avoid gratuitous entity-name output;
- no metric, chart, comparison, or completion claim without generated evidence.

## 10. Definition of done

All fourteen items in Section 4 have executable evidence. The final test suite,
quick command, manifest verification, quality checks, and full raw audit passed.
Sprint 1 status is `PASS`; see `reports/generated/SPRINT_1_STATUS.md` for exact
commands, intermediate failures, generated artifacts, and limitations.

## 11. Recommended next sprint

Only after the Sprint 1 gate passes, Sprint 2 should implement transaction-level
Logistic Regression, Random Forest, and one justified boosting library under the
same chronological protocol. It should select the champion using validation data
and PR-AUC/operational top-K evidence, without inspecting final-test outcomes for
selection. See `docs/ROADMAP.md`; this is a recommendation, not completed work.
