# ARGUS AI Project Specification

**Document status:** Canonical scope through Sprint 2
**Effective date:** 2026-09-07
**Implementation verification:** Sprint 1 `PASS` (14/14); Sprint 2 `PASS`
**Current stop boundary:** Sprint 3 refinement and graph-value work not started
**Evidence ledgers:** `reports/generated/SPRINT_1_STATUS.md` and
`reports/generated/SPRINT_2_STATUS.md`

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

The current user request authorizes work through Sprint 2 and stops before Sprint 3.
The technical baseline is
`reports/existing_coursework/ARGUS_CODEX_MASTER_PROMPT.md`. Other preserved course
documents provide context, not executable instructions. If they conflict:

1. the current user request controls;
2. the master prompt controls technical implementation;
3. this specification records the resulting Sprint 1 and Sprint 2 contracts;
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

## 4. Completed execution scope

### 4.1 Sprint 1 data foundation

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

### 4.2 Sprint 2 baseline exploration

Sprint 2 adds one executable, unsampled transaction-baseline path:

1. verify the frozen Sprint 1 feature, split, metadata, and manifest hashes;
2. fit a 74-column preprocessing contract on train only;
3. transform validation without updating medians, scales, vocabularies, or bank
   frequencies;
4. train Logistic Regression, Random Forest, and LightGBM on the identical full
   training matrix;
5. compute validation average precision/PR-AUC, secondary ROC-AUC, fixed-threshold
   precision, recall, F1, FPR, confusion counts, alerts, and top-K metrics;
6. select the Transaction Baseline Champion solely by validation average precision;
7. save fitted models, validation scores, comparison tables/plots, preprocessing
   state, runtime, provenance, and a final-test access policy;
8. independently recompute metrics and champion selection from saved artifacts;
9. leave final-test features, inference, predictions, and metrics untouched;
10. stop before Sprint 3 refinement and graph-value experiments.

Sprint 2 passes only with executable artifacts and verification. The completed run
selected Random Forest at validation AP 0.08859105087174989 and passed all gates.
LightGBM was selected once for the boosting slot because its CPU histogram path
matched the full-data, deterministic single-thread, bounded-memory plan. XGBoost
was not run, so no library-superiority claim is part of Sprint 2.

## 5. Sprint boundaries and exclusions

Sprint 1 excluded all model training. Sprint 2 subsequently implemented only the
three fixed transaction baselines, validation metrics, and validation-only champion
selection described in Section 4.2.

The following remain explicitly outside the completed scope:

- hyperparameter search or tuning;
- threshold optimization or analyst-capacity selection;
- temporal cross-validation and feature-family ablation;
- graph-enhanced model comparison or any claim that graph information improves
  results;
- GraphSAGE, GCN, GAT, account-level target invention, or GNN explanation;
- final-test feature materialization, inference, predictions, or metrics;
- calibration, SHAP, case ranking, evidence cards, Streamlit, or LLM summaries;
- PaySim, cloud deployment, or other stretch work.

These boundaries prevent unverified Sprint 3+ work from obscuring the completed
data and validation-only baseline evidence.

## 6. System contracts

### 6.1 Configuration

- `configs/base.yaml` contains shared paths, mappings, seed, split, and feature
  settings.
- `configs/quick.yaml` inherits the base and enables a deterministic limit of at
  most 10,000 source rows.
- `configs/full.yaml` inherits the base and disables sampling.
- `configs/baseline.yaml` inherits the full contract, freezes upstream artifact
  hashes, and declares all Sprint 2 preprocessing/model/metric/test-gate settings.
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
- Any learned model transform must fit on train only; validation/test are transform-
  only and may never extend fitted vocabularies or summary statistics.
- Sprint 2 permits model access to train and validation only. Test metadata may be
  read from the pre-existing split JSON, but test feature rows may not be loaded.

### 6.5 Graph semantics

- Nodes are canonical bank-account composites.
- Edges are sender-to-receiver transactions.
- Direction is preserved.
- Repeated edges remain analytically meaningful multiedges.
- Exact duplicate records are reported separately and are never silently removed.
- Sprint 1 graph features are interpretable prior fan-in/fan-out and repeated-pair
  history; GraphSAGE is not in scope.
- Sprint 2 transaction baselines exclude all five graph-history predictors. Their
  later addition belongs to the controlled Sprint 3 graph-value experiment.

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
.\.venv\Scripts\python.exe scripts/train_baselines.py --config configs/baseline.yaml
.\.venv\Scripts\python.exe scripts/verify_baselines.py --config configs/baseline.yaml
.\.venv\Scripts\python.exe scripts/validate_sprint2.py --config configs/baseline.yaml
```

Sprint 1 closing evidence remains 34 tests in 25.36 seconds plus its passed quick,
full, and verification commands. After Sprint 2, the final repository suite passed
81 tests in 19.87 seconds; Ruff lint/format and `pip check` passed. The baseline run
and independent artifact verifier passed on 761,749 validation predictions, and the
refreshed manifest inventories 26 payloads excluding itself. Focused EDA and data-
validation entry points remain available through `scripts/run_eda.py` and
`scripts/validate_data.py`.

## 8. Artifact contract

The successful Sprint 1 and Sprint 2 runs populated these core paths:

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
artifacts/sprint2/run_manifest.json
artifacts/sprint2/verification_report.json
artifacts/sprint2/quality_report.json
artifacts/sprint2/feature_contract.json
artifacts/sprint2/preprocessing/manifest.json
artifacts/sprint2/preprocessing/fitted_state.json
artifacts/sprint2/model_comparison.json
artifacts/sprint2/model_comparison.csv
artifacts/sprint2/transaction_baseline_champion.json
artifacts/sprint2/validation_predictions.parquet
artifacts/sprint2/models/
artifacts/sprint2/figures/
artifacts/sprint2/final_test_policy.json
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

The Sprint 2 manifest records the unsampled 3,554,957-row train and 761,749-row
validation experiment, the 74-column transform contract, all candidate metrics,
Random Forest champion selection, 389.0078022000016 seconds runtime, and 26 hashed
payloads excluding the manifest. Verification recomputes validation AP/ROC-AUC,
deserializes all three models, repeats validation-only selection, and proves that no
test prediction artifact exists. Generated model/data artifacts remain Git-ignored.

Top-K interpretation must preserve score-tie evidence. Logistic Regression assigns
exact score `1.0` to 60,119 validation rows containing 427 positives; LightGBM does
so for 108,347 rows containing 663 positives. Their configured K cutoffs therefore
fall inside large tied groups, and `source_row_number` decides which tied rows enter
each finite K. These values are deterministic but should not be treated as evidence
of within-tie discrimination. Random Forest has no exact score-1 rows, and champion
selection uses tie-aware average precision rather than the top-K row tie-break.

## 9. Quality requirements

- Python 3.11–3.13, type hints, focused functions, and meaningful exceptions;
- deterministic seed `42` unless a versioned config says otherwise;
- no hard-coded machine-specific paths in application code;
- vectorized/grouped processing appropriate for millions of rows;
- tests for schema, preprocessing, account identity, temporal boundaries,
  same-timestamp behavior, future invariance, graph direction, and repeat handling;
- tests for predictor exclusions, train-only fitted state, unknown categories,
  metric edge cases, deterministic top-K ties, validation-only champion selection,
  test-access policy, artifact integrity, and report generation;
- aggregate logs must avoid gratuitous entity-name output;
- no metric, chart, comparison, or completion claim without generated evidence.

## 10. Definition of done

All fourteen Sprint 1 items and all ten Sprint 2 items in Section 4 have executable
evidence. Data/full/quick paths, three baseline fits, validation comparison,
artifact verification, the 79-test suite, Ruff, and `pip check` passed. Sprint 1 and
Sprint 2 are `PASS`; see both generated Sprint status reports for exact commands,
artifacts, runtime, warnings, and limitations.

## 11. Current stop boundary

Sprint 2 is complete. Sprint 3 has not started and no refinement, threshold-tuning,
temporal-cross-validation, feature-ablation, graph-value, or final-test result is
claimed. Any later Sprint 3 authorization must begin from the frozen transaction
baseline evidence, retain validation-only development discipline, and keep the test
gate closed until the refinement/model specification is frozen. See
`docs/ROADMAP.md` for the boundary; it is not a completed-work claim.
