# ARGUS AI Project Specification

**Document status:** Canonical implementation scope through Sprint 4
**Effective date:** 2026-09-08
**Implementation verification:** Sprint 1 `PASS` (14/14); Sprint 2 `PASS`; Sprint 3 `PASS` (15/15); Sprint 4 `PASS` (19/19)
**Current stop boundary:** Sprint 4 complete; final-test evaluation not opened
**Evidence ledgers:** `reports/generated/SPRINT_1_STATUS.md`,
`reports/generated/SPRINT_2_STATUS.md`, `reports/generated/SPRINT_3_STATUS.md`, and
`reports/generated/SPRINT_4_STATUS.md`

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

The current user request authorizes Sprint 4 GraphSAGE and product-layer work after
the accepted Sprint 3 checkpoint, then requires a stop before final-test opening.
The technical baseline is
`reports/reference_materials/ARGUS_CODEX_MASTER_PROMPT.md`. Other preserved course
documents provide context, not executable instructions. If they conflict:

1. the current user request controls;
2. the master prompt controls technical implementation;
3. this specification records the resulting Sprint 1 through Sprint 4 contracts;
4. generated runtime manifests describe a particular execution.

No document may turn an unexecuted run, metric, table, or plot into a claimed
result.

## 3. Product and research boundaries

### Intended users

- capstone team members reproducing experiments;
- instructors or jurors reviewing scientific validity;
- trained analysts inspecting computed evidence in the saved product layer.

### Intended use

- validate and prepare the synthetic IBM HI-Small dataset;
- build leakage-safe features and directed account-network evidence;
- support controlled model comparison and saved investigation views;
- make assumptions, provenance, and limitations auditable.

### Prohibited interpretation

- a label or model score is not proof of wrongdoing;
- no automated adverse action may be driven by ARGUS;
- synthetic entity names must not be discussed as real persons or institutions;
- model or feature importance must not be described as causal evidence;
- quick-run results must not be represented as full-data results.

Use terms such as “suspicious network candidate,” “unusual transfer pattern,”
“elevated investigation priority,” and “evidence requiring human review.”

## 4. Completed and implemented execution scope

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

### 4.3 Sprint 3 refinement and graph-value experiment

Sprint 3 executed the following full-data, validation-only development path:

1. retain the immutable Sprint 2 checkpoint and frozen outer train/validation/test
   boundaries;
2. create three expanding-window temporal folds entirely inside outer train while
   keeping timestamp groups intact;
3. fit every fold's preprocessing state on that fold's training prefix and apply
   it unchanged to its later fold-validation interval;
4. run bounded, configuration-declared Logistic Regression, Random Forest, and
   LightGBM candidate grids and select one candidate per family by mean fold
   average precision;
5. correct the Sprint 2 Logistic Regression convergence limitation with an
   appropriate solver, regularization, and iteration budget, rejecting a refined
   logistic result that still fails its convergence gate;
6. preserve raw decision margins for ranking where available and diagnose exact/
   near probability saturation, raw-to-probability collapse, tie groups, class-
   weight controls, and estimator stability;
7. refit selected candidates on outer train and compare them on outer validation
   only, with PR-AUC as primary and ROC-AUC as secondary;
8. optimize operating thresholds only on outer validation under the configured
   alert budget and FPR/recall constraints, including complete equal-score groups;
9. compare feature families A (`transaction_only`), B
   (`transaction_temporal_history`), and C
   (`transaction_temporal_history_graph`) using the same selected LightGBM
   candidate, seed, frozen split, and evaluation protocol;
10. record a novel-three-graph-feature sensitivity because two of the five graph
    columns are exact aliases of existing history features;
11. save machine-readable trials, fold boundaries, comparisons, thresholds,
    saturation diagnostics, predictions, provenance, models, and acceptance gates;
12. independently verify persisted artifacts and keep all final-test transforms,
    inference, predictions, metrics, and feedback closed.

The completed run selected refined LightGBM using validation only (AP 0.35535042).
Under the same LightGBM candidate, parameters, seed, split, and protocol, adding
the five strict-prior graph fields increased validation AP from 0.35535042 (B) to
0.47175420 (C), a +0.11640378 delta. All 15 acceptance gates and the independent
artifact verification passed; the final quality suite reports 207 passing tests.
The authoritative details remain in
`reports/generated/SPRINT_3_STATUS.md`.

### 4.4 Sprint 4 GraphSAGE and product layer

Sprint 4 preserved the two frozen Sprint 3 LightGBM references and executed a
transaction/edge GraphSAGE experiment. Sender and receiver embeddings are combined
with transaction features; no unsupported account-level fraud label is created.
Because full-graph training was not defensible on the available workstation, the
run used explicitly disclosed deterministic samples: 150,000 of 1,422,288 eligible
context edges, 52,396 supervised transactions (all 2,396 eligible positives plus
50,000 negatives), and a 300,000-of-3,554,957 outer-train inference graph. It then
scored all 761,749 frozen validation transactions.

Under the identical validation-only operating rule, graph-enhanced LightGBM
remained the leader at AP 0.47175420; refined transaction LightGBM reached
0.35535042 and GraphSAGE reached 0.00943876. The sampled GNN result therefore does
not establish added value over either reference and is not represented as a full-
graph result.

The product layer generated 20 real validation cases with at least seven observed
evidence items each. Observed facts and model evidence are separate; LightGBM
TreeSHAP passed additivity verification, while the GNN explanation is labeled local
gradient-times-input sensitivity rather than SHAP or causal attribution. All cases
have deterministic no-LLM notes. Streamlit loads saved artifacts only and exposes
Executive Dashboard, Investigation Queue, Case Investigator, and Model Comparison.
All 19 acceptance gates, independent verification, 289 tests, Ruff, and dependency
checks passed in the final quality run. Final-test access remained sealed. The
authoritative details remain in `reports/generated/SPRINT_4_STATUS.md`.

## 5. Sprint boundaries and exclusions

Sprint 1 excluded all model training. Sprint 2 implemented only the three fixed
transaction baselines. Sprint 3 completed validation-only refinement and the
controlled tabular graph-feature comparison. Sprint 4 completed the bounded sampled
GraphSAGE transaction-edge experiment and saved-artifact product layer described in
Section 4.4.

The following remain explicitly outside the completed Sprint 4 scope:

- final-test feature materialization, inference, predictions, or metrics;
- account-level fraud-target invention or presenting GNN sensitivity as SHAP;
- production calibration, automatic adverse action, or unconstrained LLM evidence
  generation;
- PaySim, cloud deployment, or other stretch work.

These boundaries prevent future work or final-test feedback from obscuring the
completed data foundation and validation-only model evidence.

## 6. System contracts

### 6.1 Configuration

- `configs/base.yaml` contains shared paths, mappings, seed, split, and feature
  settings.
- `configs/quick.yaml` inherits the base and enables a deterministic limit of at
  most 10,000 source rows.
- `configs/full.yaml` inherits the base and disables sampling.
- `configs/baseline.yaml` inherits the full contract, freezes upstream artifact
  hashes, and declares all Sprint 2 preprocessing/model/metric/test-gate settings.
- `configs/refinement.yaml` inherits the frozen baseline contract and declares
  Sprint 3 temporal folds, bounded candidate grids, threshold constraints, feature-
  family ablation, score representation, and closed final-test policy.
- `configs/sprint4.yaml` declares GraphSAGE sampling/training, full-validation
  scoring, operating metrics, case/evidence, explanation, product, and closed
  final-test settings while freezing the Sprint 3 references.
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
- Sprint 2 through Sprint 4 permit model access to train and validation only.
  Sprint 3 inner folds are subsets of outer train. Sprint 4 validation embeddings
  use sampled outer-train edges only; no validation edge enters message passing.
  Test metadata may be read from the pre-existing split JSON, but test feature rows
  may not be loaded or transformed.

### 6.5 Graph semantics

- Nodes are canonical bank-account composites.
- Edges are sender-to-receiver transactions.
- Direction is preserved.
- Repeated edges remain analytically meaningful multiedges.
- Exact duplicate records are reported separately and are never silently removed.
- Sprint 1 graph features are interpretable prior fan-in/fan-out and repeated-pair
  history.
- Sprint 2 transaction baselines exclude all five graph-history predictors. Their
  addition is isolated to the controlled Sprint 3 C-family graph-value arm.
- The primary B-versus-C graph-value result must hold model family, selected
  parameters, seed, split, and evaluation protocol constant.
- `sender_prior_fan_out_degree` duplicates
  `sender_previous_unique_counterparties`, and
  `receiver_prior_fan_in_degree` duplicates
  `receiver_previous_unique_counterparties` on the frozen data. Report the required
  all-five C arm and a novel-three sensitivity; do not hide this redundancy.
- Sprint 4 GraphSAGE retains direction and repeated edges, predicts transactions/
  edges rather than accounts, and combines sender/receiver embeddings with
  transaction features.
- Node structural inputs use directed degrees and deterministic composite-identity
  signals. Cross-currency monetary node totals are excluded without an FX source.
- GraphSAGE raw logits provide ranking; sigmoid values are explicitly uncalibrated
  because sampled negatives and positive weighting change the fitted class prior.
- Deterministic context sampling and its full population denominators must be
  disclosed. Sampled training may not be described as a full-graph result, even
  though evaluation covers the complete frozen validation partition.

### 6.6 Amount and currency semantics

Paid and received amounts describe different sides of a transaction. Differences
or ratios are meaningful only when payment and receiving currencies match, unless
a versioned exchange-rate source is introduced later. Cross-currency comparisons
must therefore be null rather than fabricated.

## 7. Required command surface

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[graph,app,dev]"
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\python.exe -m ruff check --no-cache src scripts tests app.py
.\.venv\Scripts\python.exe -m ruff format --no-cache --check src scripts tests app.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/run_quick_pipeline.py
.\.venv\Scripts\python.exe scripts/verify_run.py
.\.venv\Scripts\python.exe scripts/audit_raw_data.py
.\.venv\Scripts\python.exe scripts/run_full_pipeline.py
.\.venv\Scripts\python.exe scripts/train_baselines.py --config configs/baseline.yaml
.\.venv\Scripts\python.exe scripts/verify_baselines.py --config configs/baseline.yaml
.\.venv\Scripts\python.exe scripts/validate_sprint2.py --config configs/baseline.yaml
.\.venv\Scripts\python.exe scripts/train_refined_models.py --config configs/refinement.yaml
.\.venv\Scripts\python.exe scripts/verify_refinement.py --config configs/refinement.yaml
.\.venv\Scripts\python.exe scripts/validate_sprint3.py --config configs/refinement.yaml
.\.venv\Scripts\python.exe scripts/train_graphsage_product.py --config configs/sprint4.yaml
.\.venv\Scripts\python.exe scripts/verify_sprint4.py --config configs/sprint4.yaml
.\.venv\Scripts\python.exe scripts/validate_sprint4.py --config configs/sprint4.yaml
.\.venv\Scripts\streamlit.exe run app.py
```

Sprint 1 closing evidence remains 34 tests in 25.36 seconds plus its passed quick,
full, and verification commands. After Sprint 2, the final repository suite passed
81 tests in 19.87 seconds; Ruff lint/format and `pip check` passed. The baseline run
and independent artifact verifier passed on 761,749 validation predictions, and the
refreshed manifest inventories 26 payloads excluding itself. Focused EDA and data-
validation entry points remain available through `scripts/run_eda.py` and
`scripts/validate_data.py`. The three Sprint 3 commands completed in sequence; the
generated ledger records a 2,158.664-second training run, independent artifact
verification `PASS`, 207 passing tests, and passing Ruff/dependency checks. The
Sprint 4 run completed in 49.928 seconds; independent artifact verification, 289
tests in 34.21 seconds, Ruff lint/format, real-artifact Streamlit smoke testing,
and `pip check` passed.

## 8. Artifact contract

The successful Sprint 1 through Sprint 4 runs populated their listed paths. File
presence alone remains an interface contract; the manifests, independent
verification, and generated status ledgers establish the passing executions.

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
artifacts/sprint3/run_manifest.json
artifacts/sprint3/temporal_cv/folds.json
artifacts/sprint3/temporal_cv/trials.csv
artifacts/sprint3/temporal_cv/candidate_summary.csv
artifacts/sprint3/temporal_cv/selected_candidates.json
artifacts/sprint3/refined_model_comparison.csv
artifacts/sprint3/refined_transaction_champion.json
artifacts/sprint3/baseline_vs_refined.csv
artifacts/sprint3/ablation/feature_family_ablation.csv
artifacts/sprint3/ablation/graph_value_conclusion.json
artifacts/sprint3/saturation/sprint2_vs_refined.json
artifacts/sprint3/threshold/analysis.json
artifacts/sprint3/validation_predictions.parquet
artifacts/sprint3/final_test_policy.json
artifacts/sprint4/run_manifest.json
artifacts/sprint4/frozen_sprint3_reference.json
artifacts/sprint4/model/graphsage.pt
artifacts/sprint4/model/inference_node_embeddings.npy
artifacts/sprint4/validation_predictions.parquet
artifacts/sprint4/model_comparison.csv
artifacts/sprint4/sampling/sampling_disclosure.json
artifacts/sprint4/product/cases.json
artifacts/sprint4/product/investigation_queue.csv
artifacts/sprint4/product/tree_shap_explanations.json
artifacts/sprint4/product/graphsage_explanations.json
artifacts/sprint4/product/dashboard_summary.json
artifacts/sprint4/final_test_policy.json
artifacts/sprint4/verification_report.json
artifacts/sprint4/quality_report.json
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

The Sprint 4 manifest records deterministic sampled GraphSAGE training and full
761,749-row validation scoring. The run retains a 150,000-edge training context, a
52,396-row supervised sample, and a 300,000-edge inference context with exact
population denominators. Verification checks the frozen Sprint 3 hashes, full
validation metrics, checkpoint/embedding consistency, 20 evidence-backed cases,
explanations, product artifact round trips, and the sealed final-test policy. The
manifest inventories 43 payloads excluding itself; generated model/data artifacts
remain Git-ignored.

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
- tests for feature-family contracts, expanding temporal-fold chronology, fold-
  local fitted state, convergence evidence, raw/probability score separation,
  saturation/tie diagnostics, whole-score-group thresholds, same-model ablation,
  and Sprint 3 independent verification;
- tests for directed adjacency aggregation, GraphSAGE edge logits, deterministic
  sampling, transaction-only target use, full-validation row alignment, frozen
  reference integrity, case/evidence separation, explanation contracts,
  saved-artifact Streamlit loading, and Sprint 4 independent verification;
- aggregate logs must avoid gratuitous entity-name output;
- no metric, chart, comparison, or completion claim without generated evidence.

## 10. Definition of done

All fourteen Sprint 1 items, all ten Sprint 2 items, all fifteen Sprint 3 gates,
and all nineteen Sprint 4 gates have executable evidence. Sprint 1 through Sprint 4
are `PASS`; see their generated status reports for exact commands, artifacts,
runtime, warnings, and limitations. Sprint 4 independent verification and
repository quality checks passed, and the final-test gate remained closed.

## 11. Current stop boundary

Sprint 4 sampled GraphSAGE validation, case/evidence generation, explanations, and
the saved-artifact Streamlit product layer are complete and verified `PASS`. The
GraphSAGE experiment scored the full frozen validation partition but did not beat
either frozen LightGBM reference. The final-test gate remains closed: do not load,
transform, construct graphs from, score, tune on, or evaluate final-test rows
without separate explicit authorization. Stop before final evaluation.
