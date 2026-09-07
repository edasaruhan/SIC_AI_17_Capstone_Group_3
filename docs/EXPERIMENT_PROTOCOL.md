# ARGUS AI Experiment Protocol

**Protocol version:** 3.0
**Completed execution scope:** Sprint 1 data proof, Sprint 2 transaction baselines, and Sprint 3 refinement/graph value
**Sprint 3 status:** `PASS` — 15/15 acceptance gates and 207 tests passed
**Current stop boundary:** Sprint 3 complete; Sprint 4/GraphSAGE not started
**Pipeline/test results:** generated artifacts control each sprint's empirical status

This document defines how ARGUS experiments become comparable and scientifically
defensible. Observed values are included only where they are read from generated
manifests and status reports. The completed Sprint 3 evidence ledger is
[`SPRINT_3_STATUS.md`](../reports/generated/SPRINT_3_STATUS.md).

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

The quick, full, baseline, and refinement runs write inventories to
`artifacts/quick/run_manifest.json`, `artifacts/full/run_manifest.json`, and
`artifacts/sprint2/run_manifest.json`, and `artifacts/sprint3/run_manifest.json`.
Generated artifacts and local datasets are
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

The Sprint 1 closing run passed 34 tests in 25.36 seconds under CPython 3.12.10.
After Sprint 2 implementation and artifact verification, the final repository run
passed 81 tests in 19.87 seconds. Ruff lint, Ruff format verification, and
`pip check` also passed. Exact commands and resolved limitations are preserved in
the generated Sprint status reports.

## 10. Executed Sprint 2 baseline protocol

Sprint 2 ran once on the full, frozen Sprint 1 feature/split snapshot configured in
`configs/baseline.yaml`. It did not change boundaries, resample rows, or create a
test feature matrix.

### Frozen partitions and temporal prevalence

| Partition | Rows | Positives | Positive rate | Modeling access |
| --- | ---: | ---: | ---: | --- |
| Train | 3,554,957 | 2,856 | 0.0008033852448848186 | Preprocessing and model fit |
| Validation | 761,749 | 760 | 0.0009977039681049794 | Transform, comparison, selection |
| Test | 761,639 | 1,561 | 0.0020495274007764834 | Pre-existing metadata only |

The validation/train positive-rate ratio is 1.2418748968286322. The
test/validation ratio is 2.0542440105448496. Because precision and fixed-threshold
alert volume are prevalence-sensitive, validation operating values cannot be
assumed to transfer unchanged to the later test period. The split was deliberately
left frozen rather than rebalanced.

### Train-only transform

The explicit allow-list produces 74 float32 columns from transaction, calendar,
and strictly-prior account-history features. Numeric median imputation and scaling,
low-cardinality vocabularies, and bank-frequency maps are fit on train only.
Unknown validation categories use declared zero encodings. The target, identifiers,
timestamp/provenance, account/node IDs, and all five prior fan-in/fan-out/pair graph
features are forbidden predictors. Withholding graph history preserves a valid
  transaction-only reference for the controlled Sprint 3 graph-value experiment.

### Executed models

The fixed candidate configurations were:

1. Logistic Regression implemented by
   `sklearn.linear_model.SGDClassifier(loss="log_loss")`, balanced class weight;
2. `sklearn.ensemble.RandomForestClassifier`, balanced subsample weight;
3. `lightgbm.LGBMClassifier`, using the train-only class ratio as
   `scale_pos_weight`.

LightGBM was chosen once for the boosting slot because its CPU histogram path fit
the intended full-data, deterministic single-thread, bounded-memory execution.
XGBoost was not run; this dependency/runtime decision is not a comparative model-
quality result.

The SGD implementation enabled bounded training at 3,554,957 rows. It reached the
configured 20-iteration maximum before convergence; the warning is retained. The
models are uncalibrated, were not hyperparameter-tuned, and used no oversampling.

### Metrics, threshold, and deterministic ranking

The primary metric is non-interpolated average precision, recorded as PR-AUC.
ROC-AUC is secondary. Precision, recall, F1, FPR, confusion counts, alert count,
and alert rate are computed at the fixed configuration threshold `0.5`; this
threshold was not optimized in Sprint 2. Accuracy is not primary. Recall@K and
Precision@K use `K = 100, 500, 1000`, score descending, then
`source_row_number` ascending for deterministic ties.

Tie diagnostics are part of interpretation: Logistic Regression has 60,119
validation rows at exact score `1.0`, including 427 positives; LightGBM has
108,347, including 663 positives. All configured K cutoffs lie inside those score
plateaus, so their top-K membership materially depends on the deterministic source-
row tie-break and does not establish within-tie discrimination. Random Forest has
no exact score-1 rows. Average precision groups score ties and remains independent
of arbitrary row order for champion selection.

Only the raster precision-recall visualization is scaled: it deterministically
keeps endpoints and at most 10,000 evenly indexed curve points per model. Numerical
AP, ROC-AUC, threshold, and Top-K metrics use every one of the 761,749 validation
rows; the sampling is not part of evaluation or selection.

### Observed validation comparison

| Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.08859105087174989 | 0.97497346510507 | 0.03127670483678004 | 0.7223684210526315 | 0.05995740730628515 | 0.022344606820860747 | 17,553 |
| Logistic Regression | 0.006634242622201133 | 0.9180836419863874 | 0.004914120533763571 | 0.9618421052631579 | 0.00977828311540648 | 0.1945152952276577 | 148,755 |
| LightGBM | 0.0054755570960638884 | 0.851071813334877 | 0.005075210957563899 | 0.8736842105263158 | 0.01009179889354976 | 0.17105109272275945 | 130,832 |

Random Forest is the Transaction Baseline Champion under the predeclared rule:
maximum validation average precision, with model-name ascending only as an exact-
score tie-break. No test metric entered selection.

### Test gate and artifacts

Final-test inference was not performed. Test rows were not used for preprocessing,
training, tuning, selection, threshold work, prediction, or evaluation. Only their
pre-existing Sprint 1 metadata is reported. No test prediction artifact exists.

The full run took 389.0078022000016 seconds. Executable outputs include:

```text
artifacts/sprint2/run_manifest.json
artifacts/sprint2/verification_report.json
artifacts/sprint2/quality_report.json
artifacts/sprint2/preprocessing/manifest.json
artifacts/sprint2/model_comparison.json
artifacts/sprint2/model_comparison.csv
artifacts/sprint2/transaction_baseline_champion.json
artifacts/sprint2/validation_predictions.parquet
artifacts/sprint2/models/
artifacts/sprint2/figures/
artifacts/sprint2/final_test_policy.json
```

The independent verifier recomputed AP/ROC-AUC from 761,749 saved validation rows,
deserialized all three estimators, repeated validation-only champion selection,
and confirmed the closed test gate. The refreshed manifest inventories 26 payloads
excluding itself. Sprint 2 closed with 81 tests passing in 19.87 seconds, Ruff
lint/format passing, and `pip check` passing.

The reproducible command sequence after frozen Sprint 1 artifacts exist is:

```powershell
.\.venv\Scripts\python.exe scripts/train_baselines.py --config configs/baseline.yaml
.\.venv\Scripts\python.exe scripts/verify_baselines.py --config configs/baseline.yaml
.\.venv\Scripts\python.exe scripts/validate_sprint2.py --config configs/baseline.yaml
```

## 11. Sprint 3 refinement protocol

Sprint 3 was executed from `configs/refinement.yaml`. Its completion status,
selected candidates, metrics, runtimes, and graph-value conclusion are recorded in
`artifacts/sprint3/run_manifest.json` and
[`SPRINT_3_STATUS.md`](../reports/generated/SPRINT_3_STATUS.md). The rules below
were fixed before those results were interpreted.

The completed run selected refined LightGBM at validation AP 0.35535042. With the
same LightGBM candidate, parameters, seed, frozen split, and protocol, the C graph
arm reached AP 0.47175420 versus 0.35535042 for B, a +0.11640378 delta. Training
runtime was 2,158.664 seconds; independent artifact verification, 207 tests, Ruff
lint/format, and `pip check` passed. The final test remained untouched.

### Outer split and expanding temporal cross-validation

The Sprint 1 outer train/validation/test boundaries remain unchanged. Candidate
tuning uses three expanding-window folds derived only from outer train at the
configured cumulative timestamp quantiles. Every fold satisfies:

```text
fold train ⊂ outer train
fold validation ⊂ outer train
max(fold train timestamp) < min(fold validation timestamp)
```

Equal timestamps may not cross a fold boundary. Each fold gets a newly fitted
preprocessor using only its training prefix; its later interval is transform-only.
No outer-validation or final-test category, median, scale, frequency, label, or
model statistic may enter fold fitting. One candidate per estimator family is
selected by mean fold average precision, with the configured candidate-ID tie-
break and complete-fold eligibility requirement.

The bounded candidate grids compare Logistic Regression solver/L2/class-weight
choices, two Random Forest configurations, and regularized LightGBM class-weight
controls. A refined Logistic Regression is eligible only if convergence evidence
shows `n_iter < max_iter`; increasing the configured iteration budget must not be
misreported as proof of convergence by itself. Non-finite score outputs are
ineligible.

### Outer validation and metrics

After inner-fold selection, each chosen estimator is refit on all outer-train rows
with a newly train-fitted transform and evaluated once on outer validation. The
refined Transaction Baseline Champion is selected by outer-validation average
precision. PR-AUC/average precision is primary, ROC-AUC is secondary, and
Recall@K, Precision@K, F1, FPR, alert count, and alert rate are operational
evidence. Accuracy is not a primary measure.

Threshold optimization is confined to outer validation and uses the raw ranking
score. Candidate operating points are exact whole score groups, so an equal-score
plateau is never split merely to hit a budget. The configured primary rule
maximizes recall subject to both an alert budget of 5,000 and FPR ceiling of 0.01;
the artifact also records maximum-F1 and individual budget/FPR alternatives. These
are research operating points, not a deployed bank policy.

### Saturation and tie policy

Sprint 2 probability outputs remain immutable evidence. Sprint 3 reproduces their
diagnostics and separates two layers:

1. upstream estimator behavior, including preprocessing tails, learned raw-margin
   range, class weighting, regularization, and LightGBM leaf-score stability;
2. downstream sigmoid/probability representation, including exact and near-zero/
   one counts, unique-score counts, raw-to-probability collapse, and label makeup
   of saturated groups.

Logistic Regression and LightGBM are ranked by raw decision margin where available;
probabilities are retained as diagnostic outputs. Random Forest uses its positive-
class probability because it has no separate margin interface in this protocol.
Top-K output is deterministically ordered by score descending and
`source_row_number` ascending, but each K also records tie-aware expected, minimum,
and maximum true positives. If K intersects a tied score group, its deterministic
membership cannot establish within-tie superiority.

### Same-model feature-family ablation

The primary graph-value comparison uses one selected LightGBM candidate with the
same parameters, seed, outer train/validation rows, preprocessing discipline,
score representation, and metric protocol for all arms:

```text
A  transaction-only
B  transaction + temporal/history
C  transaction + temporal/history + all five graph-history fields
```

The five graph fields remain strictly prior and target-free. On the frozen feature
table, `sender_prior_fan_out_degree` is an exact duplicate of
`sender_previous_unique_counterparties`, and `receiver_prior_fan_in_degree` is an
exact duplicate of `receiver_previous_unique_counterparties`. Therefore the
required all-five C arm is accompanied by a declared novel-three sensitivity using
sender prior fan-in, receiver prior fan-out, and repeated-pair count. Only the
machine-generated B-versus-C validation delta may support a graph-value statement;
the experiment is tabular feature ablation, not GraphSAGE.

### Final-test gate and executable evidence

Sprint 3 did not materialize a final-test feature matrix, transform final-test rows,
run inference, write test predictions, calculate a test metric, choose a candidate
or threshold using test information, or revise a decision after test feedback.
Reported test counts and prevalence came only from frozen Sprint 1 metadata.

Run, independently verify, and close quality checks with:

```powershell
.\.venv\Scripts\python.exe scripts/train_refined_models.py --config configs/refinement.yaml
.\.venv\Scripts\python.exe scripts/verify_refinement.py --config configs/refinement.yaml
.\.venv\Scripts\python.exe scripts/validate_sprint3.py --config configs/refinement.yaml
```

The executable evidence contract includes:

```text
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
artifacts/sprint3/verification_report.json
artifacts/sprint3/quality_report.json
reports/generated/SPRINT_3_STATUS.md
```

Stop after these gates. Sprint 4/GraphSAGE, case/evidence, explainability, and
product work are not part of this protocol version and were not started.

## 12. Human review

Metrics measure ranking/classification behavior on synthetic labels. They do not
establish guilt or justify adverse action. Any later alert or case must show
observed evidence separately from model contribution, state uncertainty, and remain
subject to trained human review.
