# ARGUS AI Roadmap

**Completed boundary:** Sprint 1 and Sprint 2
**Overall Sprint 1 status:** `PASS` — 14/14 acceptance criteria  
**Sprint 2 status:** `PASS` — three baselines and validation-only champion verified
**Current stop boundary:** Sprint 3 and later have not started

## Sprint 1 — Repository Foundation + Data Proof

The goal is one working, reproducible path from the local IBM CSV files to validated
EDA, chronological partitions, and leakage-safe initial features.

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Existing coursework preserved | Files and checksums under `reports/existing_coursework/` | PASS — all seven documents match Downloads by SHA-256 |
| Installable dependency set | Editable install and dependency check | PASS — install completed; `pip check` found no broken requirements |
| IBM transaction/accounts loading | Real files load from documented local paths | PASS — 5,078,345 transactions and 518,581 accounts loaded |
| Schema and quality validation | Machine-readable validation/profile output | PASS — full raw, canonical, and saved-output reports pass |
| Canonical preprocessing | Stable source identity, normalized fields, node IDs | PASS — 5,078,345 canonical rows generated |
| Account referential validation | Aggregate endpoint report | PASS — all 515,088 transaction nodes matched |
| Chronological split | Saved counts/boundaries and strict-order checks | PASS — 3,554,957/761,749/761,639 full rows; strict boundaries |
| Leakage protection | Same-timestamp and future-append automated tests | PASS — included in final 34-test suite |
| Real executable EDA | Generated profile, exact tables, and sampled PNG figures | PASS — all-row exact tables plus labeled 100,000-row/50,000-edge scope |
| Transaction/time/history features | Generated table plus unit tests | PASS — 5,078,345 rows, 52 total columns |
| Initial directed graph features | Prior fan-in/fan-out/pair history plus tests | PASS — five graph-history fields generated for all rows |
| Automated suite | `.venv\Scripts\pytest.exe -q` | PASS — 34 passed in 25.36 seconds |
| One-command smoke path | `python scripts/run_quick_pipeline.py` | PASS — 11.197340699996857 seconds |
| Reproduction documentation | README paths/commands agree with final manifests | PASS — paths and evidence synchronized |

Sprint 1 closed after each gate obtained executable evidence. The quick manifest
contains 38 hashed payloads plus itself. The full DuckDB manifest contains 54
hashed payloads plus itself and records a 304.93406899999536-second, `2GB`,
one-thread run. Verification reconciled 5,078,345 rows and unique transaction IDs
across all four primary Parquet outputs. See
`reports/generated/SPRINT_1_STATUS.md` for the final ledger.

## Sprint 1 execution order

1. validate configuration and source presence;
2. load/validate/canonicalize transactions and accounts;
3. prove normalized composite account references;
4. create chronological partitions without splitting timestamp groups;
5. generate transaction-local and calendar features;
6. generate strictly-prior account-history features;
7. generate strictly-prior directed fan-in/fan-out features;
8. generate EDA and run manifests;
9. run focused tests, then the full test suite;
10. run the quick pipeline from a clean command;
11. reconcile artifact inventory, row counts, and documentation.

If a gate fails, fix and rerun that gate before widening scope. Do not begin model
work to compensate for an incomplete data foundation.

## Completed Sprint 2 — Baseline + Model Exploration

Sprint 2 entered only after Sprint 1 passed and then executed the full-data,
transaction-only protocol in `configs/baseline.yaml`.

| Gate | Executed evidence | Status |
| --- | --- | --- |
| Frozen inputs | Sprint 1 feature/split hashes checked before model access | PASS |
| Train-only transform | 3,554,957 train rows fit 74 predictors; validation transform-only | PASS |
| Logistic Regression | `SGDClassifier(loss="log_loss")`, balanced weight | PASS with convergence warning retained |
| Random Forest | 80 trees, balanced subsample, bounded configuration | PASS |
| LightGBM | 250 trees, train-only `scale_pos_weight` | PASS |
| Imbalance-aware metrics | AP/PR-AUC primary; ROC-AUC, F1, FPR, alerts, top-K saved | PASS |
| Deterministic ranking | K 100/500/1000; source-row tie-break | PASS |
| Validation-only champion | Random Forest, validation AP 0.08859105087174989 | PASS |
| Final-test gate | No test matrix, inference, prediction, metric, or selection input | PASS |
| Graph-history exclusion | Five prior fan-in/fan-out/pair fields withheld | PASS |
| Machine-readable evidence | JSON, CSV, Parquet, model files, plots, manifest | PASS |
| Independent verification | 761,749 saved validation rows and model artifacts checked | PASS |
| Final quality | 81 tests in 19.87 seconds; Ruff and `pip check` PASS | PASS |

Validation AP ranked Random Forest first (0.08859105087174989), Logistic Regression
second (0.006634242622201133), and LightGBM third (0.0054755570960638884). The
decision threshold remained the fixed, unoptimized configuration value `0.5`.
The full run took 389.0078022000016 seconds, and the refreshed manifest inventories
26 payloads excluding itself.

LightGBM filled the boosting slot because its CPU histogram path matched the full-
data, deterministic single-thread, bounded-memory plan. XGBoost was not executed,
so the roadmap records no empirical superiority claim between those libraries.

The frozen positive rates are 0.080338524% for train, 0.099770397% for validation,
and 0.204952740% for test metadata. This temporal prevalence shift is preserved,
not equalized; later precision and alert-volume comparisons must account for it.

Top-K evidence also retains its tie limitation. Logistic Regression has 60,119
validation rows at score `1.0` (427 positives), and LightGBM has 108,347 (663
positives); K 100/500/1,000 cuts through those ties, so deterministic source-row
ordering materially determines membership. Random Forest has no exact score-1
rows, and the validation-AP champion rule is tie-aware.

### Sprint 2 exit status

All planned baseline families, metrics, provenance, leakage guards, and validation-
only selection evidence passed. The logistic baseline reached its configured
20-iteration limit before convergence, its scores and all other candidate scores
are uncalibrated, and no threshold/hyperparameter refinement was attempted. These
are explicit inputs to Sprint 3 rather than reasons to rewrite Sprint 2 results.

## Unstarted Sprint 3 and later work

The master implementation prompt retains model refinement, threshold work, temporal
cross-validation, feature-family ablation, the controlled graph-value experiment,
GraphSAGE, case/evidence, explainability, and Streamlit phases. None has started,
and no Sprint 3+ result is claimed.

## Stop conditions

Stop and report rather than invent when:

- an expected source file is unavailable or its schema/hash changes unexpectedly;
- the environment cannot install a required dependency;
- a chronological split cannot satisfy strict timestamp separation;
- a historical feature changes after future rows are appended;
- an artifact count cannot be reconciled;
- a command exceeds available hardware and no defensible sampled mode exists.

A blocker does not authorize synthetic substitution to be reported as IBM evidence.
Finish safe independent work, record the exact failed command, and identify the
rerun action.
