# ARGUS AI Roadmap

**Completed boundary:** Sprint 1 only  
**Overall Sprint 1 status:** `PASS` — 14/14 acceptance criteria  
**Later work in this document:** recommendations, not authorization or results

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

## Recommended Sprint 2 — Baseline + Model Exploration

Sprint 2 should start only after Sprint 1 is marked PASS. Recommended scope:

- implement Logistic Regression and Random Forest;
- choose and implement either LightGBM or XGBoost with a documented dependency
  decision;
- use the identical frozen chronological partitions for all models;
- build training-only preprocessing/encoding pipelines;
- address imbalance through justified training-time methods without contaminating
  validation/test;
- generate PR-AUC, Recall@K, Precision@K, F1, FPR, alert count, confusion matrix,
  and secondary ROC-AUC from code;
- compare multiple operational K values, configured rather than hard-coded;
- select the Transaction Baseline Champion on validation evidence only;
- save model/config/runtime/prediction provenance and machine-readable comparison
  artifacts;
- leave final-test labels untouched during selection and threshold work.

### Recommended Sprint 2 entry criteria

- Sprint 1 status is PASS;
- quick run is repeatable with the same manifest-relevant counts;
- the verified full-data split and feature schema are frozen for model comparison;
- split metadata is frozen and versioned;
- feature columns and forbidden leakage fields are reviewed;
- no unresolved schema, identity, or referential-integrity failure remains.

### Recommended Sprint 2 exit criteria

- all three baseline families run under one protocol;
- validation comparison is generated, not manually typed;
- champion selection and threshold basis are recorded;
- top-K tie behavior is deterministic;
- tests cover metric edge cases and training-only transformations;
- no final-test-driven selection occurred;
- claims are limited to the executed dataset/run scope.

## Beyond the current planning horizon

The master implementation prompt retains later refinement, graph-value, GraphSAGE,
case/evidence, explainability, and Streamlit phases. They are intentionally not
expanded here so they cannot be mistaken for current work. No Sprint 3+ status or
result is claimed.

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
