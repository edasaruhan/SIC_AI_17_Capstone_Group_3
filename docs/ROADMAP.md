# ARGUS AI — Completed Project Roadmap

ARGUS processes the synthetic IBM AML HI-Small dataset under a chronological,
leakage-safe evaluation protocol. The project is complete through data preparation,
baseline modelling, graph-feature experiments, a sampled GraphSAGE comparison, an
analyst application, and the one-shot final evaluation.

## Completion summary

| Stage | Status | Completed outcome |
| --- | :---: | --- |
| Sprint 1 — Data foundation | **PASS** | Validation, chronological splits, EDA, and leakage-safe features for 5,078,345 transactions |
| Sprint 2 — Baseline models | **PASS** | Logistic Regression, Random Forest, and LightGBM transaction baselines |
| Sprint 3 — Refinement and graph value | **PASS** | Temporal cross-validation, threshold analysis, and controlled feature-family ablation |
| Sprint 4 — GraphSAGE and product layer | **PASS** | Transaction-edge GNN comparison, case/evidence engine, explanations, and Streamlit application |
| Sprint 5 — Final evaluation | **PASS** | One-shot evaluation of frozen models on 761,639 test transactions |

## Sprint 1 — Data foundation

The two HI-Small CSV files were checked for schema, data types, missing values,
duplicate records, account references, labels, amounts, and timestamps. Bank and
account identifiers were canonicalized as
`normalized_bank_id::UPPERCASE_ACCOUNT_ID`. Timestamp groups were kept intact in
the chronological split, producing 3,554,957 training, 761,749 validation, and
761,639 test transactions.

Feature engineering covers transaction fields, calendar context, strictly prior
account history, and directed fan-in/fan-out signals. Transactions sharing a
timestamp do not contribute to one another's history. Quick and full-data runs use
separate configurations and manifests.

## Sprint 2 — Transaction baselines

Logistic Regression, Random Forest, and LightGBM were evaluated under the same
train/validation protocol. Learned transforms were fitted on training data only.
PR-AUC was the selection metric; ROC-AUC, precision, recall, F1, FPR, alert volume,
and Top-K metrics provided operational context.

Random Forest became the immutable transaction-baseline champion at validation
PR-AUC 0.08859105. Score saturation in the initial Logistic Regression and
LightGBM runs was recorded explicitly, and Top-K results that cut an equal-score
group were not treated as evidence of within-tie ranking superiority.

## Sprint 3 — Refinement and graph contribution

Model refinement used three expanding-window temporal folds inside the outer
training period. The Logistic Regression convergence issue was corrected through
solver, regularization, and iteration changes. Raw decision scores and probability
transforms were evaluated separately, confirming that the earlier probability
plateaus arose from extreme raw margins and the link transformation rather than
model serialization.

The controlled LightGBM ablation held the model family, parameters, seed, split,
and evaluation protocol constant:

| Feature family | Validation PR-AUC |
| --- | ---: |
| Transaction only | 0.09984124 |
| Transaction + temporal/history | 0.35535042 |
| Transaction + temporal/history + graph | **0.47175420** |

Adding the five graph-history fields increased PR-AUC by 0.11640378 over the
temporal/history model. Two graph fields duplicate existing history fields on the
frozen data; the duplicate-removed three-field sensitivity produced the same
0.47175420 PR-AUC. The operating threshold was selected on validation only under
a 5,000-alert budget and a 1% FPR ceiling.

## Sprint 4 — GraphSAGE and analyst product

GraphSAGE uses the IBM transaction label for edge classification; no account-level
fraud label is derived. Sender and receiver embeddings are combined with
transaction features. The training graph was deterministically bounded to 150,000
context edges and 52,396 supervised transactions because full-graph training
exceeded the available compute budget. Validation and test edges were excluded
from message passing, and all 761,749 validation transactions were scored.

Validation PR-AUC was 0.47175420 for graph-enhanced LightGBM, 0.35535042 for the
refined transaction LightGBM, and 0.00943876 for GraphSAGE. Graph-enhanced
LightGBM therefore remained the primary model; the sampled GraphSAGE model is a
research comparator.

The product layer reads saved model and graph outputs. It separates observed
transaction/network evidence from model contribution, uses additivity-checked
TreeSHAP for LightGBM, and labels GNN explanations as local
gradient-times-input sensitivity. Streamlit does not train models at page load.

## Sprint 5 — Frozen final evaluation

Graph-enhanced LightGBM was frozen with its feature family, hyperparameters, and
validation-selected threshold before test evaluation. The complete 761,639-row
test partition was evaluated once. Test results did not alter model selection,
features, parameters, or thresholds, and no post-test tuning or retraining was
performed.

| Model | Role | Final test PR-AUC |
| --- | --- | ---: |
| Graph-enhanced LightGBM | Frozen primary model | **0.69005906** |
| Refined transaction LightGBM | Comparator | 0.53079643 |
| GraphSAGE edge classifier | Research comparator | 0.01341710 |

The positive rate increased from 0.099770% in validation to 0.204953% in final
test, a 2.054-fold shift. Precision and fixed-threshold alert volume are
prevalence-sensitive and must be interpreted in that context. Full final results
are recorded in the [final evaluation report](../reports/generated/FINAL_EVALUATION_STATUS.md)
and [final model comparison](../reports/generated/FINAL_MODEL_COMPARISON.md).

## Future research

- external and prospective validation on financial-institution data;
- subgroup, fairness, calibration, robustness, and drift analyses;
- full-graph or scalable neighbor-sampling GNN experiments;
- institution-specific threshold calibration using alert capacity and review cost;
- production controls for access, audit trails, monitoring, and model governance.

These items are research directions, not completed results, and do not change the
frozen final evaluation.
