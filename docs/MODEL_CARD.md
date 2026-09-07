# Model Card — Sprint 2 Baseline and Sprint 3 Refinement Evidence

## Current state

Sprint 2 trained three transaction-level baselines on the frozen Sprint 1
chronological protocol. Random Forest remains the **Transaction Baseline Champion**
for that immutable snapshot. Sprint 3 completed temporal refinement and selected
LightGBM as the **Refined Transaction Baseline Champion** at validation AP
0.35535042. The controlled LightGBM graph-enhanced C arm reached validation AP
0.47175420, +0.11640378 over the same-model B arm.

All 15 Sprint 3 acceptance gates, artifact verification, and 207 tests passed.
These are experiment results on synthetic data, not production model approval.
Final-test inference remained closed, and Sprint 4/GraphSAGE has not started.

## Intended use

The model ranks transactions as candidates for trained human review and provides a
transaction-only reference for the controlled Sprint 3 graph-value experiment. A score
means elevated investigation priority under the experimental protocol. It is not
proof of laundering, guilt, or a basis for automatic blocking, freezing, reporting,
or another adverse action.

## Data and evaluation protocol

The source is the synthetic IBM AML-Data HI-Small dataset. Sprint 2 consumes the
unsampled 5,078,345-row Sprint 1 feature Parquet and its frozen timestamp-group-
preserving split.

| Partition | Rows | Positives | Positive rate | Sprint 2 use |
| --- | ---: | ---: | ---: | --- |
| Train | 3,554,957 | 2,856 | 0.080338524% | Preprocessing fit and model fit |
| Validation | 761,749 | 760 | 0.099770397% | Transform, metrics, and champion selection |
| Test | 761,639 | 1,561 | 0.204952740% | Metadata only; no model access |

Validation prevalence is 1.2418748968 times train prevalence; test prevalence is
2.0542440105 times validation prevalence. Precision and fixed-threshold alert
volume are prevalence-sensitive, so validation values must not be assumed to
transfer unchanged to the later test period. The partitions were not shuffled,
resampled, or rebalanced to equalize these rates.

## Predictor and preprocessing contract

All three models use the same 74 transformed predictors:

- 31 transaction, time, and strictly-prior account-history numeric fields;
- four explicit missing-value indicators;
- train-vocabulary one-hot encodings for payment currency, receiving currency, and
  payment format;
- train-frequency encodings for sender and receiver bank.

Numeric medians, means, population standard deviations, category vocabularies, and
bank frequencies were fit from training rows only. Validation uses transform-only
state. Unknown low-cardinality values map to an all-zero one-hot block; unknown
banks map to frequency zero.

The target, partition, transaction/source identity, raw timestamp, and account/node
IDs are always excluded. All five directed graph-history fields are excluded from
the Sprint 2 and Sprint 3 A/B transaction-reference arms. Sprint 3 adds them only
to the controlled C arm so graph value is measured against the same-model B arm.

## Candidate implementations and imbalance handling

| Candidate | Implementation | Imbalance handling | Fit seconds |
| --- | --- | --- | ---: |
| Logistic Regression | `SGDClassifier(loss="log_loss")` | `class_weight="balanced"` | 56.8203844 |
| Random Forest | `RandomForestClassifier` | `class_weight="balanced_subsample"` | 177.2493681 |
| LightGBM | `LGBMClassifier` | train-only `scale_pos_weight=1243.7328431372548` | 100.0102195 |

The SGD logistic implementation is a resource-bounded Logistic Regression baseline,
not `sklearn.linear_model.LogisticRegression`. It reached the configured maximum of
20 iterations before convergence; that warning is retained as evidence. None of
the three score outputs is probability-calibrated.

LightGBM was selected once over XGBoost for the boosting slot because its CPU
histogram path supported the planned full-data, deterministic single-thread,
bounded-memory configuration. XGBoost was not run, so no empirical LightGBM-versus-
XGBoost performance claim is made.

## Validation results

PR-AUC below is average precision and the sole champion-selection metric. ROC-AUC
is secondary. Remaining metrics use the fixed configuration threshold `0.5`; it
was not optimized or tuned in Sprint 2. Accuracy is intentionally not primary.

| Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Random Forest** | **0.08859105** | 0.97497347 | 0.03127670 | 0.72236842 | 0.05995741 | 0.02234461 | 17,553 |
| Logistic Regression | 0.00663424 | 0.91808364 | 0.00491412 | 0.96184211 | 0.00977828 | 0.19451530 | 148,755 |
| LightGBM | 0.00547556 | 0.85107181 | 0.00507521 | 0.87368421 | 0.01009180 | 0.17105109 | 130,832 |

Random Forest top-K validation results are:

| K | Precision@K | Recall@K | True positives |
| ---: | ---: | ---: | ---: |
| 100 | 0.34 | 0.04473684 | 34 |
| 500 | 0.172 | 0.11315789 | 86 |
| 1,000 | 0.13 | 0.17105263 | 130 |

Ranking ties use score descending and `source_row_number` ascending. The fixed
threshold produces 17,553 validation alerts for the champion, far above the three
reported K budgets; these are different operating views and not a selected
production capacity.

Logistic Regression assigns exact score `1.0` to 60,119 validation rows containing
427 positives; LightGBM assigns it to 108,347 rows containing 663 positives. Since
K 100/500/1,000 falls inside those tied groups, their source-row tie-break
materially determines which equally scored rows enter K. Those top-K values are
reproducible but do not show discrimination within the tied score plateau. Random
Forest has no exact score-1 rows; champion selection uses tie-aware average
precision and is unaffected by the source-row ordering of equal scores.

The precision-recall PNG alone uses deterministic endpoint-preserving display
decimation capped at 10,000 points per model. Every numerical metric and champion
selection calculation uses the complete 761,749-row validation score vectors.

## Reproducibility and verification

The full Sprint 2 run took 389.0078022000016 seconds. Machine-readable evidence is
in `artifacts/sprint2/model_comparison.json`,
`transaction_baseline_champion.json`, per-model metadata/metric files, and the
761,749-row `validation_predictions.parquet`. The independent verifier recomputed
AP and ROC-AUC from saved scores, deserialized all models, repeated validation-only
selection, and confirmed that no test prediction artifact exists. The refreshed
manifest inventories 26 payloads, excluding itself. The final repository suite
passed 81 tests in 19.87 seconds; Ruff lint/format and `pip check` also passed.

## Sprint 3 refinement and graph-value results

`configs/refinement.yaml` declares bounded candidate grids for Logistic Regression,
Random Forest, and LightGBM. Candidate selection uses mean average precision over
three expanding chronological folds contained entirely within outer train. Each
fold fits its own preprocessing state on its training prefix. Selected candidates
were refit on all outer train and compared on outer validation; the final test
remained unopened.

| Validation arm | Model | PR-AUC (AP) | ROC-AUC |
| --- | --- | ---: | ---: |
| Refined transaction baseline | Logistic Regression | 0.02600700 | 0.95570357 |
| Refined transaction baseline | Random Forest | 0.12967475 | 0.97718091 |
| **Refined transaction baseline** | **LightGBM** | **0.35535042** | **0.98179242** |
| Graph-enhanced C arm | LightGBM | 0.47175420 | 0.98635944 |

At the predeclared validation-only joint operating rule, refined LightGBM B emitted
4,931 alerts with precision 0.10302170, recall 0.66842105, F1 0.17852750, and FPR
0.00581217. This operating point is research evidence, not a deployment policy.

The selected refined Logistic Regression uses `LogisticRegression` with the
`newton-cholesky` solver, L2 regularization, square-root class weighting, and
`C=0.01`. Its saved convergence evidence passed; the Sprint 2 Logistic warning
remains part of the historical baseline.

Sprint 3 treats raw decision margins as the ranking representation for Logistic
Regression and LightGBM and retains probabilities for diagnostics. Saturation
artifacts report raw ranges, probability collapse, exact/near boundary counts,
unique scores, tied-group label composition, class-weight controls, and estimator
stability. Top-K output includes deterministic row selection plus tie-aware
expected/minimum/maximum true positives; a K that cuts a tie does not demonstrate
within-tie superiority.

Executable reproduction confirmed that the Sprint 2 exact-zero/one plateaus came
from sigmoid/link conversion of distinguishable extreme raw margins, not model
serialization. The upstream Sprint 2 causes were the nonconverged, weakly
regularized balanced SGD with heavy-tailed amount signals and full positive-class
weight with near-zero leaf/Hessian regularization in LightGBM. Refined LightGBM had
no exact probability-zero/one rows; refined Logistic Regression retained 135 exact
ones, while raw-score and probability AP agreed at 0.02600700.

Operating thresholds are chosen only on outer validation. The configured primary
rule includes complete equal-score groups and maximizes recall subject to a 5,000-
alert budget and 0.01 FPR ceiling, alongside maximum-F1 and single-constraint
alternatives. These values are research evidence, not a deployment policy.

Graph value was tested with the same selected LightGBM candidate, parameters, seed,
outer split, and evaluation protocol across A transaction-only, B transaction plus
temporal/history, and C B plus all five graph-history fields. Because two C fields
duplicate existing history columns on the frozen data, a separate novel-three arm
adds only sender prior fan-in, receiver prior fan-out, and repeated-pair count.
Both C and the duplicate-removed novel-three sensitivity measured AP 0.47175420,
versus 0.35535042 for B. This is a tabular feature-family experiment, not GraphSAGE.
Complete folds, thresholds, tie diagnostics, runtimes, and artifacts are in
[`SPRINT_3_STATUS.md`](../reports/generated/SPRINT_3_STATUS.md).

## Final-test policy

Final-test inference was **not performed** in Sprint 2 or Sprint 3. Neither sprint
used test rows for preprocessing fit, training, model selection, threshold work,
prediction, or evaluation. Test counts above come only from pre-existing Sprint 1
split metadata. Any later test evaluation may not revise the frozen selected
specification.

## Limitations

- IBM AML-Data is synthetic and does not establish production-bank performance.
- Severe class imbalance makes accuracy uninformative and precision highly
  sensitive to prevalence and operational capacity.
- Temporal prevalence shift is already visible across the frozen partitions.
- Sprint 2 candidate configurations were fixed baselines, not tuned estimators; its
  linear model retained a convergence warning and its `0.5` threshold was not
  optimized.
- Sprint 3 results are validation evidence after repeated development comparisons;
  they are not a final-test estimate.
- No fairness, subgroup, calibration, robustness, or deployment validation is
  provided by this scope.
- No GraphSAGE or other network-model result is available. The Sprint 3 graph
  conclusion concerns only the same-model tabular graph-feature ablation.
- Validation metrics measure synthetic-label ranking/classification behavior and
  are not causal explanations or evidence of wrongdoing.
