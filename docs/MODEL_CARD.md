# Model Card — Sprint 2 Transaction Baseline Champion

## Current state

Sprint 2 trained three transaction-level baselines on the frozen Sprint 1
chronological protocol. Random Forest is the **Transaction Baseline Champion**
because it achieved the highest validation PR-AUC, implemented as non-interpolated
average precision. Champion selection did not use final-test features, labels,
predictions, or metrics.

This is an experiment result on synthetic data, not a production model approval.
Sprint 3 refinement and graph-value comparison have not started.

## Intended use

The model ranks transactions as candidates for trained human review and provides a
transaction-only reference for a later controlled graph-value experiment. A score
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

The target, partition, transaction/source identity, raw timestamp, account/node
IDs, and all five directed graph-history fields are excluded. In particular,
prior fan-in/fan-out and repeated-pair count are withheld so Sprint 3 can measure
graph value against a transaction-only baseline.

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

## Final-test policy

Final-test inference was **not performed**. Sprint 2 did not use test rows for
preprocessing fit, training, model selection, threshold work, or evaluation. Test
counts above come only from pre-existing Sprint 1 split metadata. The test gate may
be opened only after the later refinement protocol is frozen; its outcome must not
be used to revise the selected specification.

## Limitations

- IBM AML-Data is synthetic and does not establish production-bank performance.
- Severe class imbalance makes accuracy uninformative and precision highly
  sensitive to prevalence and operational capacity.
- Temporal prevalence shift is already visible across the frozen partitions.
- Candidate configurations were fixed baselines, not tuned estimators; the linear
  model retained a convergence warning.
- Scores are uncalibrated and the `0.5` threshold was not optimized.
- No fairness, subgroup, calibration, robustness, temporal cross-validation,
  feature-ablation, or deployment validation was performed.
- Graph-history fields were intentionally excluded; no conclusion about graph
  value, GraphSAGE, or network-model superiority is available.
- Validation metrics measure synthetic-label ranking/classification behavior and
  are not causal explanations or evidence of wrongdoing.
