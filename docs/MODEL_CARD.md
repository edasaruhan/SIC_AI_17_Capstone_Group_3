# Model Card — Sprint 2–4 Validation Evidence

## Current state

<!-- ARGUS_FINAL_MODEL_CARD_START -->
## One-shot final-test evidence

| Role | Frozen model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen champion | Graph-enhanced LightGBM | 0.69005906 | 0.99127213 | 0.17816018 | 0.88212684 | 0.29644779 | 0.00835704 | 7,729 |
| Comparator | GraphSAGE edge classifier | 0.01341710 | 0.82662724 | 0.00600590 | 0.11338885 | 0.01140758 | 0.03854078 | 29,471 |
| Comparator | Refined transaction LightGBM | 0.53079643 | 0.98731443 | 0.16072332 | 0.81422165 | 0.26845496 | 0.00873200 | 7,908 |

## Operational Top-K metrics

| Model | K | Precision@K | Recall@K | True positives |
| --- | ---: | ---: | ---: | ---: |
| Graph-enhanced LightGBM | 100 | 0.98000000 | 0.06278027 | 98 |
| Graph-enhanced LightGBM | 500 | 0.96800000 | 0.31005766 | 484 |
| Graph-enhanced LightGBM | 1,000 | 0.83900000 | 0.53747598 | 839 |
| GraphSAGE edge classifier | 100 | 0.18000000 | 0.01153107 | 18 |
| GraphSAGE edge classifier | 500 | 0.08200000 | 0.02626521 | 41 |
| GraphSAGE edge classifier | 1,000 | 0.06000000 | 0.03843690 | 60 |
| Refined transaction LightGBM | 100 | 0.92000000 | 0.05893658 | 92 |
| Refined transaction LightGBM | 500 | 0.86200000 | 0.27610506 | 431 |
| Refined transaction LightGBM | 1,000 | 0.66600000 | 0.42664958 | 666 |

## Prevalence shift

- Validation: 760 / 761,749 (0.00099770).
- Final test: 1,561 / 761,639 (0.00204953).
- Test/validation positive-rate ratio: 2.05424401.
- Interpretation: Precision and fixed-threshold alert volume are prevalence-sensitive; the observed chronological shift was reported without resampling or threshold changes.

## Frozen protocol statement

`graph_enhanced_lightgbm` was selected by validation PR-AUC and frozen before final-test access. Test results did not change the model, feature family, hyperparameters, or validation-selected threshold. No post-test tuning or retraining was performed.

## Machine-readable evidence

- [Final metrics](../artifacts/sprint5/final_metrics.json)
- [Final comparison](../artifacts/sprint5/final_model_comparison.json)
- [Final Top-K metrics](../artifacts/sprint5/final_top_k_metrics.json)
- [Saved final predictions](../artifacts/sprint5/final_test_predictions.parquet)
- [Prevalence analysis](../artifacts/sprint5/prevalence_shift.json)
- [Test identity/leakage audit](../artifacts/sprint5/test_identity_audit.json)
- [Immutable run manifest](../artifacts/sprint5/run_manifest.json)
- [Read-only verification](../artifacts/sprint5/verification_report.json)
- [Quality report](../artifacts/sprint5/quality_report.json)
- [Streamlit: Executive Dashboard](../artifacts/sprint5/screenshots/executive_dashboard.png)
- [Streamlit: Investigation Queue](../artifacts/sprint5/screenshots/investigation_queue.png)
- [Streamlit: Case Investigator](../artifacts/sprint5/screenshots/case_investigator.png)
- [Streamlit: Model Comparison](../artifacts/sprint5/screenshots/model_comparison.png)

## Final quality

- Status: **PASS**
- Pytest: 377 passed
- Saved-artifact verification: PASS
<!-- ARGUS_FINAL_MODEL_CARD_END -->

Sprint 2 trained three transaction-level baselines on the frozen Sprint 1
chronological protocol. Random Forest remains the **Transaction Baseline Champion**
for that immutable snapshot. Sprint 3 completed temporal refinement and selected
LightGBM as the **Refined Transaction Baseline Champion** at validation AP
0.35535042. The controlled LightGBM graph-enhanced C arm reached validation AP
0.47175420, +0.11640378 over the same-model B arm. Sprint 4 then evaluated a
transaction-edge GraphSAGE classifier at validation AP 0.00943876 on the same full
761,749-row validation partition. It did not outperform either frozen LightGBM
reference.

All 19 Sprint 4 acceptance gates, artifact verification, and 289 tests passed.
These are experiment results on synthetic data, not production model approval.
Sprint 5 subsequently consumed the frozen final test exactly once. The
graph-enhanced LightGBM specification had already been selected on validation and
remained unchanged; saved predictions were independently re-evaluated and no
post-test tuning or retraining occurred.

## Intended use

The models rank transactions as candidates for trained human review. The
transaction-only model provides a reference for controlled graph-value experiments;
the product layer turns saved scores and graph facts into reviewable cases. A score
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

## Sprint 4 GraphSAGE and product-layer results

GraphSAGE uses the supplied transaction label for edge classification. Sender and
receiver node embeddings are combined with transaction features; no account-level
fraud label is derived. The frozen Sprint 3 LightGBM scores are reference artifacts,
not retrained comparators.

Full-graph training was not represented as feasible on the available workstation.
The deterministic run used 150,000 of 1,422,288 eligible train-prefix edges for the
training message graph, then supervised on 52,396 later train transactions: all
2,396 eligible positives and 50,000 deterministic negatives. Validation inference
used a 300,000-of-3,554,957 sampled outer-train message graph and scored all 761,749
validation transactions. No validation edge entered message passing.

| Validation model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Graph-enhanced LightGBM** | **0.47175420** | **0.98635944** | **0.11631632** | **0.76447368** | **0.20191138** | **0.00580035** | **4,995** |
| Refined transaction LightGBM | 0.35535042 | 0.98179242 | 0.10302170 | 0.66842105 | 0.17852750 | 0.00581217 | 4,931 |
| GraphSAGE edge classifier | 0.00943876 | 0.84664170 | 0.00969110 | 0.06315789 | 0.01680378 | 0.00644556 | 4,953 |

All three rows use the identical frozen validation transactions and the same
validation-only 5,000-alert/1% FPR operating rule. GraphSAGE's substantially lower
AP is negative experimental evidence for this sampled architecture and training
scope; it is not evidence that graph neural networks are categorically ineffective.
The graph-enhanced LightGBM remains the validation leader.

Twenty investigation cases were generated from real saved validation model and
graph output. Each contains at least seven observed evidence items, stored
separately from model evidence. LightGBM explanations use native TreeSHAP with
verified additivity. GNN explanations are explicitly labeled local
gradient-times-input sensitivity, not SHAP or causal attribution. All cases have a
deterministic no-LLM note, and the Streamlit application loads saved artifacts only
for Executive Dashboard, Investigation Queue, Case Investigator, and Model
Comparison. Full evidence is in
[`SPRINT_4_STATUS.md`](../reports/generated/SPRINT_4_STATUS.md).

GraphSAGE node structure uses directed in/out degree and deterministic composite-
identity inputs. Monetary node aggregates are excluded because a versioned FX
conversion source is unavailable; transaction amounts remain edge features. Raw
logits drive ordering, while displayed sigmoid values are uncalibrated ranking
scores rather than event probabilities because training uses sampled negatives and
positive weighting.

## Final-test policy

Final-test inference was **not performed** in Sprint 2, Sprint 3, or Sprint 4. No
sprint used test rows for preprocessing fit, training, graph construction, model
selection, or threshold work. Sprint 5 then performed the separately authorized
single confirmatory evaluation on all 761,639 exact frozen test rows. The access
receipt was created before the first test query; the champion, feature family,
hyperparameters, train-fitted preprocessors, and validation-selected raw-score
thresholds were fixed before access. Test results did not revise the specification,
and no additional final-test inference run is permitted.

## Limitations

- IBM AML-Data is synthetic and does not establish production-bank performance.
- Severe class imbalance makes accuracy uninformative and precision highly
  sensitive to prevalence and operational capacity.
- Temporal prevalence shift is already visible across the frozen partitions.
- Sprint 2 candidate configurations were fixed baselines, not tuned estimators; its
  linear model retained a convergence warning and its `0.5` threshold was not
  optimized.
- Sprint 3 results are validation evidence after repeated development comparisons;
  the separate Sprint 5 block above is the one-shot final-test estimate.
- Sprint 4 GraphSAGE training uses explicitly disclosed deterministic graph and
  supervised samples, not all eligible training edges; only validation scoring is
  full-partition.
- GraphSAGE underperformed both frozen LightGBM references in this experiment. The
  result applies to this sampled architecture and protocol, not every possible GNN.
- No fairness, subgroup, calibration, robustness, or deployment validation is
  provided by this scope.
- Case evidence is generated from synthetic transactions and model outputs; it is
  not independently verified real-world intelligence.
- Validation metrics measure synthetic-label ranking/classification behavior and
  are not causal explanations or evidence of wrongdoing.
- Final metrics come from one synthetic chronological holdout and are not external,
  prospective, fairness, calibration, or production validation.
