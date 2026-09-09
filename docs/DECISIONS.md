# ARGUS Scientific and Engineering Decision Log

This log records the methodological choices that define the final ARGUS study.
They apply to data semantics, feature construction, model development, evaluation,
and analyst-facing interpretation.

## 1. Preserve chronology in every split

Transactions are partitioned chronologically, and every group of transactions with
the same timestamp remains in a single partition. This avoids future-to-past
leakage and removes arbitrary row order from split boundaries.

| Partition | Time interval | Rows | Positives | Positive rate |
| --- | --- | ---: | ---: | ---: |
| Train | 2022-09-01 00:00–2022-09-07 14:55 | 3,554,957 | 2,856 | 0.000803385 |
| Validation | 2022-09-07 14:56–2022-09-09 03:16 | 761,749 | 760 | 0.000997704 |
| Test | 2022-09-09 03:17–2022-09-18 16:18 | 761,639 | 1,561 | 0.002049527 |

## 2. Construct historical features from strictly prior events

For a transaction at time `t`, history and graph-history features use only events
with timestamps earlier than `t`. Transactions sharing a timestamp are calculated
as a batch and update state only after their features have been emitted. Future-row
append invariance and timestamp-tie tests enforce this rule.

## 3. Identify accounts by bank and account number

An account is represented by a normalized composite key:
`normalized_bank_id::UPPERCASE_ACCOUNT_ID`. Account numbers are not globally
unique, and bank identifiers appear with inconsistent zero padding in the raw
tables. Composite normalization produced zero unmatched transaction endpoints in
the full source audit; eight account-number strings each mapped to two distinct
bank-account identities.

## 4. Preserve directed multigraph semantics

Transfers are directed sender-to-receiver edges. Repeated transfers remain separate
edges because frequency and direction carry behavioral information. Exact duplicate
rows are reported and retained unless a separately declared deduplication experiment
is performed. Reducing the network to an undirected simple graph would discard
fan-in, fan-out, and repeated-pair signals.

## 5. Keep labels and monetary comparisons well defined

The IBM AML label belongs to a transaction, so ARGUS predicts transactions or graph
edges. It does not derive unsupported account-level fraud labels. Amount differences
and ratios are calculated only when both amounts use the same currency; cross-currency
comparisons remain null without a versioned foreign-exchange source. For the same
reason, GraphSAGE node features exclude monetary totals that would mix currencies.

## 6. Fit transformations and select models without test feedback

Numeric statistics, category vocabularies, and frequency mappings are fitted on
training data only. Validation and test data use the frozen transformation state,
including explicit fallbacks for unseen categories. Predictor allow-lists exclude
targets, identifiers, and provenance fields.

Hyperparameter selection used three expanding temporal folds contained entirely
within the outer training partition. Selected candidates were refitted on outer
train and compared on outer validation. Model family, feature set, and operating
threshold were frozen before final-test evaluation.

## 7. Evaluate ranking under severe class imbalance

Non-interpolated average precision (PR-AUC) is the primary selection metric;
ROC-AUC is secondary. Precision, recall, F1, false-positive rate, alert volume, and
Recall/Precision@K describe operating behavior. Accuracy is not used for selection
because positive rates range from 0.0803% in train to 0.2050% in test.

The frozen graph-enhanced LightGBM threshold was selected on validation by maximizing
recall subject to at most 5,000 alerts and an FPR ceiling of 1%. The chosen raw-score
threshold, `-4.3019702136515985`, produced 4,995 validation alerts, 0.764474 recall,
0.116316 precision, and 0.005800 FPR. Equal-score groups are admitted together rather
than split arbitrarily at the threshold.

## 8. Diagnose score saturation before interpreting Top-K results

The first logistic baseline reached its 20-iteration limit without convergence.
It also produced 60,119 validation probabilities equal to 1.0; the first LightGBM
baseline produced 108,347. The immediate tie mechanism was sigmoid conversion of
extreme raw margins, with weighting and weak regularization contributing to the
margin scale.

The refined logistic model uses `newton-cholesky`, L2 regularization with `C=0.01`,
and `max_iter=100`; it converged in seven iterations. Refined LightGBM produced no
exact-zero or exact-one validation probabilities. The refined logistic probability
output still contained 135 exact-one values, while its decision margins preserved
more ranking detail. Ranking therefore uses raw margins, and any Top-K cutoff that
intersects a tie is reported with deterministic and tie-aware bounds.

## 9. Measure graph value with a controlled feature ablation

The graph-feature experiment holds the LightGBM candidate, parameters, seed,
chronological split, and evaluation protocol constant.

| Feature family | Features | Validation PR-AUC | Validation ROC-AUC |
| --- | ---: | ---: | ---: |
| Transaction only | 49 | 0.099841 | 0.964412 |
| Transaction + temporal/history | 74 | 0.355350 | 0.981792 |
| Transaction + temporal/history + graph | 79 | 0.471754 | 0.986359 |

Adding the graph family increased validation PR-AUC by `0.116404` relative to the
temporal/history model. Two of the five graph columns duplicate existing unique-
counterparty history fields on this dataset. A sensitivity run using only the three
non-duplicate additions—sender prior fan-in, receiver prior fan-out, and prior
sender-receiver transfer count—also achieved `0.471754` PR-AUC.

## 10. Treat GraphSAGE as a bounded edge-classification experiment

GraphSAGE combines sender and receiver node embeddings with transaction features to
classify edges. Training used a deterministic subset: 150,000 of 1,422,288 eligible
message-graph edges, 52,396 supervised edges (all 2,396 eligible positives and
50,000 deterministic negatives), and a 300,000-edge outer-train inference graph.
Evaluation still covered all 761,749 validation transactions.

GraphSAGE validation PR-AUC was `0.009439`, compared with `0.355350` for refined
transaction LightGBM and `0.471754` for graph-enhanced LightGBM. This is a bounded
research comparison, not a full-graph result and not evidence of GNN superiority.

## 11. Bind explanations to evidence and retain human review

LightGBM explanations use native TreeSHAP with an additivity check. GraphSAGE uses
gradient-times-input local sensitivity and labels it accordingly rather than calling
it SHAP. Observed transaction/network facts remain separate from model contributions.
Each saved case contains at least three observed facts; the 20 generated validation
cases contain at least seven each. A deterministic evidence-to-note fallback keeps
case generation reproducible without an external language model.

Outputs indicate investigation priority, not guilt. Final decisions remain with a
trained analyst, and model scores are not proof of criminal activity.

## 12. Evaluate the frozen champion once on the final test period

Graph-enhanced LightGBM was selected on validation and frozen before test scoring.
The final test partition was evaluated once; no model, feature, hyperparameter, or
threshold changes followed the result.

| Model | Role | Test PR-AUC |
| --- | --- | ---: |
| Graph-enhanced LightGBM | Frozen champion | 0.690059 |
| Refined transaction LightGBM | Comparator | 0.530796 |
| GraphSAGE edge classifier | Research comparator | 0.013417 |

For the frozen champion, test ROC-AUC was `0.991272`, precision `0.178160`, recall
`0.882127`, F1 `0.296448`, FPR `0.008357`, and alert volume `7,729`. The positive
rate increased from `0.000997704` in validation to `0.002049527` in test, a ratio of
`2.054244`. Precision and alert volume are prevalence-sensitive, so their change
between periods is interpreted alongside this shift rather than as an isolated
model-quality effect.
