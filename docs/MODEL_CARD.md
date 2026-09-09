# Model Card — Graph-Enhanced LightGBM

## Model summary

| Field | Value |
| --- | --- |
| Model key | `graph_enhanced_lightgbm` |
| Role | Frozen primary transaction-ranking model |
| Model family | LightGBM gradient-boosted decision trees |
| Prediction unit | Directed financial transaction/edge |
| Primary metric | PR-AUC / average precision |
| Selection partition | Chronological validation partition |
| Final evaluation | One-shot chronological test partition |
| Scientific reference | `v1.0-scientific-final` (`ce182474cd8ee36bcab2449a61347eb9803451ae`) |

ARGUS ranks transactions for analyst investigation by combining transaction fields,
strictly prior temporal/account history, and directed graph-history features. The
graph-enhanced LightGBM model is the primary model because it achieved the highest
validation PR-AUC under the frozen comparison protocol. The sampled GraphSAGE
edge classifier is retained only as a research comparator.

## Intended use

- prioritize suspicious transactions and connected account activity for trained
  financial-crime analysts;
- compare transaction-only, temporal/history, and graph-enhanced feature families;
- support case investigation with observed transaction/network facts and model
  contribution evidence;
- reproduce a leakage-safe experiment on the synthetic IBM AML HI-Small dataset.

The model is not designed to establish guilt, identify a person as a money
launderer, or trigger automatic blocking, freezing, regulatory reporting, or
another adverse action. Every flagged case requires source-record checks and human
analysis.

## Data

The study uses the synthetic IBM AML-Data HI-Small dataset: 5,078,345 transactions,
518,581 account records, and 5,177 positive transaction labels. The target is
transaction-level; no account-level fraud label is inferred.

| Partition | Time range | Rows | Positives | Positive rate | Role |
| --- | --- | ---: | ---: | ---: | --- |
| Train | 2022-09-01 00:00 – 2022-09-07 14:55 | 3,554,957 | 2,856 | 0.0803385% | Preprocessing and model fit |
| Validation | 2022-09-07 14:56 – 2022-09-09 03:16 | 761,749 | 760 | 0.0997704% | Model and threshold selection |
| Test | 2022-09-09 03:17 – 2022-09-18 16:18 | 761,639 | 1,561 | 0.2049527% | One-shot confirmatory evaluation |

Partitions are chronological, timestamp groups remain intact, and the rates are
not rebalanced. Test prevalence is 2.05424401 times validation prevalence.
Precision and fixed-threshold alert volume are therefore interpreted alongside
this temporal shift.

## Inputs and preprocessing

The primary model receives 79 transformed predictors:

- current transaction amount, currency, payment-format, and bank-relationship
  fields;
- calendar and elapsed-time fields;
- strictly prior sender and receiver activity, amount, counterparty, and rolling
  window history;
- five directed graph-history fields: sender prior fan-out/fan-in, receiver prior
  fan-out/fan-in, and prior repeated-pair transfer count;
- explicit missing-value indicators and train-fitted categorical/bank encodings.

Numeric medians, means, population standard deviations, categorical vocabularies,
and bank frequencies are fitted from training rows only. Validation and test rows
are transform-only. The target, partition, timestamps, transaction/source
identities, and account/node identifiers are excluded from predictors.

Historical state for a transaction at time `t` uses only events before `t`.
Same-timestamp peers cannot influence one another. Amount differences and ratios
are defined only for matching currencies; no exchange rate is imputed.

## Model specification

The frozen LightGBM configuration is:

| Parameter | Value |
| --- | ---: |
| Boosting | `gbdt` |
| Estimators | 300 |
| Learning rate | 0.04 |
| Maximum depth | 8 |
| Leaves | 31 |
| Minimum child samples | 100 |
| Minimum child weight | 1.0 |
| L1 regularization | 1.0 |
| L2 regularization | 20.0 |
| Class weight / `scale_pos_weight` | none / 1.0 |
| Seed | 42 |

Training is deterministic and column-wise. Raw LightGBM margins provide the
ranking score. The selected raw-score threshold is
`-4.3019702136515985`, chosen on validation by maximizing recall subject to at
most 5,000 alerts and FPR at most 0.01. Complete equal-score groups are included.
The output is not presented as a calibrated event probability.

## Development evidence

### Transaction baselines

The first baseline comparison used Logistic Regression, Random Forest, and
LightGBM. Random Forest led that immutable baseline snapshot at validation PR-AUC
0.08859105. The initial linear model reached its 20-iteration limit, and initial
linear/boosting probabilities contained large exact-one plateaus.

Refinement used expanding-window temporal cross-validation, stronger
regularization, corrected linear-model convergence, and raw decision scores for
ranking. Diagnostics confirmed that the earlier 0/1 probability plateaus resulted
from extreme raw margins passing through the link transformation, not from model
serialization. The refined graph-enhanced LightGBM produced no exact-zero or
exact-one probabilities on validation.

### Feature-family ablation

The same LightGBM candidate, parameters, seed, chronological split, and evaluation
protocol were used for every arm.

| Feature family | Predictors | Validation PR-AUC | Validation ROC-AUC |
| --- | ---: | ---: | ---: |
| Transaction only | 49 | 0.09984124 | 0.96441199 |
| Transaction + temporal/history | 74 | 0.35535042 | 0.98179242 |
| Transaction + temporal/history + graph | 79 | **0.47175420** | **0.98635944** |
| Duplicate-removed graph sensitivity | 77 | **0.47175420** | **0.98635944** |

Graph-history fields increased validation PR-AUC by 0.11640378 over the same-model
temporal/history arm. Two graph columns are exact aliases of existing history
columns on the frozen data; the three non-duplicate graph fields reproduced the
same result.

### Validation comparison

All rows use the full 761,749-transaction validation partition and the same
validation-only alert-budget/FPR operating rule.

| Model | Role | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Graph-enhanced LightGBM** | Primary | **0.47175420** | **0.98635944** | **0.11631632** | **0.76447368** | **0.20191138** | **0.00580035** | **4,995** |
| Refined transaction LightGBM | Comparator | 0.35535042 | 0.98179242 | 0.10302170 | 0.66842105 | 0.17852750 | 0.00581217 | 4,931 |
| GraphSAGE edge classifier | Research comparator | 0.00943876 | 0.84664170 | 0.00969110 | 0.06315789 | 0.01680378 | 0.00644556 | 4,953 |

## GraphSAGE comparator

GraphSAGE combines sender and receiver embeddings with transaction features and
predicts transaction edges. Full-graph training exceeded the available compute
budget, so the recorded experiment used deterministic bounded samples:

- 150,000 of 1,422,288 eligible context edges;
- 52,396 supervised transactions, comprising all 2,396 eligible positives and
  50,000 deterministic negatives;
- 300,000 of 3,554,957 outer-training edges for validation message passing;
- complete scoring of all 761,749 validation transactions.

No validation edge entered the message-passing graph. Raw logits determine ranking;
sigmoid values are uncalibrated because class weighting and negative sampling alter
the fitted prior. The substantially lower GraphSAGE PR-AUC is evidence about this
sampled architecture and training protocol, not graph neural networks in general.

## One-shot final-test results

Graph-enhanced LightGBM, its 79-feature contract, hyperparameters, preprocessing
state, and threshold were frozen before test access. Test metrics were confirmatory
and were not used for model, feature, parameter, or threshold selection. No model
was retrained or tuned after the result.

| Model | Role | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Graph-enhanced LightGBM** | Frozen primary | **0.69005906** | **0.99127213** | **0.17816018** | **0.88212684** | **0.29644779** | **0.00835704** | **7,729** |
| Refined transaction LightGBM | Comparator | 0.53079643 | 0.98731443 | 0.16072332 | 0.81422165 | 0.26845496 | 0.00873200 | 7,908 |
| GraphSAGE edge classifier | Research comparator | 0.01341710 | 0.82662724 | 0.00600590 | 0.11338885 | 0.01140758 | 0.03854078 | 29,471 |

The primary model produced 1,377 true positives, 6,352 false positives, 753,726
true negatives, and 184 false negatives at the frozen threshold.

### Primary-model Top-K results

| K | Precision@K | Recall@K | True positives |
| ---: | ---: | ---: | ---: |
| 100 | 0.98000000 | 0.06278027 | 98 |
| 500 | 0.96800000 | 0.31005766 | 484 |
| 1,000 | 0.83900000 | 0.53747598 | 839 |

Exact test membership was joined by frozen transaction identity rather than a time
range approximation. The audit found 761,639 unique matches, no missing feature
rows, no duplicate feature matches, and no timestamp mismatches. The saved
prediction vectors were independently re-evaluated without reopening raw test
data.

## Explanations and case presentation

The product layer builds cases from saved transaction, graph, and model outputs.
Observed facts remain separate from model contributions. LightGBM explanations use
native TreeSHAP and pass an additivity check. GraphSAGE explanations are explicitly
labeled local gradient-times-input sensitivity rather than SHAP or causal
attribution.

The Streamlit application reads saved artifacts and performs no training at page
load. It presents portfolio-level results, prioritized investigations, directed
account-network context, case evidence, and model comparison. Evidence-only case
notes have a deterministic fallback and do not require an external language-model
service.

## Limitations

- HI-Small is synthetic and does not establish production-bank performance.
- Extreme class imbalance makes accuracy uninformative and makes precision
  sensitive to prevalence and alert capacity.
- The final estimate comes from one chronological synthetic holdout, not external
  or prospective validation.
- The positive rate more than doubled between validation and test; fixed-threshold
  alarm counts and precision do not transfer unchanged across periods.
- GraphSAGE training used a deterministic subset, not every eligible graph edge.
- The primary model is not probability-calibrated.
- No fairness, subgroup, calibration, robustness, drift, or deployment validation
  is included.
- SHAP and GNN sensitivity describe model behaviour, not causal evidence.
- Synthetic case evidence is not real-world financial intelligence.

## Evidence and reproducibility

Repository-tracked summaries:

- [Final evaluation status](../reports/generated/FINAL_EVALUATION_STATUS.md)
- [Final model comparison](../reports/generated/FINAL_MODEL_COMPARISON.md)
- [Experiment protocol](EXPERIMENT_PROTOCOL.md)
- [Data dictionary](DATA_DICTIONARY.md)

Local machine-readable artifacts are generated under `artifacts/` and excluded
from version control:

```text
artifacts/sprint3/ablation/feature_family_ablation.csv
artifacts/sprint3/threshold/analysis.json
artifacts/sprint4/model_comparison.json
artifacts/sprint4/product/cases.json
artifacts/sprint5/final_metrics.json
artifacts/sprint5/final_model_comparison.json
artifacts/sprint5/final_top_k_metrics.json
artifacts/sprint5/prevalence_shift.json
artifacts/sprint5/test_identity_audit.json
artifacts/sprint5/run_manifest.json
artifacts/sprint5/verification_report.json
```

The frozen experiment rules and standard reproduction commands are documented in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). The one-shot final scoring command
is excluded from routine reproduction; persisted outputs can be verified without
reopening raw test data.
