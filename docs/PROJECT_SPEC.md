# ARGUS AI — Final Project Specification

| Field | Value |
| --- | --- |
| Project status | Sprints 1–5 complete |
| Primary model | Graph-enhanced LightGBM |
| Final evaluation | Frozen one-shot test completed |
| Scientific reports | [Final evaluation](../reports/generated/FINAL_EVALUATION_STATUS.md) and [model comparison](../reports/generated/FINAL_MODEL_COMPARISON.md) |

## 1. Purpose

ARGUS is a human-in-the-loop financial-crime research prototype. It represents
transfers as directed account-network edges, combines each transaction with
strictly prior behavioural and graph context, ranks suspicious transactions, and
presents the evidence behind each case to a financial-crime analyst.

ARGUS is decision support. A model score is not proof of laundering or guilt and
cannot justify automatic blocking, freezing, reporting, or another adverse action.

## 2. Data

The project uses the synthetic IBM AML-Data HI-Small dataset. Raw CSV files remain
local and are excluded from version control.

```text
data/raw/HI-Small_Trans.csv
data/raw/HI-Small_accounts.csv
```

Verified full-data properties:

| Property | Value |
| --- | ---: |
| Transactions | 5,078,345 |
| Account records | 518,581 |
| Positive transactions | 5,177 |
| Overall positive rate | 0.101943% |
| Time range | 2022-09-01 00:00 to 2022-09-18 16:18 |

Transaction bank identifiers contain leading zeroes while the account file uses
unpadded identifiers. A globally stable node is therefore defined as:

```text
normalized_bank_id::UPPERCASE_ACCOUNT_ID
```

Bank identifiers are validated as decimal text before zero removal. Account number
alone is not treated as a globally unique key. Direction and repeated transfers
are retained, so each transaction remains a sender-to-receiver edge in a directed
multigraph.

## 3. Chronological evaluation design

Rows are sorted deterministically by timestamp and stable source identity.
Timestamp groups are never split across partitions.

| Partition | Time range | Rows | Positives | Positive rate |
| --- | --- | ---: | ---: | ---: |
| Train | 2022-09-01 00:00 – 2022-09-07 14:55 | 3,554,957 | 2,856 | 0.0803385% |
| Validation | 2022-09-07 14:56 – 2022-09-09 03:16 | 761,749 | 760 | 0.0997704% |
| Test | 2022-09-09 03:17 – 2022-09-18 16:18 | 761,639 | 1,561 | 0.2049527% |

The protocol enforces these boundaries:

- `max(train) < min(validation) < min(test)`;
- learned preprocessing state is fitted on training rows only;
- validation and test use transform-only state;
- history for a transaction at time `t` includes only events with timestamp `< t`;
- transactions sharing `t` do not influence one another;
- the target, partition, timestamps, row identities, transaction IDs, and account
  identifiers are excluded from predictors;
- model family, feature family, hyperparameters, and threshold are selected without
  test feedback;
- the final test partition is evaluated once after the complete specification is
  frozen, with no post-test tuning or retraining.

The primary metric is PR-AUC/average precision because positives are extremely
rare. ROC-AUC is secondary. Precision, recall, F1, FPR, alert volume, Recall@K,
and Precision@K describe operational trade-offs. Accuracy is not a primary metric.

## 4. Feature system

### Transaction and amount fields

The model receives paid and received amounts, log amounts, payment format, bank
relationship, and currency fields. Amount differences and ratios are computed only
when payment and receiving currencies match. Cross-currency monetary comparisons
remain null because no versioned FX source is part of the study.

### Temporal and history fields

Calendar features include hour, day of week, and weekend status. Strictly prior
history captures sender and receiver transaction counts, elapsed time, cumulative
amounts, unique counterparties, and rolling 1-hour, 24-hour, and 7-day activity.

### Directed graph-history fields

Five interpretable graph features are computed from earlier transactions:

- `sender_prior_fan_out_degree`;
- `sender_prior_fan_in_degree`;
- `receiver_prior_fan_out_degree`;
- `receiver_prior_fan_in_degree`;
- `pair_previous_transfer_count`.

On the frozen dataset, `sender_prior_fan_out_degree` equals
`sender_previous_unique_counterparties`, and `receiver_prior_fan_in_degree` equals
`receiver_previous_unique_counterparties`. The main ablation retains all five
fields, while a separate sensitivity arm removes these two duplicate predictors.

### Model matrices

| Family | Contents | Transformed predictors |
| --- | --- | ---: |
| A | Transaction only | 49 |
| B | Transaction + temporal/history | 74 |
| C | Transaction + temporal/history + five graph fields | 79 |
| C sensitivity | Family B + three non-duplicate graph fields | 77 |

Numeric medians, means, population standard deviations, category vocabularies, and
bank-frequency encodings are learned from training rows only. Unseen categorical
values map to the defined unknown representation without expanding the fitted
feature space.

## 5. Model development

### Transaction baselines

Logistic Regression, Random Forest, and LightGBM establish transaction-level
references on the same chronological split. The immutable baseline snapshot chose
Random Forest by validation PR-AUC 0.08859105.

The initial Logistic Regression stopped at its 20-iteration limit, and both its
probabilities and the initial LightGBM probabilities contained large exact-one
plateaus. Refinement replaced the linear solver and regularization settings,
preserved raw decision margins for ranking, and measured probability ties directly.
The diagnosis showed that sigmoid/link conversion collapsed distinguishable
extreme margins; serialization was not the cause. Refined LightGBM had no exact
zero/one probability rows.

### Temporal refinement and threshold selection

Candidate selection uses three expanding-window folds contained within the outer
training partition. Each fold fits its own preprocessing state on its earlier
prefix. Selected candidates are refitted on all training rows, then compared on
outer validation.

Threshold analysis operates only on validation raw scores. The primary rule
maximizes recall subject to at most 5,000 alerts and FPR at most 0.01, and includes
complete equal-score groups. Ranking ties use score descending and stable source
row ascending; any K that cuts a tie is reported with tie-aware bounds.

### Controlled graph-value experiment

The A/B/C ablation holds the LightGBM candidate, hyperparameters, seed, split, and
evaluation protocol constant.

| Feature family | Validation PR-AUC | Validation ROC-AUC |
| --- | ---: | ---: |
| A — Transaction only | 0.09984124 | 0.96441199 |
| B — Transaction + temporal/history | 0.35535042 | 0.98179242 |
| C — B + five graph fields | **0.47175420** | **0.98635944** |
| C sensitivity — B + three non-duplicate graph fields | **0.47175420** | **0.98635944** |

Family C improves validation PR-AUC by 0.11640378 over Family B. This establishes
added value for the strict-prior graph feature family in this dataset and protocol.

The frozen primary LightGBM configuration uses 300 estimators, learning rate 0.04,
maximum depth 8, 31 leaves, minimum 100 child samples, L1 regularization 1.0, L2
regularization 20.0, deterministic column-wise training, and seed 42. Its raw-score
threshold is -4.3019702136515985, selected on validation before test evaluation.

## 6. GraphSAGE research comparison

GraphSAGE predicts the recorded IBM transaction/edge target. Sender and receiver node
embeddings are concatenated with transaction features; no unsupported account-level
fraud label is created.

Full-graph training exceeded the available compute budget. The deterministic
training design used:

- 150,000 of 1,422,288 eligible context edges;
- 52,396 supervised transactions: all 2,396 eligible positives and 50,000
  deterministic negatives;
- 300,000 of 3,554,957 outer-training edges for the validation inference graph;
- all 761,749 frozen validation transactions for scoring.

No validation edge entered message passing. Cross-currency node-level monetary
totals were excluded. Raw logits determine ranking; displayed sigmoid values are
uncalibrated because negative sampling and positive weighting change the fitted
class prior.

| Validation model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Graph-enhanced LightGBM** | **0.47175420** | **0.98635944** | **0.11631632** | **0.76447368** | **0.20191138** | **0.00580035** | **4,995** |
| Refined transaction LightGBM | 0.35535042 | 0.98179242 | 0.10302170 | 0.66842105 | 0.17852750 | 0.00581217 | 4,931 |
| GraphSAGE edge classifier | 0.00943876 | 0.84664170 | 0.00969110 | 0.06315789 | 0.01680378 | 0.00644556 | 4,953 |

The sampled GraphSAGE architecture did not outperform either LightGBM reference.
It remains a research comparator and is not presented as a full-graph result or as
evidence about every possible graph neural network.

## 7. Final evaluation

Graph-enhanced LightGBM was selected by validation PR-AUC and frozen before test
access. The final comparison used the validation-selected raw-score thresholds for
all three models.

| Role | Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen primary | Graph-enhanced LightGBM | **0.69005906** | 0.99127213 | 0.17816018 | 0.88212684 | 0.29644779 | 0.00835704 | 7,729 |
| Comparator | Refined transaction LightGBM | 0.53079643 | 0.98731443 | 0.16072332 | 0.81422165 | 0.26845496 | 0.00873200 | 7,908 |
| Research comparator | GraphSAGE edge classifier | 0.01341710 | 0.82662724 | 0.00600590 | 0.11338885 | 0.01140758 | 0.03854078 | 29,471 |

Validation prevalence was 0.099770%, while final-test prevalence was 0.204953%, a
2.054-fold increase. Precision and fixed-threshold alert volume are sensitive to
this chronological shift. The split was not resampled, the threshold was not
adjusted, and the primary model remained unchanged after the test result.

## 8. Product layer and explanations

The Streamlit application reads saved outputs and performs no model training at
page load. Its analyst workflow covers a portfolio overview, prioritized
investigations, transaction-level case inspection, directed account-network
visualization, and model evidence.

Cases are built from saved transaction, graph, and model outputs. Observed facts
remain separate from model contributions. Each generated case contains multiple
concrete transaction/network evidence items. LightGBM explanations use TreeSHAP
with an additivity check. GraphSAGE explanations are labeled local
gradient-times-input sensitivity and are not described as SHAP or causal effects.
Case notes have a deterministic evidence-only fallback and do not require an
external language-model service.

## 9. Configuration and reproducibility

Experiment settings are versioned in:

```text
configs/base.yaml
configs/quick.yaml
configs/full.yaml
configs/baseline.yaml
configs/refinement.yaml
configs/sprint4.yaml
configs/final_evaluation.yaml
```

The standard quality and data-foundation commands are:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[graph,app,dev]"
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\python.exe -m ruff check --no-cache src scripts tests app.py
.\.venv\Scripts\python.exe -m ruff format --no-cache --check src scripts tests app.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/run_quick_pipeline.py
.\.venv\Scripts\python.exe scripts/verify_run.py
.\.venv\Scripts\streamlit.exe run app.py
```

Full model-development commands and the scientific freeze rules are documented in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). The one-shot final scoring command
is not part of the routine reproduction sequence; persisted final artifacts may be
checked without reopening raw test data.

Generated outputs remain local and Git-ignored. Key artifact roots are:

```text
artifacts/quick/
artifacts/full/
artifacts/sprint2/
artifacts/sprint3/
artifacts/sprint4/
artifacts/sprint5/
```

The final machine-readable evidence includes `artifacts/sprint5/final_metrics.json`,
`final_model_comparison.json`, `final_top_k_metrics.json`,
`prevalence_shift.json`, `test_identity_audit.json`, `run_manifest.json`, and
`verification_report.json`. Repository-tracked summaries are available in the
[final evaluation report](../reports/generated/FINAL_EVALUATION_STATUS.md) and
[final comparison table](../reports/generated/FINAL_MODEL_COMPARISON.md).

## 10. Responsible-use boundaries and limitations

- HI-Small is synthetic; results do not establish performance at a production
  financial institution.
- The positive class is extremely imbalanced, and precision varies with prevalence
  and alert capacity.
- The final metrics come from one chronological synthetic holdout, not an external
  or prospective validation.
- GraphSAGE training used disclosed deterministic samples rather than all eligible
  graph edges.
- Scores are ranking signals, not calibrated probabilities unless explicitly
  stated.
- No fairness, subgroup, calibration, robustness, or deployment validation is
  included.
- Model explanations describe model behaviour; they are not causal evidence.
- Case evidence comes from synthetic transactions and model outputs and must not be
  treated as real-world intelligence.
- Any operational decision requires trained human analysis and source-record
  verification.
