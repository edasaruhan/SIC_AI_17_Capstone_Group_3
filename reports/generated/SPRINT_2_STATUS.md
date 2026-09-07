# ARGUS AI — Sprint 2 Validation Status

This report is generated from executable Sprint 2 artifacts. It contains no manually entered model metric.

## Outcome

- Transaction Baseline Champion: `random_forest`
- Selection evidence: validation average precision only
- Final test model inference/evaluation: **NOT USED**
- Threshold: fixed configuration value; not optimized in Sprint 2
- Accuracy: intentionally not used as a primary metric
- PR-curve figure: deterministic endpoint-preserving display decimation to at most 10,000 points/model; numerical metrics use all rows

## Validation comparison

| Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_forest | 0.08859105 | 0.97497347 | 0.03127670 | 0.72236842 | 0.05995741 | 0.02234461 | 17,553 |
| logistic_regression | 0.00663424 | 0.91808364 | 0.00491412 | 0.96184211 | 0.00977828 | 0.19451530 | 148,755 |
| lightgbm | 0.00547556 | 0.85107181 | 0.00507521 | 0.87368421 | 0.01009180 | 0.17105109 | 130,832 |

## Validation Top-K

| Model | Precision@100 | Recall@100 | Precision@500 | Recall@500 | Precision@1000 | Recall@1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgbm | 0.57000000 | 0.07500000 | 0.58200000 | 0.38289474 | 0.32100000 | 0.42236842 |
| logistic_regression | 0.85000000 | 0.11184211 | 0.36800000 | 0.24210526 | 0.18600000 | 0.24473684 |
| random_forest | 0.34000000 | 0.04473684 | 0.17200000 | 0.11315789 | 0.13000000 | 0.17105263 |

### Top-K cutoff tie diagnostics

Rows are ranked by descending score and then ascending immutable `source_row_number`. When a cutoff intersects an equal-score group, Precision@K and Recall@K are deterministic but depend on that declared secondary ordering; they do not measure discrimination within the tied group.

| Model | K | Cutoff score | Tied rows at cutoff | Rows selected from tie |
| --- | ---: | ---: | ---: | ---: |
| lightgbm | 100 | 1.000000000000 | 108,347 | 100 |
| lightgbm | 500 | 1.000000000000 | 108,347 | 500 |
| lightgbm | 1000 | 1.000000000000 | 108,347 | 1,000 |
| logistic_regression | 100 | 1.000000000000 | 60,119 | 100 |
| logistic_regression | 500 | 1.000000000000 | 60,119 | 500 |
| logistic_regression | 1000 | 1.000000000000 | 60,119 | 1,000 |

## Frozen chronological prevalence

| Partition | Rows | Positives | Positive rate | Timestamp range |
| --- | ---: | ---: | ---: | --- |
| train | 3,554,957 | 2,856 | 0.080338524% | `2022-09-01T00:00:00`–`2022-09-07T14:55:00` |
| validation | 761,749 | 760 | 0.099770397% | `2022-09-07T14:56:00`–`2022-09-09T03:16:00` |
| test | 761,639 | 1,561 | 0.204952740% | `2022-09-09T03:17:00`–`2022-09-18T16:18:00` |

Validation prevalence differs from train prevalence, and test prevalence is higher again. Precision and threshold alert volume are prevalence-sensitive; therefore validation precision/alert volume must not be assumed to transfer unchanged to the later test period. The frozen splits were not rebalanced.

The test row count, timestamp range, and positive count above are pre-existing Sprint 1 split metadata. Sprint 2 did not load test feature rows, labels, or predictions.

## Runtime

- `total_core_pipeline`: 389.008 seconds
- `duckdb_initialization`: 0.024 seconds
- `frozen_input_verification`: 1.129 seconds
- `source_snapshot`: 0.155 seconds
- `three_baseline_fit_and_validation`: 346.501 seconds
- `train_matrix_materialization`: 16.058 seconds
- `train_only_preprocessor_fit`: 19.574 seconds
- `validation_comparison_and_champion`: 0.912 seconds
- `validation_matrix_materialization`: 3.495 seconds
- `validation_prediction_export`: 0.760 seconds
- `work_matrix_cleanup`: 0.371 seconds

## Artifacts

- `artifacts/sprint2/run_manifest.json`
- `artifacts/sprint2/model_comparison.json`
- `artifacts/sprint2/model_comparison.csv`
- `artifacts/sprint2/transaction_baseline_champion.json`
- `artifacts/sprint2/validation_predictions.parquet`
- `artifacts/sprint2/preprocessing/manifest.json`
- `artifacts/sprint2/final_test_policy.json`
- `artifacts/sprint2/figures/metric_comparison.png`
- `artifacts/sprint2/figures/validation_precision_recall_curves.png`
- `reports/generated/SPRINT_2_STATUS.md`
- `artifacts/sprint2/verification_report.json`
- `artifacts/sprint2/quality_report.json`

## Acceptance checklist

- PASS — Logistic Regression baseline executed on frozen train/validation data.
- PASS — Random Forest baseline executed on the identical matrix and partitions.
- PASS — LightGBM baseline executed on the identical matrix and partitions.
- PASS — Comparison uses PR-AUC (non-interpolated average precision) as primary.
- PASS — ROC-AUC, fixed-threshold metrics, FPR, alert volume, and top-K metrics saved.
- PASS — Champion selected exclusively from validation average precision.
- PASS — Final test was not used for fitting, tuning, selection, or inference.
- PASS — Graph-history features were excluded from transaction baselines.
- PASS — Learned preprocessing state was fit from train only.
- NOT IN SPRINT 2 — Hyperparameter tuning/refinement and graph-value experiment.
- PASS — 81 tests passed in 19.87s; Ruff lint and format, pip check, and independent saved-artifact verification passed.
