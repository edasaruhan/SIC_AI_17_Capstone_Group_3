# ARGUS AI — Final Model Comparison

This table is generated from the persisted one-shot final-test artifacts. The model order follows pre-test role and name; it is not a test-driven ranking or selection.

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

The paths below are generated locally and intentionally excluded from Git. They become available after the frozen artifact package is restored locally.

- Final metrics: `../../artifacts/sprint5/final_metrics.json`
- Final comparison: `../../artifacts/sprint5/final_model_comparison.json`
- Final Top-K metrics: `../../artifacts/sprint5/final_top_k_metrics.json`
- Saved final predictions: `../../artifacts/sprint5/final_test_predictions.parquet`
- Prevalence analysis: `../../artifacts/sprint5/prevalence_shift.json`
- Test identity/leakage audit: `../../artifacts/sprint5/test_identity_audit.json`
- Immutable run manifest: `../../artifacts/sprint5/run_manifest.json`
- Read-only verification: `../../artifacts/sprint5/verification_report.json`
- Quality report: `../../artifacts/sprint5/quality_report.json`
- Streamlit — Executive Dashboard: `../../artifacts/sprint5/screenshots/executive_dashboard.png`
- Streamlit — Investigation Queue: `../../artifacts/sprint5/screenshots/investigation_queue.png`
- Streamlit — Case Investigator: `../../artifacts/sprint5/screenshots/case_investigator.png`
- Streamlit — Model Comparison: `../../artifacts/sprint5/screenshots/model_comparison.png`

## Frozen-run quality snapshot

- Status: **PASS**
- Pytest at scientific freeze: 377 passed
- Saved-artifact verification: PASS
