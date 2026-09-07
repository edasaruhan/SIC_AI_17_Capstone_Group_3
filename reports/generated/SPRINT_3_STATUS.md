# ARGUS AI - Sprint 3 Refinement and Graph-Value Status

This report is generated solely from executable, machine-readable Sprint 3 artifacts. No model result is entered or estimated by the report renderer.

## Outcome

- Sprint 3 status: **PASS**
- Refined Transaction Baseline Champion: `lightgbm`
- Champion/model selection partition: `validation`
- Primary selection metric: `average_precision_on_raw_ranking_score`
- Final test fitting, transformation, inference, tuning, and selection: **NOT USED**
- Sprint 4 / GraphSAGE: **NOT STARTED**
- Accuracy: intentionally not used as a primary metric
- Headline graph-value comparison: `B_transaction_temporal_history_vs_C_plus_all_graph`
- Headline graph-value AP delta: `0.11640378`
- Same-family/parameter protocol verified: `True`

## Baseline vs refined validation results

| Stage | Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sprint2_baseline_recomputed_raw_ranking | lightgbm | 0.01477564 | 0.89558547 | 0.00507521 | 0.87368421 | 0.01009180 | 0.17105109 | 130,832 |
| sprint2_baseline_recomputed_raw_ranking | logistic_regression | 0.00811875 | 0.92007851 | 0.00491412 | 0.96184211 | 0.00977828 | 0.19451530 | 148,755 |
| sprint2_baseline_recomputed_raw_ranking | random_forest | 0.08859105 | 0.97497347 | 0.03127670 | 0.72236842 | 0.05995741 | 0.02234461 | 17,553 |
| sprint3_refined_transaction_baseline | logistic_regression | 0.02600700 | 0.95570357 | 0.03339368 | 0.21842105 | 0.05793055 | 0.00631415 | 4,971 |
| sprint3_refined_transaction_baseline | random_forest | 0.12967475 | 0.97718091 | 0.08008008 | 0.52631579 | 0.13900956 | 0.00603820 | 4,995 |
| sprint3_refined_transaction_baseline | lightgbm | 0.35535042 | 0.98179242 | 0.10302170 | 0.66842105 | 0.17852750 | 0.00581217 | 4,931 |
| sprint3_graph_enhanced | lightgbm | 0.47175420 | 0.98635944 | 0.11631632 | 0.76447368 | 0.20191138 | 0.00580035 | 4,995 |

### Validation Top-K and tie bounds

| Stage | Model | K | Precision@K | Recall@K | TP deterministic | TP expected | TP min-max | Material cutoff tie |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sprint2_baseline_recomputed_raw_ranking | lightgbm | 100 | 0.00000000 | 0.00000000 | 0 | 0.000 | 0-0 | False |
| sprint2_baseline_recomputed_raw_ranking | lightgbm | 500 | 0.00400000 | 0.00263158 | 2 | 2.000 | 2-2 | False |
| sprint2_baseline_recomputed_raw_ranking | lightgbm | 1,000 | 0.02300000 | 0.03026316 | 23 | 23.000 | 23-23 | False |
| sprint2_baseline_recomputed_raw_ranking | logistic_regression | 100 | 0.00000000 | 0.00000000 | 0 | 0.000 | 0-0 | False |
| sprint2_baseline_recomputed_raw_ranking | logistic_regression | 500 | 0.00000000 | 0.00000000 | 0 | 0.000 | 0-0 | False |
| sprint2_baseline_recomputed_raw_ranking | logistic_regression | 1,000 | 0.00700000 | 0.00921053 | 7 | 7.000 | 7-7 | False |
| sprint2_baseline_recomputed_raw_ranking | random_forest | 100 | 0.34000000 | 0.04473684 | 34 | 34.000 | 34-34 | False |
| sprint2_baseline_recomputed_raw_ranking | random_forest | 500 | 0.17200000 | 0.11315789 | 86 | 86.000 | 86-86 | False |
| sprint2_baseline_recomputed_raw_ranking | random_forest | 1,000 | 0.13000000 | 0.17105263 | 130 | 130.000 | 130-130 | False |
| sprint3_refined_transaction_baseline | logistic_regression | 100 | 0.00000000 | 0.00000000 | 0 | 0.000 | 0-0 | False |
| sprint3_refined_transaction_baseline | logistic_regression | 500 | 0.04000000 | 0.02631579 | 20 | 20.000 | 20-20 | False |
| sprint3_refined_transaction_baseline | logistic_regression | 1,000 | 0.04700000 | 0.06184211 | 47 | 47.000 | 47-47 | False |
| sprint3_refined_transaction_baseline | random_forest | 100 | 0.41000000 | 0.05394737 | 41 | 41.000 | 41-41 | False |
| sprint3_refined_transaction_baseline | random_forest | 500 | 0.23000000 | 0.15131579 | 115 | 115.000 | 115-115 | False |
| sprint3_refined_transaction_baseline | random_forest | 1,000 | 0.16200000 | 0.21315789 | 162 | 162.000 | 162-162 | False |
| sprint3_refined_transaction_baseline | lightgbm | 100 | 0.93000000 | 0.12236842 | 93 | 93.000 | 93-93 | False |
| sprint3_refined_transaction_baseline | lightgbm | 500 | 0.48600000 | 0.31973684 | 243 | 243.000 | 243-243 | False |
| sprint3_refined_transaction_baseline | lightgbm | 1,000 | 0.30500000 | 0.40131579 | 305 | 305.000 | 305-305 | False |
| sprint3_graph_enhanced | lightgbm | 100 | 0.99000000 | 0.13026316 | 99 | 99.000 | 99-99 | False |
| sprint3_graph_enhanced | lightgbm | 500 | 0.59200000 | 0.38947368 | 296 | 296.000 | 296-296 | False |
| sprint3_graph_enhanced | lightgbm | 1,000 | 0.38100000 | 0.50131579 | 381 | 381.000 | 381-381 | False |

## Expanding-window temporal cross-validation

Temporal folds remain inside the frozen outer-training partition. Outer validation is not used to select hyperparameters, and the final test remains unopened.

### Fold boundaries

| Fold | Train timestamp range | Train rows | Train positives | Validation timestamp range | Validation rows | Validation positives |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 1 | `2022-09-01T00:00:00` - `2022-09-02T09:27:00` | 1,422,288 | 460 | `2022-09-02T09:28:00` - `2022-09-04T05:08:00` | 710,693 | 733 |
| 2 | `2022-09-01T00:00:00` - `2022-09-04T05:08:00` | 2,132,981 | 1,193 | `2022-09-04T05:09:00` - `2022-09-06T03:12:00` | 711,163 | 862 |
| 3 | `2022-09-01T00:00:00` - `2022-09-06T03:12:00` | 2,844,144 | 2,055 | `2022-09-06T03:13:00` - `2022-09-07T14:55:00` | 710,813 | 801 |

### Candidate results

| Candidate | Model family | Fold | Train rows | Validation rows | PR-AUC (AP) | ROC-AUC | Eligible | Fit seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| lr_newton_unweighted_c001 | logistic_regression | 1 | 1,422,288 | 710,693 | 0.01143686 | N/A | True | 8.638 |
| lr_newton_sqrt_weight_c001 | logistic_regression | 1 | 1,422,288 | 710,693 | 0.01377531 | N/A | True | 7.149 |
| lr_newton_sqrt_weight_c0001 | logistic_regression | 1 | 1,422,288 | 710,693 | 0.01329915 | N/A | True | 6.969 |
| lr_saga_sqrt_weight_c001 | logistic_regression | 1 | 1,422,288 | 710,693 | 0.00636321 | N/A | True | 56.191 |
| rf_sprint2_reference | random_forest | 1 | 1,422,288 | 710,693 | 0.02261109 | N/A | True | 28.938 |
| rf_deeper_regularized | random_forest | 1 | 1,422,288 | 710,693 | 0.02784963 | N/A | True | 49.730 |
| lgb_stable_unweighted | lightgbm | 1 | 1,422,288 | 710,693 | 0.07107480 | N/A | True | 30.564 |
| lgb_stable_sqrt_weight | lightgbm | 1 | 1,422,288 | 710,693 | 0.06782340 | N/A | True | 26.357 |
| lgb_stable_capped100_weight | lightgbm | 1 | 1,422,288 | 710,693 | 0.06246840 | N/A | True | 22.455 |
| lgb_sqrt_weight_stronger_regularization | lightgbm | 1 | 1,422,288 | 710,693 | 0.06190329 | N/A | True | 28.922 |
| lr_newton_unweighted_c001 | logistic_regression | 2 | 2,132,981 | 711,163 | 0.03017520 | N/A | True | 12.029 |
| lr_newton_sqrt_weight_c001 | logistic_regression | 2 | 2,132,981 | 711,163 | 0.03210865 | N/A | True | 10.378 |
| lr_newton_sqrt_weight_c0001 | logistic_regression | 2 | 2,132,981 | 711,163 | 0.03100588 | N/A | True | 9.817 |
| lr_saga_sqrt_weight_c001 | logistic_regression | 2 | 2,132,981 | 711,163 | 0.01433826 | N/A | True | 70.944 |
| rf_sprint2_reference | random_forest | 2 | 2,132,981 | 711,163 | 0.05624168 | N/A | True | 43.216 |
| rf_deeper_regularized | random_forest | 2 | 2,132,981 | 711,163 | 0.07019155 | N/A | True | 73.877 |
| lgb_stable_unweighted | lightgbm | 2 | 2,132,981 | 711,163 | 0.28496272 | N/A | True | 47.734 |
| lgb_stable_sqrt_weight | lightgbm | 2 | 2,132,981 | 711,163 | 0.20962604 | N/A | True | 43.074 |
| lgb_stable_capped100_weight | lightgbm | 2 | 2,132,981 | 711,163 | 0.16936378 | N/A | True | 39.988 |
| lgb_sqrt_weight_stronger_regularization | lightgbm | 2 | 2,132,981 | 711,163 | 0.17534824 | N/A | True | 45.385 |
| lr_newton_unweighted_c001 | logistic_regression | 3 | 2,844,144 | 710,813 | 0.03875566 | N/A | True | 14.450 |
| lr_newton_sqrt_weight_c001 | logistic_regression | 3 | 2,844,144 | 710,813 | 0.04060694 | N/A | True | 12.470 |
| lr_newton_sqrt_weight_c0001 | logistic_regression | 3 | 2,844,144 | 710,813 | 0.03875176 | N/A | True | 12.802 |
| lr_saga_sqrt_weight_c001 | logistic_regression | 3 | 2,844,144 | 710,813 | 0.02124877 | N/A | True | 69.819 |
| rf_sprint2_reference | random_forest | 3 | 2,844,144 | 710,813 | 0.08949349 | N/A | True | 62.924 |
| rf_deeper_regularized | random_forest | 3 | 2,844,144 | 710,813 | 0.12436848 | N/A | True | 111.573 |
| lgb_stable_unweighted | lightgbm | 3 | 2,844,144 | 710,813 | 0.35257250 | N/A | True | 51.054 |
| lgb_stable_sqrt_weight | lightgbm | 3 | 2,844,144 | 710,813 | 0.31428022 | N/A | True | 59.241 |
| lgb_stable_capped100_weight | lightgbm | 3 | 2,844,144 | 710,813 | 0.27551041 | N/A | True | 54.571 |
| lgb_sqrt_weight_stronger_regularization | lightgbm | 3 | 2,844,144 | 710,813 | 0.28348126 | N/A | True | 63.941 |

### Selected candidates

| Model family | Selected candidate | Metric | Value | Selection partition |
| --- | --- | --- | ---: | --- |
| lightgbm | lgb_stable_unweighted | mean_average_precision | 0.23620334 | inner_temporal_validation_folds |
| logistic_regression | lr_newton_sqrt_weight_c001 | mean_average_precision | 0.02883030 | inner_temporal_validation_folds |
| random_forest | rf_deeper_regularized | mean_average_precision | 0.07413655 | inner_temporal_validation_folds |

## Feature-family ablation and graph value

Primary A/B/C graph-value claims require the same model family, selected hyperparameters, seed, chronological split, preprocessing discipline, and evaluation code; only the permitted feature family changes.

| Feature family | Scope | Model family | Parameter digest | Features | PR-AUC (AP) | ROC-AUC | Delta vs B |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| transaction_only | primary | lightgbm | `894d641902a8d3361fff24a54998c34813c5367c7c3f6122667fbcbb1cba3fc7` | 49 | 0.09984124 | 0.96441199 | -0.25550918 |
| transaction_temporal_history | primary | lightgbm | `894d641902a8d3361fff24a54998c34813c5367c7c3f6122667fbcbb1cba3fc7` | 74 | 0.35535042 | 0.98179242 | 0.00000000 |
| transaction_temporal_history_graph | primary | lightgbm | `894d641902a8d3361fff24a54998c34813c5367c7c3f6122667fbcbb1cba3fc7` | 79 | 0.47175420 | 0.98635944 | 0.11640378 |
| transaction_temporal_history_graph_novel3 | sensitivity_duplicate_removed | lightgbm | `894d641902a8d3361fff24a54998c34813c5367c7c3f6122667fbcbb1cba3fc7` | 77 | 0.47175420 | 0.98635944 | 0.11640378 |

## Validation-only threshold analysis

Thresholds are selected only from complete validation score groups using the recorded alert-budget and FPR/recall trade-off. No test-set threshold feedback is permitted.

| Model | Rule | Status | Threshold | Precision | Recall | F1 | FPR | Alerts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation_transaction_only | maximum_f1 | selected | -2.703450333051 | 0.20129870 | 0.16315789 | 0.18023256 | 0.00064653 | 616 |
| ablation_transaction_only | maximum_recall_under_alert_budget | selected | -3.935199803528 | 0.06004953 | 0.38289474 | 0.10381734 | 0.00598563 | 4,846 |
| ablation_transaction_only | maximum_recall_under_fpr_ceiling | selected | -4.219675234709 | 0.04312532 | 0.44736842 | 0.07866728 | 0.00991342 | 7,884 |
| ablation_transaction_only | predeclared_joint_primary_constraint | selected | -3.935199803528 | 0.06004953 | 0.38289474 | 0.10381734 | 0.00598563 | 4,846 |
| ablation_transaction_temporal_history_graph | maximum_f1 | selected | -1.947318257926 | 0.56880734 | 0.40789474 | 0.47509579 | 0.00030881 | 545 |
| ablation_transaction_temporal_history_graph | maximum_recall_under_alert_budget | selected | -4.301970213652 | 0.11631632 | 0.76447368 | 0.20191138 | 0.00580035 | 4,995 |
| ablation_transaction_temporal_history_graph | maximum_recall_under_fpr_ceiling | selected | -5.030524981120 | 0.07970176 | 0.81578947 | 0.14521607 | 0.00940749 | 7,779 |
| ablation_transaction_temporal_history_graph | predeclared_joint_primary_constraint | selected | -4.301970213652 | 0.11631632 | 0.76447368 | 0.20191138 | 0.00580035 | 4,995 |
| ablation_transaction_temporal_history_graph_novel3 | maximum_f1 | selected | -1.947318257926 | 0.56880734 | 0.40789474 | 0.47509579 | 0.00030881 | 545 |
| ablation_transaction_temporal_history_graph_novel3 | maximum_recall_under_alert_budget | selected | -4.301970213652 | 0.11631632 | 0.76447368 | 0.20191138 | 0.00580035 | 4,995 |
| ablation_transaction_temporal_history_graph_novel3 | maximum_recall_under_fpr_ceiling | selected | -5.030524981120 | 0.07970176 | 0.81578947 | 0.14521607 | 0.00940749 | 7,779 |
| ablation_transaction_temporal_history_graph_novel3 | predeclared_joint_primary_constraint | selected | -4.301970213652 | 0.11631632 | 0.76447368 | 0.20191138 | 0.00580035 | 4,995 |
| lightgbm | maximum_f1 | selected | -1.695729447614 | 0.53846154 | 0.31315789 | 0.39600666 | 0.00026807 | 442 |
| lightgbm | maximum_recall_under_alert_budget | selected | -3.949463822202 | 0.10302170 | 0.66842105 | 0.17852750 | 0.00581217 | 4,931 |
| lightgbm | maximum_recall_under_fpr_ceiling | selected | -4.734173972431 | 0.07211002 | 0.76578947 | 0.13180840 | 0.00984114 | 8,071 |
| lightgbm | predeclared_joint_primary_constraint | selected | -3.949463822202 | 0.10302170 | 0.66842105 | 0.17852750 | 0.00581217 | 4,931 |
| logistic_regression | maximum_f1 | selected | -0.139262676239 | 0.03637251 | 0.39210526 | 0.06656986 | 0.01037466 | 8,193 |
| logistic_regression | maximum_recall_under_alert_budget | selected | 0.474977016449 | 0.03339368 | 0.21842105 | 0.05793055 | 0.00631415 | 4,971 |
| logistic_regression | maximum_recall_under_fpr_ceiling | selected | -0.073819160461 | 0.03546371 | 0.36578947 | 0.06465868 | 0.00993575 | 7,839 |
| logistic_regression | predeclared_joint_primary_constraint | selected | 0.474977016449 | 0.03339368 | 0.21842105 | 0.05793055 | 0.00631415 | 4,971 |
| random_forest | maximum_f1 | selected | 0.793702313019 | 0.15384615 | 0.23947368 | 0.18733917 | 0.00131539 | 1,183 |
| random_forest | maximum_recall_under_alert_budget | selected | 0.642896569524 | 0.08008008 | 0.52631579 | 0.13900956 | 0.00603820 | 4,995 |
| random_forest | maximum_recall_under_fpr_ceiling | selected | 0.555093113479 | 0.05872107 | 0.62105263 | 0.10729711 | 0.00994233 | 8,038 |
| random_forest | predeclared_joint_primary_constraint | selected | 0.642896569524 | 0.08008008 | 0.52631579 | 0.13900956 | 0.00603820 | 4,995 |

## Saturation and tie investigation

Raw ranking scores and transformed probabilities are reported separately. If a Top-K cutoff intersects an equal-score group, deterministic values use the declared immutable-row tie break and the expected/minimum/maximum bounds describe the unresolved ordering within that group.

| Stage | Model | Raw unique | Probability unique | Exact p=0 | Exact p=1 | Raw AP | Probability AP | Material Top-K ties |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sprint2_reproduced | logistic_regression | 750,422 | 663,495 | 35 | 60,119 | 0.00811875 | 0.00663424 | 0 |
<!-- finding: accepted_model_recomputed_from_serialized_estimator -->
| sprint2_reproduced | random_forest | 158,190 | 158,190 | 371,235 | 0 | 0.08859105 | 0.08859105 | 0 |
<!-- finding: accepted_model_recomputed_from_serialized_estimator -->
| sprint2_reproduced | lightgbm | 367,109 | 1,960 | 463,876 | 108,347 | 0.01477564 | 0.00547556 | 0 |
<!-- finding: accepted_model_recomputed_from_serialized_estimator -->
| sprint3_refined | logistic_regression | 731,426 | 731,281 | 0 | 135 | 0.02600700 | 0.02600700 | 0 |
<!-- finding: refined_model_outer_validation_diagnostic -->
| sprint3_refined | random_forest | 152,558 | 152,558 | 463,322 | 0 | 0.12967475 | 0.12967475 | 0 |
<!-- finding: refined_model_outer_validation_diagnostic -->
| sprint3_refined | lightgbm | 753,820 | 753,820 | 0 | 0 | 0.35535042 | 0.35535042 | 0 |
<!-- finding: refined_model_outer_validation_diagnostic -->
- accepted_model_recomputed_from_serialized_estimator
- accepted_model_recomputed_from_serialized_estimator
- accepted_model_recomputed_from_serialized_estimator
- refined_model_outer_validation_diagnostic
- refined_model_outer_validation_diagnostic
- refined_model_outer_validation_diagnostic

### Executable root-cause conclusion

- Saved-score/model reproduction verified: `True`
- Serialization error observed: `False`
- Immediate probability-tie mechanism: sigmoid/link conversion maps extreme distinguishable raw margins to exact floating-point 0/1 probabilities
- Logistic Regression upstream finding: non-converged weakly regularized balanced SGD plus heavy-tailed standardized amount signals produced extreme margins
- LightGBM upstream finding: full positive-class weight with near-zero leaf/Hessian regularization produced extreme leaf values and raw margins
- Refinement controls: convergent LogisticRegression C/solver/weight trials and controlled LightGBM weight/regularization trials
- Ranking policy: raw margins; probability ties are diagnosed separately

## Frozen chronological prevalence

| Partition | Rows | Positives | Positive rate | Timestamp range |
| --- | ---: | ---: | ---: | --- |
| train | 3,554,957 | 2,856 | 0.0008033852 | `2022-09-01T00:00:00` - `2022-09-07T14:55:00` |
| validation | 761,749 | 760 | 0.0009977040 | `2022-09-07T14:56:00` - `2022-09-09T03:16:00` |
| test | 761,639 | 1,561 | 0.0020495274 | `2022-09-09T03:17:00` - `2022-09-18T16:18:00` |

The chronological prevalence shift is preserved. Precision and alert volume are prevalence-sensitive, so a validation operating point must not be assumed to transfer unchanged to the later test period.
Test counts above are metadata-only; Sprint 3 did not load test features, labels, scores, or predictions.

## Runtime

- `total`: 2,158.664 seconds
- `duckdb_initialization`: 0.024 seconds
- `frozen_input_and_sprint2_verification`: 1.101 seconds
- `inner_expanding_temporal_cv`: 1,397.208 seconds
- `outer_b_matrix_and_refined_models`: 324.525 seconds
- `same_model_feature_family_ablation`: 411.347 seconds
- `saturation_attribution_summary`: 0.022 seconds
- `source_and_configuration_snapshot`: 0.739 seconds
- `sprint2_saturation_root_cause_reproduction`: 21.040 seconds
- `validation_selection_thresholds_and_export`: 1.957 seconds
- `work_matrix_cleanup`: 0.506 seconds

## Artifacts

- `artifacts/sprint3/run_manifest.json`
- `artifacts/sprint3/temporal_cv/folds.json`
- `artifacts/sprint3/temporal_cv/trials.csv`
- `artifacts/sprint3/temporal_cv/candidate_summary.csv`
- `artifacts/sprint3/temporal_cv/selected_candidates.json`
- `artifacts/sprint3/refined_model_comparison.csv`
- `artifacts/sprint3/refined_transaction_champion.json`
- `artifacts/sprint3/baseline_vs_refined.csv`
- `artifacts/sprint3/baseline_vs_refined.json`
- `artifacts/sprint3/ablation/feature_family_ablation.csv`
- `artifacts/sprint3/ablation/graph_value_conclusion.json`
- `artifacts/sprint3/saturation/sprint2_vs_refined.json`
- `artifacts/sprint3/threshold/analysis.json`
- `artifacts/sprint3/final_test_policy.json`
- `artifacts/sprint3/validation_predictions.parquet`
- `reports/generated/SPRINT_3_STATUS.md`
- `artifacts/sprint3/verification_report.json`
- `artifacts/sprint3/quality_report.json`

## Acceptance checklist

- **PASS** - automated_quality_checks
- **PASS** - expanding_temporal_cv_outer_train_only
- **PASS** - final_test_untouched
- **PASS** - frozen_full_data_unsampled
- **PASS** - graph_value_same_model_protocol
- **PASS** - logistic_convergence_resolved
- **PASS** - machine_readable_artifacts
- **PASS** - pr_auc_primary_operational_metrics_reported
- **PASS** - raw_ranking_metrics_and_tie_aware_diagnostics
- **PASS** - score_saturation_root_cause_investigated
- **PASS** - sprint2_git_checkpoint_present
- **PASS** - sprint4_graphsage_not_started
- **PASS** - three_model_families_tuned
- **PASS** - three_primary_feature_families_compared
- **PASS** - validation_only_threshold_optimization

## Automated quality evidence

- Status: **PASS**
- Pytest passed: 207
- Artifact verification: PASS
- Ruff lint/format: inspect machine-readable `quality_report.json` checks

## Scope stop

Sprint 3 stops here. Sprint 4 / GraphSAGE training, tuning, inference, and evaluation were not started.
