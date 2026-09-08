# ARGUS AI - Sprint 4 GraphSAGE and Product Layer Status

This report is rendered only from executable Sprint 4 artifacts. No metric, case, or explanation is entered manually.

## Outcome

- Sprint 4 status: **PASS**
- GraphSAGE target: transaction/edge label; unsupported account-level label: `False`
- GraphSAGE training scope: `deterministic_sampled_graphsage_training_full_validation_evaluation`
- Outer validation evaluation: **FULL 761,749 rows**
- Final test fitting, transformation, graph construction, inference, tuning, and evaluation: **NOT USED**
- Final-test opening: **STOPPED BEFORE FINAL EVALUATION**

## Fair full-validation model comparison

| Model | PR-AUC (AP) | ROC-AUC | Precision@1K | Recall@1K | Threshold precision | Threshold recall | F1 | FPR | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| graph_enhanced_lightgbm | 0.47175420 | 0.98635944 | 0.38100000 | 0.50131579 | 0.11631632 | 0.76447368 | 0.20191138 | 0.00580035 | 4,995 |
| graphsage_edge_classifier | 0.00943876 | 0.84664170 | 0.02500000 | 0.03289474 | 0.00969110 | 0.06315789 | 0.01680378 | 0.00644556 | 4,953 |
| refined_transaction_lightgbm | 0.35535042 | 0.98179242 | 0.30500000 | 0.40131579 | 0.10302170 | 0.66842105 | 0.17852750 | 0.00581217 | 4,931 |

All three score vectors use the identical frozen outer-validation rows and the same 5,000-alert / 1% FPR validation-only operating-point rule. The two LightGBM rows are immutable Sprint 3 references; GraphSAGE uses its saved raw logits.

## Explicit scalable-subset disclosure

- Train-prefix message graph: 150,000 / 1,422,288 edges (0.10546387).
- Validation-inference historical graph: 300,000 / 3,554,957 train edges (0.08438921).
- Supervised train sample: 52,396 transactions, including all 2,396 positives after the message-graph cutoff and a deterministic negative sample.
- Context sampling is target-agnostic and ordered by stable MD5 transaction-ID digest. This is not described as full-graph GraphSAGE training.
- Node structural features use directed degrees, not cross-currency monetary sums; no FX conversion table is available.

## Graph construction and leakage boundary

- Training message graph maximum timestamp: `2022-09-02T09:27:00`
- Supervised edge minimum timestamp: `2022-09-02T09:28:00`
- Outer-validation embeddings use only sampled outer-train edges; no validation edge enters message passing.
- Directed endpoints and repeated transactions are retained; repeated edges act as frequency weight in neighbor means.

## Cases, evidence, and explanations

- Saved cases: 20
- Minimum observed evidence per case: 7
- Deterministic no-LLM notes: 20
- Observed evidence and model evidence are stored separately.
- LightGBM explanations use native `pred_contrib` TreeSHAP with an additivity check. GraphSAGE explanations are labeled local gradient-times-input sensitivity, not SHAP or causal attribution.

## Streamlit application

- Screens: Executive Dashboard, Investigation Queue, Case Investigator, Model Comparison.
- Page-load training: **DISABLED**; the app reads saved Sprint 4 artifacts only.
- Optional LLM unavailable path: deterministic fallback is complete and tested.
- GraphSAGE sigmoid values are uncalibrated ranking scores because training uses sampled negatives and positive weighting; they are not event probabilities.

## Runtime

- `case_evidence_and_product_export`: 3.115 seconds
- `deterministic_training_sampling`: 7.654 seconds
- `frozen_input_preflight`: 1.625 seconds
- `full_validation_scoring`: 13.311 seconds
- `graphsage_training`: 7.892 seconds
- `inference_graph_construction`: 9.281 seconds
- `training_graph_construction`: 1.262 seconds
- `validation_only_evaluation`: 5.629 seconds
- `total`: 49.928 seconds

## Acceptance checklist

- **PASS** - automated_quality_checks
- **PASS** - deterministic_no_llm_fallback_completed
- **PASS** - deterministic_sampled_graphsage_training_completed
- **PASS** - final_test_remained_sealed
- **PASS** - frozen_sprint3_references_preserved
- **PASS** - full_frozen_validation_scored
- **PASS** - full_graph_limitation_explicitly_disclosed
- **PASS** - gnn_explanation_labeled_as_sensitivity_not_shap
- **PASS** - graphsage_checkpoint_and_node_embeddings_saved
- **PASS** - minimum_three_observed_evidence_per_case
- **PASS** - observed_and_model_evidence_separated
- **PASS** - real_validation_cases_built
- **PASS** - same_validation_rows_used_for_three_model_comparison
- **PASS** - streamlit_page_load_training_disabled
- **PASS** - streamlit_saved_artifact_round_trip_passed
- **PASS** - transaction_edge_target_only
- **PASS** - tree_shap_additivity_verified
- **PASS** - unsupported_account_level_label_not_created
- **PASS** - validation_only_threshold_optimization_completed

## Automated quality evidence

- Status: **PASS**
- Pytest passed: 289
- Artifact verification: PASS
- `pip_check`: **PASS**
- `pytest`: **PASS**
- `ruff_format`: **PASS**
- `ruff_lint`: **PASS**
- `streamlit_artifact_smoke`: **PASS**

## Stop boundary

Sprint 4 stops before final-test opening. Test features, labels, predictions, and metrics remain sealed.
