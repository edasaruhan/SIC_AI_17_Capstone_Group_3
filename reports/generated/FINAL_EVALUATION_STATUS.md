# ARGUS AI - Frozen One-Shot Final Evaluation

Status: **PASS**

The exact untouched test partition contained 761,639 transactions and 1,561 positive labels (0.204953%).

The champion remained graph_enhanced_lightgbm, frozen on validation before test access. Test metrics were confirmatory only and performed no selection or tuning.

| Model | Frozen role | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | Alerts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| graph_enhanced_lightgbm | frozen_champion | 0.69005906 | 0.99127213 | 0.17816018 | 0.88212684 | 0.29644779 | 0.00835704 | 7,729 |
| graphsage_edge_classifier | comparator | 0.01341710 | 0.82662724 | 0.00600590 | 0.11338885 | 0.01140758 | 0.03854078 | 29,471 |
| refined_transaction_lightgbm | comparator | 0.53079643 | 0.98731443 | 0.16072332 | 0.81422165 | 0.26845496 | 0.00873200 | 7,908 |

All values above were generated from the sealed prediction artifact using validation-selected raw-score thresholds. No post-test retraining or tuning was performed.
