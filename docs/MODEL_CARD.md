# Model Card — Sprint 1 Status

## Current state

No predictive model exists in Sprint 1. Consequently there are no model metrics,
thresholds, feature importances, SHAP values, rankings, or automated decisions to report.

## Intended future use

Later sprints may compare transaction-only and graph-aware methods for prioritizing
suspicious network candidates for human investigation. ARGUS is decision support, not an
automated guilt determination or punitive-action system.

## Current deliverable

Sprint 1 provides the reproducible data proof: validated IBM AML ingestion, canonical
transactions, strict chronological splits, past-only features, directed graph summaries,
EDA artifacts, and automated tests. Model development is `PENDING` Sprint 2.

## Limitations

- IBM AML-Data is synthetic and does not establish production-bank performance.
- Labels and behavior in a synthetic benchmark may not represent live typologies or drift.
- No fairness, calibration, deployment, or analyst-capacity claim can be made before models
  and appropriate evaluation exist.

