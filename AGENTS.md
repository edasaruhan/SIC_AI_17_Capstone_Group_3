# ARGUS repository guardrails

- Keep all experiment settings in `configs/`; do not hide critical choices in notebooks.
- Treat `data/raw/`, `data/interim/`, `data/processed/`, and generated artifacts as local data.
- Never fabricate a metric, chart, table, case, or comparison. Use `PENDING` when code was not run.
- At transaction time `t`, history and graph-history features may use timestamps strictly before `t` only.
- Preserve directed repeated transfers. Remove exact duplicate rows only through an explicit audited policy.
- Build account identity from normalized bank ID plus account ID; never use account ID alone.
- Select/tune future models on validation only. The final test set remains untouched until decisions freeze.
- Use human-review language: suspicious candidate or elevated investigation priority. Never declare guilt.
- Sprint 1 has no trained model, threshold, SHAP output, GraphSAGE, or Streamlit application.

