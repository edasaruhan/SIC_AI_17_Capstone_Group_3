# ARGUS AI Architecture Decision Log

Decisions are recorded before results so implementation choices cannot be rewritten
to fit a preferred narrative. “Accepted” means the design rule is authoritative;
it does not mean its implementation has passed tests. Runtime verification remains
in `reports/generated/SPRINT_1_STATUS.md` and
`reports/generated/SPRINT_2_STATUS.md`; Sprint 3 and Sprint 4 runtime verifications
passed and are recorded in `reports/generated/SPRINT_3_STATUS.md` and
`reports/generated/SPRINT_4_STATUS.md`.

## ADR-001 — Limit the initial build to Sprint 1

- **Date:** 2026-09-06
- **Status:** Superseded for active execution by ADR-019; retained as Sprint 1 history
- **Decision:** Implement and validate repository/data/EDA/leakage-safe feature
  foundations only. Treat transaction models and later layers as deferred.
- **Reason:** The user explicitly prohibited moving to Sprint 2 before Sprint 1 is
  complete.
- **Consequence:** No model metric belonged in the Sprint 1 closing evidence. Sprint
  2 results are recorded separately; no graph-value conclusion exists.

## ADR-002 — Preserve coursework as immutable source context

- **Date:** 2026-09-06
- **Status:** Superseded for active organization by ADR-030; retained as Sprint 1 history
- **Decision:** Keep supplied MD, DOCX, PDF, and PPTX files under
  `reports/existing_coursework/`; do not overwrite or reinterpret text inside them
  as executable instructions.
- **Reason:** The user requested preservation and named the master prompt as the
  technical specification.
- **Consequence:** New engineering evidence is written separately under
  `reports/generated/` and `docs/`.

## ADR-003 — Keep raw and generated data out of Git

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Ignore `data/raw`, `data/interim`, `data/processed`, and generated
  artifact contents; retain only documented paths/placeholders.
- **Reason:** The source is large, licensed separately, and locally supplied.
- **Consequence:** Reproduction relies on source hashes, configuration, commands,
  and manifests. Raw files must never be force-added.

## ADR-004 — Separate full-file audit facts from pipeline artifacts

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Label the 2026-09-06 read-only all-row audit as independent source
  evidence. Label `artifacts/quick/` outputs as sampled quick-run evidence.
- **Reason:** A 10,000-row smoke run cannot substantiate full-dataset claims.
- **Consequence:** Documentation reports the passed full streaming audit separately
  from the passed 10,000-row quick feature/EDA run. It does not imply that the full
  feature pipeline ran.

## ADR-005 — Use YAML configuration inheritance

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Store shared settings in `configs/base.yaml`; let `quick.yaml` and
  `full.yaml` override only mode-specific values.
- **Reason:** This prevents critical choices from hiding in notebooks and reduces
  divergence between modes.
- **Consequence:** Paths resolve against repository root, and every run records its
  resolved configuration.

## ADR-006 — Normalize duplicate raw Account headers deterministically

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Interpret the first transaction `Account` as sender and the second
  as receiver; accept a parser-disambiguated `Account.1` only in the corresponding
  validated position.
- **Reason:** The IBM raw header repeats `Account` and name-only selection can merge
  or overwrite endpoints.
- **Consequence:** Canonical names are `from_account` and `to_account`; schema
  validation occurs before feature generation.

## ADR-007 — Use normalized composite node identity

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Canonical node ID is
  `normalized_bank_id::UPPERCASE_ACCOUNT_ID`; digit-only bank IDs lose leading
  zeroes during canonicalization.
- **Reason:** Transaction bank IDs are zero-padded while accounts IDs are not, and
  account numbers are not globally unique.
- **Evidence informing decision:** Raw string joins failed on all transaction rows;
  normalized composite joins had zero unmatched endpoints in the independent full
  audit. Eight account-number strings occur in two composite identities each.
- **Consequence:** Retain raw identifiers for provenance and join only on composite
  canonical identity.

## ADR-008 — Keep exact duplicates and repeated edges explicit

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Report exact source duplicates, preserve stable source identity, and
  do not automatically collapse repeated directed transfers.
- **Reason:** Repetition can be a meaningful behavioral signal; deletion changes
  temporal/graph evidence and label counts.
- **Evidence informing decision:** The transaction source contains 9 exact duplicate
  occurrences in 9 pairs; accounts contains none.
- **Consequence:** Any later deduplication is a separate versioned experiment with
  before/after counts.

## ADR-009 — Use chronological splits with intact timestamp groups

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Use chronological train/validation/test partitions and assign all
  equal timestamps to one partition.
- **Reason:** A shuffled row split leaks future behavior and splitting a timestamp
  group makes strict ordering ambiguous.
- **Consequence:** Realized fractions may differ slightly from requested 70/15/15;
  metadata must prove strict boundary inequalities.

## ADR-010 — Define history as strictly earlier timestamps

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Features at `t` use state from timestamps `< t`, never the current
  event, same-timestamp peers, or future events.
- **Reason:** Row ordering inside a timestamp is arbitrary and would otherwise leak
  contemporaneous/future information.
- **Consequence:** Compute updates in timestamp batches and test future-append
  invariance.

## ADR-011 — Preserve directed multiedge semantics

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Model sender→receiver direction and retain repeated edges. Initial
  Sprint 1 graph features are strictly-prior fan-in, fan-out, and pair counts.
- **Reason:** Direction and repetition describe different financial behaviors;
  silently reducing to a simple undirected graph destroys evidence.
- **Consequence:** Global expensive graph algorithms and GraphSAGE are deferred.

## ADR-012 — Compare amounts only within a shared currency

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Amount difference/ratio fields are null for cross-currency rows.
- **Reason:** Sprint 1 has no versioned exchange-rate table; direct arithmetic across
  different units is not meaningful.
- **Consequence:** `currency_match` carries the availability condition and nulls are
  intentional, not missing-source errors.

## ADR-013 — Prefer vectorized/grouped historical computation

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Use pandas grouping/cumulative/rolling operations at entity and
  timestamp-batch granularity rather than a Python transaction-row loop.
- **Reason:** HI-Small contains 5,078,345 transactions and requires scalable,
  deterministic processing.
- **Consequence:** Toy tests must still specify exact expected values and tie
  behavior.

## ADR-014 — Keep labels at transaction/edge level

- **Date:** 2026-09-06
- **Status:** Accepted and verified in Sprint 4
- **Decision:** Do not invent account-level laundering targets from transaction
  labels. Any graph model should predict transactions/edges or document a
  defensible alternative target.
- **Reason:** The IBM source supplies transaction-level ground truth only.
- **Consequence:** Sprint 4 GraphSAGE classifies transaction edges from endpoint
  embeddings and transaction features; no account-level fraud target is created.

## ADR-015 — Require human review and non-accusatory language

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Present computed facts as investigation evidence and priority, not
  guilt or an instruction for adverse action.
- **Reason:** Synthetic labels and probabilistic patterns do not justify legal or
  operational conclusions.
- **Consequence:** Sprint 4 case output separates observed evidence from model
  evidence, uses non-accusatory notes, and remains subject to trained analyst
  review.

## ADR-016 — Require validation-first model selection for Sprint 2

- **Date:** 2026-09-06
- **Status:** Accepted and verified
- **Decision:** Compare Logistic Regression, Random Forest, and LightGBM on identical
  frozen chronological partitions. Select the Transaction Baseline Champion solely
  by maximum validation average precision; report top-K/FPR evidence as operating
  context, not as test-driven selection input.
- **Reason:** Accuracy is misleading at a 0.101942660453% positive rate, and test-
  driven selection would bias final evaluation.
- **Evidence:** Random Forest validation AP is 0.08859105087174989, Logistic
  Regression AP is 0.006634242622201133, and LightGBM AP is
  0.0054755570960638884. Independent reselection names Random Forest champion.
- **Consequence:** Random Forest is the current Transaction Baseline Champion for
  this snapshot. Final-test inference remains closed and Sprint 3 refinement may
  not rewrite the Sprint 2 validation evidence.

## ADR-017 — Isolate ARGUS from the unrelated DATATON repository

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Build ARGUS as its own Git repository in the user-requested
  `ARGUS PROJECET` desktop folder and leave the initial DATATON/GridUp repository
  read-only.
- **Reason:** DATATON was an unrelated dirty repository with user-owned changes and
  roughly 13.4 GiB of untracked material. Mixing projects would risk overwriting or
  committing unrelated work.
- **Consequence:** `ARGUS PROJECET` has its own `main` branch and ignore policy;
  DATATON/GridUp remains untouched.

## ADR-018 — Close Sprint 1 with quick proof and full out-of-core execution

- **Date:** 2026-09-06
- **Status:** Accepted and verified
- **Decision:** Retain the deterministic quick path, and additionally execute exact
  full preprocessing, splitting, and feature engineering through a bounded
  out-of-core DuckDB path. Separate exact all-row EDA tables from sampled
  plot/NetworkX artifacts.
- **Reason:** The first `1GB`/two-thread full attempt exhausted memory in the
  six-way feature export join. Materializing that join before export, using one
  thread and a bounded `2GB` limit, and releasing large temporary tables allowed
  completion without treating sampled features as full-data evidence.
- **Evidence:** 34 tests passed; the full run produced 5,078,345 rows and 52 columns
  in 304.93406899999536 seconds; all four primary outputs reconcile by row and ID;
  full-exact EDA covers every row; sampled plots use 100,000 target-independent
  hash-ranked edges and a 50,000-edge NetworkX cap.
- **Consequence:** Sprint 1 is PASS with actual full feature execution. Sampled
  graph component/local-subgraph values remain descriptive and must not be called
  population estimates. This frozen output became the verified input to Sprint 2.

## ADR-019 — Execute Sprint 2 as a transaction-only validation experiment

- **Date:** 2026-09-07
- **Status:** Accepted and verified
- **Decision:** Fit Logistic Regression, Random Forest, and LightGBM on the same
  unsampled 3,554,957-row training matrix; evaluate all three on the same 761,749
  validation rows. Exclude target/identity/provenance fields and all five graph-
  history fields. Do not materialize or score the final test partition.
- **Reason:** A clean transaction-only champion and a closed test gate are required
  before model refinement or the controlled graph-value experiment.
- **Evidence:** The run finished in 389.0078022000016 seconds. Its verifier
  recomputed validation metrics, deserialized all three models, repeated champion
  selection, and found zero test prediction artifacts. The refreshed manifest
  inventories 26 payloads excluding itself.
- **Consequence:** Sprint 2 is complete and immutable, and Random Forest remains its
  validation-selected champion. Later Sprint 3 evidence is recorded separately
  under ADR-023 through ADR-026.

## ADR-020 — Fit all preprocessing state on train only

- **Date:** 2026-09-07
- **Status:** Accepted and verified
- **Decision:** Use an explicit predictor allow-list. Fit numeric medians/scales,
  low-cardinality vocabularies, and bank-frequency mappings only on train; transform
  validation with frozen state and safe unknown-category fallbacks.
- **Reason:** Allow-list selection prevents newly introduced identity, target, or
  graph columns from silently becoming predictors, while train-only state prevents
  validation distribution leakage.
- **Evidence:** The fitted state records 3,554,957 train rows and 74 transformed
  features. Tests exercise unknown categories, state integrity, forbidden fields,
  and DuckDB/pandas parity.
- **Consequence:** Validation is transform-only. Test was neither transformed nor
  used for preprocessing fit.

## ADR-021 — Keep threshold fixed and final test closed in Sprint 2

- **Date:** 2026-09-07
- **Status:** Accepted and verified
- **Decision:** Report threshold metrics at the configured `0.5` without optimizing
  it. Use non-interpolated average precision as the primary selection metric and
  ROC-AUC as secondary. Report K 100/500/1000 with deterministic source-row tie-
  breaking. Defer threshold tuning and final-test inference until refinement is
  frozen.
- **Reason:** Tuning belongs to Sprint 3, and repeated test inspection would bias
  the final estimate. Accuracy is not informative under the observed imbalance.
- **Evidence:** `final_test_policy.json` records no test fit, training, selection,
  inference, or predictions. Validation/train prevalence ratio is
  1.2418748968286322 and test/validation metadata ratio is 2.0542440105448496.
  Logistic Regression has 60,119 validation rows at exact score `1.0` (427
  positives); LightGBM has 108,347 (663 positives). Random Forest has no exact
  score-1 rows.
- **Consequence:** Fixed-threshold precision and alert volume are exploratory and
  prevalence-sensitive. They must not be projected unchanged to the untouched test
  period. Logistic/LightGBM top-K membership cuts through large tied plateaus and
  therefore depends materially on the deterministic source-row tie-break; average-
  precision champion selection remains tie-aware.

## ADR-022 — Use bounded baseline implementations on the full training partition

- **Date:** 2026-09-07
- **Status:** Accepted and verified
- **Decision:** Implement Logistic Regression as
  `SGDClassifier(loss="log_loss")`, Random Forest with bounded depth/tree count and
  balanced subsampling, and single-thread deterministic LightGBM with a train-only
  class ratio.
- **Reason:** The full 3,554,957-row by 74-column matrix must fit the available
  Windows hardware without substituting a sampled training claim. LightGBM was
  selected once for its CPU histogram/full-data fit under the deterministic,
  single-thread, bounded-memory plan; XGBoost was not executed.
- **Evidence:** All three models ran on the full training matrix. The logistic model
  reached its configured 20-iteration limit before convergence; this warning is
  preserved rather than hidden. All outputs are uncalibrated baselines.
- **Consequence:** Sprint 2 provides initial comparison evidence, not tuned or
  production-ready estimators. Its limitations become explicit Sprint 3 inputs and
  are never erased by later results. No empirical LightGBM-over-XGBoost superiority
  claim is supported.

## ADR-023 — Execute Sprint 3 with the final-test gate closed

- **Date:** 2026-09-07
- **Status:** Accepted and verified
- **Decision:** Implement bounded model refinement, expanding temporal CV,
  validation-only threshold selection, and controlled graph-feature ablation on
  the frozen Sprint 1/Sprint 2 data snapshot. Do not load, transform, score, or
  evaluate the final-test rows. At the Sprint 3 checkpoint, stop before
  Sprint 4/GraphSAGE.
- **Reason:** Refinement and graph value require multiple development comparisons;
  using final-test feedback for those choices would bias the final estimate.
- **Evidence:** All 15 Sprint 3 acceptance gates passed. Independent artifact
  verification, 207 tests, Ruff lint/format, and `pip check` passed; the final-test
  policy records no test fit, transform, inference, prediction, metric, or feedback.
- **Consequence:** Sprint 3 completed with test counts/prevalence reported only
  from existing split metadata. This was the entry checkpoint for the later,
  separately verified Sprint 4 execution.

## ADR-024 — Tune on expanding folds wholly inside outer train

- **Date:** 2026-09-07
- **Status:** Accepted and verified
- **Decision:** Create three expanding, timestamp-group-intact folds inside outer
  train. Fit preprocessing independently on every fold train, transform only its
  later fold interval, and select one candidate per estimator family by mean fold
  average precision with deterministic candidate-ID tie-breaking.
- **Reason:** A single outer-validation search would overfit the only available
  development holdout, while shuffled cross-validation violates temporal order.
- **Evidence:** Three timestamp-intact expanding folds ran wholly inside outer
  train. Mean fold AP selected `lr_newton_sqrt_weight_c001`,
  `rf_deeper_regularized`, and `lgb_stable_unweighted` before outer-validation
  comparison.
- **Consequence:** Outer validation is reserved for the selected-candidate refit,
  refined champion comparison, threshold analysis, and A/B/C experiment. No outer-
  validation statistic may enter fold preprocessing or candidate selection.

## ADR-025 — Separate raw ranking evidence from probability saturation

- **Date:** 2026-09-07
- **Status:** Accepted and verified
- **Decision:** Preserve raw Logistic Regression and LightGBM decision margins for
  ranking and store probabilities as diagnostics. Attribute saturation across
  preprocessing tails, class weighting, regularization/leaf stability, raw-margin
  ranges, and sigmoid representation. Report exact/near-boundary groups, score
  collapse, and tie-aware Top-K expected/minimum/maximum outcomes.
- **Reason:** Sprint 2 Logistic and LightGBM probabilities contain large exact-one
  plateaus. A stable source-row tie-break reproduces membership but cannot prove
  discrimination inside an equal-score group.
- **Evidence:** Serialized-estimator reproduction passed. The immediate tie
  mechanism was sigmoid/link conversion of extreme raw margins; Sprint 2 weak
  regularization/class weighting drove those margins. Refined LightGBM had no exact
  boundary probabilities, and refined Logistic Regression convergence passed.
- **Consequence:** Top-K superiority may not be claimed when K cuts an unresolved
  tie. Refined Logistic Regression must also pass explicit convergence evidence;
  a larger `max_iter` setting alone is insufficient.

## ADR-026 — Isolate graph value with a same-model A/B/C ablation

- **Date:** 2026-09-07
- **Status:** Accepted and verified
- **Decision:** Compare A transaction-only, B transaction plus temporal/history,
  and C B plus all five graph-history features using the same selected LightGBM
  candidate, parameters, seed, outer split, preprocessing discipline, raw-score
  representation, and metrics. Also run a declared novel-three graph sensitivity.
- **Reason:** Changing estimator family or tuning protocol together with features
  would confound graph value. Two required C columns are exact duplicates of B
  history columns on the frozen data, so the all-five result needs a redundancy-
  aware sensitivity.
- **Evidence:** With the same selected LightGBM protocol, B validation AP was
  0.35535042 and C AP was 0.47175420, a +0.11640378 delta. The novel-three
  sensitivity produced the same AP. Test metrics were not used.
- **Consequence:** The headline graph statement is the generated B-versus-C
  validation delta. The sensitivity adds only sender prior fan-in, receiver prior
  fan-out, and prior repeated-pair count. This tabular experiment is not GraphSAGE
  and supports no GNN-superiority claim.

## ADR-027 — Run GraphSAGE as a sampled transaction-edge experiment

- **Date:** 2026-09-08
- **Status:** Accepted and verified
- **Decision:** Preserve the frozen Sprint 3 references and train a GraphSAGE edge
  classifier from sender/receiver node embeddings plus transaction features. Use
  deterministic bounded graph samples when the full graph exceeds the declared
  workstation budget, disclose every denominator, and still score the complete
  frozen outer-validation partition. Keep final-test rows out of graph
  construction, fitting, inference, tuning, and evaluation.
- **Reason:** The IBM label belongs to transactions, while the available 16 GB RAM,
  approximately 4.6 GB free memory, and 4 GB GPU do not justify describing an
  unbounded full-graph training run as reproducible on this workstation.
- **Evidence:** The run used 150,000 of 1,422,288 eligible message-graph edges,
  52,396 supervised edges (all 2,396 eligible positives plus 50,000 deterministic
  negatives), and a 300,000-of-3,554,957 outer-train inference graph. It scored all
  761,749 validation rows. GraphSAGE AP was 0.00943876, below the frozen refined
  transaction LightGBM AP 0.35535042 and graph-enhanced LightGBM AP 0.47175420.
- **Consequence:** Sprint 4 supplies an honest bounded GNN value experiment, not a
  full-graph claim and not evidence of GNN superiority. The graph-enhanced LightGBM
  remains the validation leader; the final test stays sealed.

## ADR-028 — Exclude cross-currency node totals and label sigmoid scores honestly

- **Date:** 2026-09-08
- **Status:** Accepted and verified
- **Decision:** Use directed node-degree and deterministic composite-identity
  inputs for GraphSAGE, but exclude node-level monetary aggregates without a
  versioned FX conversion source. Rank by raw GraphSAGE logit and label the sigmoid
  value as an uncalibrated ranking score rather than an event probability.
- **Reason:** A node may transact in multiple currencies, so raw monetary sums mix
  units. Deterministic negative sampling plus positive weighting also changes the
  fitted class prior and prevents a probability-calibration claim.
- **Consequence:** Transaction amounts remain available in leakage-safe edge
  features and observed case evidence, while node features do not make invalid
  cross-currency totals. The UI and case schema state the score limitation.

## ADR-029 — Build cases from saved evidence with a deterministic fallback

- **Date:** 2026-09-08
- **Status:** Accepted and verified
- **Decision:** Build cases only from real saved model and graph outputs. Store
  observed evidence separately from model evidence, require at least three observed
  facts, label GNN gradient-times-input output as local sensitivity rather than
  SHAP, and use an evidence-bound deterministic note generator when no LLM is
  available. Streamlit must load artifacts and must not train on page load.
- **Reason:** An investigation interface must remain reproducible and usable without
  an external API, while explanations must not be confused with observed facts or
  causal proof.
- **Evidence:** Twenty saved validation cases each contain at least seven observed
  evidence items. Native LightGBM TreeSHAP passed its additivity check; GNN saved-
  score reproduction had zero maximum error. Executive Dashboard, Investigation
  Queue, Case Investigator, and Model Comparison load from saved artifacts, and all
  20 cases use the deterministic no-LLM fallback.
- **Consequence:** The product layer supports human prioritization only. It cannot
  manufacture evidence, infer guilt, or silently change the validated models.

## ADR-030 — Separate real coursework from technical and reference material

- **Date:** 2026-09-08
- **Status:** Accepted and verified
- **Decision:** Organize only genuinely prepared academic submissions under the six
  assigned `reports/coursework/` stages. Keep the master technical specification
  and blank teacher template under `reports/reference_materials/`. Preserve Sprint
  and final scientific outputs under `reports/generated/` and `docs/`; do not
  invent a separate `Final Evaluation` coursework stage.
- **Reason:** A blank template, implementation prompt, or generated model report is
  not evidence that a distinct academic submission was assigned or completed.
  Keeping these categories separate makes the GitHub repository easier to audit.
- **Evidence:** The reachable Git history contains only the combined **Model
  Refinement + Test Submission** teacher template and no separate final-evaluation
  assignment. The supplied Turkish refinement/test PDF contains both sections; the
  supplied short progress report is filled. Current document hashes are recorded
  in `reports/SHA256SUMS.txt`.
- **Consequence:** The coursework index lists six real delivery categories and no
  hypothetical seventh stage. A prior Literature–Data–Technology revision remains
  in its phase-local archive for provenance. No source DOCX is claimed for the
  refinement/test PDF until that original file is actually supplied.
