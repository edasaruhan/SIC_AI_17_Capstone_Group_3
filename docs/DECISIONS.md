# ARGUS AI Architecture Decision Log

Decisions are recorded before results so implementation choices cannot be rewritten
to fit a preferred narrative. “Accepted” means the design rule is authoritative;
it does not mean its implementation has passed tests. Runtime verification remains
in `reports/generated/SPRINT_1_STATUS.md`.

## ADR-001 — Limit the active build to Sprint 1

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Implement and validate repository/data/EDA/leakage-safe feature
  foundations only. Treat transaction models and later layers as deferred.
- **Reason:** The user explicitly prohibited moving to Sprint 2 before Sprint 1 is
  complete.
- **Consequence:** No model metric or graph-value conclusion belongs in current
  documentation.

## ADR-002 — Preserve coursework as immutable source context

- **Date:** 2026-09-06
- **Status:** Accepted
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
- **Status:** Accepted for future work
- **Decision:** Do not invent account-level laundering targets from transaction
  labels. Any future graph model should predict transactions/edges or document a
  defensible alternative target.
- **Reason:** The IBM source supplies transaction-level ground truth only.
- **Consequence:** Sprint 2 remains transaction-baseline work; GraphSAGE is deferred
  beyond it.

## ADR-015 — Require human review and non-accusatory language

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Present computed facts as investigation evidence and priority, not
  guilt or an instruction for adverse action.
- **Reason:** Synthetic labels and probabilistic patterns do not justify legal or
  operational conclusions.
- **Consequence:** Later case output must separate observed evidence from model
  evidence, state uncertainty, and require trained analyst review.

## ADR-016 — Recommend validation-first model selection for Sprint 2

- **Date:** 2026-09-06
- **Status:** Proposed for Sprint 2; not implemented
- **Decision:** Compare Logistic Regression, Random Forest, and one boosting family
  on identical chronological partitions; select using validation PR-AUC and
  operational top-K/FPR evidence.
- **Reason:** Accuracy is misleading at a 0.101942660453% positive rate, and test-
  driven selection would bias final evaluation.
- **Consequence:** Sprint 1 now passes, but model and threshold work remains a
  separately authorized Sprint 2 task. No model result is presently claimed.

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
  population estimates. Sprint 2 has not started.
