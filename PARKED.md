# Branch status: PARKED (proposed)

**Branch:** `feat/control-store-stateful-ha` — StateStore-backed server state (SQL/SQLite,
Postgres-capable) replacing filesystem-backed job/study/disabled-client state.

**Decision context (June 2026):** Multistudy went GA in 2.8 on the legacy JSON registry, so the
legacy-migration cost is already locked in regardless of when this lands. HA — the feature this
infrastructure exists for — has no committed timeline. Merging in 2.9 would add two core
dependencies (SQLAlchemy, Alembic), a new operational surface (state DB, migration marker,
`nvflare-state-store-migrate`), and a maturity reset on the server's core state path, with no
user-facing benefit until HA ships.

**State of the branch:** Complete and heavily reviewed — three adversarial review rounds with
empirical reproductions, ~3.7k lines of tests, all known defects fixed. See the review record:

- `fable-review.md` — round 1 findings (15 verified issues in the original design)
- `fable-review-2.md` — round 2 findings (incl. 4 reproduced concurrency/bootstrap defects)
- `fable-changes.md` — full fix log across all rounds

(Review docs are kept alongside this branch; ask Peter Cnudde for copies if absent.)

The hard-won design knowledge — dual-store consistency invariants, SQLite write-lock behavior
(`BEGIN IMMEDIATE` writes-only), disabled-client cache epochs, the deployment bootstrap matrix
(POC / Docker / bare-metal / K8s / simulator / vault-mode / in-place upgrade), and the documented
Postgres READ COMMITTED races that full HA must still close (jobs→studies FK or SERIALIZABLE) —
lives in those documents and in the test suite.

**Revival triggers (revive ~2 quarters before the dependent release, to soak single-server first):**

1. HA gets a committed release target.
2. Job-listing scalability complaints (the legacy path reads every job's meta file from disk per
   scheduler pass / admin listing).
3. Study-state consistency tickets against the JSON registry.

**Expected re-landing cost:** rebase against multistudy-area churn plus one fresh review round —
estimate 30–40% of original effort.
