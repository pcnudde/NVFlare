# Slide 1 — StateStore branch: what it is, what it costs

## SQL-backed server state (`feat/control-store-stateful-ha`, candidate for 2.9)

Replaces filesystem-backed server state (job meta files, `study_registry.json`,
`disabled_clients.json`) with a SQL StateStore — SQLite by default, Postgres-capable.
**Purpose: foundation for HA / stateful multi-server.**

**Scope:** ~6.6k lines across 70 files (~2.9k production, ~3.7k tests/docs) ·
schema + Alembic migrations · legacy-migration CLI · startup bootstrap for every
deployment mode (POC / Docker / bare-metal / K8s / simulator)

| What we'd gain in 2.9 | What every user pays in 2.9 |
|---|---|
| Indexed job queries (legacy path re-reads every job's meta file from disk, every scheduler pass) | 2 new core dependencies for all installs, clients included (SQLAlchemy, Alembic) |
| Transactional job / study / submit-token state | New ops surface: state DB file, migration marker, migrate-CLI, upgrade step |
| SQL-queryable server state for operators | New failure classes: DB locked, schema mismatch, missing marker |
| Foundation in place when HA arrives | Maturity reset on the server's core state path (battle-tested → 3 weeks old) |

**The catch:** roughly half the production code exists just to make the new design as robust
as what it replaces (migration, bootstrap, dual-store consistency, caching, concurrency).
And it's still only the *first layer* of HA — leader election, failover semantics, and the
remaining cross-server races are ahead, and may reshape this schema.

---

# Slide 2 — Proposal: park it, with explicit revival triggers

## Why now is the wrong time

- **HA has no committed timeline** — infrastructure soak only pays off against a launch date.
- **The migration-debt argument expired:** multistudy GA'd in 2.8 on the JSON registry, so the
  legacy format already exists in the field. Landing later migrates the same formats.
- **The one standalone benefit** (job-listing scalability) would be a few-hundred-line targeted
  fix if it ever becomes a real complaint — not a 6.6k-line replacement.

## What parking looks like (the work is not lost)

- Branch pushed to `origin/feat/control-store-stateful-ha` with a `PARKED.md` status note.
- **Best-reviewed version of this design that will ever exist:** 3 adversarial review rounds,
  reproduced-and-fixed concurrency/bootstrap defects, ~3.7k lines of tests, full review record
  (dual-store invariants, SQLite locking, deployment bootstrap matrix, open Postgres races).

## Revival triggers — revive ~2 quarters *before* the dependent release (single-server soak first)

1. HA gets a committed release target
2. Job-listing scalability complaints from the field
3. Study-state consistency tickets against the JSON registry

**Known cost of deferring:** rebase against multistudy churn + one fresh review round
(~30–40% of original effort). We accept that as the price of not carrying unused
infrastructure risk through 2.9.

**Ask: agree to park; revisit at the first trigger.**
