# NVFlare Ephemeral-Key Architecture — Complete Picture and Required Changes

Status: draft, 2026-07-02; restructured around independent feature axes
2026-07-06.

Notation: **W1–W10** are work items (defined in §5; W8 was removed as
out of scope, the label is retired), **D1–D4** are decisions (§6; all
closed except D3); both are cited by label throughout.
Synthesizes and supersedes-as-index: `ephemeral_keys.md` (modes + external-CA
architecture), `fable-minimal-key-rotation.md` (recommended rotation posture),
`fable-key-rot.md` (full live-rotation design, kept as escalation appendix),
`sso-study.md` (cert-carried study entitlements).

## 1. Goal, and the shape of the change

**Goal:** make "external CA with ephemeral endpoint keys" the recommended
production posture of NVFlare.

This is deliberately **not a new operating mode**. It is a set of
independent features — axes with a baseline and an upgraded setting —
each adoptable on its own, most per site or per admin. The two endpoint
presets get names purely for convenience:

- **Baseline** (today's behavior, NVFlare 2.8) — the defaults on every
  axis: `nvflare provision` generates a project root CA plus every
  participant's private key and certificate, and distributes them in
  startup kits.
- **Target** (recommended production posture) — every axis upgraded:
  FLARE is not a CA and holds no private keys; endpoints generate their
  own keys, enroll against an external issuer (step-ca is the reference
  issuer, D4; any equivalent works through the same contract), and renew
  short-lived certificates without interrupting FL jobs.

| Axis | Baseline (today, 2.8) | Target | Enabled by | Adoptable per |
|---|---|---|---|---|
| Root origin | FLARE generates root | external ceremony | BYO root (**shipped**) → W1 | project |
| Root custody | root key in provisioning state | key never touches FLARE (custody ladder, §3.1) | W1 | project |
| Endpoint issuance | provisioning generates + distributes keys | local keygen + CSR: manual approve (**shipped**, `cert request/approve`) → automated against step-ca | W2 | site |
| Cert lifetime + renewal | ~10-year static | 30 days (D1), renewed live | W2 runtime | site |
| Admin credentials | kit-issued files | SSO-issued ephemeral | PR #4846 | admin |
| Authorization source | registry lists | + cert-carried entitlements | W5 | admin |
| Job identity | inherited site token | per-job certs, token retired | W3 (**prototyped**, §9) + W10 | federation (one capability window) |
| Key rotation cadence | never | ≤ 1 year, at restarts | D2 — pure ops policy | participant |

Two design rules make the axes compose:

1. **The runtime is posture-blind.** One invariant contract everywhere:
   FLARE processes read credentials from the startup paths and validate
   against `rootCA.pem`. Whether those files came from a provisioning
   kit, a manual approval zip, `step ca renew`, or a kubelet Secret
   mount is invisible to every FLARE process. (This is also why live
   renewal works: rotation never introduces a second trust path.)
2. **No posture flag.** Provisioning grows per-feature inputs (a root
   cert without its key; an issuance method per participant); the
   presets live in documentation and project templates, never in code —
   no `if mode == 2`-style branch may exist.

Dependencies between axes form a small DAG (§5), not a switch. Mixed
federations are first-class: a K8s site on automated issuance, a
single-VM site on manual approval, and a POC site on kit credentials
coexist under one root, because the server validates certificates, not
enrollment methods. Only two things are federation-global: the trust
anchor itself, and the job-identity/token axis (W10's deprecation
window).

**Non-goals:**

- Removing the baseline. It stays for POCs, dev/test, and small
  deployments that accept centralized key custody.
- Building issuance, attestation, or CA infrastructure into FLARE. Those stay
  external (step-ca, cert-manager, SPIRE, KMS/HSM).
- Live private-key rotation (restart-free key change). Deliberately excluded
  from the base posture; analyzed as decision D3 (§6), full design in
  Appendix A.
- Certificate revocation infrastructure (CRL/OCSP). Short cert lifetime is
  the containment mechanism.

## 2. Threat model — what changes and why

Baseline weaknesses this architecture removes:

| Weakness today | Fixed by |
|---|---|
| Provisioning workspace holds root CA key + every participant key — one high-value boundary | Root key non-exportable in KMS/HSM; endpoint keys generated locally, never transported |
| Long-lived root and endpoint credentials; renewal = reprovision + redistribute + restart | Short-lived certs renewed live; keys bounded, rotated at restarts |
| No eviction/containment short of reprovisioning everyone | Revocation-by-expiry: stop certifying, access dies within one cert lifetime — **requires W2's live-session expiry sweep**: TLS checks expiry only at handshake, and today an established connection survives its peer cert's expiry indefinitely |
| Admin credentials are long-lived files handed out by the project admin | Short-lived admin certs issued after SSO/OIDC auth |

Consciously accepted residual risks:

- **Undetected key theft** bounded by key lifetime (≤ 1 year, D2), not
  cert lifetime.
- **Recorded-traffic exposure:** the end-to-end cipher of F3 (FLARE's
  cellnet communication layer, which encrypts payloads inside the TLS
  envelope) has no forward secrecy, so a stolen key retroactively decrypts
  payloads recorded during its lifetime by any observer inside that
  envelope. The fix — forward secrecy in F3 — is an independent effort
  outside this design (sketch in `fable-minimal-key-rotation.md`); until
  it lands, key lifetime (D2) is the only bound.

## 3. Target architecture — four layers

### 3.1 Trust layer (external root)

```text
Root private key — non-exportable, isolated KMS/HSM, offline authorization
        │  signs (rare, audited ceremony)
        ▼
Issuing intermediate(s) — step-ca or equivalent online CA
        │  signs endpoint CSRs (short-lived leaves)
        ▼
Admin, server, client, and internal-listener certificates

FLARE receives and distributes ONLY: rootCA.pem (or trust bundle),
topology/config, expected identities, enrollment metadata.
```

Rules:

- Root key never signs endpoint certs or arbitrary content; root KMS
  authorization is unavailable to any NVFlare process or the online issuer.
- No root, intermediate, or endpoint private key ever enters provisioning
  state (`state/cert.json`).
- Root cert lifetime 10 years; routine renewal happens beneath it, so
  routine root rotation does not exist. End-of-life / compromise replacement
  is a planned rare event (rebootstrap, or one overlapping-root migration).
- **Custody is a ladder, not a requirement:** step-ca's encrypted
  file-backed root (small projects) → cloud KMS (recommended) → HSM
  (strict). Every other property of this design — local key generation,
  no keys in kits, short-lived certs, revocation-by-expiry — is unchanged
  by the rung chosen; even the bottom rung strictly improves on the
  baseline (the root at least never sits in provisioning state).

### 3.2 Identity & enrollment layer (who gets certs, how)

| Population | Auth to issuer | Key location | Cert lifetime | Status |
|---|---|---|---|---|
| Admins | interactive SSO/OIDC → step-ca | admin machine, generated locally | ~24 h (open question 2) | PR #4846 (open) |
| Servers / clients | machine identity (agent, init container, wrapper) → CSR | endpoint, generated locally | 30 days default (D1) | partial: `feat/external-workload-certs` branch |
| Internal listeners (a client with a listening host needs both its client identity and its listener server identity) | same workload path | endpoint | 30 days default (D1) | not designed (W3) |
| Job processes (per-job keys) | issued at job start by the site agent / parent, injected via the launch channel (§9) | site, delivered with the job payload | job lifetime | prototype (§9, W3) |
| External trainers (Client-API / CellPipe, e.g. Slurm-launched — today handed the site token) | per-job leaf (trainer purpose) read from the shared job workspace, which the trainer already mounts for code and task data | trainer node, from job workspace | job lifetime | not designed (W3, OQ12) |
| CC/HE (confidential computing / homomorphic encryption) startup content signing | dedicated content-signing service | signing service | n/a | not designed (W7) |

Admin and workload issuance must be purpose-constrained and enforced by
FLARE, not just by issuer configuration — details in §7 (work item W4).

Deliberately absent from the table: **relays** (out of scope for the
target posture; see §9) and the **HA overseer** (HA/overseer deployments
are out of scope for target v1 — if added later, the overseer is one
more workload-path row).

The enrollment agent contract is deliberately thin: it writes the renewed
chain + key to the startup-kit paths FLARE already reads
(`server.crt`/`server.key`, `client.crt`/`client.key`). No FLARE-specific
manifest or staging protocol — cert-manager, SPIRE, `step ca renew`, and K8s
projected secrets already produce exactly this.

### 3.3 Credential-lifecycle layer (rotation posture)

Key architectural decision: **split certificate renewal from key rotation.**

| Credential | Lifetime | Mechanism | Interruption |
|---|---|---|---|
| Certificate | 30 days default, per-project override (D1); renew at half-life | **live**: watch startup paths → validate → swap | none |
| Private key | ≤ 1 year (D2) | **controlled restart** (ride upgrade/patch windows) | a restart already being taken |

Architecture points (mechanism details in §8; a spike implementation
exists):

- The rotation signal is the credential files FLARE already reads at
  startup; the enrollment agent needs zero FLARE-specific glue.
- A single validation gate guarantees nothing invalid — torn write, wrong
  identity, or an unexpected *key* change — ever displaces a working
  credential. Live renewal is cert-only by design; a key change fails
  closed and requires a restart.
- Always on, zero configuration.
- Key rotation, when taken, is per-participant and uncoordinated: clients
  restart on their own schedule (FL churn tolerance is the drain), the
  server restart is a single-org ops scheduling problem. No
  cross-organization choreography exists anywhere.
- Emergency response to suspected compromise = the same restart procedure,
  immediately, for the affected participant only; the old cert's remaining
  lifetime is the containment bound.

### 3.4 Authorization layer (what certs carry)

Identity fields stay as today: CN = identity, O = org,
unstructuredName = FLARE role.

Additive extension (from `sso-study.md`): study entitlements in a dedicated
non-critical X.509 extension (enterprise OID, versioned JSON schema,
`studies: [...]`), OR'd with the registry `admins` list:

```text
study exists AND (registry.has_user(user, study) OR cert_claim.allows(study))
```

Registry keeps study existence, site enrollment, scheduling. Malformed
extension ⇒ reject the login (never silently ignore a signed claim). Audit
log records authorization source. Separate PR from ephemeral-cert
acquisition. Full spec, truth table, and acceptance criteria: `sso-study.md`.

**Review finding — ephemeral admin certs vs signed jobs (OQ13):** app/job
signatures are verified against the *submitter's* cert chain **at
verification time** (`verify_cert_chain(..., now)`,
`nvflare/lighter/utils.py`, called from
`nvflare/private/fed/utils/app_deployer.py`). A job signed by a ~24 h
admin cert therefore fails deployment verification for any site deploying
it later — late joiners, client restarts mid-job (redeploy), queued jobs.
`feat/require-signed-jobs-pki` makes this mandatory. Semantics must be
decided before PR #4846 + signed jobs coexist: verify-at-submission-time
(server-attested timestamp), job re-signing, or a longer-lived admin
*signing* identity distinct from the login cert.

## 4. Current state (repo, 2026-07-02)

| Piece | State |
|---|---|
| Configurable root CA validity (`root_valid_days`) | merged, PR #4848 — baseline knob only; root key still in provisioning state |
| BYO root certificate (`serialized_root_cert` + `root_private_key`) | shipped — external root *origin*, but root key still enters provisioning (see §4.1) |
| Distributed provisioning (`nvflare cert request/approve`) | shipped — local key custody for participants, manual human-approval enrollment (see §4.1) |
| Ephemeral admin certs via step-ca | PR #4846 **open** — initial SSO admin flow |
| Kits without FLARE-generated workload creds | `feat/external-workload-certs` branch — allows omission; **no** enrollment, renewal, or reload |
| Live cert renewal runtime | spike implementation |
| Per-job certs for job cells | **prototype** — branch `claude/upbeat-joliot-932cfa`: root-signed job CA in the server kit, SP issues at deploy; own design doc `per_job_certs_design.md` (see §9) |
| Trust-only provisioning | not started — seam exists in the BYO root path |
| Internal-listener enrollment | not started |
| Issuer policy enforcement (admin vs workload) | not started |
| Study entitlements extension | designed (`sso-study.md`), not started |

### 4.1 Relation to existing certificate features

**BYO root certificate** — `Project(serialized_root_cert, root_private_key)`
(`nvflare/lighter/entity.py`, consumed by CertBuilder in
`nvflare/lighter/impl/cert.py`; the Dashboard uses it and stores the root
key in its database). It changes the root's *origin*, not its *custody*:
the root private key must be supplied and is used by provisioning to sign
every participant cert — still the baseline custody model, and
incompatible with a KMS-held non-exportable root by construction. It is,
however, exactly the code seam W1 builds on: trust-only provisioning ≈
accept `serialized_root_cert` *without* `root_private_key`, skip
signing, require enrollment instead.
Today that combination is explicitly rejected.

**Distributed provisioning** — `nvflare cert request/approve`
(`docs/design/distributed_provisioning.md`). Already delivers the custody
half of the enrollment story: participants generate private keys locally;
only CSRs and signed certs travel. In this design's vocabulary it is
*manual, human-approval enrollment against FLARE-as-CA*. Gaps vs the
target:
the Project Admin still holds the root key (human-operated CA instead of
KMS + step-ca), certs are long-lived with no renewal path (re-request =
another manual approval), and identity is vouched by the approver rather
than SSO or machine identity. W2's external enrollment agent is the
automated, short-lived version of this flow, and its approval-metadata
concepts (expected identity, endpoint, scheme signed into `signed.json`)
largely prefigure the identity metadata W1 emits.

**Not related:** the `custom_ca_cert` network property
(`nvflare/fuel/f3/drivers/net_utils.py`) is a client-side CA override for
proxy / one-way-TLS validation only.

So the honest gap statement for W1/W2 is not "not started" but: extend
the BYO root path to cert-only input, and automate what
`cert request/approve` already does manually. Both shipped features are
intermediate rungs on the §1 axes — evidence the axis decomposition is
how this system already evolves, not a new doctrine.

## 5. Work items and dependencies

```text
W1 trust-only provisioning ──┬──> W2 workload enrollment+live renewal ──> W3 listeners/job keys
                             │                                  │
PR #4846 admin flow ─────────┼──> W5 study entitlements         └──> W6 failure behavior
                             │
                             └──> W4 issuer policy enforcement
W7 CC/HE content signing  (blocks W1 for CC/HE kits — see W7)
W9 adoption docs — new deployments first (after W1–W2)
W10 token retirement, baseline + target (after W3; deprecation window)
```

- **W1 — Trust-only provisioning.** `nvflare provision` accepts
  `rootCA.pem`/trust bundle as input; generates no CA, stores no private key;
  emits config, expected identities (CN/org/role/SAN/purpose), and enrollment
  metadata. Kits are non-runnable until the endpoint enrolls. Builds on the
  BYO root seam (§4.1: cert without key) and `feat/external-workload-certs`;
  reuse distributed provisioning's approval-metadata concepts for the
  identity metadata.
- **W2 — Workload enrollment + live certificate renewal.** External-agent
  contract (write to startup paths) + the §3.3 runtime (harden the spike).
  Conceptually: automate what `cert request/approve` does manually, against
  step-ca instead of a human approver (§4.1). Restart procedure documented
  as *the* key-rotation mechanism (D2: rotate, ≤ 1 year). Two deliverables
  found in review and now in scope: **live-session expiry enforcement**
  (server-side sweep disconnecting peers whose exchanged cert has expired —
  no auth-layer expiry check exists today, and without it
  revocation-by-expiry does not bind connected participants) and
  **renewal observability** (renewal status, expiry countdown, and the
  fail-closed "unexpected key change" alert exposed through FLARE's
  metrics/status surface — the single most load-bearing op in the target).
- **W3 — Internal listeners and per-job keys.** Per-job certs are
  prototyped (§9); remaining: CSR mode (site-local key generation),
  chaining or replacing the job CA under the external issuer, stopping
  site keys from shipping into job workspaces, sub-worker / Client-API
  subprocess cells — including CellPipe, which today carries the site
  token into training subprocesses (`nvflare/fuel/utils/pipe/cell_pipe.py`)
  — **job-cert renewal for jobs that outlive their cert** (prototype
  issues fixed ~30 d with no renewal; an expired job cert strands the job
  cell at its next reconnect — SP re-issuing into the run dir rides the
  §8 watcher contract), and every internal-listener certificate
  (dual-identity clients per-role-pair). Token retirement is W10.
- **W4 — Issuer policy definition and enforcement.** See §7.
- **W5 — Study entitlements extension.** Per `sso-study.md`; needs OID
  allocation, encoding, limits, malformed-claim scope decided first.
- **W6 — Failure behavior.** Define semantics for: CA unavailable at
  enrollment/renewal, expired cached credentials at startup, partial renewal
  (some participants renewed, some not), emergency intermediate compromise
  (issuer allowlisting / trust-bundle update path).
- **W7 — CC/HE content-signing identity.** Dedicated external
  content-signing cert + service; never a CA key. **Not independent
  (review finding):** `signature.json` is generated only for CC/HE kits
  and is signed with the root private key
  (`nvflare/lighter/impl/signature.py`) — trust-only provisioning (W1)
  has no root key, so CC/HE deployments cannot provision in the target
  posture until W7 lands. Plain kits are unaffected (no signature.json;
  mTLS is the anchor). Either W7 joins the release or the release states
  "CC/HE stays baseline". CC composition and the CoCo wave: §11.
- **W9 — Adoption documentation (new deployments first).** The target is
  officially supported for *new* deployments: provision with trust-only
  input and enroll from day one. Existing baseline deployments get a
  **re-bootstrap checklist** (new root, new kits, one coordinated
  restart), not migration tooling — overlapping-root migration
  (two-root trust bundles, re-enroll under the new root) is deliberately
  not built; it remains a documented escalation should a deployment be
  unable to take the outage. Note the per-admin axes (SSO admin certs,
  study entitlements) are additive and need no migration support at all:
  a baseline deployment can adopt them in place today. Deliverables: a
  user-guide chapter for the target posture (the §1 axis table is the
  outline), per-deployment-model how-tos with copy-paste configs
  (systemd timer, compose sidecar, cert-manager manifests — §10/§10.1),
  a project-operator step-ca guide (root-ceremony ladder, provisioners,
  OIDC, one-time tokens), the adoption/re-bootstrap guide, and token
  deprecation release notes.
- **W10 — Token retirement, baseline and target.** Replace token auth with
  cert-bound message auth (origin-FQCN ⊆ authenticated connection
  identity), require mTLS or local-only transport on internal listeners,
  keep the token branch as a legacy path for one deprecation window, then
  delete `sign_auth_token`/`TokenVerifier` and the `-t`/`-ts`/`-d` args.
  Details §9.

### 5.1 Release packaging — land in one release

Target release: **2.10 (Q4 2026)**. Everything ships together except
two items whose structure forbids it.

| In the release | Deferred, with reason |
|---|---|
| W1 trust-only provisioning | W10 phase 2 (delete token code) — the deprecation window structurally spans two releases: ship new path + warning in 2.10, delete in 2.11 |
| W2 enrollment contract + hardened renewal spike, incl. expiry sweep + observability | W7 CC/HE content signing — deferring means CC/HE deployments stay baseline this release (W7 blocks W1 for them) |
| W3 per-job certs (land the prototype; CSR mode as follow-up) | D3 escalations — already build-on-trigger only |
| W4 issuer policy enforcement | overlapping-root migration — dropped per W9 (re-bootstrap only) |
| W5 study entitlements + PR #4846 merge | CC1–CC4 CoCo wave (§11.4) — rides after W7 |
| W10 phase 1: cert-bound auth ships, token path deprecated | |
| W6 as fail-closed defaults + documented failure semantics | |
| W9 documentation workstream | |

D1 (30-day certs) and D2 (rotate keys ≤ 1 year at restarts) are closed
(§6), so the documentation workstream is unblocked; D3 stays open
(escalation-only) and does not block anything.

## 6. Decisions

D1, D2, and D4 are closed; D3 remains open (escalation-only, does not
block the release). Closed decisions first, the open one last.

### D1 — Closed 2026-07-06: certificate lifetime 30 days

Thirty-day certificates, renewed at half-life, as the project default; a
per-project override (down to ~7 days) for deployments that want tighter
containment. Rationale: with live renewal always on, renewal cost is
identical at any lifetime, so the deciding axis is operational slack —
a ~15-day buffer between a failed renewal and an outage tolerates CA
downtime, holidays, and flaky site links, which fits the mixed-skill
federations this design targets (§10.1). The accepted cost: the
revocation-by-expiry containment window for a stolen cert or evicted
participant is up to a month (still bounded, vs years today); projects
with stricter containment needs take the shorter override.

### D2 — Closed 2026-07-06: keys rotate, ≤ 1 year, at restarts

Private keys are bounded at ≤ 1 year and rotated at the controlled
restarts that upgrades and patching already force; the same procedure is
the rehearsed emergency path for detected compromise. Rationale: bounds
silent impersonation and recorded-traffic exposure (F3 has no forward
secrecy today), passes cryptoperiod policy review without argument, and
costs only issuer key-reuse configuration (§8) plus restarts already
being taken. Consequence: issuers must be configured for key reuse on
routine renewals; an unexpected key change fails closed and pages.

### D4 — Closed 2026-07-06: step-ca is the reference issuer; the contract stays issuer-neutral

step-ca is what we document, template, and test against — it is never a
code dependency. The runtime contract (§8: read credential files,
validate against rootCA) means FLARE never talks to a CA; anything
step-specific lives in docs, examples, and provisioner templates only
(PR #4846 must respect this boundary too). Rationale: one OSS Apache-2.0
binary covers every §3.2 population — OIDC for admin SSO, AWS/GCP/Azure
instance-identity provisioners for workload bootstrap, JWK one-time
tokens for mixed-skill sites — its `renew`/`rekey` semantics match
D1/D2 (same-key renewal is its default), and its KMS backends are the
§3.1 custody ladder natively. Alternatives (Vault PKI, cloud CAs, SPIRE,
EJBCA) still work through the same file contract for deployments that
have them; they are not documented or tested. Risk noted: Smallstep's
commercial tier — everything needed is OSS today; pin tested versions in
the W9 docs.

### D3 — Server key rotation without a job-killing window: which shape?

(Process abbreviations for this section: SP/CP = server/client parent
control process; SJ/CJ = server/client job process, spawned per job by its
parent.)

This is an escalation decision — none of it is base-posture work. The
trigger: a deployment with continuous back-to-back jobs, no workable
snapshot/recovery, *and* a policy demanding frequent key rotation.
Clients are never the trigger — they are settled regardless of this
decision: plain restart, accept the in-flight task loss; FL algorithms
tolerate site churn by design, and only that site's current task is
redone. Three server-side options:

**Option 1 — live key rotation** (server-scoped subset of
`fable-key-rot.md`): swap the key in-process; grace window, connection
recycle, session pinning.

**Option 2 — job-surviving restart**: let SP restart while running SJ/CJ
processes survive and are re-adopted. Today four things prevent this, all
deliberate design rather than accident:

1. **Self-termination**: CJ/SJ monitor the parent PID and stop themselves
   within ~1 s of parent exit (`nvflare/private/fed/app/utils.py:45`,
   `monitor_parent_process`) — orphan cleanup that would become a bounded
   reconnect-grace window.
2. **Parent amnesia**: the running-job registry (`engine.run_processes`)
   is in-memory only, and the client resync path *aborts* jobs the
   restarted server does not recognize
   (`nvflare/private/fed/server/fed_server.py` `_sync_client_jobs`).
   Needs a persisted/rediscovered run registry and adopt-instead-of-abort
   sync.
3. **Token signatures**: job cells present launch-time tokens signed with
   the parent's private key; same-key restart verifies fine, a key-change
   restart needs token re-issuance on reconnect (job cells hold site
   certs, so cryptographic re-auth works — no session pinning needed).
4. Existing snapshot recovery *relaunches* SJ from component state; it is
   a fallback, not the mechanism.

This decomposes into two stages: **(a)** same-key restart survivability
(grace window + run registry + adoption sync) — valuable on its own for
parent upgrades and crash recovery; **(b)** key-change restart, adding
token re-issuance.

**Option 3 — server blue-green drain**: start a new SP with the new key;
all new jobs go to it; the old SP accepts no new jobs and lives until its
last job finishes, then is retired with its key. No job is ever
interrupted. The crux is the drain overlap: old jobs still need clients,
so sites must talk to both SPs until the drain ends. Two sub-shapes:

- *dual-root cells* — one CP holds connections to both SPs with
  job-affinity routing; real new cellnet code; or
- *dual CP per site* — each site briefly runs a second CP pointed at the
  new SP (same site cert/key, different endpoint); two disjoint
  federations, zero cellnet changes; cost is site-local resource
  partitioning plus an endpoint switch (client kits bake in the SP
  endpoint; the HA machinery — SP endpoint lists, overseer — is existing
  scaffolding for multi-SP awareness).

Structural costs regardless of sub-shape: the old key stays live and
certified until the last old job ends, so detected compromise still means
"terminate old SP now, lose its jobs"; live cert renewal (§3.3) is
load-bearing during long drains (the old SP's certs must keep renewing on
the old key); new-job capacity is split while sites drain unless they run
dual CP; and the old SP needs a "refuse new jobs" mode.

| | 1: live key rotation | 2: job-surviving restart | 3: blue-green drain |
|---|---|---|---|
| Pro | No process restart at all; no persistence or adoption protocol; design fully specified in `fable-key-rot.md`. | Mechanisms exercised on every ordinary restart and crash — no rarely-run path rot. Also fixes parent upgrades/crashes killing jobs. One story: restart is the only rotation mechanism, doubles as the emergency path. Rotation is instant. | Zero interruption to running jobs. Smallest FLARE-code footprint (dual-CP shape needs almost none). No delicate liveness/persistence/adoption changes. Rehearses the same shape as root replacement (§3.1). |
| Con | Machinery runs only during rotations — classic rarely-exercised path. Grace window, recycle, session pinning exist solely to avoid a restart. Does nothing for upgrades/crashes. | Touches delicate code: liveness detection, run-registry persistence, resync/adoption semantics. Not obviously cheaper than option 1. Brief admin/registration blip. | Rotation latency is job-bounded, not restart-bounded — old key stays valid until the longest job ends (emergency path degrades to job loss, same as today). Two SPs' infra + endpoint management during drain. Capacity split or dual-CP burden on sites. |

Recommendation: option 3 (dual-CP shape) is the cheapest way to satisfy a
no-window deployment and needs no delicate FLARE changes — prefer it as
the first escalation. Option 2 stage (a) remains independently attractive
for upgrade/crash resilience and can be pursued on its own merits; option
1 (full design in `fable-key-rot.md`, Appendix A) only if rotation
latency must be seconds rather than job-bounded. Build none until a real
deployment hits the trigger.

(A fourth shape — generation-scoped blue-green, rotating the whole
federation as a closed mesh — is rejected outright: zero FLARE code, but
a coordinated deploy across autonomous organizations and job-bounded
rotation latency. Noted because live cert renewal is load-bearing for
it: a draining generation must keep renewing same-key certs for the
length of its longest job.)

## 7. Detail: issuer policy enforcement (W4)

Admin and workload issuance must be purpose-constrained — distinct EKU,
identity fields, SANs, roles, and allowed issuer chains per population.
Separate step-ca provisioners or templates alone are not sufficient
protection after an intermediate key compromise; the constraints must be
cryptographically or explicitly enforced by FLARE at every trust decision:

- an admin certificate must never be accepted where a workload certificate
  is expected, and vice versa;
- SANs and CN must match the expected identity metadata that provisioning
  emitted for that participant;
- role and organization fields are validated against the project
  configuration, not merely read;
- allowed issuer chains can be allowlisted per population so a compromised
  or misconfigured intermediate cannot mint credentials for the other
  population.

Open sub-question: one shared validator (extend
`nvflare/fuel/sec/admin_cert.py`) vs per-population validators.

## 8. Detail: live certificate renewal mechanism (spike)

The cert-only subset of `fable-key-rot.md`, as specified in
`fable-minimal-key-rotation.md`:

1. **Watcher** per process: poll the startup-config cert/key paths,
   SHA-256(cert‖key) as generation id, dedupe vs active + last-failed.
   Startup load *is* activation — no special path for new/job processes.
2. **Validation gate**: key-matches-leaf, chain to active trust bundle,
   CN/SAN/EKU/lifetime checks, **and public key must equal the active
   public key**. A key change is staged-not-activated: reject, log, alert
   "restart required". Torn writes just fail validation and retry.
3. **Atomic swap** — one immutable snapshot per (cert_path, key_path);
   single slot, no grace window (key never changes live).
4. **Listener refresh** — live `ssl.SSLContext` objects (tcp/http/websocket;
   http is the provisioning default) reloaded in place; client contexts
   rebuilt per connect. Established connections untouched. gRPC live reload
   deliberately omitted — `scheme: grpc` deployments restart before expiry.
5. Fixed defaults (poll ~2 s, `min_remaining_time` 900 s); no configuration.
   Status: fingerprint, expiry, last result, loud "key change staged —
   restart required" state.
6. **Expiry sweep (W2, review finding)**: TLS validates expiry only at
   handshake and nothing at the auth/message layer checks
   `not_valid_after` (`nvflare/private/fed/authenticator.py` delegates to
   SSL), so an established connection outlives its peer cert. The server
   periodically re-checks exchanged peer certs and disconnects expired
   peers; the peer's reconnect then fails cleanly at handshake. Without
   this, eviction only bites at the next network flap.

Issuer sharp edge (live now that D2 closed as rotate): most issuers
rotate the key on every renewal by default (SPIRE always; cert-manager
`rotationPolicy: Always`; certbot without `--reuse-key`). Deployments must
configure key reuse; an unexpected key change fails closed and must page —
the cert marches toward expiry with no valid replacement until the issuer
is fixed or a restart is taken.

## 9. Detail: job-process token retirement (W3)

Today every job process is handed a bearer token pair at launch
(`JobProcessArgs.AUTH_TOKEN` / `TOKEN_SIGNATURE`,
`nvflare/private/fed/client/client_executor.py`): CJ receives the CP's
**site-wide registration token** (uuid4, no expiry, no revocation short
of client deregistration or server restart); SJ receives
`token = job_id`. Both are signed with the server's private key
(`sign_auth_token`, `nvflare/private/fed/server/fed_server.py`), and
every post-registration message is authenticated by this header pair
(`validate_auth_headers`, `nvflare/private/fed/authenticator.py`).

Why it is dangerous:

- **Pure bearer after registration.** Site-key possession is proven only
  at registration; afterwards a captured token+signature is replayable
  from any host that can reach the server.
- **Not bound to the TLS peer.** On message paths, `client_name` resolves
  from the server's token table, not the connection cert CN — so *any*
  federation cert plus a stolen token impersonates that client
  cross-site. Internal listeners default to CLEAR, where the token is the
  only gate.
- **Broad exposure surface.** argv (`ps`, `/proc/*/cmdline`, any local
  user) with ProcessJobLauncher; `docker inspect` with the Docker
  launcher; pod-spec args with the K8s launcher; and a file on disk for
  external launchers (Slurm) — a site-scoped, non-expiring credential
  persisted on a shared filesystem.

With per-job keys (W3) the token contract is **retired, not replaced**:

| Token role today | Replacement |
|---|---|
| Message-level identity in the routed mesh | per-job cert; proof of possession, not a replayable header |
| Job scoping (SJ token = job_id) | job-id claim/SAN in the per-job cert, validated per W4 |
| Launch authorization (server-signed token) | the cert-issuance act itself |
| Bootstrap | the launch channel (below) |

Issuance path — **launch-channel injection**: the site agent / parent
obtains the per-job key+cert before spawn and delivers it with the rest
of the job payload (workspace file, K8s secret mount). No enrollment
step runs inside the job, so no bootstrap secret exists: the launch
channel already delivers the code the job executes, so a job-lifetime
key adds no new trust assumption to it. The same delivery covers
**externally launched trainers** (Client-API / CellPipe, e.g. a
Slurm-launched AV trainer): the shared job workspace the trainer already
mounts *is* its launch channel — it reads the per-job leaf from there
and does mTLS to the (W10-hardened) CP/CJ listener, replacing the site
token CellPipe hands it today. The CellPipe pairing label in the cell
name is just a name and is unaffected. The alternative —
self-enrollment inside the job (in-memory key + CSR, authenticated by a
one-time enrollment token / K8s service account / munge) — keeps the key
off the launch channel entirely but adds a CSR round-trip and an
issuer-availability dependency to every job start (a W6 failure mode).
It is an optional hardening, not the base design.

**Prototype** (branch `claude/upbeat-joliot-932cfa`, design doc
`per_job_certs_design.md` there): a root-signed **job CA** (CA:TRUE,
pathlen:0, marker extension) lands in the *server* kit only; SP issues
per-job leaves at deploy time (CN = site name plus a job-ID extension,
~30 d clamped to the job CA's expiry); SJ's credential is written into
the job run dir, CJ's is pushed per-site on the existing mTLS deploy
channel and written into the run dir — i.e. exactly the launch-channel
delivery above (detached launchers that bundle the run dir, like the
K8s no-shared-PVC transfer, inherit it for free). Site-scope replay is
blocked twice: any cert carrying the job-ID extension is rejected at
site-scope identity assertion, and any chain containing the marked job
CA is rejected — a stolen job CA key's blast radius is job cells only.
Old clients fall back to site certs.

Deltas from the target above — all named in the prototype's own future
work: CJ keys are born at SP and transit the deploy channel (its CSR
mode moves generation site-local); the job CA is a FLARE-held CA key —
a deliberate, marker-bounded exception to the no-CA-keys rule, which
the target either accepts as-is (job scope only) or replaces by chaining
the job CA under the external issuer; and site keys still ship into job
workspaces in this phase — the §10 isolation win and W10 token
retirement land only once job cells stop needing them.

Concretely: `JobProcessArgs` drops `-t`/`-ts`/`-d` (SSID is unverified
metadata today); `validate_auth_headers` validates cert-derived identity
plus job claim on job-cell traffic; `sign_auth_token` collapses into
cert issuance. The CP registration token survives only as a CP↔SP
session handle — a candidate for the same retirement once identity is
cert-bound end to end.

**Removing the token code entirely — baseline and target alike.** Token
authentication is not target-specific, and per-job keys are not a
prerequisite for killing it: every participant holds a cert in the
baseline too. The replacement is one rule — a message's claimed origin
FQCN must match, or be a child of, the mTLS-authenticated identity of
the connection it arrived on (a cell vouches only for its own subtree).
Baseline job cells keep authenticating with the site cert they already
possess (the kit is already shared with them); per-job keys stay a
target-posture feature, motivated by custody and user-code isolation
(§10), not by token removal. Requirements and caveats:

- **Every hop must be authenticated once tokens are gone.** Internal
  listeners default to CLEAR and are network-reachable in Docker/K8s;
  they must move to mTLS (listener certs exist in both postures) or be
  restricted to genuinely local transport.
- **Relay deployments are out of scope.** Relays exist in the code (the
  relay hierarchy connects via internal listeners,
  `nvflare/lighter/impl/cert.py`) and forward traffic for other cells,
  so the subtree rule structurally does not apply to them. Deployments
  using relays keep the token legacy path until a relay-aware design
  exists — stated explicitly rather than silently.
- **Migration:** old clients speak token auth, so the token branch
  survives one deprecation window behind a capability check, then dies.
  The honest intermediate state is "legacy path", not "removed".
- **Edge-device paths** have their own cookie/report scheme and are out
  of scope here.

What survives afterwards: a server-side registration record keyed by
cert identity (duplicate-login detection, heartbeat liveness) containing
no secret; admin HCI session tokens (a separate, in-memory,
cert-gated-login mechanism); nothing else. `ssid` is unverified HA
metadata and drops out of job args regardless.

Interim mitigations (worth taking before W3): scope the CJ token to the
job — sign `client_name + job_id`, exactly as SJ already does — so a
leaked launch file burns one job rather than the site; and pass tokens
via env/stdin with delete-after-read instead of argv and launch files.

## 10. The target across deployment models

The job process runs user code (training scripts, custom components),
so the discriminator between job launchers is what the launch channel
lets that code see. Today all three expose the site's long-lived
credentials to it: in-process shares the filesystem, the Docker launcher
bind-mounts the startup dir read-only into the container, and the K8s
launcher mounts the startup kit as a Secret into the job pod. Per-job
keys (W3) end that — where a boundary exists to enforce it.

- **In-process (ProcessJobLauncher).** Host enrollment agent
  (`step ca renew`, systemd timer) renews the site cert at the startup
  paths; CP writes the per-job cert into the job workspace. The per-job
  key is cosmetic here — CJ runs as the same user on the same host and
  can read the site key anyway. Production fit: single-tenant sites
  whose training code is as trusted as FLARE itself.
- **Docker.** Same host agent for CP. The job container's volume carries
  only rootCA + the per-job cert — the site-credential mount disappears,
  so compromised training code burns one job for one job's lifetime.
  Production fit: single-host sites wanting code/credential isolation.
- **K8s — the reference deployment.** The ecosystem already provides
  every ingredient of the target: cert-manager or SPIRE against a step-ca
  issuer; the site cert as a `Certificate` whose Secret is RBAC-scoped
  to the CP pod; per-job certs as short-lived per-job Secrets mounted
  only into that job pod; trust anchor via trust-manager; and kubelet
  Secret auto-update is exactly the file-change signal the live-renewal
  watcher (§8) consumes — renewal needs zero extra plumbing. FLARE's
  remaining job: read certs from paths, validate job claims (W4), keep
  secrets out of pod-spec args.

| | In-process | Docker | K8s |
|---|---|---|---|
| **Site cert/key custody** | host FS, startup paths | host FS; **not** mounted into job containers | Secret, RBAC-scoped to CP pod |
| **Enrollment/renewal agent** | host agent (`step ca renew`, systemd) | host agent | cert-manager / SPIRE + step-ca issuer |
| **Renewal signal** | file watch (§8) | file watch | kubelet Secret auto-update → same file watch |
| **Per-job key delivery** | job workspace file | job-only volume mount | per-job short-lived Secret, mounted only into that pod |
| **What user code sees** | everything (same user/host) | rootCA + per-job cert only | rootCA + per-job cert only |
| **Tokens after W3** | none | none | none |
| **Key-rotation restart** | process restart | container restart | pod restart / rollout |
| **Residual risk** | job code can read site key — trust required | Docker daemon/root on host | K8s API RBAC, etcd at-rest encryption |
| **Production fit (target)** | single-tenant trusted sites only | single-host sites needing isolation | **reference deployment** |

One line: the target is deployment-model-agnostic for SP/CP — a
file-based contract everywhere — and the models differ only in how well
the launch
channel isolates user code from credentials: none (in-process),
container boundary (Docker), RBAC-scoped Secrets with free rotation
plumbing (K8s).

### 10.1 Minimal-footprint deployments (mixed-skill federations)

The infrastructure cost of the target is asymmetric by design: nearly
all of it lands on the project operator, once; the per-site footprint is
one static binary and one timer.

| Component | Who | What it concretely is | Effort |
|---|---|---|---|
| Online CA (step-ca) | project operator, once per federation | 1 small container/VM with a DNS name, reachable outbound-HTTPS from sites | hours, once |
| Root custody | project operator | the §3.1 ladder: encrypted file → cloud KMS → HSM | config choice at ceremony |
| Admin SSO | the org's existing IdP wired to step-ca's OIDC provisioner | config once | — |
| Site enrollment agent | each site | `step` CLI + systemd timer / cron / sidecar running `step ca renew` | minutes; ship it in the kit |
| Per-site CA / KMS / K8s | — | **none required** | — |

**Single cloud VM site:** one-time enrollment from the kit instructions
(`step ca bootstrap` with the CA fingerprint, then
`step ca certificate <site> client.crt client.key --token <one-time
token>` from the project admin), plus a systemd timer rewriting the same
files at half-life — the §8 watcher does the rest, zero FLARE config.
Outbound 443 only. If the VM dies, recovery is *simpler* than baseline:
issue a new enrollment token and re-enroll; no reprovisioning or kit
redistribution.

**Single on-prem Docker host:** CP container plus a host cron or a
`step-cli` sidecar sharing the startup volume. Job containers get only
rootCA + per-job cert (§10) — most relevant exactly here, the
"hospital box running someone else's training code" case.

**Why mixed skill composes:** the enrollment contract is the lowest
common denominator — *files appear at the startup paths*. One federation
can hold a K8s site on cert-manager, a VM site on cron, and a POC site
on kit credentials; the issuer neither knows nor cares. The skill-heavy
pieces sit with the party that already runs the server, dashboard, and
provisioning.

Caveats, eyes open: step-ca is a renewal-time dependency — the ~15-day
renewal buffer at the 30-day default (D1) is precisely the tolerance
for it being down or a site's link flaking. A project that cannot
operate even one CA container stays on the baseline preset; W9 is the
path up when it matures.

## 11. Confidential computing

Reference: `docs/user_guide/confidential_computing/` (notably
`on_premises/cc_architecture.rst`, the NVIDIA CC architecture for FLARE).

### 11.1 What FLARE has today

Two attestation stages, split between infrastructure and application:

- **Boot stage (infrastructure, outside FLARE code):** a locked-down CVM
  image (no shell, no open ports, `dm-crypt` + `dm-verity` partitions)
  whose chain of trust runs hardware → kernel → InitApp. Disk decryption
  keys live in a **CoCo Trustee** service (Attestation Service + Key
  Broker Service); the LUKS key is released only when boot measurements
  match the KBS resource policy (rego). Trustee is explicitly swappable
  infrastructure. FLARE starts only after this succeeds.
- **Runtime stage (FLARE `CCManager`,
  `nvflare/app_opt/confidential_computing/`):** mutual attestation woven
  into the FL lifecycle — clients and server exchange fresh,
  nonce-protected attestation tokens at registration, all sites are
  cross-verified before a job schedules, and a background thread
  re-verifies every `check_frequency` seconds. Any failure is
  fail-closed: system shutdown, not degradation. Verifier plugins:
  Intel TDX, AMD SEV-SNP, Azure CVM (MAA), Azure Confidential Containers
  (ACI), and **NVIDIA GPU attestation** via NRAS
  (`gpu_authorizer.py`, `nv_attestation_sdk`).
- **Provisioning:** per-participant `cc_config` YAML (compute env,
  CPU/GPU mechanism, issuers/verifiers, `class_allow_list`), CVM image
  packaging, a locked-down authorization template (no BYOC, no shell, no
  job download), and `signature.json` over kit/workspace content —
  signed with the **root private key**, which is the W7×W1 collision
  already flagged in §5.

Known limits today: CC is all-or-nothing (no per-job CC requirement, no
partial attestation), attestation status has no observability surface,
and the runtime is whole-CVM — no confidential-container (CoCo/Kata) pod
support.

### 11.2 The trust-model shift CC introduces

Everything before this chapter trusts the site host: §9's launch-channel
argument rests on it explicitly ("the channel already delivers the code
the job executes"), and §10 stores credentials in host filesystems and
K8s Secrets. CC removes the host from the TCB — host admins, the
hypervisor, and the K8s control plane become adversaries. Consequences:

- Plain-file / K8s-Secret credential delivery violates the CC threat
  model (the host can read both).
- §9's "optional hardening" — in-TEE key generation, or release gated on
  attestation — is **the mandatory CC path**. The two issuance shapes
  map exactly to the two threat models: trusted host → launch-channel
  injection; untrusted host → KBS-released or TEE-generated keys.
- The custody story improves past the §3.1 ladder's top rung: a key
  generated inside a TEE and sealed to it never exists on any storage an
  attacker can read, and dies with the TEE.

### 11.3 Composition with the six features

| Feature | With CC |
|---|---|
| 1. SSO admin certs | Orthogonal — admins are outside TEEs. CC's locked-down authorization template stacks on top. |
| 2. Ephemeral workload certs | **Bootstrap identity becomes attestation evidence**: the enrollment secret (or the site key itself) sits in KBS and is released only to an attested CVM; renewal/rekey then proceed per §8 unchanged. Certs and attestation stay complementary: the cert answers *who*, the CCManager token answers *what environment, right now* — issuance is gated on attestation, freshness stays with CCManager. |
| 3. Per-job keys | Delivery upgrades from launch channel to a per-job **KBS resource** (rego-scoped) or in-TEE CSR; a job cert exists only for an attested job environment. Also closes the last token exposure that CC alone never fixed: CC job processes today still carry the site token. |
| 4. Trust-only provisioning + KMS | Composes; and CC makes W7 mandatory (content signing needs an identity that is not the root key). Measured boot covers the base image; content signing covers job/workspace content on top. |
| 5. Key rotation at restarts | Composes best of all: a CVM restart re-runs boot attestation, and TEE-sealed keys make D2's residual risk (undetected key theft) require TEE compromise rather than host compromise. |
| 6. Study entitlements | Orthogonal. |

W10 token retirement applies within CC unchanged — the subtree rule does
not care whether the peer is a TEE.

### 11.4 What CoCo-runtime support adds (post-release wave, after W7)

Today's CC is whole-CVM images; CoCo (Confidential Containers: Kata +
TEE pods, the same CNCF project whose Trustee FLARE already uses) brings
CC to the §10 K8s reference deployment. Work items, labeled CC1–CC4 to
avoid renumbering W:

- **CC1 — Attestation-gated enrollment and key release.** Site and
  per-job keys / enrollment tokens as KBS resources behind resource
  policies; wire the §3.2 bootstrap column to attestation evidence.
  Trustee — already the boot-stage attestation and key-release service
  in FLARE's CC architecture — is the composition point.
- **CC2 — Per-job CC policy.** Jobs declare "attested sites only";
  scheduler treats attestation as a resource; partial-attestation
  semantics replace today's all-or-nothing shutdown.
- **CC3 — CoCo/Kata as a fourth §10 column.** Confidential pods for
  CP/CJ; signed/encrypted images; KBS replaces K8s Secrets for
  credential delivery — the K8s API and etcd drop out of the §10
  residual-risk row entirely.
- **CC4 — Attestation observability.** Attestation status joins W2's
  renewal observability as one ops surface, replacing shutdown-is-the-
  only-signal.

### 11.5 One consistency rule

Identity and attestation remain separate concerns, composed at exactly
two points: **issuance/release gating** (a credential comes into
existence only for an attested environment) and **runtime freshness**
(CCManager keeps verifying the environment while credentials stay
valid). Do not encode attestation claims into long-lived certs — they
go stale the moment the environment degrades; short-lived per-job certs
are the only defensible carrier, and that is an open question (OQ14).

## 12. End state, one job per layer

| Layer | Job |
|---|---|
| External root in KMS + step-ca issuance | No FLARE key custody; identity from real IdP / machine identity |
| Short-lived certs, live renewal | Eviction, policy agility, post-detection containment |
| Bounded keys, rotated at restarts (per D2) | Silent-impersonation bound, cryptoperiod hygiene |
| Cert-carried entitlements | Authorization from the issuer/IdP, additive to registry |

## Appendix A — full live-rotation design

`fable-key-rot.md`: two-slot CredentialStore with grace deadline,
connection recycle, bearer-token session pinning, per-driver refresh
contract, no-peer-coordination rationale. Kept as the escalation blueprint;
its cert-only subset is exactly the §8 mechanism.

## Appendix B — source docs

- `ephemeral_keys.md` — mode definitions, external-CA architecture, security
  requirements, original follow-up list.
- `fable-minimal-key-rotation.md` — recommended posture, HA analysis, issuer
  key-reuse edge, FS design sketch.
- `fable-key-rot.md` — full live rotation.
- `sso-study.md` — study-entitlement extension implementation spec.

## Open questions

1. W1 kit shape: does a not-yet-enrolled kit fail at startup with a clear
   "enroll first" error, or block-and-wait for credentials to appear?
2. Admin cert lifetime: 24 h (sso-study assumption) vs shorter; and does the
   admin console need live renewal or is per-login issuance enough? Note:
   console sessions (30-min idle timeout, `nvflare/fuel/hci/server/sess.py`)
   can straddle cert expiry mid-session — define whether an active session
   survives its login cert.
3. gRPC deployments: accept "restart before expiry" permanently, or schedule
   the fetcher-based reload as a W2 follow-on?
4. W4 enforcement point: one shared validator (extend
   `nvflare/fuel/sec/admin_cert.py`) or per-population validators?
5. Issuer allowlisting for intermediate-compromise containment (W6): trust
   bundle update mechanism — file-based like cert renewal, or admin command?
6. Does PR #4846 need changes to align with W4 policy enforcement before
   merge, or retrofit after?
7. Where does this doc land in-repo — stays at `docs/design/` next to
   `multistudy.md`?
8. D3 option 2 stage (a) — job-surviving same-key restart — is valuable
   independent of ephemeral keys (parent upgrades, crash recovery). Pursue
   it as its own roadmap item now, or keep it parked behind the escalation
   trigger?
9. Verify the internal-listener assumption in D3 option 2: cellnet peer
   validation is against rootCA (not a pinned leaf), so a restarted
   parent's new listener cert with the same CN should handshake — confirm
   in `nvflare/fuel/f3` before relying on it.
10. D3 option 3 mechanics: how does the old SP refuse new jobs (scheduler
    pause vs admin convention), and how do admin CLI and sites address the
    new SP endpoint — reuse the HA SP-list/overseer machinery or plain
    config switch at restart?
11. Job CA in SP (per-job cert prototype, §9): accept as a permanent,
    marker-bounded exception to the no-CA-keys-in-FLARE rule, or chain /
    replace it under the external issuer in the target? Related: HA with
    multiple SPs — per-server-kit job CA or one shared job CA identity?
12. Sub-worker cells (multi-GPU `sub_worker_process`, Client-API
    subprocesses): per-job credential or internal links only? Must cover
    CellPipe/TaskExchanger, which today receives the site token
    (`nvflare/fuel/utils/pipe/cell_pipe.py`). Proposed (§9): per-job leaf
    with a trainer purpose, delivered via the shared job workspace the
    trainer already mounts; the trainer↔CP/CJ hop moves to mTLS.
13. **Release-gating (review finding):** signed-job verification checks
    the submitter chain at verification time (§3.4), which breaks under
    ~24 h admin certs for any deploy later than the cert — decide
    verify-at-submission semantics, job re-signing, or a separate
    longer-lived admin signing identity, before PR #4846 and
    `feat/require-signed-jobs-pki` coexist.
14. Attestation-gated issuance mechanism (CC1, §11): KBS-released
    one-time enrollment tokens vs the site key itself as a KBS resource
    vs ACME device-attestation against step-ca — and whether short-lived
    per-job certs may carry attestation claims (§11.5).
15. Per-job CC policy shape (CC2, §11): job-meta flag vs resource-based
    scheduling, and partial-attestation semantics (mixed attested /
    unattested federations) replacing today's all-or-nothing shutdown.
