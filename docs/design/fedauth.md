# Federated Authentication Design and Requirements

Status: Draft  
Audience: NVFlare platform maintainers, security architects, and deployment owners  
Last updated: 2026-03-09

## 1. Purpose

Define the target authentication model for NVFlare where:

- Infrastructure participants (server, clients, relays) continue using PKI/mTLS.
- Human participants (admins/research users) authenticate via enterprise SSO.

This document captures product-level design intent and requirements. Detailed implementation and test execution details are in [fedauth_implementation.md](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/docs/design/fedauth_implementation.md).

## 2. Problem

Human identities are currently provisioned similarly to sites (certificates + startup kits). This creates operational friction:

- manual provisioning and distribution of human startup kits
- cert/key handling burden for humans
- slow onboarding/offboarding and role change propagation
- poor fit with enterprise MFA and SSO governance
- hidden coupling of human certs into admin transport and job submission signing
- inability to remove human entries from `project.yml` without redesigning integrity and provenance controls

## 3. Scope

In scope:

- Human authentication and identity source transition from cert to SSO.
- Role/org derivation for authorization context.
- Job submission integrity and provenance after human cert removal.
- Backward-compatible migration strategy.

Out of scope:

- Replacing site/server/client/relay mTLS trust model.
- Replacing federated authorization policy model.
- Replacing site-local custom security plugin mechanisms.

## 4. Target Security Model

### 4.1 Trust-plane split

- **Infrastructure trust plane:** unchanged PKI/mTLS + existing explicit message authentication.
- **Human identity plane:** SSO token-based auth (OIDC, or SAML via broker/token translation).

### 4.2 Human key-management target

- Normal human users shall not receive project-provisioned private keys.
- Normal SSO-based human access shall not require human participants in `project.yml`.
- Human console/API profiles may include server trust material and SSO configuration, but not human client certificates or signing keys.

Current Option A implementation status:

- fedauth provisioning can synthesize and sign a lightweight admin console profile from a site-only `project.yml`
- the resulting console workspace still has an admin directory name (for example `admin@nvidia.com/`), but that directory is no longer backed by a human participant entry in project provisioning
- the current user-facing bootstrap artifact is an exported `invite.zip` that is imported into a local admin workspace before running `fl_admin.sh`

### 4.3 Standards-first federation pattern (recommended)

Use one trusted OIDC issuer for NVFlare.  
That issuer can be:

- a direct enterprise IdP, or
- a federation broker that fronts multiple organization IdPs.

- Org A users authenticate through upstream IdP A.
- Org B users authenticate through upstream IdP B.
- The trusted issuer emits normalized OIDC claims consumed by NVFlare.

Benefits:

- NVFlare validates one issuer/JWKS surface.
- federation complexity is isolated in the IdP/broker layer.
- claim normalization is centralized.

Reference deployment:

- Keycloak is the reference broker for local/CI testing and initial rollout, but the NVFlare auth design must remain provider-agnostic.

### 4.4 Job integrity and provenance target

Equivalent security to today's human-cert model means preserving these properties even after human cert removal:

- the package executed by server/clients is exactly the package accepted by the trusted NVFlare control plane
- submitter identity, org, role, issuer, and auth source are durably bound to the accepted package
- tampering after submission, including storage or fanout tampering, is detected before execution

The recommended design is server-issued submission attestation:

- the human authenticates with SSO and uploads the job over server-authenticated TLS
- the server validates the SSO session, computes canonical job digests, snapshots submitter identity from validated claims, and signs an immutable submission attestation with a server-held job-signing key
- execution nodes verify this server-issued attestation before deploy/run

This preserves cryptographic tamper evidence without distributing human private keys.

Security tradeoff:

- this shifts package-origin trust from "independent human signer" to "trusted NVFlare control plane accepted and attested this package"
- execution nodes still verify cryptographic integrity, but the signer they trust is now the server-side attester, not the human submitter directly
- this is acceptable only if the deployment already treats the server as the authoritative job-acceptance trust anchor
- if independent end-user provenance is required, a separate human-held or keyless user-signing design is needed in addition to server acceptance

### 4.5 Provenance decision options to keep documented

Two provenance options must stay documented because the deployment may switch between them later.

**Option A: Server-issued submission attestation**

- the human submits an unsigned package after SSO login
- the NVFlare server canonicalizes the accepted package, snapshots validated submitter claims, and signs a submission attestation
- executors verify the server-issued attestation and the accepted package digest

Use this when:

- the server is already the accepted trust anchor for job acceptance
- the priority is removing human keys and simplifying operations
- independent end-user non-repudiation is not a hard requirement

**Option C: Keyless end-user signing plus server countersignature**

- the human authenticates with SSO
- the client obtains a short-lived signing certificate bound to that OIDC identity
- the client signs the job manifest with an ephemeral private key
- the server verifies the end-user signature bundle, authorizes submission, and countersigns acceptance
- executors verify both the end-user signature bundle and the server countersignature

Use this when:

- executors must validate provenance that is independent of the NVFlare server
- the deployment wants evidence closer to today's "the submitter signed this package" model
- the added signing-service and trust-policy complexity is acceptable

Current implementation target:

- implement Option A first
- keep Option C fully documented so the trust model can be upgraded later without redesigning the whole auth plane

## 5. Requirements

### 5.1 Functional requirements

1. NVFlare shall support human login using SSO tokens.
2. NVFlare shall preserve cert login during migration (`dual-stack` mode).
3. NVFlare shall map token claims into existing authorization context fields:
   - user name
   - user org
   - user role
4. NVFlare shall keep site authentication unchanged (mTLS/cert).
5. NVFlare shall preserve existing authorization policy semantics unless explicitly configured otherwise.
6. NVFlare shall support multi-org deployments where different humans authenticate through different upstream IdPs.
7. NVFlare shall support non-interactive authentication for automation/service principals.
8. NVFlare shall be standards-based and provider-agnostic (OIDC/SAML integration without hard dependency on one IdP product).
9. Normal SSO-based human access shall not require human participants in `project.yml`.
10. NVFlare shall preserve job submission integrity and submitter provenance without human client certificates.

### 5.2 Security requirements

1. JWT validation must enforce issuer, audience, signature, expiry, and algorithm allowlist.
2. JWT key resolution must support key rotation (`kid`) with bounded JWKS cache behavior.
3. Missing required claims (`org`, mapped role, or required identity claim) must fail closed.
4. Raw bearer tokens must not be logged.
5. Session lifetime must not exceed configured security constraints relative to token lifetime.
6. Auth source (`cert` vs `sso`) must be auditable.
7. Execution nodes must reject jobs whose submission attestation or attested digests do not verify.
8. Submission attestation must bind job digests to validated submitter identity attributes and accepted auth source.
9. The server-held job-signing key must be rotatable and auditable.

### 5.3 Compatibility and migration requirements

1. Existing cert-based human workflows shall continue in compatibility mode.
2. Upgrades shall support mixed-mode operation during rollout.
3. Documentation/tooling migration shall be phased; no abrupt removal of cert mode.
4. A break-glass cert path shall remain available for outage scenarios during transition.

### 5.4 Operational requirements

1. Identity lifecycle (join/leave/role change) shall be IdP-driven for SSO users.
2. Runbooks shall exist for JWKS rotation, IdP outage, and fallback mode activation.
3. Multi-org claim mapping rules shall be explicit and deterministic.
4. Release validation shall include at least one reference provider in CI and at least one additional provider in periodic compatibility testing.

## 6. Architecture Requirements

### 6.1 Canonical token contract

NVFlare requires a normalized claim contract from the trusted issuer:

- stable subject identifier (`sub` or explicitly configured equivalent)
- human display/login identifier (`preferred_username` or `email`)
- organization claim (`org`)
- role/group claims mapped to NVFlare roles

### 6.2 Authorization context continuity

Post-login execution path should continue consuming `USER_*`/`SUBMITTER_*` style context to minimize disruption to federated authorization and site-specific security handlers.

### 6.3 Human transport model

For SSO humans, the admin channel shall move from mutual TLS to server-authenticated TLS plus token-based login.

- Sites remain on mTLS.
- Humans authenticate at the application/session layer with validated SSO tokens.
- Stronger proof-of-possession modes such as DPoP or token binding may be added, but human project-provisioned certs are not required in the steady state.

### 6.4 Job submission attestation model

The attested object shall include, at minimum:

- canonical job digest or manifest digest
- accepted submission timestamp
- submitter identity snapshot
- submitter org and mapped role snapshot
- trusted issuer identifier and stable subject identifier
- auth source and policy/mapping version

The signer of this attestation shall be a server-controlled deployment or project signing identity, not a human identity.

### 6.5 Option C trust model requirements

If Option C is selected later:

- Keycloak or another standards-compliant broker remains the OIDC identity source only
- a separate signing service must mint short-lived signing certificates from validated OIDC identities
- executors trust the signing-service root and verification policy, not a distributed human private key
- executors should verify offline from a signature bundle whenever possible, not by performing live IdP calls at job runtime

### 6.6 Plugin boundary

- Keep core token authentication and validation in NVFlare core code path.
- Keep site-local plugin/event-handler mechanisms for deployment-specific authorization and policy extensions.

### 6.7 Job authorization semantics decision required

Define whether submitter role/org are:

- snapshotted at submission time (current behavior), or
- re-evaluated at schedule/deploy time (dynamic behavior).

This is a core policy decision and must be explicitly selected.

## 7. Acceptance Criteria

The design is accepted for implementation when:

1. A dual-stack auth architecture is agreed (`cert` + `sso`).
2. Claim contract and mapping policy are approved for multi-org federation.
3. Token-only human operation without `project.yml` human participants is designed and approved.
4. Security control checklist is approved (validation, redaction, attestation, auditing).
5. Rollout and rollback requirements are approved.
6. Open decisions (Section 8) have owners and target resolution milestones.

## 8. Open Decisions

1. Snapshot vs runtime role evaluation for scheduled jobs.
2. Canonical human identifier (`sub`, email, UPN).
3. Org claim source policy (IdP alias vs mapped claim vs hybrid).
4. Bearer-only vs token binding in first release.
5. Preferred non-interactive flow for automation.
6. Deprecation horizon for cert-based human auth.
7. Identity topology model (shared broker realm vs per-project realm vs direct IdP trust).
8. Account-linking policy for cross-IdP identities in the selected broker/IdP platform.
9. Server-held job-signing key topology: per deployment, per project, or external KMS/HSM-backed signer.
10. Whether delayed job scheduling uses submit-time attested role snapshots only, or also requires runtime re-authorization.
11. Is server-attested job provenance sufficient, or must execution nodes verify an end-user signature that remains independent of the server?

## 9. Implementation Reference

Detailed engineering breakdown, component-level changes, migration phases, and full testing plan are tracked in:

- [fedauth_implementation.md](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/docs/design/fedauth_implementation.md)
