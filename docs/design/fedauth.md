# Federated Authentication Design and Requirements

Status: Draft  
Audience: NVFlare platform maintainers, security architects, and deployment owners  
Last updated: 2026-03-04

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

## 3. Scope

In scope:

- Human authentication and identity source transition from cert to SSO.
- Role/org derivation for authorization context.
- Backward-compatible migration strategy.

Out of scope:

- Replacing site/server/client/relay mTLS trust model.
- Replacing federated authorization policy model.
- Replacing site-local custom security plugin mechanisms.

## 4. Target Security Model

### 4.1 Trust-plane split

- **Infrastructure trust plane:** unchanged PKI/mTLS + existing explicit message authentication.
- **Human identity plane:** SSO token-based auth (OIDC, or SAML via broker/token translation).

### 4.2 Keycloak federation pattern (recommended)

Use one Keycloak broker realm as the issuer trusted by NVFlare.  
Keycloak federates to multiple upstream organization IdPs.

- Org A users authenticate through upstream IdP A.
- Org B users authenticate through upstream IdP B.
- Keycloak emits normalized OIDC claims consumed by NVFlare.

Benefits:

- NVFlare validates one issuer/JWKS surface.
- federation complexity is isolated in Keycloak.
- claim normalization is centralized.

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

### 5.2 Security requirements

1. JWT validation must enforce issuer, audience, signature, expiry, and algorithm allowlist.
2. JWT key resolution must support key rotation (`kid`) with bounded JWKS cache behavior.
3. Missing required claims (`org`, mapped role, or required identity claim) must fail closed.
4. Raw bearer tokens must not be logged.
5. Session lifetime must not exceed configured security constraints relative to token lifetime.
6. Auth source (`cert` vs `sso`) must be auditable.

### 5.3 Compatibility and migration requirements

1. Existing cert-based human workflows shall continue in compatibility mode.
2. Upgrades shall support mixed-mode operation during rollout.
3. Documentation/tooling migration shall be phased; no abrupt removal of cert mode.
4. A break-glass cert path shall remain available for outage scenarios during transition.

### 5.4 Operational requirements

1. Identity lifecycle (join/leave/role change) shall be IdP-driven for SSO users.
2. Runbooks shall exist for JWKS rotation, IdP outage, and fallback mode activation.
3. Multi-org claim mapping rules shall be explicit and deterministic.

## 6. Architecture Requirements

### 6.1 Canonical token contract

NVFlare requires a normalized claim contract from the trusted issuer (Keycloak broker):

- stable subject identifier (`sub` or explicitly configured equivalent)
- human display/login identifier (`preferred_username` or `email`)
- organization claim (`org`)
- role/group claims mapped to NVFlare roles

### 6.2 Authorization context continuity

Post-login execution path should continue consuming `USER_*`/`SUBMITTER_*` style context to minimize disruption to federated authorization and site-specific security handlers.

### 6.3 Job authorization semantics decision required

Define whether submitter role/org are:

- snapshotted at submission time (current behavior), or
- re-evaluated at schedule/deploy time (dynamic behavior).

This is a core policy decision and must be explicitly selected.

## 7. Acceptance Criteria

The design is accepted for implementation when:

1. A dual-stack auth architecture is agreed (`cert` + `sso`).
2. Claim contract and mapping policy are approved for multi-org Keycloak federation.
3. Security control checklist is approved (validation, redaction, auditing).
4. Rollout and rollback requirements are approved.
5. Open decisions (Section 8) have owners and target resolution milestones.

## 8. Open Decisions

1. Snapshot vs runtime role evaluation for scheduled jobs.
2. Canonical human identifier (`sub`, email, UPN).
3. Org claim source policy (IdP alias vs mapped claim vs hybrid).
4. Bearer-only vs token binding in first release.
5. Preferred non-interactive flow for automation.
6. Deprecation horizon for cert-based human auth.
7. Keycloak topology model (shared broker realm vs per-project realm).
8. Keycloak account-linking policy for cross-IdP identities.

## 9. Implementation Reference

Detailed engineering breakdown, component-level changes, migration phases, and full testing plan are tracked in:

- [fedauth_implementation.md](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/docs/design/fedauth_implementation.md)

