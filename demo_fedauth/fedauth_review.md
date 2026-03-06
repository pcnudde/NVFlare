# FedAuth Review Summary

This document captures the current review status of the fedauth branch so follow-up cleanup can stay focused and minimize additional code.

For the deeper follow-up review, see:

- `demo_fedauth/fedauth_review_detailed.md`

## Resolved In Current Working Tree

These review items were valid at the start of the cleanup pass, and are now addressed in the current uncommitted tree:

- `AdminAPI.connect()` again verifies the configured `server_identity` before login for all admin auth modes.
- invite import no longer silently reuses an existing extracted workspace; reruns must remove the old directory or choose a new one.
- admin-port identity verification now challenges the admin cell target while validating against the root server cert identity, so `server.admin` works in the secure token-auth flow again.
- `invite.zip` no longer carries `local/signature.json`; local defaults are treated as mutable local state instead of implied signed bootstrap.
- production `SessionManager` now verifies session-token signatures on decode when the server id-asserter is configured.
- token-validation config now enforces a minimum required-claims floor (`iss`, `aud`, `exp`, `iat`) and has boundary coverage for clock-skew semantics.

## Scope

Reviewed range:

- `1e975646a581b662c22a7673d453dc429fa44328..HEAD`

Reviewed areas:

- token/OIDC login flow
- server-side token validation and claim mapping
- admin port / `server.admin` cell changes
- server-side submission attestation path
- POC and `demo_fedauth` provisioning/demo flow

Review inputs:

- direct code review by Codex
- joined Claude review notes kept locally under `.codex/.local_review/...`

## Main Findings

### 1. Medium: child-cell readiness is too optimistic

File:

- `nvflare/fuel/f3/cellnet/core_cell.py`

Relevant lines:

- `577-585`

What changed:

- `is_ready()` returns `True` for child cells whenever `ext_listeners` exist

Why this matters:

- a listener existing does not prove the child can route to its parent or root
- this matches the failure mode seen during admin-port bring-up: listener present, but `target_unreachable` at runtime

Minimum-change fix strategy:

1. tighten readiness for child cells so it reflects routability, not just listener creation
2. add a focused test around admin child-cell readiness semantics

### 2. Low: OIDC error-path coverage is still thin

Files:

- `nvflare/fuel/hci/client/oidc.py`
- `tests/unit_test/fuel/hci/client/oidc_test.py`

Why this matters:

- the happy path is well covered now, but callback/server/network failures are still under-tested
- that is an operational robustness issue more than a security blocker

Minimum-change fix strategy:

1. add callback mismatch / missing-code tests
2. add malformed JSON / non-200 token endpoint tests
3. classify refresh fallback failures more precisely in logs

## Items Reviewed But Not Raised As Main Findings

These areas changed substantially, but did not currently justify top-level findings:

- server-side submission attestation design for Option A
- Keycloak-backed OIDC browser flow and PKCE token manager
- generated signed admin console profile path
- `demo_fedauth` operational flow after README and Marp cleanup

These still need cleanup, but they are not currently the main blockers.

## Open Questions

1. Is cert-mode admin connectivity still part of the supported compatibility matrix?
2. What exact semantics should `clock_skew_seconds` have?
3. Is Option A server-side attestation the intended long-term trust model, or only the current implementation phase before Option C?
4. Should readiness for `server.admin` mean:
   - listener bound, or
   - command-routable to the FL server engine?

## Prioritized Cleanup Tasks

### High Priority

1. Tighten child-cell readiness semantics and test them.
2. Add focused OIDC callback/network error-path tests.
3. Decide whether refresh fallback logging should distinguish revocation vs transient failure.

### Medium Priority

4. Add a required containerized smoke test for `demo_fedauth`:
   - Keycloak up
   - server + 2 clients up
   - OIDC login
   - `check_status`
   - `submit_job`
   - completion check
5. Improve reprovision safety in `demo_fedauth/prepare_startup_kits.sh`.
6. Rename the generated admin console profile away from `admin@nvidia.com` to a neutral console-profile name.
7. Keep `demo_fedauth/README.md` and the slide deck aligned as commands or topology change.

## Cleanup Strategy To Minimize New Code

The goal for the next round should be to reduce divergence, not add abstraction.

Recommended approach:

1. prefer restoring proven legacy behavior over inventing a new common auth path
2. isolate token/OIDC logic to the paths that truly need it
3. add tests before broad refactors
4. remove stale code and imports quickly once behavior is settled
5. avoid new provisioning layers unless they remove existing duplication

## Handoff Notes

Local joined-review artifacts exist here:

- `.codex/.local_review/20260305-155739-codex-fedauth-keycloak-design-review/joint_review.md`
- `.codex/.local_review/20260305-155739-codex-fedauth-keycloak-design-review/round-1-claude.md`

These are useful for reviewer context, but they are not tracked repo artifacts.
