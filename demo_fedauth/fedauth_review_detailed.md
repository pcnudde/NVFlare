# FedAuth Detailed Review

This document is the detailed review handoff for the `codex/fedauth-keycloak-design` branch, including the current uncommitted `invite.zip` workspace import changes.

Update after the current cleanup pass:

- `AdminAPI.connect()` now restores explicit server-identity verification before login.
- invite import now refuses to reuse an already extracted workspace.
- admin-port verification now challenges the `server.admin` cell while validating the root server cert identity.
- invite packaging no longer includes `local/signature.json`, so mutable local defaults are no longer presented as signed bootstrap content.

## Scope

Reviewed range:

- branch diff: `origin/main..HEAD`
- plus current local changes in:
  - `nvflare/fuel/hci/tools/admin.py`
  - `nvflare/tool/poc/poc_commands.py`
  - `nvflare/lighter/constants.py`
  - `tests/unit_test/lighter/poc_fedauth_test.py`
  - `demo_fedauth/README.md`

Reviewed areas:

- admin auth transport and identity verification
- token/OIDC login and session lifecycle
- server-side submission attestation
- POC/demo provisioning and invite packaging
- demo operator UX and presentation artifacts

## Validation Inputs

Local code review:

- `nvflare/fuel/hci/client/api.py`
- `nvflare/private/fed/authenticator.py`
- `nvflare/fuel/f3/drivers/net_utils.py`
- `nvflare/fuel/hci/server/login.py`
- `nvflare/fuel/hci/server/sess.py`
- `nvflare/fuel/hci/server/token_auth.py`
- `nvflare/private/fed/app/utils.py`
- `nvflare/apis/impl/job_def_manager.py`
- `nvflare/fuel/hci/tools/admin.py`
- `nvflare/tool/poc/poc_commands.py`
- `demo_fedauth/README.md`

Targeted unit validation:

```bash
/bin/zsh -lc 'mkdir -p .tmp && PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit_test/fuel/hci/client/api_login_test.py \
  tests/unit_test/fuel/hci/client/oidc_test.py \
  tests/unit_test/fuel/hci/server/login_test.py \
  tests/unit_test/fuel/hci/server/sess_test.py \
  tests/unit_test/fuel/hci/server/token_auth_test.py \
  tests/unit_test/apis/impl/job_def_manager_test.py \
  tests/unit_test/fuel/f3/cellnet/core_cell_test.py \
  tests/unit_test/private/fed/app/utils_test.py \
  tests/unit_test/lighter/poc_fedauth_test.py'
```

Result:

- `70 passed`

Subagent review artifacts:

- joined review:
  - `.codex/.local_review/20260306-120559-codex-fedauth-keycloak-design-review/joint_review.md`
  - `.codex/.local_review/20260306-120559-codex-fedauth-keycloak-design-review/round-1-claude.md`
  - `.codex/.local_review/20260306-120559-codex-fedauth-keycloak-design-review/round-2-claude.md`
- targeted Claude checks:
  - `.codex/.local_review/ask-claude/20260306-121242-security-fedauth.md`
  - `.codex/.local_review/ask-claude/20260306-121407-simplification-fedauth.md`
  - `.codex/.local_review/ask-claude/20260306-121937-server-identity-regression.md`

## Executive Summary

The fedauth branch direction is sound:

- token/OIDC login path is coherent
- server-side attestation for Option A is implemented consistently
- Keycloak demo and containerized demo flow are credible
- invite packaging can now reuse the existing workspace model with little new code

The main blocker is still transport/server identity verification on the admin client side. The branch removed the explicit `Authenticator` handshake from `AdminAPI.connect()`, but the underlying TLS stack still runs with `check_hostname = False`. That means the admin client no longer verifies that it is actually connected to the configured `server_identity`.

Beyond that blocker, the remaining issues are mostly cleanup and integration polish:

- stale invite reuse
- invite artifact semantics versus what is actually verified
- overgrown config plumbing in provisioning/runtime helpers
- a few demo/documentation inconsistencies

## Main Findings

### 1. High: admin client no longer verifies the configured server identity

Files:

- [api.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/client/api.py#L395)
- [authenticator.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/private/fed/authenticator.py#L135)
- [net_utils.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/f3/drivers/net_utils.py#L105)

What changed:

- `AdminAPI.connect()` now starts the cell directly and no longer runs the `Authenticator` challenge/registration block.
- The old handshake explicitly checked `server_cn == expected_sp_identity`.
- The TLS client context still sets `check_hostname = False`.

Why this matters:

- before this branch, the admin client had an application-layer identity check on top of CA verification
- after this branch, the client accepts any server certificate chained to the trusted CA for that host/port
- that affects `cert`, `token`, and `oidc` admin flows equally
- in token/OIDC mode, this also means the bearer token can be presented to the wrong peer if the connection is misdirected

Why the current tests do not catch it:

- current unit tests cover login behavior and TLS credential wiring
- there is no regression test for `expected_sp_identity` enforcement in the admin connect path

Minimum-change cleanup:

1. Restore explicit server-identity verification in the admin connect path.
2. For `cert` mode, the safest fix is to restore the old `Authenticator` block as-is.
3. For `token`/`oidc` modes, either:
   - reuse `_challenge_server()` without client registration, or
   - add a separate identity-pinning path before sending login tokens.
4. Add a regression test that fails when the server CN does not match `server_identity`.

### 2. Medium: invite import silently reuses stale extracted workspaces

File:

- [admin.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/tools/admin.py#L48)

What changed:

- `prepare_workspace()` returns an existing workspace immediately if it already contains `startup/<fed_admin>` and `local/`.

Why this matters:

- rerunning `python -m nvflare.fuel.hci.tools.admin -i invite.zip` against a newer invite will silently keep using the old extracted workspace
- that is a bad fit for an invite/bootstrap model, where refreshed trust material or endpoint changes should replace the previous import

Impact:

- primarily UX/integration
- can turn security rotations into confusing “why am I still using the old config?” failures

Minimum-change cleanup:

1. Make invite import refuse reuse unless `--reuse-existing` is specified, or
2. make it overwrite only when the invite path is explicitly provided, or
3. at minimum, emit a warning that the existing workspace is being reused unchanged

### 3. Medium: the new invite artifact is not treated as a fully verified signed bundle

Files:

- [poc_commands.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/tool/poc/poc_commands.py#L635)
- [poc_commands.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/tool/poc/poc_commands.py#L642)
- [config.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/client/config.py#L140)

What changed:

- `_sign_fedauth_admin_profile()` signs both `startup/` and `local/`
- `_create_fedauth_admin_invite()` copies both `startup/` and `local/` into `invite.zip`
- `secure_load_admin_config()` only verifies `startup/fed_admin.json` before loading the workspace

Why this matters:

- the codebase still treats `local/` as mutable local workspace state
- the new invite model packages `local/` as if it is part of the signed bootstrap artifact
- but the runtime load path does not actually verify `local/`

Current impact:

- not an immediate exploit in the current demo, because `local/resources.json` is now mostly prompt/timeout defaults
- but it is a design mismatch and weakens the claim that `invite.zip` is the signed bootstrap artifact

Minimum-change cleanup:

1. Best minimal fix: remove `local/` from `invite.zip` entirely and recreate default `local/resources.json` on first import.
2. Alternative: actually verify `local/signature.json` on import/load if `local/` stays inside the invite.

### 4. Low: refresh-token failures are swallowed silently

File:

- [oidc.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/client/oidc.py#L94)

What changed:

- failed refresh attempts fall through to browser re-auth without any log or surfaced reason

Why this matters:

- operational diagnosis becomes harder
- it hides the distinction between normal expiry, revoked refresh token, and provider-side failure

Minimum-change cleanup:

1. Log the refresh failure reason at debug or warning level before falling back to browser login.

### 5. Low: child-cell readiness semantics are still too optimistic

File:

- [core_cell.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/f3/cellnet/core_cell.py#L579)

What changed:

- `is_ready()` for child cells returns `True` as soon as `ext_listeners` exist

Why this matters:

- listener creation does not prove the child admin cell is routable to its parent/root
- this was already the failure mode seen during admin-port bring-up

Current assessment:

- not a security issue
- still an integration quality issue because it can mask startup problems

Minimum-change cleanup:

1. Tighten readiness semantics, or
2. rename/clarify the method usage so “listener bound” is not read as “command-routable”

### 6. Low: demo README still contains absolute machine-specific paths

File:

- [README.md](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/demo_fedauth/README.md#L103)

Why this matters:

- it makes the operator guide less portable
- it also obscures the intended repo-relative workflow for the next reviewer

Minimum-change cleanup:

1. Replace absolute worktree paths with repo-relative commands.

## Security Review

### Blocking security issue

1. Admin server identity verification regression.

That is the one issue I would treat as the true blocker.

### Security hardening items, but not blockers

1. `SessionManager.get_session()` decodes the session token without signature verification before looking up the in-memory session map:
   - [sess.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/server/sess.py#L230)
   - current assessment: defense-in-depth only, because the real gate is the in-memory UUID-keyed session table
2. `build_admin_token_login_kwargs()` allows operator config to replace `required_claims` entirely:
   - [utils.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/private/fed/app/utils.py#L239)
   - current assessment: trusted-operator config, not an exploit path, but worth enforcing a minimum floor of `iss`, `aud`, and `exp`
3. OIDC callback host is configurable and the redirect URI is `http://` loopback:
   - [oidc.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/client/oidc.py#L179)
   - current assessment: acceptable for native loopback flow, but should stay loopback-only unless explicitly overridden

### Security items reviewed and closed

1. Zip-slip on invite import:
   - reviewed via CPython `ZipFile._extract_member`
   - current assessment: not a real issue on supported Python versions; `extractall()` already strips drive letters, absolute paths, `.` and `..`
2. Session token hot-path verification:
   - reviewed with Claude follow-up
   - current assessment: hardening opportunity, not a merge blocker

## Integration Assessment

### Integration points that look good

1. Token validation and claim mapping split:
   - [token_auth.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/server/token_auth.py)
   - clean separation of responsibilities
2. Server-side attestation implementation:
   - [job_def_manager.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/apis/impl/job_def_manager.py#L167)
   - implementation matches the current Option A design
3. Invite import built on top of the existing workspace model:
   - avoids inventing a new local profile subsystem

### Integration points I would change

1. `invite.zip` should be a bootstrap artifact, not a full copy of the mutable admin workspace.
2. `server_identity` should stay security-relevant in the admin runtime, not just a routing target string.
3. `server.admin` readiness should not be inferred from listener presence alone.

## Simplification Opportunities

### High-value simplifications

1. Collapse the verbose config readers in [utils.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/private/fed/app/utils.py#L201).
   - `build_admin_token_login_kwargs()` is correct but overly manual.
   - This is a good candidate for a compact dataclass-or-dict normalization pass.

2. Reduce duplication between [utils.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/private/fed/app/utils.py#L45) and [oidc.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/client/oidc.py#L24).
   - duplicated JSON fetch and string-config helpers

3. Simplify [poc_commands.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/tool/poc/poc_commands.py#L698).
   - `apply_fedauth_to_poc_startup_kit()` currently does too much:
     - server token-login config
     - admin bootstrap rewrite
     - local resource cleanup
     - signing
     - invite packaging
   - split into:
     - configure server token auth
     - generate admin bootstrap workspace
     - package invite

### Small cleanups

1. Remove dead `_get_numeric_claim()` from [token_auth.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/server/token_auth.py#L200).
2. Remove stale imports in [api.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/fuel/hci/client/api.py#L58) once the server-identity fix is settled.
3. Replace raw `_LOCAL_ADMIN_OVERRIDE_KEYS` strings with a narrower purpose-built list or direct rewrites:
   - [poc_commands.py](/Users/pcnudde/.codex/worktrees/ad00/NVFlare/nvflare/tool/poc/poc_commands.py#L666)

## Recommended Cleanup Order

1. Restore explicit server identity verification in the admin client path.
2. Add a regression test for mismatched `server_identity`.
3. Decide what `invite.zip` actually is:
   - bootstrap-only artifact, or
   - exported mutable workspace
4. If bootstrap-only, trim `invite.zip` to `startup/` plus minimal generated local defaults.
5. Fix stale invite reuse semantics.
6. Remove portability issues from `demo_fedauth/README.md`.
7. Refactor duplicated config plumbing only after the security and invite semantics are settled.

## Notes For The Next Reviewer

Things that are easy to overstate:

1. Zip-slip on invite import is not a real issue on supported Python versions.
2. Session token signature verification in `get_session()` is worth tightening, but it is not the main branch risk.
3. The biggest problem is not JWT validation; it is loss of expected-server identity verification on the admin connection path.

Things that are easy to miss:

1. `check_hostname = False` means CA validation alone is not enough here.
2. `invite.zip` currently inherits assumptions from the old workspace model that do not fully match “signed bootstrap artifact” semantics.
3. The current tests are strong on token/session logic but weak on transport identity assertions.
