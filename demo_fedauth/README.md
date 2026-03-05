# FedAuth Production Demo

This folder is the concise operator guide for the production-style federated-auth demo.

It brings up:

- 1 Keycloak container
- 1 NVFlare server container
- 2 NVFlare client containers
- 1 host-side admin startup kit using browser OIDC login

This README is the canonical operator guide for the `demo_fedauth/` demo.

## What This Demo Proves

- Keycloak is configured entirely from imported JSON, with no manual UI setup.
- Human admin login uses OIDC browser auth instead of an admin client cert.
- The production `project.yml` contains only site participants; the human admin console profile is generated afterward.
- Job submission works end to end in secure mode.
- The submitted job is attested by the server-side Option A path.

## Key Files

- `project.yml`: production provisioning spec.
- `keycloak/nvflare_realm.json`: realm import file (client `nvflare-admin`, user `alice`).
- `prepare_startup_kits.sh`: provisions startup kits and applies fedauth resource config.
- `docker-compose.yml`: launches Keycloak + server + 2 clients.
- `Dockerfile`: local NVFlare image build (from this repo).
- `fedauth_demo_slides.md`: slide source for the demo narrative.
- `fedauth_demo_slides.pdf`: exported presentation deck.

## Prerequisites

- Podman
- `.venv` available in repo root and already activated in your shell
- demo Python deps installed into this `.venv`

From repo root:

```bash
uv pip install --python .venv/bin/python -r demo_fedauth/requirements.txt
```

## Quick Start

### 1. Prepare startup kits

From repo root:

```bash
./demo_fedauth/prepare_startup_kits.sh
```

This does all required startup-kit work:

- provisions production startup kits
- applies fedauth resource settings
- generates a signed host-side admin console profile even though `project.yml` has no human participant
- rewrites the signed host-side admin startup config to use `127.0.0.1:8003`
- copies `hello-numpy-sag` into the admin `transfer/` folder

Important:

- do not rerun `prepare_startup_kits.sh` while `server`, `site-1`, or `site-2` containers are running
- reprovisioning replaces the mounted workspace and will kill the live FL processes
- stop the stack first with `podman compose down`

Output:

```text
demo_fedauth/workspace/fedauth_prod_demo/prod_00
```

### 2. Start the stack

```bash
cd demo_fedauth
podman compose up --build -d keycloak server site-1 site-2
```

Useful checks:

```bash
podman compose ps
podman compose logs -f server
```

### 3. Login from the host admin CLI

In a new terminal:

```bash
cd /Users/pcnudde/.codex/worktrees/ad00/NVFlare
cd demo_fedauth/workspace/fedauth_prod_demo/prod_00/admin@nvidia.com/startup
./fl_admin.sh
```

The CLI opens the browser-based OIDC flow.

Use:

- username: `alice`
- password: `alicepass`

Then in `fl_admin`, run:

```text
check_status client
list_jobs
submit_job hello-numpy-sag
list_jobs
```

Repeat `list_jobs` until job is `FINISHED:*`.

### 4. Stop the stack

```bash
cd /Users/pcnudde/.codex/worktrees/ad00/NVFlare/demo_fedauth
podman compose down
```

## Notes

- Issuer is set to `http://localhost:38080/realms/nvflare` so host-side admin browser flow works.
- Server/clients fetch JWKS via `http://host.docker.internal:38080/...` (reachable inside Podman containers).
- Admin startup config is rewritten and re-signed to connect to `127.0.0.1:8003` for host-side CLI.
- Demo job `hello-numpy-sag` is copied into admin `transfer/` by `prepare_startup_kits.sh`.
