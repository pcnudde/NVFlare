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
- `presentation/fedauth_demo_slides.md`: Marp-compatible slide source for the demo narrative.
- `presentation/diagrams/*.mmd`: Mermaid source for the presentation diagrams.

## Prerequisites

- Podman
- `.venv` available in repo root and already activated in your shell
- demo Python deps installed into this `.venv`

All commands below assume your current directory is `demo_fedauth/`.

One-time dependency install:

```bash
uv pip install --python ../.venv/bin/python -r requirements.txt
```

## Quick Start

If you want a clean starting point before the demo:

```bash
./clean_demo.sh
```

### 1. Prepare startup kits

```bash
./prepare_startup_kits.sh
```

This does all required startup-kit work:

- provisions production startup kits
- applies fedauth resource settings
- generates an internal signed host-side admin console profile even though `project.yml` has no human participant
- packages that bootstrap workspace as a user-facing invite
- exports the user-facing invite as `distribution/invite.zip`
- rewrites the signed host-side admin startup config to use `127.0.0.1:8003`
- copies `hello-numpy-sag` into the admin `transfer/` folder

Important:

- do not rerun `prepare_startup_kits.sh` while `server`, `site-1`, or `site-2` containers are running
- reprovisioning replaces the mounted workspace and will kill the live FL processes
- stop the stack first with `podman compose down`

Output:

```text
workspace/fedauth_prod_demo/prod_00
distribution/invite.zip
```

Optional invite import flow:

```bash
mkdir -p user_demo
cp distribution/invite.zip user_demo/
cd user_demo
python -m nvflare.fuel.hci.tools.admin -i invite.zip
cd ..
```

This simulates what the invited user does in a separate local folder, not from the provisioned `workspace/`.
It only unpacks the invite into `./invite/` next to the zip.
If `./invite/` already exists, remove it first or import into a different workspace path.
The invite is bootstrap-only and does not carry demo jobs. Stage the demo job into the imported workspace:

```bash
cp -R ../tests/integration_test/data/jobs/hello-numpy-sag user_demo/invite/transfer/
```

Do not launch `fl_admin.sh` yet. Launch only after the stack is up in step 3.

### 2. Build the demo image

Do this once, or whenever the NVFlare code or `Dockerfile` changes:

```bash
podman compose build server
```

### 3. Start the stack

```bash
podman compose up -d keycloak server site-1 site-2
```

Useful checks:

```bash
podman compose ps
podman compose logs -f server
```

### 4. Login from the host admin CLI

In a new terminal:

```bash
cd user_demo/invite/startup
./fl_admin.sh
```

This requires the stack from step 2 to already be up.
The CLI then opens the browser-based OIDC flow.

Use:

- Keycloak username: `alice`
- Keycloak password: `alicepass`

Then in `fl_admin`, run:

```text
check_status client
list_jobs
submit_job hello-numpy-sag
list_jobs
```

Repeat `list_jobs` until job is `FINISHED:*`.

### 5. Stop the stack

```bash
podman compose down
```

## Presenting The Slides

```bash
cd presentation && npm ci && npm run render-diagrams && marp -p fedauth_demo_slides.md
```

## Notes

- Issuer is set to `http://localhost:38080/realms/nvflare` so host-side admin browser flow works.
- Server/clients fetch JWKS via `http://host.docker.internal:38080/...` (reachable inside Podman containers).
- `workspace/fedauth_prod_demo/prod_00/admin@nvidia.com/` is an internal generated bootstrap workspace used to produce the invite bundle. It is not the user-facing artifact.
- The user-facing artifact for the demo is `distribution/invite.zip`.
- The invite is bootstrap-only. Demo jobs are copied separately into `user_demo/invite/transfer/`.
- Admin startup config is rewritten and re-signed to connect to `127.0.0.1:8003` for host-side CLI.
- Demo job `hello-numpy-sag` is copied into admin `transfer/` by `prepare_startup_kits.sh`.
