#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKSPACE="$SCRIPT_DIR/workspace"
PROJECT_FILE="$SCRIPT_DIR/project.yml"
PROJECT_NAME="fedauth_prod_demo"
PROD_DIR="$WORKSPACE/$PROJECT_NAME/prod_00"

ISSUER="${FEDAUTH_ISSUER:-http://localhost:38080/realms/nvflare}"
AUDIENCE="${FEDAUTH_AUDIENCE:-nvflare-admin}"
JWKS_URI="${FEDAUTH_JWKS_URI:-http://host.docker.internal:38080/realms/nvflare/protocol/openid-connect/certs}"
DISCOVERY_URL="${FEDAUTH_DISCOVERY_URL:-http://localhost:38080/realms/nvflare/.well-known/openid-configuration}"

cd "$REPO_DIR"

if command -v podman >/dev/null 2>&1; then
  RUNNING_DEMO_CONTAINERS="$(podman ps --format '{{.Names}}' 2>/dev/null | rg '^(fedauth-server|fedauth-site-1|fedauth-site-2)$' || true)"
  if [ -n "$RUNNING_DEMO_CONTAINERS" ]; then
    echo "ERROR: demo workspace is currently mounted by running FL containers:" >&2
    echo "$RUNNING_DEMO_CONTAINERS" >&2
    echo "Stop the demo stack before reprovisioning startup kits:" >&2
    echo "  cd $SCRIPT_DIR && podman compose down" >&2
    exit 1
  fi
fi

echo "[1/3] Provisioning production startup kits into $WORKSPACE"
rm -rf "$WORKSPACE"
"$REPO_DIR/.venv/bin/nvflare" provision -p "$PROJECT_FILE" -w "$WORKSPACE"

if [ ! -d "$PROD_DIR" ]; then
  echo "ERROR: expected prod directory not found: $PROD_DIR" >&2
  exit 1
fi

# In some provision layouts, usable startup kits are left under prod_00/wip/.
# Normalize by copying them to prod_00/<participant>/ so downstream scripts can
# use stable paths.
if [ -d "$PROD_DIR/wip/server" ]; then
  for p in server site-1 site-2; do
    rm -rf "$PROD_DIR/$p"
    cp -R "$PROD_DIR/wip/$p" "$PROD_DIR/$p"
  done
fi

echo "[2/3] Applying fedauth config to server/admin resources"
"$REPO_DIR/.venv/bin/python" - "$PROD_DIR" "$ISSUER" "$AUDIENCE" "$JWKS_URI" "$DISCOVERY_URL" <<'PY'
import json
import os
import sys
from argparse import Namespace

from nvflare.tool.poc.poc_commands import apply_fedauth_to_poc_startup_kit, _sign_fedauth_admin_profile

prod_dir = sys.argv[1]
issuer = sys.argv[2]
audience = sys.argv[3]
jwks_uri = sys.argv[4]
discovery_url = sys.argv[5]

args = Namespace(
    fedauth_issuer=issuer,
    fedauth_audience=audience,
    fedauth_jwks_uri=jwks_uri,
    fedauth_discovery_url=discovery_url,
    fedauth_alg_allowlist=["RS256"],
    fedauth_required_claims=["iss", "aud", "exp", "iat"],
    fedauth_user_name_claims=["preferred_username", "email"],
    fedauth_user_org_claim="org",
    fedauth_user_role_claim="nvf_role",
    fedauth_role_mappings=["lead=project_admin"],
    fedauth_admin_mode="oidc",
    fedauth_admin_token_file="/tmp/nvflare_alice.token",
    fedauth_oidc_client_id=audience,
    fedauth_oidc_scopes="openid profile email",
    fedauth_oidc_discovery_url=discovery_url,
    fedauth_oidc_callback_host="127.0.0.1",
    fedauth_oidc_callback_port=39123,
    fedauth_oidc_callback_path="/callback",
    fedauth_oidc_refresh_skew_seconds=60,
    fedauth_oidc_open_browser=True,
)

apply_fedauth_to_poc_startup_kit(
    prod_dir=prod_dir,
    server_name="server",
    admin_name="admin@nvidia.com",
    fedauth_args=args,
)

# Host-side admin CLI should connect through published server admin port.
admin_startup_file = os.path.join(prod_dir, "admin@nvidia.com", "startup", "fed_admin.json")
with open(admin_startup_file, "r") as f:
    admin_startup = json.load(f)
admin_cfg = admin_startup.setdefault("admin", {})
admin_cfg["host"] = "127.0.0.1"
admin_cfg["port"] = 8003
admin_cfg["oidc_callback_host"] = "127.0.0.1"
admin_cfg["oidc_callback_port"] = 39123
admin_cfg["oidc_callback_path"] = "/callback"
with open(admin_startup_file, "w") as f:
    json.dump(admin_startup, f, indent=2)

_sign_fedauth_admin_profile(prod_dir=prod_dir, admin_name="admin@nvidia.com")

print(f"Configured fedauth resources under: {prod_dir}")
PY

echo "[3/3] Copying demo job to admin transfer folder"
ADMIN_TRANSFER_DIR="$PROD_DIR/admin@nvidia.com/transfer"
mkdir -p "$ADMIN_TRANSFER_DIR"
rm -rf "$ADMIN_TRANSFER_DIR/hello-numpy-sag"
cp -R "$REPO_DIR/tests/integration_test/data/jobs/hello-numpy-sag" "$ADMIN_TRANSFER_DIR/hello-numpy-sag"

echo

echo "Startup kits prepared: $PROD_DIR"
echo "Next: cd $SCRIPT_DIR && podman compose up --build -d keycloak server site-1 site-2"
