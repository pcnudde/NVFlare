# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from argparse import Namespace
from zipfile import ZipFile

import pytest

from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.config import secure_load_admin_config
from nvflare.fuel.hci.tools.admin import prepare_workspace
from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key, verify_folder_signature
from nvflare.tool.poc.poc_commands import apply_fedauth_to_poc_startup_kit


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_text(path, content):
    with open(path, "w") as f:
        f.write(content)


def _prepare_server_material(prod_dir):
    root_key, root_pub_key = generate_keys()
    root_identity = Identity("root", "nvidia")
    root_cert = generate_cert(root_identity, root_identity, root_key, root_pub_key, ca=True)

    state_dir = prod_dir.parent / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        state_dir / "cert.json",
        {
            "root_pri_key": serialize_pri_key(root_key).decode("ascii"),
            "root_cert": serialize_cert(root_cert).decode("ascii"),
        },
    )

    server_startup = prod_dir / "server" / "startup"
    server_startup.mkdir(parents=True, exist_ok=True)
    _write_text(server_startup / "rootCA.pem", serialize_cert(root_cert).decode("ascii"))
    _write_json(
        server_startup / "fed_server.json",
        {
            "format_version": 2,
            "servers": [
                {
                    "name": "example_project",
                    "service": {"target": "server:8002", "scheme": "http"},
                    "admin_server": "server",
                    "admin_port": 8003,
                    "connection_security": "mtls",
                }
            ],
        },
    )


def test_apply_fedauth_writes_server_and_admin_resources(tmp_path):
    prod_dir = tmp_path / "example_project" / "prod_00"
    server_local = prod_dir / "server" / "local"
    admin_local = prod_dir / "admin@nvidia.com" / "local"
    server_local.mkdir(parents=True, exist_ok=True)
    admin_local.mkdir(parents=True, exist_ok=True)

    _write_json(server_local / "resources.json", {"servers": [{}], "format_version": 1})
    _prepare_server_material(prod_dir)

    args = Namespace(
        fedauth_issuer="http://127.0.0.1:38080/realms/nvflare",
        fedauth_audience="nvflare-admin",
        fedauth_jwks_uri="http://127.0.0.1:38080/realms/nvflare/protocol/openid-connect/certs",
        fedauth_discovery_url=None,
        fedauth_alg_allowlist=["RS256"],
        fedauth_required_claims=["iss", "aud", "exp", "iat"],
        fedauth_user_name_claims=["preferred_username", "email"],
        fedauth_user_org_claim="org",
        fedauth_user_role_claim="nvf_role",
        fedauth_role_mappings=["lead=project_admin"],
        fedauth_admin_mode="oidc",
        fedauth_admin_token_file="/tmp/nvflare_alice.token",
        fedauth_oidc_client_id="nvflare-admin",
        fedauth_oidc_scopes="openid profile email offline_access",
        fedauth_oidc_callback_host="127.0.0.1",
        fedauth_oidc_callback_port=39123,
        fedauth_oidc_callback_path="/callback",
        fedauth_oidc_refresh_skew_seconds=60,
        fedauth_oidc_open_browser=True,
        fedauth_oidc_discovery_url="http://127.0.0.1:38080/realms/nvflare/.well-known/openid-configuration",
    )

    apply_fedauth_to_poc_startup_kit(
        prod_dir=str(prod_dir),
        server_name="server",
        admin_name="admin@nvidia.com",
        fedauth_args=args,
    )

    server_cfg = _read_json(server_local / "resources.json")
    token_login = server_cfg["servers"][0]["admin_auth"]["token_login"]
    assert token_login["issuer"] == args.fedauth_issuer
    assert token_login["audience"] == args.fedauth_audience
    assert token_login["jwks_uri"] == args.fedauth_jwks_uri
    assert token_login["claim_mappings"]["role_mappings"]["lead"] == "project_admin"
    assert server_cfg["servers"][0]["admin_connection_security"] == "tls"
    assert server_cfg["servers"][0]["admin_interface_identity"] == "server.admin"

    admin_cfg = _read_json(admin_local / "resources.json")
    admin_section = admin_cfg["admin"]
    assert admin_section["idle_timeout"] == 900.0
    assert "auth_mode" not in admin_section
    assert "server_identity" not in admin_section

    config = secure_load_admin_config(Workspace(root_dir=str(prod_dir / "admin@nvidia.com"))).get_admin_config()
    assert config["auth_mode"] == "oidc"
    assert config["oidc_issuer"] == args.fedauth_issuer
    assert config["oidc_client_id"] == args.fedauth_oidc_client_id
    assert config["oidc_callback_port"] == args.fedauth_oidc_callback_port
    assert config["connection_security"] == "tls"
    assert config["server_identity"] == "server.admin"
    assert config["uid_source"] == "user_input"
    assert config["client_key"] == ""
    assert config["client_cert"] == ""


def test_apply_fedauth_admin_token_mode(tmp_path):
    prod_dir = tmp_path / "example_project" / "prod_00"
    server_local = prod_dir / "server" / "local"
    admin_local = prod_dir / "admin@nvidia.com" / "local"
    server_local.mkdir(parents=True, exist_ok=True)
    admin_local.mkdir(parents=True, exist_ok=True)

    _write_json(server_local / "resources.json", {"servers": [{}], "format_version": 1})
    _write_json(admin_local / "resources.json", {"format_version": 1, "admin": {"idle_timeout": 120}})
    _prepare_server_material(prod_dir)

    args = Namespace(
        fedauth_issuer="http://127.0.0.1:38080/realms/nvflare",
        fedauth_audience="nvflare-admin",
        fedauth_jwks_uri=None,
        fedauth_discovery_url="http://127.0.0.1:38080/realms/nvflare/.well-known/openid-configuration",
        fedauth_alg_allowlist=["RS256"],
        fedauth_required_claims=["iss", "aud", "exp", "iat"],
        fedauth_user_name_claims=["preferred_username"],
        fedauth_user_org_claim="org",
        fedauth_user_role_claim="nvf_role",
        fedauth_role_mappings=[],
        fedauth_admin_mode="token",
        fedauth_admin_token_file="/tmp/nvflare_alice.token",
        fedauth_oidc_client_id="nvflare-admin",
        fedauth_oidc_scopes="openid profile email offline_access",
        fedauth_oidc_callback_host="127.0.0.1",
        fedauth_oidc_callback_port=39123,
        fedauth_oidc_callback_path="/callback",
        fedauth_oidc_refresh_skew_seconds=60,
        fedauth_oidc_open_browser=True,
        fedauth_oidc_discovery_url=None,
    )

    apply_fedauth_to_poc_startup_kit(
        prod_dir=str(prod_dir),
        server_name="server",
        admin_name="admin@nvidia.com",
        fedauth_args=args,
    )

    admin_cfg = _read_json(admin_local / "resources.json")
    admin_section = admin_cfg["admin"]
    assert admin_section["idle_timeout"] == 120
    assert "auth_mode" not in admin_section
    assert "token_file" not in admin_section

    config = secure_load_admin_config(Workspace(root_dir=str(prod_dir / "admin@nvidia.com"))).get_admin_config()
    assert config["auth_mode"] == "token"
    assert config["token_file"] == "/tmp/nvflare_alice.token"
    assert config["connection_security"] == "tls"
    assert config["server_identity"] == "server.admin"
    assert config["uid_source"] == "user_input"
    assert config["client_key"] == ""
    assert config["client_cert"] == ""


def test_apply_fedauth_falls_back_to_default_resource_files(tmp_path):
    prod_dir = tmp_path / "example_project" / "prod_00"
    server_local = prod_dir / "server" / "local"
    admin_local = prod_dir / "admin@nvidia.com" / "local"
    server_local.mkdir(parents=True, exist_ok=True)
    admin_local.mkdir(parents=True, exist_ok=True)

    _write_json(server_local / "resources.json.default", {"servers": [{}], "format_version": 1})
    _write_json(admin_local / "resources.json.default", {"format_version": 1, "admin": {"idle_timeout": 300}})
    _prepare_server_material(prod_dir)

    args = Namespace(
        fedauth_issuer="http://127.0.0.1:38080/realms/nvflare",
        fedauth_audience="nvflare-admin",
        fedauth_jwks_uri=None,
        fedauth_discovery_url=None,
        fedauth_alg_allowlist=["RS256"],
        fedauth_required_claims=["iss", "aud", "exp", "iat"],
        fedauth_user_name_claims=["preferred_username"],
        fedauth_user_org_claim="org",
        fedauth_user_role_claim="nvf_role",
        fedauth_role_mappings=["lead=project_admin"],
        fedauth_admin_mode="oidc",
        fedauth_admin_token_file="/tmp/nvflare_alice.token",
        fedauth_oidc_client_id="nvflare-admin",
        fedauth_oidc_scopes="openid profile email offline_access",
        fedauth_oidc_callback_host="127.0.0.1",
        fedauth_oidc_callback_port=39123,
        fedauth_oidc_callback_path="/callback",
        fedauth_oidc_refresh_skew_seconds=60,
        fedauth_oidc_open_browser=True,
        fedauth_oidc_discovery_url="http://127.0.0.1:38080/realms/nvflare/.well-known/openid-configuration",
    )

    apply_fedauth_to_poc_startup_kit(
        prod_dir=str(prod_dir),
        server_name="server",
        admin_name="admin@nvidia.com",
        fedauth_args=args,
    )

    assert (server_local / "resources.json").exists()
    assert (admin_local / "resources.json").exists()
    admin_cfg = _read_json(admin_local / "resources.json")
    assert admin_cfg["admin"]["idle_timeout"] == 300
    assert "auth_mode" not in admin_cfg["admin"]


def test_apply_fedauth_creates_signed_admin_workspace_when_project_has_no_admin(tmp_path):
    prod_dir = tmp_path / "example_project" / "prod_00"
    server_local = prod_dir / "server" / "local"
    server_local.mkdir(parents=True, exist_ok=True)
    _write_json(server_local / "resources.json", {"servers": [{}], "format_version": 1})
    _prepare_server_material(prod_dir)

    args = Namespace(
        fedauth_issuer="http://127.0.0.1:38080/realms/nvflare",
        fedauth_audience="nvflare-admin",
        fedauth_jwks_uri="http://127.0.0.1:38080/realms/nvflare/protocol/openid-connect/certs",
        fedauth_discovery_url=None,
        fedauth_alg_allowlist=["RS256"],
        fedauth_required_claims=["iss", "aud", "exp", "iat"],
        fedauth_user_name_claims=["preferred_username", "email"],
        fedauth_user_org_claim="org",
        fedauth_user_role_claim="nvf_role",
        fedauth_role_mappings=["lead=project_admin"],
        fedauth_admin_mode="oidc",
        fedauth_admin_token_file="/tmp/nvflare_alice.token",
        fedauth_oidc_client_id="nvflare-admin",
        fedauth_oidc_scopes="openid profile email offline_access",
        fedauth_oidc_callback_host="127.0.0.1",
        fedauth_oidc_callback_port=39123,
        fedauth_oidc_callback_path="/callback",
        fedauth_oidc_refresh_skew_seconds=60,
        fedauth_oidc_open_browser=True,
        fedauth_oidc_discovery_url="http://127.0.0.1:38080/realms/nvflare/.well-known/openid-configuration",
    )

    apply_fedauth_to_poc_startup_kit(
        prod_dir=str(prod_dir),
        server_name="server",
        admin_name="admin@nvidia.com",
        fedauth_args=args,
    )

    admin_dir = prod_dir / "admin@nvidia.com"
    assert (admin_dir / "startup" / "fed_admin.json").exists()
    assert (admin_dir / "startup" / "fl_admin.sh").exists()
    assert (admin_dir / "transfer").is_dir()
    assert verify_folder_signature(
        str(admin_dir / "startup"),
        str(admin_dir / "startup" / "rootCA.pem"),
        single_signer=True,
        signature_file=ProvFileName.SIGNATURE_JSON,
    )

    config = secure_load_admin_config(Workspace(root_dir=str(admin_dir))).get_admin_config()
    assert config["project_name"] == "example_project"
    assert config["port"] == 8003
    assert config["auth_mode"] == "oidc"
    assert config["server_identity"] == "server.admin"
    assert config["client_key"] == ""
    assert config["client_cert"] == ""
    local_cfg = _read_json(admin_dir / "local" / "resources.json")
    assert "auth_mode" not in local_cfg["admin"]

    invite_zip = prod_dir / ProvFileName.INVITE_ZIP
    assert invite_zip.exists()
    with ZipFile(invite_zip, "r") as zf:
        names = set(zf.namelist())
    assert "startup/fed_admin.json" in names
    assert "startup/fl_admin.sh" in names
    assert "startup/rootCA.pem" in names
    assert "startup/signature.json" in names
    assert "local/resources.json" in names
    assert "local/signature.json" not in names
    assert any(name.startswith("transfer") for name in names)


def test_prepare_workspace_imports_invite_into_workspace_layout(tmp_path):
    prod_dir = tmp_path / "example_project" / "prod_00"
    server_local = prod_dir / "server" / "local"
    server_local.mkdir(parents=True, exist_ok=True)
    _write_json(server_local / "resources.json", {"servers": [{}], "format_version": 1})
    _prepare_server_material(prod_dir)

    args = Namespace(
        fedauth_issuer="http://127.0.0.1:38080/realms/nvflare",
        fedauth_audience="nvflare-admin",
        fedauth_jwks_uri="http://127.0.0.1:38080/realms/nvflare/protocol/openid-connect/certs",
        fedauth_discovery_url=None,
        fedauth_alg_allowlist=["RS256"],
        fedauth_required_claims=["iss", "aud", "exp", "iat"],
        fedauth_user_name_claims=["preferred_username", "email"],
        fedauth_user_org_claim="org",
        fedauth_user_role_claim="nvf_role",
        fedauth_role_mappings=["lead=project_admin"],
        fedauth_admin_mode="oidc",
        fedauth_admin_token_file="/tmp/nvflare_alice.token",
        fedauth_oidc_client_id="nvflare-admin",
        fedauth_oidc_scopes="openid profile email",
        fedauth_oidc_callback_host="127.0.0.1",
        fedauth_oidc_callback_port=39123,
        fedauth_oidc_callback_path="/callback",
        fedauth_oidc_refresh_skew_seconds=60,
        fedauth_oidc_open_browser=True,
        fedauth_oidc_discovery_url="http://127.0.0.1:38080/realms/nvflare/.well-known/openid-configuration",
    )

    apply_fedauth_to_poc_startup_kit(
        prod_dir=str(prod_dir),
        server_name="server",
        admin_name="admin@nvidia.com",
        fedauth_args=args,
    )

    imported_workspace = tmp_path / "imported_admin"
    workspace_dir = prepare_workspace(
        workspace=str(imported_workspace),
        invite_file=str(prod_dir / ProvFileName.INVITE_ZIP),
        fed_admin=ProvFileName.FED_ADMIN_JSON,
    )

    assert workspace_dir == str(imported_workspace.resolve())
    assert (imported_workspace / "startup" / "fed_admin.json").exists()
    assert (imported_workspace / "startup" / "fl_admin.sh").exists()
    assert (imported_workspace / "local" / "resources.json").exists()
    assert (imported_workspace / "transfer").is_dir()

    config = secure_load_admin_config(Workspace(root_dir=str(imported_workspace))).get_admin_config()
    assert config["auth_mode"] == "oidc"
    assert config["server_identity"] == "server.admin"
    assert config["client_key"] == ""
    assert config["client_cert"] == ""


def test_prepare_workspace_defaults_to_local_folder_next_to_invite(tmp_path):
    prod_dir = tmp_path / "example_project" / "prod_00"
    server_local = prod_dir / "server" / "local"
    server_local.mkdir(parents=True, exist_ok=True)
    _write_json(server_local / "resources.json", {"servers": [{}], "format_version": 1})
    _prepare_server_material(prod_dir)

    args = Namespace(
        fedauth_issuer="http://127.0.0.1:38080/realms/nvflare",
        fedauth_audience="nvflare-admin",
        fedauth_jwks_uri="http://127.0.0.1:38080/realms/nvflare/protocol/openid-connect/certs",
        fedauth_discovery_url=None,
        fedauth_alg_allowlist=["RS256"],
        fedauth_required_claims=["iss", "aud", "exp", "iat"],
        fedauth_user_name_claims=["preferred_username", "email"],
        fedauth_user_org_claim="org",
        fedauth_user_role_claim="nvf_role",
        fedauth_role_mappings=["lead=project_admin"],
        fedauth_admin_mode="oidc",
        fedauth_admin_token_file="/tmp/nvflare_alice.token",
        fedauth_oidc_client_id="nvflare-admin",
        fedauth_oidc_scopes="openid profile email",
        fedauth_oidc_callback_host="127.0.0.1",
        fedauth_oidc_callback_port=39123,
        fedauth_oidc_callback_path="/callback",
        fedauth_oidc_refresh_skew_seconds=60,
        fedauth_oidc_open_browser=True,
        fedauth_oidc_discovery_url="http://127.0.0.1:38080/realms/nvflare/.well-known/openid-configuration",
    )

    apply_fedauth_to_poc_startup_kit(
        prod_dir=str(prod_dir),
        server_name="server",
        admin_name="admin@nvidia.com",
        fedauth_args=args,
    )

    invite_zip = prod_dir / ProvFileName.INVITE_ZIP
    workspace_dir = prepare_workspace(invite_file=str(invite_zip), fed_admin=ProvFileName.FED_ADMIN_JSON)

    expected_dir = prod_dir / "invite"
    assert workspace_dir == str(expected_dir.resolve())
    assert (expected_dir / "startup" / "fed_admin.json").exists()
    assert (expected_dir / "local" / "resources.json").exists()


def test_prepare_workspace_refuses_to_reuse_existing_imported_workspace(tmp_path):
    prod_dir = tmp_path / "example_project" / "prod_00"
    server_local = prod_dir / "server" / "local"
    server_local.mkdir(parents=True, exist_ok=True)
    _write_json(server_local / "resources.json", {"servers": [{}], "format_version": 1})
    _prepare_server_material(prod_dir)

    args = Namespace(
        fedauth_issuer="http://127.0.0.1:38080/realms/nvflare",
        fedauth_audience="nvflare-admin",
        fedauth_jwks_uri="http://127.0.0.1:38080/realms/nvflare/protocol/openid-connect/certs",
        fedauth_discovery_url=None,
        fedauth_alg_allowlist=["RS256"],
        fedauth_required_claims=["iss", "aud", "exp", "iat"],
        fedauth_user_name_claims=["preferred_username", "email"],
        fedauth_user_org_claim="org",
        fedauth_user_role_claim="nvf_role",
        fedauth_role_mappings=["lead=project_admin"],
        fedauth_admin_mode="oidc",
        fedauth_admin_token_file="/tmp/nvflare_alice.token",
        fedauth_oidc_client_id="nvflare-admin",
        fedauth_oidc_scopes="openid profile email",
        fedauth_oidc_callback_host="127.0.0.1",
        fedauth_oidc_callback_port=39123,
        fedauth_oidc_callback_path="/callback",
        fedauth_oidc_refresh_skew_seconds=60,
        fedauth_oidc_open_browser=True,
        fedauth_oidc_discovery_url="http://127.0.0.1:38080/realms/nvflare/.well-known/openid-configuration",
    )

    apply_fedauth_to_poc_startup_kit(
        prod_dir=str(prod_dir),
        server_name="server",
        admin_name="admin@nvidia.com",
        fedauth_args=args,
    )

    invite_zip = prod_dir / ProvFileName.INVITE_ZIP
    workspace_dir = prepare_workspace(invite_file=str(invite_zip), fed_admin=ProvFileName.FED_ADMIN_JSON)
    assert workspace_dir == str((prod_dir / "invite").resolve())

    with pytest.raises(ConfigError, match="already exists"):
        prepare_workspace(invite_file=str(invite_zip), fed_admin=ProvFileName.FED_ADMIN_JSON)


def test_prepare_workspace_rejects_non_workspace_directory_for_invite_import(tmp_path):
    bad_dir = tmp_path / "bad_workspace"
    bad_dir.mkdir()
    _write_text(bad_dir / "unexpected.txt", "not a workspace")

    with pytest.raises(ConfigError):
        prepare_workspace(workspace=str(bad_dir), invite_file=str(tmp_path / ProvFileName.INVITE_ZIP))


def test_prepare_workspace_reports_missing_invite_as_config_error(tmp_path):
    with pytest.raises(ConfigError, match="failed to import invite"):
        prepare_workspace(workspace=str(tmp_path / "imported_admin"), invite_file=str(tmp_path / ProvFileName.INVITE_ZIP))
