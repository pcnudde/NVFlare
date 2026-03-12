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
import os
import shutil
import socket
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from nvflare.apis.workspace import Workspace
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.client.config import secure_load_admin_config
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
from tests.integration_test.src.poc_site_launcher import POCSiteLauncher
from tests.integration_test.src.utils import check_client_status_ready, check_job_done

ISSUER = "https://token-auth-demo.nvflare"
AUDIENCE = "nvflare-admin"
TOKEN_LOGIN_USER = "admin@nvidia.com"
TOKEN_LOGIN_ORG = "nvidia"
TOKEN_LOGIN_ROLE = "project_admin"
KEY_ID = "demo-kid"
EXAMPLE_JOB = "hello-numpy-sag"


def _make_signing_material():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk["kid"] = KEY_ID
    public_jwk["use"] = "sig"
    public_jwk["alg"] = "RS256"
    return private_key_pem, {"keys": [public_jwk]}


def _make_access_token(private_key_pem: str):
    now = int(time.time())
    claims = {
        "sub": "token-auth-demo-subject",
        "preferred_username": TOKEN_LOGIN_USER,
        "org": TOKEN_LOGIN_ORG,
        "nvf_role": TOKEN_LOGIN_ROLE,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now - 1,
        "nbf": now - 1,
        "exp": now + 600,
    }
    return jwt.encode(claims, private_key_pem, algorithm="RS256", headers={"kid": KEY_ID})


def _make_fedauth_args():
    return Namespace(
        fedauth_issuer=ISSUER,
        fedauth_audience=AUDIENCE,
        fedauth_jwks_uri="http://127.0.0.1:39080/jwks",
        fedauth_discovery_url=None,
        fedauth_alg_allowlist=["RS256"],
        fedauth_required_claims=["iss", "aud", "exp", "iat"],
        fedauth_user_name_claims=["preferred_username", "email"],
        fedauth_user_org_claim="org",
        fedauth_user_role_claim="nvf_role",
        fedauth_role_mappings=["project_admin=project_admin"],
        fedauth_oidc_client_id=AUDIENCE,
        fedauth_oidc_scopes="openid profile email offline_access",
        fedauth_oidc_callback_host="127.0.0.1",
        fedauth_oidc_callback_port=39123,
        fedauth_oidc_callback_path="/callback",
        fedauth_oidc_refresh_skew_seconds=60,
        fedauth_oidc_open_browser=False,
        fedauth_oidc_discovery_url=None,
    )


def _server_resource_file(workspace_root: str) -> str:
    return os.path.join(workspace_root, "server", "local", "resources.json")


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _enable_token_auth_on_server(workspace_root: str, jwks: dict):
    resource_file = _server_resource_file(workspace_root)
    with open(resource_file, "r") as f:
        resources = json.load(f)

    if "servers" not in resources or not resources["servers"]:
        raise RuntimeError(f"invalid server resources file: {resource_file}")

    server_entry = resources["servers"][0]
    server_entry["admin_connection_security"] = "tls"
    server_entry["admin_interface_identity"] = "server.admin"
    server_entry["admin_auth"] = {
        "token_login": {
            "enabled": True,
            "issuer": ISSUER,
            "audience": AUDIENCE,
            "alg_allowlist": ["RS256"],
            "jwks": jwks,
            "claim_mappings": {
                "user_name_claims": ["preferred_username", "email"],
                "user_org_claim": "org",
                "user_role_claim": "nvf_role",
            },
        }
    }

    with open(resource_file, "w") as f:
        json.dump(resources, f)


def _read_admin_config(workspace_root: str, admin_user_name: str) -> dict:
    admin_dir = os.path.join(workspace_root, admin_user_name)
    workspace = Workspace(root_dir=admin_dir)
    conf = secure_load_admin_config(workspace)
    return conf.get_admin_config()


def _create_token_admin_api(workspace_root: str, upload_dir: str, download_dir: str, token: str, admin_port: int):
    admin_config = _read_admin_config(workspace_root=workspace_root, admin_user_name=TOKEN_LOGIN_USER)
    admin_config[AdminConfigKey.AUTH_MODE] = "token"
    admin_config[AdminConfigKey.TOKEN] = token
    admin_config[AdminConfigKey.CONNECTION_SECURITY] = "tls"
    admin_config[AdminConfigKey.HOST] = "127.0.0.1"
    admin_config[AdminConfigKey.PORT] = admin_port
    admin_config[AdminConfigKey.SERVER_IDENTITY] = "server.admin"
    admin_config[AdminConfigKey.UID_SOURCE] = "user_input"
    admin_config[AdminConfigKey.CLIENT_CERT] = ""
    admin_config[AdminConfigKey.CLIENT_KEY] = ""

    api = FLAdminAPI(
        upload_dir=upload_dir,
        download_dir=download_dir,
        user_name=TOKEN_LOGIN_USER,
        admin_config=admin_config,
        auto_login_max_tries=20,
    )
    api.connect(10.0)
    api.login()
    return api


def _wait_for_clients_ready(admin_api: FLAdminAPI, expected_clients: int, timeout_seconds: int = 180):
    end_time = time.time() + timeout_seconds
    while time.time() < end_time:
        response = admin_api.check_status(target_type=TargetType.CLIENT)
        if check_client_status_ready(response):
            rows = response["details"].get("client_statuses", [])
            # header row + expected clients
            if len(rows) >= expected_clients + 1:
                return
        time.sleep(1.0)
    raise RuntimeError(f"clients did not become ready in {timeout_seconds} seconds")


def _wait_for_job_done(admin_api: FLAdminAPI, job_id: str, timeout_seconds: int = 300):
    end_time = time.time() + timeout_seconds
    while time.time() < end_time:
        if check_job_done(job_id=job_id, admin_api=admin_api):
            return
        time.sleep(2.0)
    raise RuntimeError(f"job {job_id} did not finish in {timeout_seconds} seconds")


def test_token_auth_poc_example_e2e():
    if os.environ.get("TOKEN_AUTH_E2E_REQUIRED", "0").lower() not in ("1", "true", "yes"):
        pytest.skip("set TOKEN_AUTH_E2E_REQUIRED=1 to run token-auth e2e POC demo test")

    jobs_root_dir = str(Path(__file__).parent / "data" / "jobs")
    private_key_pem, jwks = _make_signing_material()
    access_token = _make_access_token(private_key_pem=private_key_pem)
    fl_port = _get_free_port()
    admin_port = _get_free_port()

    site_launcher = POCSiteLauncher(
        n_servers=1,
        n_clients=2,
        fed_learn_port=fl_port,
        admin_port=admin_port,
        include_project_admin=False,
        fedauth_args=_make_fedauth_args(),
    )
    workspace_root = site_launcher.prepare_workspace()
    with open(site_launcher.project_conf_path, "r") as f:
        project_config = yaml.safe_load(f)
    assert not [p for p in project_config["participants"] if p.get("type") == "admin"]
    _enable_token_auth_on_server(workspace_root=workspace_root, jwks=jwks)

    download_dir = tempfile.mkdtemp(prefix="nvflare-token-auth-download-")
    admin_api = None
    try:
        site_launcher.start_servers()
        site_launcher.start_clients()

        admin_api = _create_token_admin_api(
            workspace_root=workspace_root,
            upload_dir=jobs_root_dir,
            download_dir=download_dir,
            token=access_token,
            admin_port=admin_port,
        )
        assert admin_api.is_ready(), "admin API failed to login with token auth"

        _wait_for_clients_ready(admin_api=admin_api, expected_clients=2, timeout_seconds=180)

        submit_response = admin_api.submit_job(EXAMPLE_JOB)
        assert submit_response.get("status") == APIStatus.SUCCESS, f"job submission failed: {submit_response}"
        job_id = submit_response.get("details", {}).get("job_id")
        assert isinstance(job_id, str) and job_id, f"missing job id in submit response: {submit_response}"

        _wait_for_job_done(admin_api=admin_api, job_id=job_id, timeout_seconds=300)
    finally:
        if admin_api:
            admin_api.close()
        site_launcher.stop_all_sites()
        site_launcher.cleanup()
        shutil.rmtree(download_dir, ignore_errors=True)
