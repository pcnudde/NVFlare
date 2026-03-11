# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import sys
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.request import urlopen

import psutil
import jwt

from nvflare.apis.fl_constant import ConnectionSecurity, FLContextKey, SystemConfigs, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.hci.server.token_auth import (
    ClaimMapper,
    ClaimMappingConfig,
    TokenValidationConfig,
    TokenValidator,
)
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.defs import CellChannel, CellChannelTopic
from nvflare.private.fed.runner import Runner
from nvflare.private.fed.server.admin import FedAdminServer
from nvflare.private.fed.server.fed_server import FederatedServer


def _get_required_non_empty_string(config: Mapping[str, Any], key: str, full_name: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{full_name} must be a non-empty string")
    return value.strip()


def _get_optional_non_empty_string(config: Mapping[str, Any], key: str, full_name: str) -> Optional[str]:
    if key not in config:
        return None
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{full_name} must be a non-empty string")
    return value.strip()


def _get_optional_non_negative_int(config: Mapping[str, Any], key: str, full_name: str) -> Optional[int]:
    if key not in config:
        return None
    value = config.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{full_name} must be a non-negative integer")
    return value


def _get_optional_positive_number(config: Mapping[str, Any], key: str, full_name: str) -> Optional[float]:
    if key not in config:
        return None
    value = config.get(key)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{full_name} must be a number > 0")
    return float(value)


def _get_optional_string_sequence(config: Mapping[str, Any], key: str, full_name: str) -> Optional[Sequence[str]]:
    if key not in config:
        return None
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{full_name} must be a non-empty list of strings")
    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{full_name} must be a non-empty list of strings")
        normalized.append(item.strip())
    return normalized


def _get_required_audience(config: Mapping[str, Any], key: str, full_name: str) -> Any:
    value = config.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value:
        normalized = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{full_name} must be a non-empty string or list of non-empty strings")
            normalized.append(item.strip())
        return normalized
    raise ValueError(f"{full_name} must be a non-empty string or list of non-empty strings")


def _get_optional_string_mapping(config: Mapping[str, Any], key: str, full_name: str) -> Optional[Dict[str, str]]:
    if key not in config:
        return None
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{full_name} must be a mapping of strings")
    result = {}
    for map_key, map_value in value.items():
        if not isinstance(map_key, str) or not map_key.strip():
            raise ValueError(f"{full_name} must contain non-empty string keys")
        if not isinstance(map_value, str) or not map_value.strip():
            raise ValueError(f"{full_name} must contain non-empty string values")
        result[map_key.strip()] = map_value.strip()
    return result


def _collect_validated_kwargs(
    config: Mapping[str, Any], full_name: str, key_specs: Sequence[tuple[str, Any]]
) -> Dict[str, Any]:
    result = {}
    for key, validator in key_specs:
        value = validator(config, key, f"{full_name}.{key}")
        if value is not None:
            result[key] = value
    return result


def _load_jwks_from_file(path: str, workspace_dir: Optional[str]) -> Dict[str, Any]:
    if not os.path.isabs(path) and workspace_dir:
        path = os.path.join(workspace_dir, path)
    with open(path, "r") as f:
        content = json.load(f)
    if not isinstance(content, Mapping):
        raise ValueError(f"jwks file '{path}' must contain a JSON object")
    return dict(content)


def _fetch_json_from_url(url: str, timeout: float) -> Dict[str, Any]:
    with urlopen(url, timeout=timeout) as resp:
        content = json.loads(resp.read().decode("utf-8"))
    if not isinstance(content, Mapping):
        raise ValueError(f"url '{url}' did not return a JSON object")
    return dict(content)


def _build_jwks_fetcher_from_remote(token_login: Mapping[str, Any], issuer: str):
    jwks_uri = _get_optional_non_empty_string(token_login, "jwks_uri", "server.admin_auth.token_login.jwks_uri")
    discovery_url = _get_optional_non_empty_string(
        token_login, "discovery_url", "server.admin_auth.token_login.discovery_url"
    )
    cache_ttl = _get_optional_non_negative_int(
        token_login, "jwks_cache_ttl_seconds", "server.admin_auth.token_login.jwks_cache_ttl_seconds"
    )
    if cache_ttl is None:
        cache_ttl = 300
    request_timeout = _get_optional_positive_number(
        token_login, "jwks_request_timeout_seconds", "server.admin_auth.token_login.jwks_request_timeout_seconds"
    )
    if request_timeout is None:
        request_timeout = 5.0

    if not jwks_uri:
        openid_config_url = discovery_url
        if not openid_config_url:
            openid_config_url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"

        metadata = _fetch_json_from_url(openid_config_url, timeout=request_timeout)
        discovered_uri = metadata.get("jwks_uri")
        if not isinstance(discovered_uri, str) or not discovered_uri.strip():
            raise ValueError(f"openid metadata from '{openid_config_url}' missing non-empty jwks_uri")
        jwks_uri = discovered_uri.strip()

    jwks_client = jwt.PyJWKClient(
        jwks_uri,
        cache_jwk_set=cache_ttl > 0,
        lifespan=cache_ttl if cache_ttl > 0 else 300,
        timeout=request_timeout,
    )

    def _fetcher():
        return jwks_client

    return _fetcher


def build_admin_token_login_kwargs(
    server_conf: Optional[Mapping[str, Any]], workspace_dir: Optional[str] = None
) -> Dict[str, Any]:
    if not isinstance(server_conf, Mapping):
        return {}

    admin_auth = server_conf.get("admin_auth")
    if admin_auth is None:
        return {}
    if not isinstance(admin_auth, Mapping):
        raise ValueError("server.admin_auth must be a mapping")

    token_login = admin_auth.get("token_login")
    if token_login is None:
        return {}
    if not isinstance(token_login, Mapping):
        raise ValueError("server.admin_auth.token_login must be a mapping")

    enabled = token_login.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("server.admin_auth.token_login.enabled must be a bool")
    if not enabled:
        return {}

    issuer = _get_required_non_empty_string(token_login, "issuer", "server.admin_auth.token_login.issuer")
    audience = _get_required_audience(token_login, "audience", "server.admin_auth.token_login.audience")

    validation_config_kwargs = {
        "issuer": issuer,
        "audience": audience,
        **_collect_validated_kwargs(
            token_login,
            "server.admin_auth.token_login",
            (
                ("alg_allowlist", _get_optional_string_sequence),
                ("clock_skew_seconds", _get_optional_non_negative_int),
                ("required_claims", _get_optional_string_sequence),
            ),
        ),
    }
    token_validator = TokenValidator(TokenValidationConfig(**validation_config_kwargs))

    claim_mappings = token_login.get("claim_mappings", {})
    if not isinstance(claim_mappings, Mapping):
        raise ValueError("server.admin_auth.token_login.claim_mappings must be a mapping")

    claim_mapping_kwargs = _collect_validated_kwargs(
        claim_mappings,
        "server.admin_auth.token_login.claim_mappings",
        (
            ("user_name_claims", _get_optional_string_sequence),
            ("user_org_claim", _get_optional_non_empty_string),
            ("user_role_claim", _get_optional_non_empty_string),
            ("groups_claim", _get_optional_non_empty_string),
            ("role_mappings", _get_optional_string_mapping),
            ("group_role_mappings", _get_optional_string_mapping),
            ("allowed_roles", _get_optional_string_sequence),
        ),
    )
    claim_mapper = ClaimMapper(ClaimMappingConfig(**claim_mapping_kwargs))

    has_inline_jwks = "jwks" in token_login
    has_jwks_file = "jwks_file" in token_login
    has_jwks_uri = "jwks_uri" in token_login
    has_discovery_url = "discovery_url" in token_login

    jwks_sources_count = sum([has_inline_jwks, has_jwks_file, has_jwks_uri or has_discovery_url])
    if jwks_sources_count > 1:
        raise ValueError(
            "server.admin_auth.token_login must specify only one JWKS source: "
            "'jwks', 'jwks_file', or remote ('jwks_uri'/'discovery_url')"
        )

    if has_inline_jwks:
        jwks = token_login.get("jwks")
        if not isinstance(jwks, Mapping):
            raise ValueError("server.admin_auth.token_login.jwks must be a mapping")
        token_jwks = dict(jwks)
        return {
            "token_validator": token_validator,
            "claim_mapper": claim_mapper,
            "token_jwks": token_jwks,
        }
    elif has_jwks_file:
        jwks_file = _get_required_non_empty_string(
            token_login, "jwks_file", "server.admin_auth.token_login.jwks_file"
        )
        token_jwks = _load_jwks_from_file(jwks_file, workspace_dir=workspace_dir)
        return {
            "token_validator": token_validator,
            "claim_mapper": claim_mapper,
            "token_jwks": token_jwks,
        }
    elif has_jwks_uri or has_discovery_url:
        jwks_fetcher = _build_jwks_fetcher_from_remote(token_login=token_login, issuer=issuer)
        return {
            "token_validator": token_validator,
            "claim_mapper": claim_mapper,
            "jwks_fetcher": jwks_fetcher,
        }
    else:
        raise ValueError("server.admin_auth.token_login requires 'jwks', 'jwks_file', 'jwks_uri', or 'discovery_url'")


def _merge_admin_server_config_with_resources(server_conf: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = dict(server_conf or {})
    resource_servers = ConfigService._get_from_config(
        lambda _name, value: value, name="servers", conf=SystemConfigs.RESOURCES_CONF, default=None
    )
    if isinstance(resource_servers, list) and resource_servers and isinstance(resource_servers[0], Mapping):
        merged.update(dict(resource_servers[0]))
    elif isinstance(resource_servers, Mapping):
        merged.update(dict(resource_servers))
    return merged


def monitor_parent_process(runner: Runner, parent_pid, stop_event: threading.Event):
    while True:
        if stop_event.is_set() or not psutil.pid_exists(parent_pid):
            runner.stop()
            break
        time.sleep(1)


def check_parent_alive(parent_pid, stop_event: threading.Event):
    while True:
        if stop_event.is_set() or not psutil.pid_exists(parent_pid):
            pid = os.getpid()
            kill_child_processes(pid)
            os.killpg(os.getpgid(pid), 9)
            break
        time.sleep(1)


def kill_child_processes(parent_pid):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.kill()


def create_admin_server(fl_server: FederatedServer, server_conf=None, args=None):
    """To create the admin server.

    Args:
        fl_server: fl_server
        server_conf: server config
        args: command args

    Returns:
        A FedAdminServer.
    """
    admin_server_conf = _merge_admin_server_config_with_resources(server_conf)
    token_login_kwargs = build_admin_token_login_kwargs(server_conf=admin_server_conf, workspace_dir=args.workspace)
    target = admin_server_conf.get("service", {}).get("target", "0.0.0.0:6007")
    scheme = admin_server_conf.get("service", {}).get("scheme", "grpc")
    target_parts = target.split(":")
    fl_port = int(target_parts[-1])
    admin_port = int(admin_server_conf.get("admin_port", fl_port))
    admin_conn_sec = str(
        admin_server_conf.get(
            "admin_connection_security",
            admin_server_conf.get("connection_security", ConnectionSecurity.MTLS),
        )
    ).strip()
    admin_identity = admin_server_conf.get("admin_interface_identity", f"{FQCN.ROOT_SERVER}.admin")

    def _resolve_startup_file(file_name: str) -> str:
        if os.path.isabs(file_name):
            return file_name
        return os.path.join(args.workspace, WorkspaceConstants.STARTUP_FOLDER_NAME, file_name)

    command_cell = fl_server.cell
    own_command_cell = False
    if admin_port != fl_port:
        secure_conn = admin_conn_sec != ConnectionSecurity.CLEAR
        credentials = {}
        if secure_conn:
            credentials = {
                DriverParams.CA_CERT.value: _resolve_startup_file(admin_server_conf.get("ssl_root_cert")),
                DriverParams.SERVER_CERT.value: _resolve_startup_file(admin_server_conf.get("ssl_cert")),
                DriverParams.SERVER_KEY.value: _resolve_startup_file(admin_server_conf.get("ssl_private_key")),
                DriverParams.CONNECTION_SECURITY.value: admin_conn_sec,
            }

        command_cell = Cell(
            fqcn=admin_identity,
            root_url=f"{scheme}://0:{admin_port}",
            secure=secure_conn,
            credentials=credentials,
            create_internal_listener=False,
            parent_url=None,
        )
        command_cell.register_request_cb(
            channel=CellChannel.SERVER_MAIN,
            topic=CellChannelTopic.Challenge,
            cb=fl_server.client_challenge,
        )
        command_cell.start()
        mpm.add_cleanup_cb(command_cell.stop)
        own_command_cell = True

    admin_server = FedAdminServer(
        cell=command_cell,
        fed_admin_interface=fl_server.engine,
        cmd_modules=fl_server.cmd_modules,
        file_upload_dir=os.path.join(args.workspace, admin_server_conf.get("admin_storage", "tmp")),
        file_download_dir=os.path.join(args.workspace, admin_server_conf.get("admin_storage", "tmp")),
        download_job_url=admin_server_conf.get("download_job_url", "http://"),
        timeout=admin_server_conf.get("admin_timeout", 10.0),
        network_cell=fl_server.cell,
        own_command_cell=own_command_cell,
        **token_login_kwargs,
    )
    return admin_server


def version_check():
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "Python versions 3.13 and above are not yet supported. Please use Python version between 3.9 and 3.12."
        )
    if sys.version_info < (3, 9):
        raise RuntimeError(
            "Python versions 3.8 and below are not supported. Please use Python version between 3.9 and 3.12."
        )


def init_security_content_service(workspace_dir):
    content_folder_path = os.path.join(workspace_dir, WorkspaceConstants.STARTUP_FOLDER_NAME)
    os.makedirs(content_folder_path, exist_ok=True)
    SecurityContentService.initialize(content_folder=content_folder_path)


def component_security_check(fl_ctx: FLContext):
    exceptions = fl_ctx.get_prop(FLContextKey.EXCEPTIONS)
    if exceptions:
        for _, exception in exceptions.items():
            if isinstance(exception, UnsafeComponentError):
                print(f"Unsafe component configured, could not start {fl_ctx.get_identity_name()}!!")
                raise RuntimeError(exception)
