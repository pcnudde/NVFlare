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

import psutil

from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.fuel.hci.server.token_auth import (
    ClaimMapper,
    ClaimMappingConfig,
    TokenValidationConfig,
    TokenValidator,
)
from nvflare.fuel.sec.security_content_service import SecurityContentService
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


def _load_jwks_from_file(path: str, workspace_dir: Optional[str]) -> Dict[str, Any]:
    if not os.path.isabs(path) and workspace_dir:
        path = os.path.join(workspace_dir, path)
    with open(path, "r") as f:
        content = json.load(f)
    if not isinstance(content, Mapping):
        raise ValueError(f"jwks file '{path}' must contain a JSON object")
    return dict(content)


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

    validation_config_kwargs = {"issuer": issuer, "audience": audience}
    alg_allowlist = _get_optional_string_sequence(
        token_login, "alg_allowlist", "server.admin_auth.token_login.alg_allowlist"
    )
    if alg_allowlist is not None:
        validation_config_kwargs["alg_allowlist"] = alg_allowlist
    clock_skew_seconds = _get_optional_non_negative_int(
        token_login, "clock_skew_seconds", "server.admin_auth.token_login.clock_skew_seconds"
    )
    if clock_skew_seconds is not None:
        validation_config_kwargs["clock_skew_seconds"] = clock_skew_seconds
    required_claims = _get_optional_string_sequence(
        token_login, "required_claims", "server.admin_auth.token_login.required_claims"
    )
    if required_claims is not None:
        validation_config_kwargs["required_claims"] = required_claims
    token_validator = TokenValidator(TokenValidationConfig(**validation_config_kwargs))

    claim_mappings = token_login.get("claim_mappings", {})
    if not isinstance(claim_mappings, Mapping):
        raise ValueError("server.admin_auth.token_login.claim_mappings must be a mapping")

    claim_mapping_kwargs = {}
    user_name_claims = _get_optional_string_sequence(
        claim_mappings, "user_name_claims", "server.admin_auth.token_login.claim_mappings.user_name_claims"
    )
    if user_name_claims is not None:
        claim_mapping_kwargs["user_name_claims"] = user_name_claims

    user_org_claim = _get_optional_non_empty_string(
        claim_mappings, "user_org_claim", "server.admin_auth.token_login.claim_mappings.user_org_claim"
    )
    if user_org_claim is not None:
        claim_mapping_kwargs["user_org_claim"] = user_org_claim

    user_role_claim = _get_optional_non_empty_string(
        claim_mappings, "user_role_claim", "server.admin_auth.token_login.claim_mappings.user_role_claim"
    )
    if user_role_claim is not None:
        claim_mapping_kwargs["user_role_claim"] = user_role_claim

    groups_claim = _get_optional_non_empty_string(
        claim_mappings, "groups_claim", "server.admin_auth.token_login.claim_mappings.groups_claim"
    )
    if groups_claim is not None:
        claim_mapping_kwargs["groups_claim"] = groups_claim

    role_mappings = _get_optional_string_mapping(
        claim_mappings, "role_mappings", "server.admin_auth.token_login.claim_mappings.role_mappings"
    )
    if role_mappings is not None:
        claim_mapping_kwargs["role_mappings"] = role_mappings

    group_role_mappings = _get_optional_string_mapping(
        claim_mappings, "group_role_mappings", "server.admin_auth.token_login.claim_mappings.group_role_mappings"
    )
    if group_role_mappings is not None:
        claim_mapping_kwargs["group_role_mappings"] = group_role_mappings

    allowed_roles = _get_optional_string_sequence(
        claim_mappings, "allowed_roles", "server.admin_auth.token_login.claim_mappings.allowed_roles"
    )
    if allowed_roles is not None:
        claim_mapping_kwargs["allowed_roles"] = allowed_roles
    claim_mapper = ClaimMapper(ClaimMappingConfig(**claim_mapping_kwargs))

    has_inline_jwks = "jwks" in token_login
    has_jwks_file = "jwks_file" in token_login
    if has_inline_jwks and has_jwks_file:
        raise ValueError("server.admin_auth.token_login must specify only one of 'jwks' or 'jwks_file'")

    if has_inline_jwks:
        jwks = token_login.get("jwks")
        if not isinstance(jwks, Mapping):
            raise ValueError("server.admin_auth.token_login.jwks must be a mapping")
        token_jwks = dict(jwks)
    elif has_jwks_file:
        jwks_file = _get_required_non_empty_string(
            token_login, "jwks_file", "server.admin_auth.token_login.jwks_file"
        )
        token_jwks = _load_jwks_from_file(jwks_file, workspace_dir=workspace_dir)
    else:
        raise ValueError("server.admin_auth.token_login requires 'jwks' or 'jwks_file'")

    return {
        "token_validator": token_validator,
        "claim_mapper": claim_mapper,
        "token_jwks": token_jwks,
    }


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
    token_login_kwargs = build_admin_token_login_kwargs(server_conf=server_conf, workspace_dir=args.workspace)

    admin_server = FedAdminServer(
        cell=fl_server.cell,
        fed_admin_interface=fl_server.engine,
        cmd_modules=fl_server.cmd_modules,
        file_upload_dir=os.path.join(args.workspace, server_conf.get("admin_storage", "tmp")),
        file_download_dir=os.path.join(args.workspace, server_conf.get("admin_storage", "tmp")),
        download_job_url=server_conf.get("download_job_url", "http://"),
        timeout=server_conf.get("admin_timeout", 10.0),
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
