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
import copy
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple

import yaml
from pyhocon import ConfigFactory as CF

from nvflare.cli_exception import CLIException
from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.fuel.utils.zip_utils import zip_directory_to_file
from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.gpu_utils import get_host_gpu_ids
from nvflare.lighter.constants import ProvisionMode, ProvFileName
from nvflare.lighter.prov_utils import prepare_builders, prepare_packager
from nvflare.lighter.provision import gen_default_project_config, prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import (
    load_yaml,
    load_private_key,
    sign_folders,
    update_project_server_name_config,
    update_server_default_host,
    update_storage_locations,
)
from nvflare.tool.api_utils import shutdown_system
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC
from nvflare.utils.cli_utils import get_hidden_nvflare_config_path, get_or_create_hidden_nvflare_dir, hocon_to_string

DEFAULT_WORKSPACE = "/tmp/nvflare/poc"
DEFAULT_PROJECT_NAME = "example_project"

CMD_PREPARE_POC = "prepare"
CMD_PREPARE_JOBS_DIR = "prepare-jobs-dir"
CMD_START_POC = "start"
CMD_STOP_POC = "stop"
CMD_CLEAN_POC = "clean"


def client_gpu_assignments(clients: List[str], gpu_ids: List[int]) -> Dict[str, List[int]]:
    n_gpus = len(gpu_ids)
    n_clients = len(clients)
    gpu_assignments = {}
    if n_gpus == 0:
        for client in clients:
            gpu_assignments[client] = []

    if 0 < n_gpus <= n_clients:
        for client_id, client in enumerate(clients):
            gpu_index = client_id % n_gpus
            gpu_assignments[client] = [gpu_ids[gpu_index]]
    elif n_gpus > n_clients > 0:
        client_name_map = {}
        for client_id, client in enumerate(clients):
            client_name_map[client_id] = client

        for gpu_index, gpu_id in enumerate(gpu_ids):
            client_id = gpu_index % n_clients
            client = client_name_map[client_id]
            if client not in gpu_assignments:
                gpu_assignments[client] = []
            gpu_assignments[client].append(gpu_ids[gpu_index])
    return gpu_assignments


def get_service_command(cmd_type: str, prod_dir: str, service_dir, service_config: Dict) -> str:
    cmd = ""
    proj_admin_dir_name = service_config.get(SC.FLARE_PROJ_ADMIN, SC.FLARE_PROJ_ADMIN)
    admin_dirs = list(service_config.get(SC.FLARE_OTHER_ADMINS, []))
    admin_dirs.append(proj_admin_dir_name)

    if cmd_type == SC.CMD_START:
        if not service_config.get(SC.IS_DOCKER_RUN):
            if service_dir in admin_dirs:
                cmd = get_cmd_path(prod_dir, service_dir, "fl_admin.sh")
            else:
                cmd = get_cmd_path(prod_dir, service_dir, "start.sh")
        else:
            if service_dir in admin_dirs:
                cmd = get_cmd_path(prod_dir, service_dir, "fl_admin.sh")
            else:
                cmd = get_cmd_path(prod_dir, service_dir, "docker.sh -d")

    elif cmd_type == SC.CMD_STOP:
        if not service_config.get(SC.IS_DOCKER_RUN):
            cmd = get_stop_cmd(prod_dir, service_dir)
        else:
            if service_dir in admin_dirs:
                cmd = get_stop_cmd(prod_dir, service_dir)
            else:
                cmd = f"docker stop {service_dir}"

    else:
        raise CLIException("unknown cmd_type :", cmd_type)
    return cmd


def get_stop_cmd(poc_workspace: str, service_dir_name: str):
    service_dir = os.path.join(poc_workspace, service_dir_name)
    stop_file = os.path.join(service_dir, "shutdown.fl")
    return f"touch {stop_file}"


def get_nvflare_home() -> Optional[str]:
    nvflare_home = None
    if "NVFLARE_HOME" in os.environ:
        nvflare_home = os.getenv("NVFLARE_HOME")
        if nvflare_home:
            if nvflare_home.endswith("/"):
                nvflare_home = nvflare_home[:-1]
    return nvflare_home


def get_upload_dir(startup_dir) -> str:
    console_config_path = os.path.join(startup_dir, "fed_admin.json")
    try:
        with open(console_config_path, "r") as f:
            console_config = json.load(f)
            upload_dir = console_config["admin"]["upload_dir"]
    except IOError as e:
        raise CLIException(f"failed to load {console_config_path} {e}")
    except json.decoder.JSONDecodeError as e:
        raise CLIException(f"failed to load {console_config_path}, please double check the configuration {e}")
    return upload_dir


def is_dir_empty(path: str):
    targe_dir = os.listdir(path)
    return len(targe_dir) == 0


def prepare_jobs_dir(cmd_args):
    poc_workspace = get_poc_workspace()
    _prepare_jobs_dir(cmd_args.jobs_dir, poc_workspace)


def _prepare_jobs_dir(jobs_dir: str, workspace: str, config_packages: Optional[Tuple] = None):
    project_config, service_config = config_packages if config_packages else setup_service_config(workspace)
    project_name = project_config.get("name")
    if jobs_dir is None or jobs_dir == "":
        raise CLIException("jobs_dir is required")
    src = os.path.abspath(jobs_dir)
    if not os.path.isdir(src):
        raise CLIException(f"jobs_dir '{jobs_dir}' is not valid directory")

    prod_dir = get_prod_dir(workspace, project_name)
    if not os.path.exists(prod_dir):
        raise CLIException("please use nvflare poc prepare to create workspace first")

    console_dir = os.path.join(prod_dir, f"{service_config[SC.FLARE_PROJ_ADMIN]}")
    startup_dir = os.path.join(console_dir, SC.STARTUP)
    transfer = get_upload_dir(startup_dir)
    dst = os.path.join(console_dir, transfer)
    if not is_dir_empty(dst):
        print(" ")
        answer = input(f"job directory at {dst} is already exists, replace with new one ? (y/N) ")
        if answer.strip().upper() == "Y":
            if os.path.islink(dst):
                os.unlink(dst)
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)

            print(f"link job directory from {src} to {dst}")
            os.symlink(src, dst)
    else:
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        print(f"link job directory from {src} to {dst}")
        os.symlink(src, dst)


def get_prod_dir(workspace, project_name: str = DEFAULT_PROJECT_NAME):
    project_name = project_name if project_name else DEFAULT_PROJECT_NAME
    prod_dir = os.path.join(workspace, project_name, "prod_00")
    return prod_dir


def gen_project_config_file(workspace: str) -> str:
    project_file = os.path.join(workspace, "project.yml")
    if not os.path.isfile(project_file):
        gen_default_project_config("dummy_project.yml", project_file)
    return project_file


def verify_host(host_name: str) -> bool:
    try:
        host_name = socket.gethostbyname(host_name)
        return True
    except:
        return False


def verify_hosts(project_config: OrderedDict):
    hosts: List[str] = get_project_hosts(project_config)
    for h in hosts:
        if not verify_host(h):
            print(f"host name: '{h}' is not defined, considering modify /etc/hosts to add localhost alias")
            exit(0)


def get_project_hosts(project_config) -> List[str]:
    participants: List[dict] = project_config["participants"]
    return [p["name"] for p in participants if p["type"] == "client" or p["type"] == "server"]


def get_fl_server_name(project_config: OrderedDict) -> str:
    participants: List[dict] = project_config["participants"]
    servers = [p["name"] for p in participants if p["type"] == "server"]
    if len(servers) == 1:
        return servers[0]
    else:
        raise CLIException(f"project should only have one server, but {len(servers)} are provided: {servers}")


def get_fl_admins(project_config: OrderedDict, is_project_admin: bool):
    participants: List[dict] = project_config["participants"]
    return [
        p["name"]
        for p in participants
        if p["type"] == "admin" and (p["role"] == "project_admin" if is_project_admin else p["role"] != "project_admin")
    ]


def get_other_admins(project_config: OrderedDict):
    return get_fl_admins(project_config, is_project_admin=False)


def get_proj_admin(project_config: OrderedDict, default_project_admin: Optional[str] = None):
    admins = get_fl_admins(project_config, is_project_admin=True)
    if len(admins) == 1:
        return admins[0]
    if len(admins) == 0 and default_project_admin:
        return default_project_admin
    else:
        raise CLIException(f"project should have only one project admin, but {len(admins)} are provided: {admins}")


def get_fl_client_names(project_config: OrderedDict) -> List[str]:
    participants: List[dict] = project_config["participants"]
    client_names = [p["name"] for p in participants if p["type"] == "client"]
    return client_names


def local_provision(
    clients: List[str],
    number_of_clients: int,
    workspace: str,
    docker_image: str,
    use_he: bool = False,
    project_conf_path: str = "",
    default_project_admin: Optional[str] = None,
) -> Tuple:
    user_provided_project_config = False
    if project_conf_path:
        src_project_file = project_conf_path
        dst_project_file = os.path.join(workspace, "project.yml")
        user_provided_project_config = True
    else:
        src_project_file = gen_project_config_file(workspace)
        dst_project_file = src_project_file

    print(f"provision at {workspace} for {number_of_clients} clients with {src_project_file}")
    project_config: OrderedDict = load_yaml(src_project_file)
    if not project_config:
        raise CLIException(f"empty or invalid project config from project yaml file: {src_project_file}")

    if not user_provided_project_config:
        project_config = update_server_name(project_config)
        project_config = update_clients(clients, number_of_clients, project_config)
        project_config = add_he_builder(use_he, project_config)
        if docker_image:
            project_config = update_static_file_builder(docker_image, project_config)
    project_config = update_server_default_host(project_config, "localhost")
    save_project_config(project_config, dst_project_file)
    service_config = get_service_config(project_config, default_project_admin=default_project_admin)
    return_project_config = copy.deepcopy(project_config)
    project = prepare_project(project_config)
    builders = prepare_builders(project_config)
    packager = prepare_packager(project_config)
    provisioner = Provisioner(workspace, builders, packager)
    provisioner.provision(project, mode=ProvisionMode.POC)

    return return_project_config, service_config


def get_service_config(project_config, default_project_admin: Optional[str] = None):
    service_config = {
        SC.FLARE_SERVER: get_fl_server_name(project_config),
        SC.FLARE_PROJ_ADMIN: get_proj_admin(project_config, default_project_admin=default_project_admin),
        SC.FLARE_OTHER_ADMINS: get_other_admins(project_config),
        SC.FLARE_CLIENTS: get_fl_client_names(project_config),
        SC.IS_DOCKER_RUN: is_docker_run(project_config),
    }
    return service_config


def save_project_config(project_config, project_file):
    with open(project_file, "w") as file:
        yaml.dump(project_config, file)


def update_server_name(project_config):
    old_server_name = get_fl_server_name(project_config)
    server_name = "server"
    if old_server_name != server_name:
        update_project_server_name_config(project_config, old_server_name, server_name)
    return project_config


def is_docker_run(project_config: OrderedDict):
    if "builders" not in project_config:
        return False
    static_builder = [
        b
        for b in project_config.get("builders")
        if b.get("path") == "nvflare.lighter.impl.static_file.StaticFileBuilder"
    ][0]
    return "docker_image" in static_builder["args"]


def update_static_file_builder(docker_image: str, project_config: OrderedDict):
    # need to keep the order of the builders
    for b in project_config.get("builders"):
        if b.get("path") == "nvflare.lighter.impl.static_file.StaticFileBuilder":
            b["args"]["docker_image"] = docker_image

    return project_config


def add_docker_builder(use_docker: bool, project_config: OrderedDict):
    if use_docker:
        docker_builder = {
            "path": "nvflare.lighter.impl.docker.DockerBuilder",
            "args": {"base_image": "python:3.8", "requirements_file": "requirements.txt"},
        }
        project_config["builders"].append(docker_builder)

    return project_config


def add_he_builder(use_he: bool, project_config: OrderedDict):
    if use_he:
        he_builder = {
            "path": "nvflare.lighter.impl.he.HEBuilder",
            "args": {},
        }
        project_config["builders"].insert(-1, he_builder)

    return project_config


def update_clients(clients: List[str], n_clients: int, project_config: OrderedDict) -> OrderedDict:
    requested_clients = prepare_clients(clients, n_clients)

    participants: List[dict] = project_config["participants"]
    new_participants = [p for p in participants if p["type"] != "client"]

    for client in requested_clients:
        client_dict = {"name": client, "type": "client", "org": "nvidia"}
        new_participants.append(client_dict)

    project_config["participants"] = new_participants

    return project_config


def prepare_clients(clients, number_of_clients):
    if not clients:
        clients = []
        for i in range(number_of_clients):
            clients.append(f"site-{(i + 1)}")

    return clients


def save_startup_kit_dir_config(workspace, project_name):
    dst = get_or_create_hidden_nvflare_config_path()
    config = None
    if os.path.isfile(dst):
        try:
            config = CF.parse_file(dst)
        except Exception:
            config = None

    prod_dir = get_prod_dir(workspace, project_name)
    conf = f"""
        startup_kit {{
            path = {prod_dir}
        }}
        poc_workspace {{
            path = {workspace}
        }}
    """
    if config:
        new_config = CF.parse_string(conf)
        config = new_config.with_fallback(config)
        config_str = hocon_to_string(ConfigFormat.PYHOCON, config)
    else:
        config_str = conf

    with open(dst, "w") as file:
        file.write(f"{config_str}\n")


def prepare_poc(cmd_args):
    poc_workspace = get_poc_workspace()
    project_conf_path = ""
    if cmd_args.project_input:
        project_conf_path = cmd_args.project_input

    _prepare_poc(
        cmd_args.clients,
        cmd_args.number_of_clients,
        poc_workspace,
        cmd_args.docker_image,
        cmd_args.he,
        project_conf_path,
        cmd_args if cmd_args.enable_fedauth else None,
    )


def _prepare_poc(
    clients: List[str],
    number_of_clients: int,
    workspace: str,
    docker_image: Optional[str] = None,
    use_he: bool = False,
    project_conf_path: str = "",
    examples_dir: Optional[str] = None,
    fedauth_args=None,
) -> bool:
    if clients:
        number_of_clients = len(clients)
    if not project_conf_path:
        print(f"prepare poc at {workspace} for {number_of_clients} clients")
    else:
        print(f"prepare poc at {workspace} with {project_conf_path}")

    project_config = None
    if os.path.exists(workspace):
        answer = input(
            f"This will delete poc workspace directory: '{workspace}' and create a new one. Is it OK to proceed? (y/N) "
        )
        if answer.strip().upper() == "Y":

            workspace_path = Path(workspace)
            project_file = Path(project_conf_path)
            if workspace_path in project_file.parents:
                raise CLIException(
                    f"\nProject file: '{project_conf_path}' is under workspace directory:"
                    f"'{workspace}', which is to be deleted. "
                    f"Please copy {project_conf_path} to different location before running this command."
                )

            shutil.rmtree(workspace, ignore_errors=True)
        else:
            return False

    default_project_admin = SC.FLARE_PROJ_ADMIN if fedauth_args else None
    project_config = prepare_poc_provision(
        clients,
        number_of_clients,
        workspace,
        docker_image,
        use_he,
        project_conf_path,
        examples_dir,
        default_project_admin=default_project_admin,
    )
    if fedauth_args:
        service_config = get_service_config(project_config, default_project_admin=default_project_admin)
        project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
        prod_dir = get_prod_dir(workspace, project_name)
        apply_fedauth_to_poc_startup_kit(
            prod_dir=prod_dir,
            server_name=service_config[SC.FLARE_SERVER],
            admin_name=service_config[SC.FLARE_PROJ_ADMIN],
            fedauth_args=fedauth_args,
        )

    project_name = project_config.get("name") if project_config else None
    save_startup_kit_dir_config(workspace, project_name)
    return True


def _load_json_file(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise CLIException(f"invalid JSON object in {path}")
    return payload


def _write_json_file(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _parse_role_mappings(role_mapping_items: List[str]) -> Dict[str, str]:
    mappings = {}
    for item in role_mapping_items:
        if "=" not in item:
            raise CLIException(f"invalid --fedauth_role_mappings value '{item}': expected <source>=<target>")
        source, target = item.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise CLIException(f"invalid --fedauth_role_mappings value '{item}': expected non-empty source/target")
        mappings[source] = target
    return mappings


@dataclass(frozen=True)
class FedAuthAdminBootstrapConfig:
    issuer: str
    audience: str
    alg_allowlist: List[str]
    required_claims: List[str]
    user_name_claims: List[str]
    user_org_claim: str
    user_role_claim: str
    role_mappings: Dict[str, str]
    jwks_uri: Optional[str]
    discovery_url: Optional[str]
    admin_mode: str
    oidc_client_id: str
    oidc_scopes: str
    oidc_discovery_url: Optional[str]
    oidc_callback_host: str
    oidc_callback_port: int
    oidc_callback_path: str
    oidc_refresh_skew_seconds: int
    oidc_open_browser: bool
    admin_token_file: str

    @classmethod
    def from_args(cls, fedauth_args):
        issuer = str(getattr(fedauth_args, "fedauth_issuer", "")).strip()
        if not issuer:
            raise CLIException("--enable_fedauth requires --fedauth_issuer")

        audience = str(getattr(fedauth_args, "fedauth_audience", "nvflare-admin")).strip()
        if not audience:
            raise CLIException("fedauth_audience must be non-empty")

        return cls(
            issuer=issuer,
            audience=audience,
            alg_allowlist=list(getattr(fedauth_args, "fedauth_alg_allowlist", ["RS256"])),
            required_claims=list(getattr(fedauth_args, "fedauth_required_claims", ["iss", "aud", "exp", "iat"])),
            user_name_claims=list(getattr(fedauth_args, "fedauth_user_name_claims", ["preferred_username", "email"])),
            user_org_claim=str(getattr(fedauth_args, "fedauth_user_org_claim", "org")).strip() or "org",
            user_role_claim=str(getattr(fedauth_args, "fedauth_user_role_claim", "nvf_role")).strip() or "nvf_role",
            role_mappings=_parse_role_mappings(
                list(getattr(fedauth_args, "fedauth_role_mappings", ["lead=project_admin"]))
            ),
            jwks_uri=str(getattr(fedauth_args, "fedauth_jwks_uri", "")).strip() or None,
            discovery_url=str(getattr(fedauth_args, "fedauth_discovery_url", "")).strip() or None,
            admin_mode=str(getattr(fedauth_args, "fedauth_admin_mode", "oidc")).strip().lower(),
            oidc_client_id=str(
                getattr(fedauth_args, "fedauth_oidc_client_id", getattr(fedauth_args, "fedauth_audience", audience))
            ).strip(),
            oidc_scopes=str(getattr(fedauth_args, "fedauth_oidc_scopes", "openid profile email")).strip(),
            oidc_discovery_url=str(getattr(fedauth_args, "fedauth_oidc_discovery_url", "")).strip() or None,
            oidc_callback_host=str(getattr(fedauth_args, "fedauth_oidc_callback_host", "127.0.0.1")),
            oidc_callback_port=int(getattr(fedauth_args, "fedauth_oidc_callback_port", 39123)),
            oidc_callback_path=str(getattr(fedauth_args, "fedauth_oidc_callback_path", "/callback")),
            oidc_refresh_skew_seconds=int(getattr(fedauth_args, "fedauth_oidc_refresh_skew_seconds", 60)),
            oidc_open_browser=bool(getattr(fedauth_args, "fedauth_oidc_open_browser", True)),
            admin_token_file=str(getattr(fedauth_args, "fedauth_admin_token_file", "/tmp/nvflare_admin.token")),
        )

    def token_login_config(self) -> dict:
        token_login = {
            "enabled": True,
            "issuer": self.issuer,
            "audience": self.audience,
            "alg_allowlist": self.alg_allowlist,
            "required_claims": self.required_claims,
            "claim_mappings": {
                "user_name_claims": self.user_name_claims,
                "user_org_claim": self.user_org_claim,
                "user_role_claim": self.user_role_claim,
                "role_mappings": self.role_mappings,
            },
        }
        if self.jwks_uri:
            token_login["jwks_uri"] = self.jwks_uri
        else:
            token_login["discovery_url"] = self.discovery_url or f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"
        return token_login

    def admin_startup_updates(self) -> dict:
        startup_updates = {
            "connection_security": "tls",
            "server_identity": "server.admin",
            "uid_source": "user_input",
            "client_key": "",
            "client_cert": "",
        }
        if self.admin_mode == "oidc":
            startup_updates.update(
                {
                    "auth_mode": "oidc",
                    "oidc_issuer": self.issuer,
                    "oidc_client_id": self.oidc_client_id,
                    "oidc_scopes": self.oidc_scopes,
                    "oidc_audience": self.audience,
                    "oidc_callback_host": self.oidc_callback_host,
                    "oidc_callback_port": self.oidc_callback_port,
                    "oidc_callback_path": self.oidc_callback_path,
                    "oidc_refresh_skew_seconds": self.oidc_refresh_skew_seconds,
                    "oidc_open_browser": self.oidc_open_browser,
                }
            )
            if self.oidc_discovery_url:
                startup_updates["oidc_discovery_url"] = self.oidc_discovery_url
            return startup_updates

        if self.admin_mode == "token":
            startup_updates.update({"auth_mode": "token", "token_file": self.admin_token_file})
            return startup_updates

        raise CLIException(f"invalid fedauth_admin_mode '{self.admin_mode}': expected 'oidc' or 'token'")


def _read_server_startup_config(prod_dir: str, server_name: str) -> dict:
    server_startup_path = os.path.join(prod_dir, server_name, "startup", ProvFileName.FED_SERVER_JSON)
    if not os.path.isfile(server_startup_path):
        raise CLIException(f"missing server startup config in {os.path.dirname(server_startup_path)}")

    server_startup = _load_json_file(server_startup_path)
    servers = server_startup.get("servers")
    if not isinstance(servers, list) or not servers or not isinstance(servers[0], dict):
        raise CLIException(f"invalid server startup config: {server_startup_path}")
    return servers[0]


def _load_project_root_signing_key(prod_dir: str):
    cert_state_path = os.path.join(str(Path(prod_dir).parent), "state", "cert.json")
    if not os.path.isfile(cert_state_path):
        raise CLIException(f"missing project cert state: {cert_state_path}")

    cert_state = _load_json_file(cert_state_path)
    root_pri_key = cert_state.get("root_pri_key")
    if not isinstance(root_pri_key, str) or not root_pri_key.strip():
        raise CLIException(f"missing root_pri_key in project cert state: {cert_state_path}")
    return load_private_key(root_pri_key)


def _ensure_fedauth_admin_profile(prod_dir: str, server_name: str, admin_name: str):
    admin_dir = os.path.join(prod_dir, admin_name)
    startup_dir = os.path.join(admin_dir, "startup")
    local_dir = os.path.join(admin_dir, "local")
    transfer_dir = os.path.join(admin_dir, SC.TRANSFER)
    os.makedirs(startup_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(transfer_dir, exist_ok=True)

    server_entry = _read_server_startup_config(prod_dir, server_name)
    service = server_entry.get("service") if isinstance(server_entry.get("service"), dict) else {}
    admin_port = server_entry.get("admin_port")
    if not isinstance(admin_port, int):
        raise CLIException(f"invalid or missing admin_port in server startup config for {server_name}")

    root_ca_src = os.path.join(prod_dir, server_name, "startup", "rootCA.pem")
    root_ca_dst = os.path.join(startup_dir, "rootCA.pem")
    if not os.path.isfile(root_ca_src):
        raise CLIException(f"missing root CA file: {root_ca_src}")
    if not os.path.isfile(root_ca_dst):
        shutil.copyfile(root_ca_src, root_ca_dst)

    fed_admin_path = os.path.join(startup_dir, ProvFileName.FED_ADMIN_JSON)
    if not os.path.isfile(fed_admin_path):
        _write_json_file(
            fed_admin_path,
            {
                "format_version": 1,
                "admin": {
                    "project_name": server_entry.get("name") or Path(prod_dir).parent.name,
                    "username": "",
                    "server_identity": server_name,
                    "scheme": service.get("scheme", "http"),
                    "host": server_entry.get("admin_server", server_name),
                    "port": admin_port,
                    "connection_security": "tls",
                    "uid_source": "user_input",
                    "with_file_transfer": True,
                    "upload_dir": SC.TRANSFER,
                    "download_dir": SC.TRANSFER,
                    "client_key": "",
                    "client_cert": "",
                    "ca_cert": "rootCA.pem",
                },
            },
        )

    fl_admin_path = os.path.join(startup_dir, ProvFileName.FL_ADMIN_SH)
    if not os.path.isfile(fl_admin_path):
        with open(fl_admin_path, "w") as f:
            f.write(
                "#!/usr/bin/env bash\n"
                'DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"\n'
                "mkdir -p $DIR/../transfer\n"
                "python3 -m nvflare.fuel.hci.tools.admin -m $DIR/.. -s fed_admin.json\n"
            )
        os.chmod(fl_admin_path, 0o755)

    admin_resources = os.path.join(local_dir, "resources.json")
    admin_resources_default = os.path.join(local_dir, "resources.json.default")
    if not os.path.isfile(admin_resources) and not os.path.isfile(admin_resources_default):
        _write_json_file(
            admin_resources,
            {
                "format_version": 1,
                "admin": {
                    "idle_timeout": 900.0,
                    "login_timeout": 10.0,
                    "with_debug": False,
                    "authenticate_msg_timeout": 2.0,
                    "prompt": "> ",
                },
            },
        )


def _sign_fedauth_admin_profile(prod_dir: str, admin_name: str):
    admin_dir = os.path.join(prod_dir, admin_name)
    root_pri_key = _load_project_root_signing_key(prod_dir)
    sign_folders(os.path.join(admin_dir, "startup"), root_pri_key, signature_file=ProvFileName.SIGNATURE_JSON)
    local_signature = os.path.join(admin_dir, "local", ProvFileName.SIGNATURE_JSON)
    if os.path.isfile(local_signature):
        os.remove(local_signature)


def _create_fedauth_admin_invite(prod_dir: str, admin_name: str) -> str:
    admin_dir = os.path.join(prod_dir, admin_name)
    if not os.path.isdir(admin_dir):
        raise CLIException(f"missing admin workspace to package invite from: {admin_dir}")

    invite_path = os.path.join(prod_dir, ProvFileName.INVITE_ZIP)
    with tempfile.TemporaryDirectory() as td:
        startup_dir = os.path.join(admin_dir, "startup")
        if not os.path.isdir(startup_dir):
            raise CLIException(f"missing admin workspace directory: {startup_dir}")
        shutil.copytree(startup_dir, os.path.join(td, "startup"))

        local_src = os.path.join(admin_dir, "local")
        local_dst = os.path.join(td, "local")
        if not os.path.isdir(local_src):
            raise CLIException(f"missing admin workspace directory: {local_src}")
        shutil.copytree(local_src, local_dst)
        local_signature = os.path.join(local_dst, ProvFileName.SIGNATURE_JSON)
        if os.path.isfile(local_signature):
            os.remove(local_signature)

        if not os.listdir(local_dst):
            os.rmdir(local_dst)

        transfer_src = os.path.join(admin_dir, SC.TRANSFER)
        transfer_dst = os.path.join(td, SC.TRANSFER)
        if os.path.isdir(transfer_src):
            shutil.copytree(transfer_src, transfer_dst)
        else:
            os.makedirs(transfer_dst, exist_ok=True)

        zip_directory_to_file(td, "", invite_path)
    return invite_path


_LOCAL_ADMIN_OVERRIDE_KEYS = {
    "auth_mode",
    "token",
    "token_file",
    "token_env_var",
    "project_name",
    "username",
    "server_identity",
    "scheme",
    "host",
    "port",
    "connection_security",
    "uid_source",
    "client_key",
    "client_cert",
    "ca_cert",
    "oidc_issuer",
    "oidc_client_id",
    "oidc_scopes",
    "oidc_audience",
    "oidc_discovery_url",
    "oidc_authorization_endpoint",
    "oidc_token_endpoint",
    "oidc_callback_host",
    "oidc_callback_port",
    "oidc_callback_path",
    "oidc_auth_timeout_seconds",
    "oidc_refresh_skew_seconds",
    "oidc_open_browser",
}


def apply_fedauth_to_poc_startup_kit(prod_dir: str, server_name: str, admin_name: str, fedauth_args):
    config = FedAuthAdminBootstrapConfig.from_args(fedauth_args)

    server_local_dir = os.path.join(prod_dir, server_name, "local")
    server_resources = os.path.join(server_local_dir, "resources.json")
    server_resources_default = os.path.join(server_local_dir, "resources.json.default")
    if os.path.exists(server_resources):
        server_payload = _load_json_file(server_resources)
    elif os.path.exists(server_resources_default):
        server_payload = _load_json_file(server_resources_default)
    else:
        raise CLIException(f"missing server resources config in {server_local_dir}")
    servers = server_payload.get("servers")
    if not isinstance(servers, list) or not servers:
        raise CLIException(f"invalid server resources config: {server_resources}")
    server_entry = servers[0]
    if not isinstance(server_entry, dict):
        raise CLIException(f"invalid first server entry in resources config: {server_resources}")
    admin_auth = server_entry.get("admin_auth")
    if not isinstance(admin_auth, dict):
        admin_auth = {}
    admin_auth["token_login"] = config.token_login_config()
    server_entry["admin_auth"] = admin_auth
    server_entry["admin_connection_security"] = "tls"
    server_entry["admin_interface_identity"] = "server.admin"
    _write_json_file(server_resources, server_payload)

    _ensure_fedauth_admin_profile(prod_dir=prod_dir, server_name=server_name, admin_name=admin_name)
    admin_startup_dir = os.path.join(prod_dir, admin_name, "startup")
    admin_startup_config = os.path.join(admin_startup_dir, ProvFileName.FED_ADMIN_JSON)
    admin_startup_payload = _load_json_file(admin_startup_config)
    startup_admin_section = admin_startup_payload.get("admin")
    if not isinstance(startup_admin_section, dict):
        raise CLIException(f"invalid admin startup config: {admin_startup_config}")

    admin_local_dir = os.path.join(prod_dir, admin_name, "local")
    admin_resources = os.path.join(admin_local_dir, "resources.json")
    admin_resources_default = os.path.join(admin_local_dir, "resources.json.default")
    if os.path.exists(admin_resources):
        admin_payload = _load_json_file(admin_resources)
    elif os.path.exists(admin_resources_default):
        admin_payload = _load_json_file(admin_resources_default)
    else:
        admin_payload = {"format_version": 1, "admin": {}}
    admin_section = admin_payload.get("admin")
    if not isinstance(admin_section, dict):
        admin_section = {}
    for key in _LOCAL_ADMIN_OVERRIDE_KEYS:
        admin_section.pop(key, None)
    startup_admin_section.update(config.admin_startup_updates())
    admin_startup_payload["admin"] = startup_admin_section
    _write_json_file(admin_startup_config, admin_startup_payload)

    admin_payload["admin"] = admin_section
    _write_json_file(admin_resources, admin_payload)
    _sign_fedauth_admin_profile(prod_dir=prod_dir, admin_name=admin_name)
    _create_fedauth_admin_invite(prod_dir=prod_dir, admin_name=admin_name)


def get_or_create_hidden_nvflare_config_path() -> str:
    """
    Get the path for the hidden nvflare configuration file.

    Returns:
        str: The path to the hidden nvflare configuration file.
    """
    hidden_nvflare_dir = get_or_create_hidden_nvflare_dir()

    hidden_nvflare_config_file = get_hidden_nvflare_config_path(str(hidden_nvflare_dir))
    return hidden_nvflare_config_file


def prepare_poc_provision(
    clients: List[str],
    number_of_clients: int,
    workspace: str,
    docker_image: str,
    use_he: bool = False,
    project_conf_path: str = "",
    examples_dir: Optional[str] = None,
    default_project_admin: Optional[str] = None,
) -> Dict:
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(os.path.join(workspace, "data"), exist_ok=True)
    project_config, service_config = local_provision(
        clients,
        number_of_clients,
        workspace,
        docker_image,
        use_he,
        project_conf_path,
        default_project_admin=default_project_admin,
    )
    project_name = project_config.get("name")
    server_name = service_config[SC.FLARE_SERVER]
    # update storage
    if workspace != DEFAULT_WORKSPACE:
        prod_dir = get_prod_dir(workspace, project_name)
        update_storage_locations(local_dir=f"{prod_dir}/{server_name}/local", workspace=workspace)
    examples_dir = get_examples_dir(examples_dir)
    if examples_dir is not None:
        _prepare_jobs_dir(examples_dir, workspace, None)

    return project_config


def get_examples_dir(examples_dir):
    if examples_dir:
        return examples_dir
    nvflare_home = get_nvflare_home()
    default_examples_dir = os.path.join(nvflare_home, SC.EXAMPLES) if nvflare_home else None
    return default_examples_dir


def _sort_service_cmds(cmd_type, service_cmds: list, service_config) -> list:
    def sort_first(val):
        return val[0]

    order_services = []
    for service_name, cmd_path in service_cmds:
        if service_name == service_config[SC.FLARE_SERVER]:
            order_services.append((0, service_name, cmd_path))
        elif service_name == service_config[SC.FLARE_PROJ_ADMIN]:
            order_services.append((sys.maxsize, service_name, cmd_path))
        else:
            if len(service_cmds) == 1:
                order_services.append((0, service_name, cmd_path))
            else:
                order_services.append((random.randint(2, len(service_cmds)), service_name, cmd_path))

    order_services.sort(key=sort_first)
    if cmd_type == SC.CMD_STOP:
        order_services.reverse()
    return [(service_name, cmd_path) for n, service_name, cmd_path in order_services]


def get_cmd_path(poc_workspace, service_name, cmd):
    service_dir = os.path.join(poc_workspace, service_name)
    bin_dir = os.path.join(service_dir, SC.STARTUP)
    cmd_path = os.path.join(bin_dir, cmd)
    return cmd_path


def is_poc_ready(poc_workspace: str, service_config, project_config):
    # check server and admin directories exist
    project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
    prod_dir = get_prod_dir(poc_workspace, project_name)
    console_dir = os.path.join(prod_dir, service_config[SC.FLARE_PROJ_ADMIN])
    server_dir = os.path.join(prod_dir, service_config[SC.FLARE_SERVER])
    return os.path.isdir(server_dir) and os.path.isdir(console_dir)


def validate_poc_workspace(poc_workspace: str, service_config, project_config=None):
    if not is_poc_ready(poc_workspace, service_config, project_config):
        raise CLIException(f"workspace {poc_workspace} is not ready, please use poc prepare to prepare poc workspace")


def validate_gpu_ids(gpu_ids: list, host_gpu_ids: list):
    for gpu_id in gpu_ids:
        if gpu_id not in host_gpu_ids:
            raise CLIException(
                f"gpu_id provided is not available in the host machine, available GPUs are {host_gpu_ids}"
            )


def get_gpu_ids(user_input_gpu_ids, host_gpu_ids) -> List[int]:
    if type(user_input_gpu_ids) == int and user_input_gpu_ids == -1:
        gpu_ids = host_gpu_ids
    else:
        gpu_ids = user_input_gpu_ids
        validate_gpu_ids(gpu_ids, host_gpu_ids)
    return gpu_ids


def start_poc(cmd_args):
    poc_workspace = get_poc_workspace()

    services_list = get_service_list(cmd_args)
    excluded = get_excluded(cmd_args)
    gpu_ids = get_gpis(cmd_args)

    _start_poc(poc_workspace, gpu_ids, excluded, services_list)


def get_gpis(cmd_args):
    if cmd_args.gpu is not None and isinstance(cmd_args.gpu, list) and len(cmd_args.gpu) > 0:
        gpu_ids = get_gpu_ids(cmd_args.gpu, get_local_host_gpu_ids())
    else:
        gpu_ids = []
    return gpu_ids


def get_excluded(cmd_args):
    excluded = None
    if cmd_args.exclude != "":
        excluded = [cmd_args.exclude]
    return excluded


def get_service_list(cmd_args):
    if cmd_args.service != "all":
        services_list = [cmd_args.service]
    else:
        services_list = []
    return services_list


def _start_poc(poc_workspace: str, gpu_ids: List[int], excluded=None, services_list=None):
    project_config, service_config = setup_service_config(poc_workspace)
    if services_list is None:
        services_list = []
    if excluded is None:
        excluded = []
    other_admins = service_config.get(SC.FLARE_OTHER_ADMINS, [])
    for admin_dir in other_admins:
        if admin_dir not in services_list:
            excluded.append(admin_dir)

    validate_services(project_config, services_list, excluded, service_config=service_config)
    validate_poc_workspace(poc_workspace, service_config, project_config)
    _run_poc(
        SC.CMD_START,
        poc_workspace,
        gpu_ids,
        service_config,
        project_config,
        excluded=excluded,
        services_list=services_list,
    )


def validate_services(project_config, services_list: List, excluded: List, service_config=None):
    participant_names = [p["name"] for p in project_config["participants"]]
    if service_config:
        admin_names = [service_config.get(SC.FLARE_PROJ_ADMIN)] + list(service_config.get(SC.FLARE_OTHER_ADMINS, []))
        for admin_name in admin_names:
            if admin_name and admin_name not in participant_names:
                participant_names.append(admin_name)
    validate_participants(participant_names, services_list)
    validate_participants(participant_names, excluded)


def validate_participants(participant_names, list_participants):
    for p in list_participants:
        if p not in participant_names:
            print(f"participant '{p}' is not defined, expecting one of followings: {participant_names}")
            exit(1)


def setup_service_config(poc_workspace) -> Tuple:
    project_file = os.path.join(poc_workspace, "project.yml")
    if os.path.isfile(project_file):
        project_config = load_yaml(project_file)
        default_project_admin = None
        if project_config:
            project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
            prod_dir = get_prod_dir(poc_workspace, project_name)
            admin_dir = os.path.join(prod_dir, SC.FLARE_PROJ_ADMIN)
            if not get_fl_admins(project_config, is_project_admin=True) and os.path.isdir(admin_dir):
                default_project_admin = SC.FLARE_PROJ_ADMIN
        service_config = (
            get_service_config(project_config, default_project_admin=default_project_admin) if project_config else None
        )
        return project_config, service_config
    else:
        raise CLIException(f"{project_file} is missing, make sure you have first run 'nvflare poc prepare'")


def stop_poc(cmd_args):
    poc_workspace = get_poc_workspace()
    excluded = get_excluded(cmd_args)
    services_list = get_service_list(cmd_args)
    _stop_poc(poc_workspace, excluded, services_list)


def _stop_poc(poc_workspace: str, excluded=None, services_list=None):
    project_config, service_config = setup_service_config(poc_workspace)

    if services_list is None:
        services_list = []
    if excluded is None:
        excluded = [service_config[SC.FLARE_PROJ_ADMIN]]
    else:
        excluded.append(service_config[SC.FLARE_PROJ_ADMIN])

    validate_services(project_config, services_list, excluded, service_config=service_config)

    validate_poc_workspace(poc_workspace, service_config, project_config)
    gpu_ids: List[int] = []
    project_name = project_config.get("name")
    prod_dir = get_prod_dir(poc_workspace, project_name)

    p_size = len(services_list)
    if p_size == 0 or service_config[SC.FLARE_SERVER] in services_list:
        print("Starting shutdown of NVFLARE")
        shutdown_system(prod_dir, username=service_config[SC.FLARE_PROJ_ADMIN])
    else:
        print(f"Starting shutdown of {services_list} using the stop_fl.sh script")

        _run_poc(
            SC.CMD_STOP,
            poc_workspace,
            gpu_ids,
            service_config,
            project_config,
            excluded=excluded,
            services_list=services_list,
        )


def _get_clients(service_commands: list, service_config) -> List[str]:
    clients = [
        service_dir_name
        for service_dir_name, _ in service_commands
        if service_dir_name != service_config[SC.FLARE_PROJ_ADMIN]
        and service_dir_name not in service_config.get(SC.FLARE_OTHER_ADMINS, [])
        and service_dir_name != service_config[SC.FLARE_SERVER]
    ]
    return clients


def _build_commands(
    cmd_type: str, poc_workspace: str, service_config, project_config, excluded: list, services_list=None
) -> list:
    """Builds commands.

    Args:
        cmd_type (str): start/stop
        poc_workspace (str): poc workspace directory path
        service_config (_type_): service_config
        excluded (list): excluded service/participants name
        services_list (_type_, optional): Service names. If empty, include every service/participants

    Returns:
        list: built commands
    """

    def is_fl_service_dir(p_dir_name: str) -> bool:
        fl_service = (
            p_dir_name == service_config[SC.FLARE_PROJ_ADMIN]
            or p_dir_name in service_config[SC.FLARE_OTHER_ADMINS]
            or p_dir_name == service_config[SC.FLARE_SERVER]
            or p_dir_name in service_config[SC.FLARE_CLIENTS]
        )
        return fl_service

    project_name = project_config.get("name")
    prod_dir = get_prod_dir(poc_workspace, project_name)

    if services_list is None:
        services_list = []
    service_commands = []
    for root, dirs, files in os.walk(prod_dir):
        if root == prod_dir:
            fl_dirs = [d for d in dirs if is_fl_service_dir(d)]
            for service_dir_name in fl_dirs:
                if service_dir_name not in excluded:
                    if len(services_list) == 0 or service_dir_name in services_list:
                        cmd = get_service_command(cmd_type, prod_dir, service_dir_name, service_config)
                        if cmd:
                            service_commands.append((service_dir_name, cmd))
    return _sort_service_cmds(cmd_type, service_commands, service_config)


def prepare_env(service_name, gpu_ids: Optional[List[int]], service_config: Dict):
    import os

    my_env = None
    if gpu_ids:
        my_env = os.environ.copy()
        if len(gpu_ids) > 0:
            my_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gid) for gid in gpu_ids])

    if service_config.get(SC.IS_DOCKER_RUN):
        my_env = os.environ.copy() if my_env is None else my_env
        if gpu_ids and len(gpu_ids) > 0:
            my_env["GPU2USE"] = f"--gpus={my_env['CUDA_VISIBLE_DEVICES']}"

        my_env["MY_DATA_DIR"] = os.path.join(get_poc_workspace(), "data")
        my_env["SVR_NAME"] = service_name

    return my_env


def async_process(service_name, cmd_path, gpu_ids: Optional[List[int]], service_config: Dict):
    my_env = prepare_env(service_name, gpu_ids, service_config)
    if my_env:
        subprocess.Popen(cmd_path.split(" "), env=my_env)
    else:
        subprocess.Popen(cmd_path.split(" "))


def sync_process(service_name, cmd_path):
    my_env = os.environ.copy()
    subprocess.run(cmd_path.split(" "), env=my_env)


def _run_poc(
    cmd_type: str,
    poc_workspace: str,
    gpu_ids: List[int],
    service_config: Dict,
    project_config: Dict,
    excluded: list,
    services_list=None,
):
    if services_list is None:
        services_list = []
    service_commands = _build_commands(cmd_type, poc_workspace, service_config, project_config, excluded, services_list)
    clients = _get_clients(service_commands, service_config)
    gpu_assignments: Dict[str, List[int]] = client_gpu_assignments(clients, gpu_ids)
    for service_name, cmd_path in service_commands:
        if service_name == service_config[SC.FLARE_PROJ_ADMIN]:
            # give other commands a chance to start first
            if len(service_commands) > 1:
                time.sleep(2)
            sync_process(service_name, cmd_path)
        elif service_name == service_config[SC.FLARE_SERVER]:
            async_process(service_name, cmd_path, None, service_config)
        else:
            time.sleep(1)
            gpu_ids = gpu_assignments[service_name] if service_name in clients else None
            async_process(service_name, cmd_path, gpu_ids, service_config)


def clean_poc(cmd_args):
    poc_workspace = get_poc_workspace()
    _clean_poc(poc_workspace)


def is_poc_running(poc_workspace, service_config, project_config):
    project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
    prod_dir = get_prod_dir(poc_workspace, project_name)
    server_dir = os.path.join(prod_dir, service_config[SC.FLARE_SERVER])
    pid_file = os.path.join(server_dir, "pid.fl")
    return os.path.exists(pid_file)


def _clean_poc(poc_workspace: str):
    import shutil

    if os.path.isdir(poc_workspace):
        project_config, service_config = setup_service_config(poc_workspace)
        if project_config is not None:
            if is_poc_ready(poc_workspace, service_config, project_config):
                if not is_poc_running(poc_workspace, service_config, project_config):
                    shutil.rmtree(poc_workspace, ignore_errors=True)
                    print(f"{poc_workspace} is removed")
                else:
                    print("system is still running, please stop the system first.")
            else:
                raise CLIException(f"{poc_workspace} is not valid poc directory")
    else:
        raise CLIException(f"{poc_workspace} is not valid poc directory")


poc_sub_cmd_handlers = {
    CMD_PREPARE_POC: prepare_poc,
    CMD_PREPARE_JOBS_DIR: prepare_jobs_dir,
    CMD_START_POC: start_poc,
    CMD_STOP_POC: stop_poc,
    CMD_CLEAN_POC: clean_poc,
}


def def_poc_parser(sub_cmd):
    cmd = "poc"
    parser = sub_cmd.add_parser(cmd)
    add_legacy_options(parser)

    poc_parser = parser.add_subparsers(title=cmd, dest="poc_sub_cmd", help="poc subcommand")
    define_prepare_parser(poc_parser)
    define_prepare_jobs_parser(poc_parser)
    define_start_parser(poc_parser)
    define_stop_parser(poc_parser)
    define_clean_parser(poc_parser)
    return {cmd: parser}


def add_legacy_options(parser):
    parser.add_argument(
        "--prepare",
        dest="old_prepare_poc",
        action="store_const",
        const=old_prepare_poc,
        help="deprecated, suggest use 'nvflare poc prepare'",
    )
    parser.add_argument(
        "--start",
        dest="old_start_poc",
        action="store_const",
        const=old_start_poc,
        help="deprecated, suggest use 'nvflare poc start'",
    )
    parser.add_argument(
        "--stop",
        dest="old_stop_poc",
        action="store_const",
        const=old_stop_poc,
        help="deprecated, suggest use 'nvflare poc stop'",
    )
    parser.add_argument(
        "--clean",
        dest="old_clean_poc",
        action="store_const",
        const=old_clean_poc,
        help="deprecated, suggest use 'nvflare poc clean'",
    )


def old_start_poc():
    print(f"'nvflare poc --{CMD_START_POC}' is deprecated, please use 'nvflare poc {CMD_START_POC}' ")


def old_stop_poc():
    print(f"'nvflare poc --{CMD_STOP_POC}' is deprecated, please use 'nvflare poc {CMD_STOP_POC}' ")


def old_clean_poc():
    print(f"'nvflare poc --{CMD_CLEAN_POC}' is deprecated, please use 'nvflare poc {CMD_CLEAN_POC}' ")


def old_prepare_poc():
    print(f"'nvflare poc --{CMD_PREPARE_POC}' is deprecated, please use 'nvflare poc {CMD_PREPARE_POC}' ")


def define_prepare_parser(poc_parser, cmd: Optional[str] = None, help_str: Optional[str] = None):
    cmd = CMD_PREPARE_POC if cmd is None else cmd
    help_str = "prepare poc environment by provisioning local project" if help_str is None else help_str
    prepare_parser = poc_parser.add_parser(cmd, help=help_str)

    prepare_parser.add_argument(
        "-n", "--number_of_clients", type=int, nargs="?", default=2, help="number of sites or clients, default to 2"
    )
    prepare_parser.add_argument(
        "-c",
        "--clients",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=[],  # default if nothing is provided
        help="Space separated client names. If specified, number_of_clients argument will be ignored.",
    )
    prepare_parser.add_argument(
        "-he",
        "--he",
        action="store_true",
        help="enable homomorphic encryption. ",
    )

    prepare_parser.add_argument(
        "-i",
        "--project_input",
        type=str,
        nargs="?",
        default="",
        help="project.yaml file path, If specified, "
        + "'number_of_clients','clients' and 'docker' specific options will be ignored.",
    )
    prepare_parser.add_argument(
        "-d",
        "--docker_image",
        nargs="?",
        default=None,
        const="nvflare/nvflare",
        help="generate docker.sh based on the docker_image, used in '--prepare' command. and generate docker.sh "
        + " 'start/stop' commands will start with docker.sh ",
    )

    prepare_parser.add_argument(
        "--enable_fedauth",
        action="store_true",
        help="automatically configure server/admin startup kits for OIDC token-based admin auth",
    )
    prepare_parser.add_argument(
        "--fedauth_issuer",
        type=str,
        default="",
        help="OIDC issuer URL used to validate admin tokens (required when --enable_fedauth is set)",
    )
    prepare_parser.add_argument(
        "--fedauth_audience",
        type=str,
        default="nvflare-admin",
        help="OIDC audience for admin access tokens",
    )
    prepare_parser.add_argument(
        "--fedauth_jwks_uri",
        type=str,
        default="",
        help="optional JWKS URI for token signature verification (if omitted, discovery is used)",
    )
    prepare_parser.add_argument(
        "--fedauth_discovery_url",
        type=str,
        default="",
        help="optional OIDC discovery URL for JWKS lookup",
    )
    prepare_parser.add_argument(
        "--fedauth_alg_allowlist",
        nargs="+",
        default=["RS256"],
        help="allowed JWT signature algorithms",
    )
    prepare_parser.add_argument(
        "--fedauth_required_claims",
        nargs="+",
        default=["iss", "aud", "exp", "iat"],
        help="required JWT claims",
    )
    prepare_parser.add_argument(
        "--fedauth_user_name_claims",
        nargs="+",
        default=["preferred_username", "email"],
        help="ordered user-name claim candidates",
    )
    prepare_parser.add_argument(
        "--fedauth_user_org_claim",
        type=str,
        default="org",
        help="claim name for user org",
    )
    prepare_parser.add_argument(
        "--fedauth_user_role_claim",
        type=str,
        default="nvf_role",
        help="claim name for user role",
    )
    prepare_parser.add_argument(
        "--fedauth_role_mappings",
        nargs="*",
        default=["lead=project_admin"],
        help="claim-role mappings in SOURCE=TARGET format",
    )
    prepare_parser.add_argument(
        "--fedauth_admin_mode",
        choices=["oidc", "token"],
        default="oidc",
        help="admin auth mode to configure in startup kit",
    )
    prepare_parser.add_argument(
        "--fedauth_admin_token_file",
        type=str,
        default="/tmp/nvflare_admin.token",
        help="token file path used when --fedauth_admin_mode token",
    )
    prepare_parser.add_argument(
        "--fedauth_oidc_client_id",
        type=str,
        default="nvflare-admin",
        help="OIDC client ID used by admin CLI when --fedauth_admin_mode oidc",
    )
    prepare_parser.add_argument(
        "--fedauth_oidc_scopes",
        type=str,
        default="openid profile email",
        help="OIDC scopes used by admin CLI browser login",
    )
    prepare_parser.add_argument(
        "--fedauth_oidc_discovery_url",
        type=str,
        default="",
        help="optional OIDC discovery URL for admin CLI browser login",
    )
    prepare_parser.add_argument(
        "--fedauth_oidc_callback_host",
        type=str,
        default="127.0.0.1",
        help="loopback callback host for admin browser login",
    )
    prepare_parser.add_argument(
        "--fedauth_oidc_callback_port",
        type=int,
        default=39123,
        help="loopback callback port for admin browser login",
    )
    prepare_parser.add_argument(
        "--fedauth_oidc_callback_path",
        type=str,
        default="/callback",
        help="loopback callback path for admin browser login",
    )
    prepare_parser.add_argument(
        "--fedauth_oidc_refresh_skew_seconds",
        type=int,
        default=60,
        help="refresh lead time in seconds for admin access token",
    )
    prepare_parser.add_argument(
        "--fedauth_oidc_no_open_browser",
        dest="fedauth_oidc_open_browser",
        action="store_false",
        default=True,
        help="disable automatic browser open for admin OIDC login",
    )

    prepare_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def define_prepare_jobs_parser(poc_parser):
    prepare_jobs_dir_parser = poc_parser.add_parser(CMD_PREPARE_JOBS_DIR, help="prepare jobs directory")
    prepare_jobs_dir_parser.add_argument("-j", "--jobs_dir", type=str, nargs="?", default=None, help="jobs directory")
    prepare_jobs_dir_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def define_clean_parser(poc_parser):
    clean_parser = poc_parser.add_parser(CMD_CLEAN_POC, help="clean up poc workspace")
    clean_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def define_start_parser(poc_parser):
    start_parser = poc_parser.add_parser(CMD_START_POC, help="start services in poc mode")

    start_parser.add_argument(
        "-p",
        "--service",
        type=str,
        nargs="?",
        default="all",
        help="participant, Default to all participants",
    )

    start_parser.add_argument(
        "-ex",
        "--exclude",
        type=str,
        nargs="?",
        default="",
        help="exclude service directory during 'start', default to " ", i.e. nothing to exclude",
    )
    start_parser.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        nargs="*",
        default=None,
        help="gpu device ids will be used as CUDA_VISIBLE_DEVICES. used for poc start command",
    )
    start_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def define_stop_parser(poc_parser):
    stop_parser = poc_parser.add_parser(CMD_STOP_POC, help="stop services in poc mode")

    stop_parser.add_argument(
        "-p",
        "--service",
        type=str,
        nargs="?",
        default="all",
        help="participant, Default to all participants",
    )
    stop_parser.add_argument(
        "-ex",
        "--exclude",
        type=str,
        nargs="?",
        default="",
        help="exclude service directory during 'stop', default to " ", i.e. nothing to exclude",
    )
    stop_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def get_local_host_gpu_ids():
    try:
        return get_host_gpu_ids()
    except Exception as e:
        raise CLIException(f"Failed to get host gpu ids:{e}")


def handle_poc_cmd(cmd_args):
    if cmd_args.poc_sub_cmd:
        poc_cmd_handler = poc_sub_cmd_handlers.get(cmd_args.poc_sub_cmd, None)
        poc_cmd_handler(cmd_args)
    elif cmd_args.old_start_poc:
        old_start_poc()
    elif cmd_args.old_stop_poc:
        old_stop_poc()
    elif cmd_args.old_clean_poc:
        old_clean_poc()
    elif cmd_args.old_prepare_poc:
        old_prepare_poc()
    else:
        raise CLIUnknownCmdException("unknown command")


def get_poc_workspace():
    poc_workspace = os.getenv("NVFLARE_POC_WORKSPACE")

    if not poc_workspace:
        src_path = get_or_create_hidden_nvflare_config_path()
        if os.path.isfile(src_path):
            from pyhocon import ConfigFactory as CF

            config = CF.parse_file(src_path)
            poc_workspace = config.get("poc_workspace.path", None)

    if poc_workspace is None or len(poc_workspace.strip()) == 0:
        poc_workspace = DEFAULT_WORKSPACE

    return poc_workspace
