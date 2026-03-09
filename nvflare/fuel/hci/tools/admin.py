# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os
from zipfile import BadZipFile

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.client.cli import AdminClient
from nvflare.fuel.hci.client.config import secure_load_admin_config
from nvflare.fuel.hci.client.file_transfer import FileTransferModule
from nvflare.fuel.utils.zip_utils import unzip_all_from_file
from nvflare.security.logging import secure_format_exception

DEFAULT_CLI_HIST_SIZE = 1000


def _get_default_invite_workspace(invite_file: str) -> str:
    invite_path = os.path.abspath(invite_file)
    invite_name = os.path.splitext(os.path.basename(invite_path))[0] or "invite"
    return os.path.join(os.path.dirname(invite_path), invite_name)


def _normalize_imported_workspace(workspace_dir: str):
    fl_admin_path = os.path.join(workspace_dir, WorkspaceConstants.STARTUP_FOLDER_NAME, "fl_admin.sh")
    if os.path.isfile(fl_admin_path):
        os.chmod(fl_admin_path, 0o755)


def prepare_workspace(workspace: str = "", invite_file: str = "", fed_admin: str = WorkspaceConstants.ADMIN_STARTUP_CONFIG):
    workspace_dir = os.path.abspath(workspace) if workspace else ""
    if not invite_file:
        if not workspace_dir:
            raise ConfigError("workspace is required unless an invite zip is provided")
        return workspace_dir

    if not workspace_dir:
        workspace_dir = _get_default_invite_workspace(invite_file)

    if os.path.exists(workspace_dir):
        if not os.path.isdir(workspace_dir):
            raise ConfigError(f"invalid workspace {workspace_dir}: not a directory")
        existing_entries = os.listdir(workspace_dir)
        if existing_entries:
            raise ConfigError(
                f"workspace {workspace_dir} already exists - remove it or use a different workspace for invite import"
            )
    else:
        os.makedirs(workspace_dir, exist_ok=True)

    try:
        unzip_all_from_file(invite_file, workspace_dir)
    except (BadZipFile, FileNotFoundError, NotADirectoryError, OSError, ValueError) as e:
        raise ConfigError(f"failed to import invite {invite_file}: {e}") from e
    startup_config = os.path.join(workspace_dir, WorkspaceConstants.STARTUP_FOLDER_NAME, fed_admin)
    if not os.path.isfile(startup_config):
        raise ConfigError(f"invite {invite_file} does not contain startup/{fed_admin}")
    _normalize_imported_workspace(workspace_dir)
    return workspace_dir


def main():
    """
    Script to launch the admin client to issue admin commands to the server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder")
    parser.add_argument("--invite_file", "-i", type=str, help="zip file containing an admin bootstrap workspace")

    parser.add_argument(
        "--fed_admin",
        "-s",
        type=str,
        help="json file with configurations for launching admin client",
        default=WorkspaceConstants.ADMIN_STARTUP_CONFIG,
    )
    parser.add_argument("--cli_history_size", type=int, default=DEFAULT_CLI_HIST_SIZE)
    parser.add_argument("--with_debug", action="store_true")

    args = parser.parse_args()
    if not args.workspace and not args.invite_file:
        parser.error("one of --workspace/-m or --invite_file/-i is required")

    try:
        workspace_dir = prepare_workspace(workspace=args.workspace, invite_file=args.invite_file, fed_admin=args.fed_admin)
        if args.invite_file:
            print(f"Invite imported to: {workspace_dir}")
            print(f"Next step: run {os.path.join(workspace_dir, WorkspaceConstants.STARTUP_FOLDER_NAME, 'fl_admin.sh')}")
            return
        os.chdir(workspace_dir)
        workspace = Workspace(root_dir=workspace_dir)
        conf = secure_load_admin_config(workspace)
    except ConfigError as e:
        print(f"{secure_format_exception(e)}")
        return

    admin_config = conf.get_admin_config()
    if not admin_config:
        print(f"Missing '{AdminConfigKey.ADMIN}' section in fed_admin configuration.")
        return

    modules = []
    if admin_config.get(AdminConfigKey.WITH_FILE_TRANSFER):
        modules.append(
            FileTransferModule(
                upload_dir=admin_config.get(AdminConfigKey.UPLOAD_DIR),
                download_dir=admin_config.get(AdminConfigKey.DOWNLOAD_DIR),
            )
        )

    if args.with_debug:
        with_debug = True
    else:
        with_debug = admin_config.get(AdminConfigKey.WITH_DEBUG, False)

    cli_history_size = admin_config.get(AdminConfigKey.CLI_HISTORY_SIZE)
    if not cli_history_size:
        cli_history_size = args.cli_history_size

    if not isinstance(cli_history_size, int) or cli_history_size <= 0:
        print(f"invalid cli_history_size {cli_history_size}: set it to {DEFAULT_CLI_HIST_SIZE}")
        cli_history_size = DEFAULT_CLI_HIST_SIZE

    if with_debug:
        with_file_transfer = admin_config.get(AdminConfigKey.WITH_FILE_TRANSFER)
        print(f"CLI History Size: {cli_history_size}")
        print(f"File Transfer: {with_file_transfer}")
        if with_file_transfer:
            print(f"  Upload Dir: {admin_config.get(AdminConfigKey.UPLOAD_DIR)}")
            print(f"  Download Dir: {admin_config.get(AdminConfigKey.DOWNLOAD_DIR)}")

    client = AdminClient(
        admin_config=admin_config,
        cmd_modules=modules,
        debug=with_debug,
        username=admin_config.get(AdminConfigKey.USERNAME, ""),
        handlers=conf.handlers,
        cli_history_dir=workspace_dir,
        cli_history_size=cli_history_size,
    )

    client.run()


if __name__ == "__main__":
    main()
