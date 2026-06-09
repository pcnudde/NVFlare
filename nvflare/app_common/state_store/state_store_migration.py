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

import argparse
import json
import os
import sys
from pathlib import Path

from nvflare.apis.fl_constant import SiteType, SystemComponents
from nvflare.apis.workspace import Workspace
from nvflare.app_common.state_store.legacy_migration import MigrationSkipError, migrate_legacy_state_store
from nvflare.app_common.state_store.sql_store import (
    SqlStateStore,
    default_state_store_db_url,
    migrate_database,
    resolve_db_url,
    resolve_relative_db_url,
)
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage


def _build_parser():
    parser = argparse.ArgumentParser(description="Migrate legacy filesystem state into the StateStore database.")
    parser.add_argument("--server-root", required=True, help="server workspace root containing local/ and startup/")
    parser.add_argument("--db-url", help="optional SQLAlchemy database URL override")
    parser.add_argument("--schema-revision", default="head", help="Alembic revision to migrate to before import")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail the migration on conditions that are otherwise skipped with a warning",
    )
    return parser


def _load_resources(workspace: Workspace) -> dict:
    resources_path = workspace.get_resources_file_path()
    if not resources_path:
        raise RuntimeError(f"missing resources.json or resources.json.default under {workspace.get_site_config_dir()}")
    with open(resources_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _component(resources: dict, component_id: str):
    for component in resources.get("components", []) or []:
        if component.get("id") == component_id:
            return component
    return None


def _state_store_db_url(resources: dict):
    component = _component(resources, SystemComponents.STATE_STORE)
    if not component:
        return None
    args = component.get("args") or {}
    return resolve_db_url(db_url=args.get("db_url"), db_url_env=args.get("db_url_env"))


def _resolve_path(path: str, server_root: Path):
    if not path:
        return str(server_root)
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = server_root / path
    return str(path)


def _filesystem_job_storage(resources: dict, server_root: Path, strict: bool):
    """Build the legacy FilesystemStorage job store, or skip job import with a warning.

    Returns (job_storage, jobs_uri_root, warnings). A non-FilesystemStorage job store cannot
    be imported from the filesystem; job import is skipped (prominently) unless --strict.
    """
    job_manager = _component(resources, SystemComponents.JOB_MANAGER)
    if not job_manager:
        return None, "jobs", []

    job_manager_args = job_manager.get("args") or {}
    jobs_uri_root = os.environ.get("NVFL_JOB_STORE_ROOT") or job_manager_args.get("uri_root", "jobs")
    job_store_id = job_manager_args.get("job_store_id", "job_store")
    job_store = _component(resources, job_store_id)
    if not job_store:
        raise RuntimeError(f"job manager references missing job store component '{job_store_id}'")

    path = job_store.get("path", "")
    if path and not path.endswith(".FilesystemStorage"):
        message = (
            f"job store component '{job_store_id}' uses '{path}', not FilesystemStorage; "
            "legacy job import is SKIPPED — existing jobs and submit records will NOT be migrated"
        )
        if strict:
            raise MigrationSkipError(message)
        print(f"WARNING: {message}", file=sys.stderr)
        return None, jobs_uri_root, [message]

    job_store_args = job_store.get("args") or {}
    root_dir = job_store_args.get("root_dir")
    if root_dir is None and Path(jobs_uri_root).is_absolute():
        root_dir = "/"
    return (
        FilesystemStorage(
            root_dir=_resolve_path(root_dir, server_root),
            uri_root=job_store_args.get("uri_root", "/"),
        ),
        jobs_uri_root,
        [],
    )


def _existing_path(path: str):
    return path if Path(path).exists() else None


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        workspace = Workspace(args.server_root, SiteType.SERVER)
        resources = _load_resources(workspace)
        db_url = args.db_url or _state_store_db_url(resources)
        if not db_url:
            # Mirror the server deployer: a workspace without a state_store component falls
            # back to the default SQLite DB under the server root, so legacy workspaces
            # provisioned before the state store existed can still be migrated.
            db_url = default_state_store_db_url(args.server_root)
            print(
                "NOTICE: no state_store component (or --db-url) configured; "
                f"defaulting to '{db_url}'. Add a state_store component to local/resources.json "
                "to control the state store database location.",
                file=sys.stderr,
            )
        # Relative SQLite paths must resolve against the server root, not the CLI's CWD,
        # so the server (which chdirs into the workspace) opens the same file.
        db_url = resolve_relative_db_url(db_url, args.server_root)

        job_storage, jobs_uri_root, warnings = _filesystem_job_storage(
            resources, Path(args.server_root).expanduser(), args.strict
        )
        migrate_database(db_url, args.schema_revision)
        result = migrate_legacy_state_store(
            state_store=SqlStateStore(db_url),
            job_storage=job_storage,
            jobs_uri_root=jobs_uri_root,
            study_registry_path=_existing_path(workspace.get_file_path_in_site_config("study_registry.json")),
            disabled_clients_path=_existing_path(workspace.get_file_path_in_root("disabled_clients.json")),
            strict=args.strict,
            warnings=warnings,
        )
    except Exception as e:
        print(f"state store migration failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
