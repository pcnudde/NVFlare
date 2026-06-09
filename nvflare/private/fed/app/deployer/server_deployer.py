# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""FL Server deployer."""
import threading
from pathlib import Path
from typing import Optional

from sqlalchemy.engine import make_url

from nvflare.apis import study_store
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, SiteType, SystemComponents
from nvflare.apis.signal import Signal
from nvflare.apis.storage import StorageException
from nvflare.apis.workspace import Workspace
from nvflare.app_common.state_store import (
    SqlStateStore,
    bootstrap_fresh_state_store,
    classify_legacy_state,
    default_state_store_db_url,
    migrate_database,
    resolve_relative_db_url,
)
from nvflare.app_common.state_store.legacy_migration import migrate_legacy_state_store, validate_state_store_migrated
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.fed.app.utils import component_security_check
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.server.job_runner import JobRunner
from nvflare.private.fed.server.run_manager import RunManager
from nvflare.private.fed.server.server_cmd_modules import ServerCommandModules
from nvflare.private.fed.server.server_status import ServerStatus
from nvflare.widgets.fed_event import ServerFedEventRunner


def _ensure_sqlite_parent_dir(db_url: str):
    """Create the parent directory of a file-backed SQLite db_url if absent.

    SQLite can create a missing DB file but not missing parent directories, and schema
    validation connects before the bootstrap/migration path gets a chance to create them.
    """
    url = make_url(db_url)
    if url.drivername.startswith("sqlite") and url.database and url.database != ":memory:":
        Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


class ServerDeployer:
    """FL Server deployer."""

    def __init__(self):
        """Init the ServerDeployer."""
        self.cmd_modules = ServerCommandModules.cmd_modules
        self.logger = get_obj_logger(self)
        self.server_config = None
        self.secure_train = None
        self.app_validator = None
        self.host = None
        self.snapshot_persistor = None
        self.components = None
        self.handlers = None

    def build(self, build_ctx):
        """To build the ServerDeployer.

        Args:
            build_ctx: build context

        """
        self.server_config = build_ctx["server_config"]
        self.secure_train = build_ctx["secure_train"]
        self.app_validator = build_ctx["app_validator"]
        self.host = build_ctx["server_host"]
        self.snapshot_persistor = build_ctx["snapshot_persistor"]
        self.components = build_ctx["server_components"]
        self.handlers = build_ctx["server_handlers"]

    def create_fl_server(self, args, secure_train=False):
        """To create the FL Server.

        Args:
            args: command args
            secure_train: True/False

        Returns: FL Server

        """
        # We only deploy the first server right now .....
        first_server = sorted(self.server_config)[0]
        heart_beat_timeout = first_server.get("heart_beat_timeout", 600)
        self.logger.info(f"server heartbeat timeout set to {heart_beat_timeout}")

        if self.host:
            target = first_server["service"].get("target", None)
            first_server["service"]["target"] = self.host + ":" + target.split(":")[1]

        services = FederatedServer(
            project_name=first_server.get("name", ""),
            min_num_clients=first_server.get("min_num_clients", 1),
            max_num_clients=first_server.get("max_num_clients", 100),
            cmd_modules=self.cmd_modules,
            heart_beat_timeout=heart_beat_timeout,
            args=args,
            secure_train=secure_train,
            snapshot_persistor=self.snapshot_persistor,
            shutdown_period=first_server.get("shutdown_period", 30.0),
            check_engine_frequency=first_server.get("check_engine_frequency", 3.0),
        )
        return first_server, services

    def deploy(self, args):
        """To deploy the FL server services.

        Args:
            args: command args.

        Returns: FL Server

        """
        workspace = Workspace(args.workspace, SiteType.SERVER, args.config_folder)
        first_server, services = self.create_fl_server(args, secure_train=self.secure_train)
        self._initialize_state_store(services, args.workspace)
        services.deploy(args, grpc_args=first_server, secure_train=self.secure_train)

        job_runner = JobRunner(workspace_root=args.workspace)
        run_manager = RunManager(
            server_name=SiteType.SERVER,
            engine=services.engine,
            job_id="",
            workspace=workspace,
            components=self.components,
            handlers=self.handlers,
        )
        job_manager = self.components.get(SystemComponents.JOB_MANAGER)
        services.engine.set_run_manager(run_manager)
        services.engine.set_job_runner(job_runner, job_manager)

        fed_event_runner = ServerFedEventRunner()
        run_manager.add_handler(fed_event_runner)

        run_manager.add_handler(job_runner)
        run_manager.add_component(SystemComponents.JOB_RUNNER, job_runner)

        with services.engine.new_context() as fl_ctx:
            fl_ctx.set_prop(ReservedKey.RUN_ABORT_SIGNAL, Signal(), private=True, sticky=True)
            fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True)
            fl_ctx.set_prop(FLContextKey.ARGS, args, private=True, sticky=True)
            fl_ctx.set_prop(FLContextKey.SITE_OBJ, services, private=True, sticky=True)
            services.engine.fire_event(EventType.SYSTEM_BOOTSTRAP, fl_ctx)

            component_security_check(fl_ctx)

            threading.Thread(target=self._start_job_runner, args=[job_runner, fl_ctx]).start()
            services.status = ServerStatus.STARTED

            services.engine.fire_event(EventType.SYSTEM_START, fl_ctx)
            self.logger.info("deployed FLARE Server.")

        return services

    def _initialize_state_store(self, services: FederatedServer, server_root: str):
        store = self.components.get(SystemComponents.STATE_STORE)
        if store is None:
            db_url = default_state_store_db_url(server_root)
            self.logger.warning(
                f"component '{SystemComponents.STATE_STORE}' is not configured in local/resources.json; "
                f"defaulting to SqlStateStore with db_url '{db_url}'. Add a '{SystemComponents.STATE_STORE}' "
                "component to local/resources.json to control the state store database location."
            )
            store = SqlStateStore(db_url=db_url)
            self.components[SystemComponents.STATE_STORE] = store
        if not isinstance(store, SqlStateStore):
            # Startup needs SqlStateStore-only APIs (migration markers, schema bootstrap),
            # which are deliberately not part of the minimal StateStore ABC.
            raise TypeError(
                f"component '{SystemComponents.STATE_STORE}' must be a SqlStateStore; other StateStore "
                f"implementations are not yet supported for server startup, got {type(store)}"
            )
        store = self._anchor_relative_db_url(store, server_root)
        _ensure_sqlite_parent_dir(store.db_url)
        self._validate_or_bootstrap(store, server_root)
        study_store.configure(store)
        services.client_manager.set_state_store(store)

    def _anchor_relative_db_url(self, store: SqlStateStore, server_root: str) -> SqlStateStore:
        """Resolve a relative SQLite db_url against the server workspace root.

        The server process chdirs into the workspace, but relative paths must not depend on the
        CWD; the migrate CLI resolves the same way against --server-root, so both open one file.
        The component's engine was created from the relative URL at construction, so a resolved
        URL requires rebuilding the store before anything connects to it.
        """
        resolved = resolve_relative_db_url(store.db_url, server_root)
        if resolved == store.db_url:
            return store
        self.logger.info(f"resolved relative state store db_url to '{resolved}' (anchored at the workspace root)")
        store.engine.dispose()
        store = SqlStateStore(db_url=resolved)
        self.components[SystemComponents.STATE_STORE] = store
        return store

    def _validate_or_bootstrap(self, store: SqlStateStore, server_root: str):
        """Require the migration marker, but self-bootstrap fresh installs.

        A workspace with no legacy filesystem state gets its schema applied and a fresh-install
        marker written inline, so POC, Docker, bare-metal, and K8s fresh installs start cleanly.
        A workspace whose ONLY legacy artifact is local/study_registry.json is a freshly
        provisioned kit with studies defined in project.yml (provision-time config, not runtime
        data), so its studies are imported inline at startup. If legacy runtime data exists
        (jobs or disabled clients), importing it implicitly at startup is too risky, so the
        operator must run nvflare-state-store-migrate explicitly.
        """
        try:
            validate_state_store_migrated(store)
            return
        except RuntimeError as e:
            jobs_dir = self._resolve_legacy_jobs_dir(server_root)
            state = classify_legacy_state(server_root, jobs_dir)
            if state["jobs"] or state["disabled_clients"]:
                found = []
                if state["jobs"]:
                    found.append(f"legacy jobs under '{jobs_dir}'")
                if state["study_registry"]:
                    found.append(f"study registry at '{state['study_registry']}'")
                if state["disabled_clients"]:
                    found.append(
                        f"disabled clients file at '{Path(server_root).expanduser() / 'disabled_clients.json'}'"
                    )
                raise RuntimeError(
                    f"{e}. Legacy filesystem state was found: {'; '.join(found)}. Run "
                    f"'nvflare-state-store-migrate --server-root {server_root}' to import it, "
                    "then restart the server."
                ) from e
            if state["study_registry"]:
                self._import_provisioned_study_registry(store, state["study_registry"])
                return
        result = bootstrap_fresh_state_store(store)
        if result.get("bootstrapped"):
            self.logger.info("state store bootstrapped for a fresh install (no legacy filesystem state found)")

    def _import_provisioned_study_registry(self, store: SqlStateStore, registry_path: str):
        """Import a provision-time study registry inline at startup.

        Provisioning writes local/study_registry.json into a new server kit when project.yml
        defines studies. With no other legacy artifacts this is a fresh install, not an
        upgrade, so the studies are imported here (writing the real migration marker) instead
        of failing startup and demanding a manual nvflare-state-store-migrate run.
        """
        try:
            store.initialize()
        except RuntimeError:
            migrate_database(store.db_url)
        result = migrate_legacy_state_store(store, job_storage=None, study_registry_path=registry_path)
        summary = (result.get("marker") or {}).get("summary_json") or {}
        self.logger.info(
            f"state store bootstrapped from provisioned study registry '{registry_path}': "
            f"imported studies {summary.get('imported_studies', [])}"
        )

    def _resolve_legacy_jobs_dir(self, server_root: str) -> Optional[str]:
        """Resolve where legacy job objects would live, from the live job-store components.

        Legacy job objects live under the job-store storage component's root_dir + the job
        manager's uri_root (exactly how state_store_migration._filesystem_job_storage resolves
        them) — NOT under the workspace root: the default uri_root is an absolute machine-global
        path, and a relative uri_root pairs with the storage root_dir, not the workspace.
        Returns None when the job store is not a FilesystemStorage (cannot check; treated as
        "no legacy jobs").
        """
        job_manager = self.components.get(SystemComponents.JOB_MANAGER)
        if job_manager is None:
            # No job manager component: fall back to the conventional workspace-relative dir.
            return str(Path(server_root).expanduser() / "jobs")
        uri_root = getattr(job_manager, "uri_root", None) or "jobs"
        job_store_id = getattr(job_manager, "job_store_id", None) or "job_store"
        job_store = self.components.get(job_store_id)
        if not isinstance(job_store, FilesystemStorage):
            self.logger.warning(
                f"job store component '{job_store_id}' is not a FilesystemStorage "
                f"({type(job_store)}); cannot check for legacy filesystem jobs"
            )
            return None
        try:
            return job_store._object_path(uri_root)
        except StorageException as e:
            self.logger.warning(
                f"cannot resolve legacy jobs dir from job store '{job_store_id}' "
                f"(uri_root '{uri_root}'): {e}; cannot check for legacy filesystem jobs"
            )
            return None

    def _start_job_runner(self, job_runner, fl_ctx):
        job_runner.run(fl_ctx)

    def close(self):
        """To close the services."""
        pass
