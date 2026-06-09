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
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nvflare.apis import study_store
from nvflare.apis.fl_constant import SystemComponents
from nvflare.apis.state_store import StateStore
from nvflare.app_common.state_store import SqlStateStore
from nvflare.app_common.state_store.legacy_migration import (
    FRESH_INSTALL_SOURCE_FORMAT,
    LEGACY_MIGRATION_MARKER,
    LEGACY_SOURCE_FORMAT,
    validate_state_store_migrated,
)
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage
from nvflare.private.fed.app.deployer.server_deployer import ServerDeployer


@pytest.fixture(autouse=True)
def _reset_study_store():
    yield
    study_store.reset()


def _make_deployer(components=None) -> ServerDeployer:
    deployer = ServerDeployer()
    deployer.components = components if components is not None else {}
    return deployer


def _make_services() -> MagicMock:
    return MagicMock()


def _workspace_store(workspace: Path) -> SqlStateStore:
    return SqlStateStore.sqlite(str(workspace / "state-store.db"))


def _write_legacy_study_registry(workspace: Path, studies=None):
    if studies is None:
        studies = {"cancer-research": {"site_orgs": {"org_a": ["hospital-a"]}, "admins": ["admin@org_a.com"]}}
    local_dir = workspace / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    registry_path = local_dir / "study_registry.json"
    registry_path.write_text(json.dumps({"format_version": "1.0", "studies": studies}))
    return registry_path


def _job_store_components(jobs_root: Path) -> dict:
    """Live job-manager + FilesystemStorage components matching how a real server wires them.

    The job manager's uri_root is absolute (the provisioned default is the machine-global
    /tmp/nvflare/jobs-storage) and the storage root_dir is "/", so legacy job objects live at
    root_dir + uri_root — typically OUTSIDE the workspace.
    """
    job_manager = MagicMock()
    job_manager.uri_root = str(jobs_root)
    job_manager.job_store_id = "job_store"
    return {
        SystemComponents.JOB_MANAGER: job_manager,
        "job_store": FilesystemStorage(root_dir="/", uri_root="/"),
    }


class TestInitializeStateStore:
    def test_fresh_workspace_bootstraps_and_writes_marker(self, tmp_path):
        store = _workspace_store(tmp_path)
        deployer = _make_deployer({SystemComponents.STATE_STORE: store})
        services = _make_services()

        deployer._initialize_state_store(services, str(tmp_path))

        marker = validate_state_store_migrated(store)
        assert marker["source_format"] == FRESH_INSTALL_SOURCE_FORMAT
        assert marker["name"] == LEGACY_MIGRATION_MARKER
        services.client_manager.set_state_store.assert_called_once_with(store)
        assert study_store.get_state_store() is store

    def test_bootstrap_is_idempotent_across_restarts(self, tmp_path):
        store = _workspace_store(tmp_path)
        deployer = _make_deployer({SystemComponents.STATE_STORE: store})
        deployer._initialize_state_store(_make_services(), str(tmp_path))
        first_marker = validate_state_store_migrated(store)

        # second startup against the same DB must not rewrite the marker or fail
        deployer._initialize_state_store(_make_services(), str(tmp_path))
        assert validate_state_store_migrated(store) == first_marker

    def test_registry_only_workspace_imports_studies_inline(self, tmp_path):
        # A freshly provisioned kit with studies in project.yml: the registry is provision-time
        # config, not legacy runtime data, so startup imports it instead of failing.
        registry_path = _write_legacy_study_registry(tmp_path)
        store = _workspace_store(tmp_path)
        deployer = _make_deployer({SystemComponents.STATE_STORE: store})

        deployer._initialize_state_store(_make_services(), str(tmp_path))

        marker = validate_state_store_migrated(store)
        assert marker["source_format"] == LEGACY_SOURCE_FORMAT
        assert marker["summary_json"]["imported_studies"] == ["cancer-research"]
        assert marker["summary_json"]["study_registry_path"] == str(registry_path)
        study = store.get_study("cancer-research")
        assert study is not None
        assert study["config_json"]["admins"] == ["admin@org_a.com"]
        assert study["config_json"]["site_orgs"] == {"org_a": ["hospital-a"]}

    def test_registry_only_import_is_idempotent_across_restarts(self, tmp_path):
        _write_legacy_study_registry(tmp_path)
        store = _workspace_store(tmp_path)
        deployer = _make_deployer({SystemComponents.STATE_STORE: store})
        deployer._initialize_state_store(_make_services(), str(tmp_path))
        first_marker = validate_state_store_migrated(store)

        # restart with the registry file still in place: marker present -> no re-import
        deployer._initialize_state_store(_make_services(), str(tmp_path))
        assert validate_state_store_migrated(store) == first_marker
        assert store.get_study("cancer-research") is not None

    def test_registry_plus_jobs_keeps_hard_error_with_enumerated_message(self, tmp_path):
        registry_path = _write_legacy_study_registry(tmp_path)
        jobs_root = tmp_path / "jobs-storage"
        (jobs_root / "job1").mkdir(parents=True)
        store = _workspace_store(tmp_path)
        components = {SystemComponents.STATE_STORE: store}
        components.update(_job_store_components(jobs_root))
        deployer = _make_deployer(components)

        with pytest.raises(RuntimeError, match="nvflare-state-store-migrate") as exc:
            deployer._initialize_state_store(_make_services(), str(tmp_path))
        message = str(exc.value)
        # the error enumerates exactly what was found and where
        assert f"legacy jobs under '{jobs_root.resolve()}'" in message
        assert f"study registry at '{registry_path}'" in message
        # no marker was written: the operator must run the migration explicitly
        with pytest.raises(RuntimeError):
            validate_state_store_migrated(store)

    def test_disabled_clients_file_keeps_hard_error(self, tmp_path):
        (tmp_path / "disabled_clients.json").write_text(json.dumps({"disabled_clients": ["site-1"]}))
        store = _workspace_store(tmp_path)
        deployer = _make_deployer({SystemComponents.STATE_STORE: store})

        with pytest.raises(RuntimeError, match="disabled clients file at") as exc:
            deployer._initialize_state_store(_make_services(), str(tmp_path))
        assert str(tmp_path / "disabled_clients.json") in str(exc.value)

    def test_legacy_jobs_dir_without_marker_keeps_hard_error(self, tmp_path):
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        (jobs_dir / "some-job").mkdir()
        store = _workspace_store(tmp_path)
        deployer = _make_deployer({SystemComponents.STATE_STORE: store})

        with pytest.raises(RuntimeError, match="nvflare-state-store-migrate"):
            deployer._initialize_state_store(_make_services(), str(tmp_path))

    def test_missing_component_creates_default_workspace_store_with_warning(self, tmp_path, caplog):
        deployer = _make_deployer({})
        services = _make_services()

        with caplog.at_level("WARNING"):
            deployer._initialize_state_store(services, str(tmp_path))

        store = deployer.components[SystemComponents.STATE_STORE]
        assert isinstance(store, SqlStateStore)
        assert store.db_url == f"sqlite:///{(tmp_path / 'state-store.db').resolve()}"
        assert (tmp_path / "state-store.db").is_file()
        validate_state_store_migrated(store)
        assert any("is not configured" in rec.message for rec in caplog.records)
        services.client_manager.set_state_store.assert_called_once_with(store)

    def test_relative_db_url_is_anchored_at_workspace(self, tmp_path):
        store = SqlStateStore(db_url="sqlite:///state-store.db")
        deployer = _make_deployer({SystemComponents.STATE_STORE: store})
        services = _make_services()

        deployer._initialize_state_store(services, str(tmp_path))

        resolved_store = deployer.components[SystemComponents.STATE_STORE]
        assert resolved_store is not store
        assert resolved_store.db_url == f"sqlite:///{(tmp_path / 'state-store.db').resolve()}"
        assert (tmp_path / "state-store.db").is_file()
        validate_state_store_migrated(resolved_store)
        services.client_manager.set_state_store.assert_called_once_with(resolved_store)

    def test_misconfigured_component_raises_type_error(self, tmp_path):
        deployer = _make_deployer({SystemComponents.STATE_STORE: object()})
        with pytest.raises(TypeError, match="must be a SqlStateStore"):
            deployer._initialize_state_store(_make_services(), str(tmp_path))

    def test_non_sql_state_store_raises_clear_type_error(self, tmp_path):
        # A custom StateStore passes the ABC check but lacks the SqlStateStore-only
        # migration-marker/bootstrap APIs: startup must reject it up front, not
        # crash later with AttributeError.
        store = MagicMock(spec=StateStore)
        assert isinstance(store, StateStore)
        deployer = _make_deployer({SystemComponents.STATE_STORE: store})
        with pytest.raises(TypeError, match="not yet supported for server startup"):
            deployer._initialize_state_store(_make_services(), str(tmp_path))

    def test_legacy_jobs_dir_from_job_store_storage_component(self, tmp_path):
        # absolute uri_root outside the workspace, resolved through the live FilesystemStorage
        jobs_root = tmp_path / "elsewhere" / "jobs-storage"
        (jobs_root / "job1").mkdir(parents=True)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        store = _workspace_store(workspace)
        components = {SystemComponents.STATE_STORE: store}
        components.update(_job_store_components(jobs_root))
        deployer = _make_deployer(components)

        with pytest.raises(RuntimeError, match="nvflare-state-store-migrate"):
            deployer._initialize_state_store(_make_services(), str(workspace))

    def test_relative_uri_root_resolves_against_job_store_root_dir(self, tmp_path):
        # Relative uri_root + a job-store root_dir elsewhere: legacy jobs live OUTSIDE the
        # workspace at root_dir/uri_root. Anchoring uri_root at the workspace would miss them
        # (false negative -> fresh marker -> legacy jobs permanently stranded).
        storage_root = tmp_path / "elsewhere" / "storage-root"
        (storage_root / "jobs" / "job1").mkdir(parents=True)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        job_manager = MagicMock()
        job_manager.uri_root = "jobs"
        job_manager.job_store_id = "job_store"
        store = _workspace_store(workspace)
        deployer = _make_deployer(
            {
                SystemComponents.STATE_STORE: store,
                SystemComponents.JOB_MANAGER: job_manager,
                "job_store": FilesystemStorage(root_dir=str(storage_root), uri_root="/"),
            }
        )

        with pytest.raises(RuntimeError, match="nvflare-state-store-migrate") as exc:
            deployer._initialize_state_store(_make_services(), str(workspace))
        assert f"legacy jobs under '{(storage_root / 'jobs').resolve()}'" in str(exc.value)
        # no marker was written: the legacy jobs are not stranded behind a fresh-install marker
        with pytest.raises(RuntimeError):
            validate_state_store_migrated(store)

    def test_empty_shared_absolute_jobs_dir_does_not_block_fresh_start(self, tmp_path):
        # The provisioned default uri_root is an absolute machine-global path that
        # SimpleJobDefManager pre-creates (empty) at construction: an empty dir must not make
        # a fresh workspace look legacy.
        jobs_root = tmp_path / "shared" / "jobs-storage"
        jobs_root.mkdir(parents=True)  # exists but empty
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        store = _workspace_store(workspace)
        components = {SystemComponents.STATE_STORE: store}
        components.update(_job_store_components(jobs_root))
        deployer = _make_deployer(components)

        deployer._initialize_state_store(_make_services(), str(workspace))
        marker = validate_state_store_migrated(store)
        assert marker["source_format"] == FRESH_INSTALL_SOURCE_FORMAT

    def test_non_filesystem_job_store_cannot_check_jobs_and_bootstraps(self, tmp_path, caplog):
        # A non-FilesystemStorage job store cannot be checked for legacy filesystem jobs;
        # the deployer logs that and proceeds (matching the migrate CLI's skip-with-warning).
        job_manager = MagicMock()
        job_manager.uri_root = "jobs"
        job_manager.job_store_id = "job_store"
        store = _workspace_store(tmp_path)
        deployer = _make_deployer(
            {
                SystemComponents.STATE_STORE: store,
                SystemComponents.JOB_MANAGER: job_manager,
                "job_store": object(),
            }
        )

        with caplog.at_level("WARNING"):
            deployer._initialize_state_store(_make_services(), str(tmp_path))

        validate_state_store_migrated(store)
        assert any("cannot check for legacy filesystem jobs" in rec.message for rec in caplog.records)
