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

import pytest

pytest.importorskip("alembic")
pytest.importorskip("sqlalchemy")

from nvflare.apis.job_def import JobMetaKey, RunStatus, SubmitRecordKey, SubmitRecordState
from nvflare.apis.storage import StorageException
from nvflare.app_common.state_store.legacy_migration import (
    _SUBMIT_RECORD_URIS_KEY,
    LEGACY_MIGRATION_MARKER,
    load_legacy_disabled_clients,
    load_legacy_study_registry,
    migrate_legacy_state_store,
    validate_state_store_migrated,
)
from nvflare.app_common.state_store.sql_store import SqlStateStore, migrate_database, sqlite_url
from nvflare.app_common.state_store.state_store_migration import main as migration_main
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage


def _store(tmp_path):
    store = SqlStateStore.sqlite(str(tmp_path / "state_store.db"))
    migrate_database(store.db_url)
    store.initialize()
    return store


def _write_registry(path):
    path.write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "studies": {
                    "study-a": {
                        "admins": ["admin@nvidia.com"],
                        "site_orgs": {"org-a": ["site-a"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_disabled_clients(path):
    path.write_text(json.dumps({"disabled_clients": ["site-b"]}), encoding="utf-8")


def _write_job_store(tmp_path):
    storage = FilesystemStorage(root_dir=str(tmp_path / "job_store"), uri_root="/")
    job_meta = {
        JobMetaKey.JOB_ID.value: "job-1",
        JobMetaKey.STATUS.value: RunStatus.SUBMITTED.value,
        JobMetaKey.JOB_NAME.value: "hello",
        JobMetaKey.JOB_FOLDER_NAME.value: "hello",
    }
    storage.create_object("jobs/job-1", b"job bytes", job_meta, overwrite_existing=False)

    record_uri = "job_submit_records/study/submitter/token"
    record = {
        SubmitRecordKey.SCHEMA_VERSION.value: 1,
        SubmitRecordKey.STATE.value: SubmitRecordState.CREATED.value,
        SubmitRecordKey.SUBMIT_TOKEN.value: "retry-1",
        SubmitRecordKey.JOB_ID.value: "job-1",
        SubmitRecordKey.STUDY.value: "study-a",
        SubmitRecordKey.SUBMITTER_NAME.value: "submitter@nvidia.com",
        SubmitRecordKey.SUBMITTER_ORG.value: "nvidia",
        SubmitRecordKey.SUBMITTER_ROLE.value: "lead",
        SubmitRecordKey.JOB_CONTENT_HASH.value: "sha256:abc",
    }
    storage.create_object(record_uri, b"", record, overwrite_existing=False)
    storage.create_object(
        "job_submit_record_index/job-1",
        b"",
        {SubmitRecordKey.JOB_ID.value: "job-1", _SUBMIT_RECORD_URIS_KEY: [record_uri]},
        overwrite_existing=False,
    )
    return storage


def _write_server_root(path, db_path=None, job_storage=None, db_url_env=None):
    local_dir = path / "local"
    startup_dir = path / "startup"
    local_dir.mkdir(parents=True)
    startup_dir.mkdir()

    state_store_args = {"db_url_env": db_url_env} if db_url_env else {"db_url": sqlite_url(str(db_path))}
    components = [
        {
            "id": "state_store",
            "path": "nvflare.app_common.state_store.sql_store.SqlStateStore",
            "args": state_store_args,
        }
    ]
    if job_storage:
        components.extend(
            [
                {
                    "id": "job_manager",
                    "path": "nvflare.apis.impl.job_def_manager.SimpleJobDefManager",
                    "args": {"uri_root": "jobs", "job_store_id": "job_store"},
                },
                {
                    "id": "job_store",
                    "path": "nvflare.app_common.storages.filesystem_storage.FilesystemStorage",
                    "args": {"root_dir": job_storage.root_dir, "uri_root": job_storage.uri_root},
                },
            ]
        )
    (local_dir / "resources.json").write_text(json.dumps({"components": components}), encoding="utf-8")


def _write_default_job_store_server_root(path, db_path):
    local_dir = path / "local"
    startup_dir = path / "startup"
    local_dir.mkdir(parents=True)
    startup_dir.mkdir()
    (local_dir / "resources.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "id": "state_store",
                        "path": "nvflare.app_common.state_store.sql_store.SqlStateStore",
                        "args": {"db_url": sqlite_url(str(db_path))},
                    },
                    {
                        "id": "job_manager",
                        "path": "nvflare.apis.impl.job_def_manager.SimpleJobDefManager",
                        "args": {"uri_root": "jobs", "job_store_id": "job_store"},
                    },
                    {
                        "id": "job_store",
                        "path": "nvflare.app_common.storages.filesystem_storage.FilesystemStorage",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


class _BrokenListStorage:
    def list_objects(self, path):
        raise StorageException("backend unavailable")


def test_migration_imports_legacy_state_once(tmp_path):
    store = _store(tmp_path)
    job_storage = _write_job_store(tmp_path)
    registry_path = tmp_path / "study_registry.json"
    disabled_path = tmp_path / "disabled_clients.json"
    _write_registry(registry_path)
    _write_disabled_clients(disabled_path)

    result = migrate_legacy_state_store(
        store,
        job_storage=job_storage,
        jobs_uri_root="jobs",
        study_registry_path=str(registry_path),
        disabled_clients_path=str(disabled_path),
    )

    assert result["migrated"] is True
    marker = store.get_migration_marker(LEGACY_MIGRATION_MARKER)
    assert marker["summary_json"]["status"] == "complete"
    assert marker["summary_json"]["imported_studies"] == ["study-a"]
    assert marker["summary_json"]["imported_jobs"] == ["job-1"]
    assert marker["summary_json"]["imported_submit_records"] == 1
    assert marker["summary_json"]["imported_disabled_clients"] == ["site-b"]
    assert store.get_study("study-a")["config_json"]["admins"] == ["admin@nvidia.com"]
    assert store.get_job("job-1")["content_uri"] == "jobs/job-1"
    assert store.get_disabled_client("site-b") is not None
    assert validate_state_store_migrated(store)["name"] == LEGACY_MIGRATION_MARKER

    second = migrate_legacy_state_store(store, job_storage=job_storage, study_registry_path=str(registry_path))

    assert second["migrated"] is False


def test_migration_refuses_existing_data_without_marker(tmp_path):
    store = _store(tmp_path)
    store.upsert_study("existing-study", {"admins": [], "site_orgs": {}})

    with pytest.raises(RuntimeError, match="marker is missing"):
        migrate_legacy_state_store(store)

    assert store.get_migration_marker(LEGACY_MIGRATION_MARKER) is None
    assert store.get_study("existing-study") is not None


def test_validation_requires_migration_marker(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(RuntimeError, match="nvflare-state-store-migrate"):
        validate_state_store_migrated(store)


def test_cli_migrates_schema_and_legacy_state(tmp_path):
    job_storage = _write_job_store(tmp_path)
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_server_root(server_root, db_path, job_storage)
    registry_path = server_root / "local" / "study_registry.json"
    disabled_path = server_root / "disabled_clients.json"
    _write_registry(registry_path)
    _write_disabled_clients(disabled_path)

    rc = migration_main(["--server-root", str(server_root)])

    assert rc == 0
    store = SqlStateStore.sqlite(str(db_path))
    assert validate_state_store_migrated(store)["summary_json"]["imported_jobs"] == ["job-1"]
    assert store.get_study("study-a")["config_json"]["site_orgs"] == {"org-a": ["site-a"]}
    assert store.get_disabled_client("site-b") is not None


def test_cli_resolves_db_url_env_from_resources(tmp_path, monkeypatch):
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_server_root(server_root, db_url_env="NVFLARE_TEST_STATE_STORE_DB_URL")
    monkeypatch.setenv("NVFLARE_TEST_STATE_STORE_DB_URL", sqlite_url(str(db_path)))

    assert migration_main(["--server-root", str(server_root)]) == 0

    store = SqlStateStore.sqlite(str(db_path))
    marker = validate_state_store_migrated(store)
    assert marker["summary_json"]["imported_jobs"] == []


def test_cli_requires_server_root():
    with pytest.raises(SystemExit):
        migration_main([])


def test_cli_creates_empty_marker_when_legacy_files_are_absent(tmp_path):
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_server_root(server_root, db_path)

    assert migration_main(["--server-root", str(server_root)]) == 0

    store = SqlStateStore.sqlite(str(db_path))
    marker = validate_state_store_migrated(store)
    assert marker["summary_json"]["imported_jobs"] == []


def test_migration_does_not_mark_complete_when_job_listing_fails(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(RuntimeError, match="failed to list legacy jobs"):
        migrate_legacy_state_store(store, job_storage=_BrokenListStorage(), jobs_uri_root="jobs")

    assert store.get_migration_marker(LEGACY_MIGRATION_MARKER) is None


def test_cli_migrates_default_relative_job_root_from_server_root(tmp_path):
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_default_job_store_server_root(server_root, db_path)
    job_storage = FilesystemStorage(root_dir=str(server_root), uri_root="/")
    job_meta = {
        JobMetaKey.JOB_ID.value: "job-1",
        JobMetaKey.STATUS.value: RunStatus.SUBMITTED.value,
        JobMetaKey.JOB_NAME.value: "hello",
        JobMetaKey.JOB_FOLDER_NAME.value: "hello",
    }
    job_storage.create_object("jobs/job-1", b"job bytes", job_meta, overwrite_existing=False)

    assert migration_main(["--server-root", str(server_root)]) == 0

    store = SqlStateStore.sqlite(str(db_path))
    marker = validate_state_store_migrated(store)
    assert marker["summary_json"]["imported_jobs"] == ["job-1"]
    assert marker["summary_json"]["legacy_jobs_uri_root_missing"] is False


def test_cli_records_missing_jobs_root_for_fresh_server(tmp_path):
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_default_job_store_server_root(server_root, db_path)

    assert migration_main(["--server-root", str(server_root)]) == 0

    store = SqlStateStore.sqlite(str(db_path))
    marker = validate_state_store_migrated(store)
    assert marker["summary_json"]["imported_jobs"] == []
    assert marker["summary_json"]["legacy_jobs_uri_root_missing"] is True


def test_migration_validates_legacy_files(tmp_path):
    disabled_path = tmp_path / "disabled_clients.json"
    disabled_path.write_text(json.dumps(["site-a"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_legacy_disabled_clients(str(disabled_path))

    registry_path = tmp_path / "study_registry.json"
    registry_path.write_text(json.dumps({"studies": {"study-a": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_legacy_study_registry(str(registry_path))
