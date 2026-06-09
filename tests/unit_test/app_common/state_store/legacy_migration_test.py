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

from nvflare.apis.job_def import JobMetaKey, SubmitRecordKey
from nvflare.apis.storage import StorageException
from nvflare.app_common.state_store.legacy_migration import (
    _SUBMIT_RECORD_URIS_KEY,
    LEGACY_MIGRATION_MARKER,
    MigrationSkipError,
    bootstrap_fresh_state_store,
    classify_legacy_state,
    has_legacy_state,
    load_legacy_disabled_clients,
    load_legacy_study_registry,
    migrate_legacy_state_store,
    validate_state_store_migrated,
)
from nvflare.app_common.state_store.sql_store import SqlStateStore, sqlite_url
from nvflare.app_common.state_store.state_store_migration import main as migration_main
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage
from tests.unit_test.app_common.state_store.state_store_helpers import (
    job_meta,
    make_sqlite_store,
    write_disabled_clients,
    write_legacy_job_store,
    write_registry,
)


def _write_server_root(path, db_path=None, job_storage=None, db_url_env=None, db_url=None, job_store_path=None):
    local_dir = path / "local"
    startup_dir = path / "startup"
    local_dir.mkdir(parents=True)
    startup_dir.mkdir()

    if db_url_env:
        state_store_args = {"db_url_env": db_url_env}
    else:
        state_store_args = {"db_url": db_url or sqlite_url(str(db_path))}
    components = [
        {
            "id": "state_store",
            "path": "nvflare.app_common.state_store.sql_store.SqlStateStore",
            "args": state_store_args,
        }
    ]
    if job_storage or job_store_path:
        job_store = {
            "id": "job_store",
            "path": job_store_path or "nvflare.app_common.storages.filesystem_storage.FilesystemStorage",
        }
        if job_storage:
            job_store["args"] = {"root_dir": job_storage.root_dir, "uri_root": job_storage.uri_root}
        components.extend(
            [
                {
                    "id": "job_manager",
                    "path": "nvflare.apis.impl.job_def_manager.SimpleJobDefManager",
                    "args": {"uri_root": "jobs", "job_store_id": "job_store"},
                },
                job_store,
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
    store = make_sqlite_store(tmp_path)
    job_storage = write_legacy_job_store(tmp_path / "job_store")
    registry_path = tmp_path / "study_registry.json"
    disabled_path = tmp_path / "disabled_clients.json"
    write_registry(registry_path)
    write_disabled_clients(disabled_path)

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
    assert marker["summary_json"]["warnings"] == []
    assert store.get_study("study-a")["config_json"]["admins"] == ["admin@nvidia.com"]
    assert store.get_job("job-1")["content_uri"] == "jobs/job-1"
    assert store.get_disabled_client("site-b") is not None
    assert validate_state_store_migrated(store)["name"] == LEGACY_MIGRATION_MARKER

    second = migrate_legacy_state_store(store, job_storage=job_storage, study_registry_path=str(registry_path))

    assert second["migrated"] is False


def test_migration_preserves_org_with_zero_sites(tmp_path):
    store = make_sqlite_store(tmp_path)
    registry_path = tmp_path / "study_registry.json"
    write_registry(
        registry_path,
        studies={
            "study-a": {
                "admins": ["admin@nvidia.com"],
                "site_orgs": {"org-a": ["site-a"], "org-empty": []},
            }
        },
    )

    migrate_legacy_state_store(store, study_registry_path=str(registry_path))

    assert store.get_study("study-a")["config_json"]["site_orgs"] == {"org-a": ["site-a"], "org-empty": []}


def test_migration_refuses_existing_data_without_marker(tmp_path):
    store = make_sqlite_store(tmp_path)
    store.upsert_study("existing-study", {"admins": [], "site_orgs": {}})

    with pytest.raises(RuntimeError, match="marker is missing"):
        migrate_legacy_state_store(store)

    assert store.get_migration_marker(LEGACY_MIGRATION_MARKER) is None
    assert store.get_study("existing-study") is not None


def test_validation_requires_migration_marker(tmp_path):
    store = make_sqlite_store(tmp_path)

    with pytest.raises(RuntimeError, match="nvflare-state-store-migrate"):
        validate_state_store_migrated(store)


def test_migration_skips_dangling_index_entries(tmp_path, capsys):
    store = make_sqlite_store(tmp_path)
    storage = write_legacy_job_store(tmp_path / "job_store")
    storage.create_object(
        "job_submit_record_index/job-dangling",
        b"",
        {SubmitRecordKey.JOB_ID.value: "job-dangling", _SUBMIT_RECORD_URIS_KEY: ["job_submit_records/missing/record"]},
        overwrite_existing=False,
    )

    result = migrate_legacy_state_store(store, job_storage=storage, jobs_uri_root="jobs")

    assert result["migrated"] is True
    summary = store.get_migration_marker(LEGACY_MIGRATION_MARKER)["summary_json"]
    assert summary["imported_submit_records"] == 1
    assert any("job_submit_records/missing/record" in w for w in summary["warnings"])
    assert "job_submit_records/missing/record" in capsys.readouterr().err


def test_migration_skips_job_with_missing_status(tmp_path, capsys):
    store = make_sqlite_store(tmp_path)
    storage = write_legacy_job_store(tmp_path / "job_store")
    storage.create_object(
        "jobs/job-no-status", b"job bytes", {JobMetaKey.JOB_ID.value: "job-no-status"}, overwrite_existing=False
    )

    result = migrate_legacy_state_store(store, job_storage=storage, jobs_uri_root="jobs")

    assert result["migrated"] is True
    summary = store.get_migration_marker(LEGACY_MIGRATION_MARKER)["summary_json"]
    assert summary["imported_jobs"] == ["job-1"]
    assert any("missing status" in w for w in summary["warnings"])
    assert "missing status" in capsys.readouterr().err
    assert store.get_job("job-no-status") is None


def test_migration_skips_invalid_study_definition(tmp_path):
    store = make_sqlite_store(tmp_path)
    registry_path = tmp_path / "study_registry.json"
    write_registry(
        registry_path,
        studies={
            "good-study": {"admins": [], "site_orgs": {"org-a": ["site-a"]}},
            "bad-study": {"admins": "not-a-list", "site_orgs": {}},
        },
    )

    result = migrate_legacy_state_store(store, study_registry_path=str(registry_path))

    assert result["migrated"] is True
    summary = store.get_migration_marker(LEGACY_MIGRATION_MARKER)["summary_json"]
    assert summary["imported_studies"] == ["good-study"]
    assert any("bad-study" in w for w in summary["warnings"])


def test_strict_mode_turns_skips_into_errors(tmp_path):
    store = make_sqlite_store(tmp_path)
    storage = write_legacy_job_store(tmp_path / "job_store")
    storage.create_object(
        "jobs/job-no-status", b"job bytes", {JobMetaKey.JOB_ID.value: "job-no-status"}, overwrite_existing=False
    )

    with pytest.raises(MigrationSkipError, match="missing status"):
        migrate_legacy_state_store(store, job_storage=storage, jobs_uri_root="jobs", strict=True)

    # nothing committed, marker not written
    assert store.get_migration_marker(LEGACY_MIGRATION_MARKER) is None
    assert store.get_job("job-1") is None


def test_migration_scans_records_root_when_index_is_missing(tmp_path):
    store = make_sqlite_store(tmp_path)
    storage = write_legacy_job_store(tmp_path / "job_store", with_index=False)

    result = migrate_legacy_state_store(store, job_storage=storage, jobs_uri_root="jobs")

    assert result["migrated"] is True
    summary = store.get_migration_marker(LEGACY_MIGRATION_MARKER)["summary_json"]
    assert summary["imported_submit_records"] == 1
    assert (
        store.get_submit_record(
            "study-a",
            {"name": "submitter@nvidia.com", "org": "nvidia", "role": "project_admin"},
            "retry-1",
        )[SubmitRecordKey.JOB_ID.value]
        == "job-1"
    )


def test_cli_migrates_schema_and_legacy_state(tmp_path):
    job_storage = write_legacy_job_store(tmp_path / "job_store")
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_server_root(server_root, db_path, job_storage)
    registry_path = server_root / "local" / "study_registry.json"
    disabled_path = server_root / "disabled_clients.json"
    write_registry(registry_path)
    write_disabled_clients(disabled_path)

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


def test_cli_resolves_relative_sqlite_db_url_against_server_root(tmp_path):
    server_root = tmp_path / "server"
    _write_server_root(server_root, db_url="sqlite:///state-store.db")

    assert migration_main(["--server-root", str(server_root)]) == 0

    db_path = server_root / "state-store.db"
    assert db_path.exists()
    store = SqlStateStore.sqlite(str(db_path))
    assert validate_state_store_migrated(store)["name"] == LEGACY_MIGRATION_MARKER


def test_cli_requires_server_root():
    with pytest.raises(SystemExit):
        migration_main([])


def test_cli_defaults_db_url_when_state_store_component_is_missing(tmp_path, capsys):
    # Legacy workspaces provisioned before the state store existed have no state_store
    # component: the CLI must fall back to the server's default DB, not dead-end.
    server_root = tmp_path / "server"
    local_dir = server_root / "local"
    local_dir.mkdir(parents=True)
    (server_root / "startup").mkdir()
    (local_dir / "resources.json").write_text(json.dumps({"components": []}), encoding="utf-8")
    write_registry(local_dir / "study_registry.json")

    assert migration_main(["--server-root", str(server_root)]) == 0

    assert "defaulting to" in capsys.readouterr().err
    db_path = server_root / "state-store.db"
    assert db_path.is_file()
    store = SqlStateStore.sqlite(str(db_path))
    assert validate_state_store_migrated(store)["name"] == LEGACY_MIGRATION_MARKER
    assert store.get_study("study-a") is not None


def test_cli_creates_empty_marker_when_legacy_files_are_absent(tmp_path):
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_server_root(server_root, db_path)

    assert migration_main(["--server-root", str(server_root)]) == 0

    store = SqlStateStore.sqlite(str(db_path))
    marker = validate_state_store_migrated(store)
    assert marker["summary_json"]["imported_jobs"] == []


def test_cli_skips_job_import_for_non_filesystem_job_store(tmp_path, capsys):
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_server_root(server_root, db_path, job_store_path="my.custom.S3Storage")

    assert migration_main(["--server-root", str(server_root)]) == 0

    assert "SKIPPED" in capsys.readouterr().err
    store = SqlStateStore.sqlite(str(db_path))
    summary = validate_state_store_migrated(store)["summary_json"]
    assert summary["imported_jobs"] == []
    assert any("not FilesystemStorage" in w for w in summary["warnings"])


def test_cli_strict_fails_on_non_filesystem_job_store(tmp_path, capsys):
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_server_root(server_root, db_path, job_store_path="my.custom.S3Storage")

    assert migration_main(["--server-root", str(server_root), "--strict"]) == 1
    assert "not FilesystemStorage" in capsys.readouterr().err


def test_migration_does_not_mark_complete_when_job_listing_fails(tmp_path):
    store = make_sqlite_store(tmp_path)

    with pytest.raises(RuntimeError, match="failed to list legacy jobs"):
        migrate_legacy_state_store(store, job_storage=_BrokenListStorage(), jobs_uri_root="jobs")

    assert store.get_migration_marker(LEGACY_MIGRATION_MARKER) is None


def test_cli_migrates_default_relative_job_root_from_server_root(tmp_path):
    server_root = tmp_path / "server"
    db_path = tmp_path / "state_store.db"
    _write_default_job_store_server_root(server_root, db_path)
    job_storage = FilesystemStorage(root_dir=str(server_root), uri_root="/")
    job_storage.create_object("jobs/job-1", b"job bytes", job_meta("job-1"), overwrite_existing=False)

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


def test_has_legacy_state(tmp_path):
    server_root = tmp_path / "server"
    (server_root / "local").mkdir(parents=True)
    assert has_legacy_state(str(server_root)) is False

    (server_root / "jobs").mkdir()
    assert has_legacy_state(str(server_root)) is False  # empty jobs dir is not legacy data

    write_registry(server_root / "local" / "study_registry.json")
    assert has_legacy_state(str(server_root)) is True
    (server_root / "local" / "study_registry.json").unlink()

    write_disabled_clients(server_root / "disabled_clients.json")
    assert has_legacy_state(str(server_root)) is True
    (server_root / "disabled_clients.json").unlink()

    (server_root / "jobs" / "job-1").mkdir()
    assert has_legacy_state(str(server_root)) is True


def test_classify_legacy_state(tmp_path):
    server_root = tmp_path / "server"
    (server_root / "local").mkdir(parents=True)
    jobs_dir = tmp_path / "job_store" / "jobs"

    # nothing present; None jobs_dir means "cannot check, treat as no jobs"
    assert classify_legacy_state(str(server_root), None) == {
        "jobs": False,
        "study_registry": None,
        "disabled_clients": False,
    }
    assert classify_legacy_state(str(server_root), str(jobs_dir))["jobs"] is False

    jobs_dir.mkdir(parents=True)
    assert classify_legacy_state(str(server_root), str(jobs_dir))["jobs"] is False  # empty dir is not legacy

    (jobs_dir / "job-1").mkdir()
    state = classify_legacy_state(str(server_root), str(jobs_dir))
    assert state == {"jobs": True, "study_registry": None, "disabled_clients": False}

    registry_path = server_root / "local" / "study_registry.json"
    write_registry(registry_path)
    write_disabled_clients(server_root / "disabled_clients.json")
    state = classify_legacy_state(str(server_root), None)
    assert state == {"jobs": False, "study_registry": str(registry_path), "disabled_clients": True}


def test_bootstrap_fresh_state_store(tmp_path):
    # works on an unmigrated database: applies the schema and writes the marker
    store = SqlStateStore.sqlite(str(tmp_path / "fresh.db"))
    with pytest.raises(RuntimeError):
        store.initialize()

    result = bootstrap_fresh_state_store(store)

    assert result["bootstrapped"] is True
    assert result["marker"]["name"] == LEGACY_MIGRATION_MARKER
    assert result["marker"]["source_format"] == "fresh-install"
    validate_state_store_migrated(store)

    # idempotent: a second bootstrap (or one after a legacy migration) keeps the marker
    again = bootstrap_fresh_state_store(store)
    assert again["bootstrapped"] is False
    assert again["marker"]["name"] == LEGACY_MIGRATION_MARKER


def test_bootstrap_keeps_existing_legacy_marker(tmp_path):
    store = make_sqlite_store(tmp_path)
    migrate_legacy_state_store(store)

    result = bootstrap_fresh_state_store(store)

    assert result["bootstrapped"] is False
    assert result["marker"]["source_format"] == "legacy-filesystem"


def test_bootstrap_refuses_populated_database_without_marker(tmp_path):
    # A populated database whose marker is missing (renamed marker, partial restore, shared
    # database) must not be silently stamped as a fresh install.
    store = make_sqlite_store(tmp_path)
    store.upsert_study("existing-study", {"admins": [], "site_orgs": {}})

    with pytest.raises(RuntimeError, match="state data already exists"):
        bootstrap_fresh_state_store(store)

    # the failed bootstrap did not stamp the marker
    assert store.get_migration_marker(LEGACY_MIGRATION_MARKER) is None


def test_migration_converges_when_marker_insert_loses_race(tmp_path, monkeypatch):
    # Simulate a concurrent migration winning the marker insert: ours converges on the
    # existing row and must return the graceful migrated=False path (not proceed into
    # _ensure_no_existing_state_data or the import).
    store = make_sqlite_store(tmp_path)
    original_insert = SqlStateStore._insert_migration_marker

    def racing_insert(self, conn, name, source_format, summary):
        original_insert(self, conn, name, source_format, {"status": "complete"})  # concurrent winner
        return False

    monkeypatch.setattr(SqlStateStore, "_insert_migration_marker", racing_insert)

    result = migrate_legacy_state_store(store)

    assert result["migrated"] is False
    assert result["marker"]["name"] == LEGACY_MIGRATION_MARKER
    assert result["marker"]["summary_json"] == {"status": "complete"}
