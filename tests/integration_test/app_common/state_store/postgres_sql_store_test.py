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
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, text

from nvflare.apis.job_def import JobMetaKey, RunStatus, SubmitRecordKey, SubmitRecordState
from nvflare.app_common.state_store.legacy_migration import (
    _SUBMIT_RECORD_URIS_KEY,
    migrate_legacy_state_store,
    validate_state_store_migrated,
)
from nvflare.app_common.state_store.sql_store import SqlStateStore, metadata, migrate_database
from nvflare.app_common.state_store.state_store_migration import main as migration_main
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage

POSTGRES_DB_URL = os.environ.get("NVFLARE_TEST_STATE_STORE_DB_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_DB_URL,
    reason="set NVFLARE_TEST_STATE_STORE_DB_URL to a disposable PostgreSQL database URL",
)


def _reset_database(db_url: str):
    engine = create_engine(db_url, future=True)
    try:
        metadata.drop_all(engine, checkfirst=True)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    finally:
        engine.dispose()


@pytest.fixture()
def postgres_db_url():
    _reset_database(POSTGRES_DB_URL)
    try:
        yield POSTGRES_DB_URL
    finally:
        _reset_database(POSTGRES_DB_URL)


@pytest.fixture()
def store(postgres_db_url):
    migrate_database(postgres_db_url)
    state_store = SqlStateStore(postgres_db_url)
    state_store.initialize()
    migrate_legacy_state_store(state_store)
    validate_state_store_migrated(state_store)
    try:
        yield state_store
    finally:
        state_store.engine.dispose()


def _job_meta(job_id: str):
    return {
        JobMetaKey.JOB_ID.value: job_id,
        JobMetaKey.STUDY.value: "study-a",
        JobMetaKey.STATUS.value: RunStatus.SUBMITTED.value,
        JobMetaKey.JOB_NAME.value: "hello",
        JobMetaKey.JOB_FOLDER_NAME.value: "hello_job",
        JobMetaKey.SUBMITTER_NAME.value: "admin@nvidia.com",
        JobMetaKey.SUBMITTER_ORG.value: "nvidia",
        JobMetaKey.SUBMITTER_ROLE.value: "project_admin",
        JobMetaKey.SUBMIT_TIME.value: 1.0,
        JobMetaKey.SUBMIT_TIME_ISO.value: "2026-06-08T00:00:00+00:00",
    }


def _submit_record(job_id: str, token: str = "token-1"):
    return {
        SubmitRecordKey.SCHEMA_VERSION.value: 1,
        SubmitRecordKey.STATE.value: SubmitRecordState.CREATING.value,
        SubmitRecordKey.SUBMIT_TOKEN.value: token,
        SubmitRecordKey.JOB_ID.value: job_id,
        SubmitRecordKey.STUDY.value: "study-a",
        SubmitRecordKey.SUBMITTER_NAME.value: "admin@nvidia.com",
        SubmitRecordKey.SUBMITTER_ORG.value: "nvidia",
        SubmitRecordKey.SUBMITTER_ROLE.value: "project_admin",
        SubmitRecordKey.JOB_NAME.value: "hello",
        SubmitRecordKey.JOB_FOLDER_NAME.value: "hello_job",
        SubmitRecordKey.JOB_CONTENT_HASH.value: "sha256:abc",
        SubmitRecordKey.SUBMIT_TIME.value: "2026-06-08T00:00:00+00:00",
    }


def _write_legacy_server_root(server_root, db_url_env: str):
    local_dir = server_root / "local"
    startup_dir = server_root / "startup"
    local_dir.mkdir(parents=True)
    startup_dir.mkdir()
    (local_dir / "resources.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "id": "state_store",
                        "path": "nvflare.app_common.state_store.sql_store.SqlStateStore",
                        "args": {"db_url_env": db_url_env},
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
    (local_dir / "study_registry.json").write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "studies": {
                    "study-a": {
                        "admins": ["admin@nvidia.com"],
                        "site_orgs": {"nvidia": ["site-a"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (server_root / "disabled_clients.json").write_text(json.dumps({"disabled_clients": ["site-disabled"]}))

    storage = FilesystemStorage(root_dir=str(server_root), uri_root="/")
    storage.create_object("jobs/job-legacy", b"job bytes", _job_meta("job-legacy"), overwrite_existing=False)
    record_uri = "job_submit_records/study/submitter/token"
    storage.create_object(record_uri, b"", _submit_record("job-legacy"), overwrite_existing=False)
    storage.create_object(
        "job_submit_record_index/job-legacy",
        b"",
        {SubmitRecordKey.JOB_ID.value: "job-legacy", _SUBMIT_RECORD_URIS_KEY: [record_uri]},
        overwrite_existing=False,
    )


def test_postgres_state_store_crud(store):
    store.upsert_study("study-a", {"admins": ["admin@nvidia.com"], "site_orgs": {"nvidia": ["site-a"]}})
    store.add_study_sites("study-a", {"partner": ["site-b"]})
    assert store.get_study("study-a")["config_json"]["site_orgs"] == {
        "nvidia": ["site-a"],
        "partner": ["site-b"],
    }

    job_id = str(uuid.uuid4())
    store.create_job(_job_meta(job_id), content_uri="artifact://jobs/hello.zip", content_hash="sha256:abc")
    store.update_job_meta(job_id, {JobMetaKey.STATUS.value: RunStatus.RUNNING.value, "custom": "db-only"})
    job = store.get_job(job_id)
    assert job["status"] == RunStatus.RUNNING.value
    assert job["meta_json"]["custom"] == "db-only"
    assert [row["job_id"] for row in store.list_jobs(status=RunStatus.RUNNING.value)] == [job_id]

    assert store.create_submit_record(_submit_record(job_id)) is True
    assert store.create_submit_record(_submit_record(job_id)) is False
    deleted = store.mark_submit_records_job_deleted(
        job_id, {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}
    )
    assert deleted[0][SubmitRecordKey.STATE.value] == SubmitRecordState.JOB_DELETED.value

    assert store.disable_client("site-a", disabled_by="admin@nvidia.com")["client_name"] == "site-a"
    assert store.get_disabled_client("site-a") is not None
    assert store.enable_client("site-a") is True


def test_postgres_migration_cli_imports_server_root_end_to_end(postgres_db_url, tmp_path, monkeypatch):
    server_root = tmp_path / "server"
    db_url_env = "NVFLARE_STATE_STORE_DB_URL"
    _write_legacy_server_root(server_root, db_url_env)
    monkeypatch.setenv(db_url_env, postgres_db_url)

    assert migration_main(["--server-root", str(server_root)]) == 0

    store = SqlStateStore(db_url_env=db_url_env)
    try:
        marker = validate_state_store_migrated(store)
        assert marker["summary_json"]["imported_studies"] == ["study-a"]
        assert marker["summary_json"]["imported_jobs"] == ["job-legacy"]
        assert marker["summary_json"]["imported_submit_records"] == 1
        assert marker["summary_json"]["imported_disabled_clients"] == ["site-disabled"]
        assert store.get_study("study-a")["config_json"]["site_orgs"] == {"nvidia": ["site-a"]}
        assert store.get_job("job-legacy")["content_uri"] == "jobs/job-legacy"
        assert (
            store.get_submit_record(
                "study-a",
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"},
                "token-1",
            )[SubmitRecordKey.JOB_ID.value]
            == "job-legacy"
        )
        assert store.get_disabled_client("site-disabled") is not None
    finally:
        store.engine.dispose()


def test_postgres_submit_record_uniqueness_under_concurrency(store):
    job_id = str(uuid.uuid4())
    store.create_job(_job_meta(job_id), content_uri="artifact://jobs/hello.zip")
    record = _submit_record(job_id, token="retry-token")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _i: store.create_submit_record(dict(record)), range(8)))

    assert results.count(True) == 1
    assert results.count(False) == 7
