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

import uuid

import pytest

pytest.importorskip("alembic")
pytest.importorskip("sqlalchemy")

from sqlalchemy import inspect

from nvflare.apis.job_def import JobMetaKey, RunStatus, SubmitRecordKey, SubmitRecordState
from nvflare.app_common.state_store.sql_store import SqlStateStore, migrate_database, sqlite_url


@pytest.fixture
def store(tmp_path):
    store = SqlStateStore.sqlite(str(tmp_path / "state_store.db"))
    migrate_database(store.db_url)
    store.initialize()
    return store


def _job_meta(job_id: str, study: str = "study_a", status: str = RunStatus.SUBMITTED.value):
    return {
        JobMetaKey.JOB_ID.value: job_id,
        JobMetaKey.STUDY.value: study,
        JobMetaKey.STATUS.value: status,
        JobMetaKey.JOB_NAME.value: "hello",
        JobMetaKey.JOB_FOLDER_NAME.value: "hello_job",
        JobMetaKey.SUBMITTER_NAME.value: "admin@nvidia.com",
        JobMetaKey.SUBMITTER_ORG.value: "nvidia",
        JobMetaKey.SUBMITTER_ROLE.value: "project_admin",
        JobMetaKey.SUBMIT_TIME.value: 1.0,
        JobMetaKey.SUBMIT_TIME_ISO.value: "2026-06-08T00:00:00+00:00",
    }


def _submit_record(job_id: str, token: str = "token-1", submitter_name: str = "admin@nvidia.com"):
    return {
        SubmitRecordKey.SCHEMA_VERSION.value: 1,
        SubmitRecordKey.STATE.value: SubmitRecordState.CREATING.value,
        SubmitRecordKey.SUBMIT_TOKEN.value: token,
        SubmitRecordKey.JOB_ID.value: job_id,
        SubmitRecordKey.STUDY.value: "study_a",
        SubmitRecordKey.SUBMITTER_NAME.value: submitter_name,
        SubmitRecordKey.SUBMITTER_ORG.value: "nvidia",
        SubmitRecordKey.SUBMITTER_ROLE.value: "project_admin",
        SubmitRecordKey.JOB_NAME.value: "hello",
        SubmitRecordKey.JOB_FOLDER_NAME.value: "hello_job",
        SubmitRecordKey.JOB_CONTENT_HASH.value: "sha256:abc",
        SubmitRecordKey.SUBMIT_TIME.value: "2026-06-08T00:00:00+00:00",
    }


def test_migration_creates_minimal_tables(store):
    inspector = inspect(store.engine)

    assert set(inspector.get_table_names()) >= {
        "alembic_version",
        "studies",
        "study_admins",
        "study_sites",
        "jobs",
        "submit_records",
        "disabled_clients",
        "state_store_migrations",
    }

    submit_indexes = {index["name"] for index in inspector.get_indexes("submit_records")}
    assert "idx_submit_records_job_id" in submit_indexes


def test_initialize_requires_migrated_database(tmp_path):
    store = SqlStateStore.sqlite(str(tmp_path / "unmigrated.db"))

    with pytest.raises(RuntimeError, match="expected"):
        store.initialize()


def test_db_url_env_configures_database_url(monkeypatch, tmp_path):
    db_url = sqlite_url(str(tmp_path / "env_configured.db"))
    monkeypatch.setenv("NVFLARE_STATE_STORE_DB_URL", db_url)

    store = SqlStateStore(db_url_env="NVFLARE_STATE_STORE_DB_URL")

    assert store.db_url == db_url
    migrate_database(store.db_url)
    store.initialize()


def test_db_url_env_is_required_when_configured(monkeypatch):
    monkeypatch.delenv("NVFLARE_STATE_STORE_DB_URL", raising=False)

    with pytest.raises(ValueError, match="NVFLARE_STATE_STORE_DB_URL"):
        SqlStateStore(db_url_env="NVFLARE_STATE_STORE_DB_URL")


def test_jobs(store):
    job_id = str(uuid.uuid4())

    job = store.create_job(_job_meta(job_id), content_uri="artifact://jobs/hello.zip", content_hash="sha256:abc")
    assert job["job_id"] == job_id
    assert job["status"] == RunStatus.SUBMITTED.value
    assert job["content_uri"] == "artifact://jobs/hello.zip"

    running = store.set_job_status(job_id, RunStatus.RUNNING.value)
    assert running["status"] == RunStatus.RUNNING.value
    assert running["meta_json"][JobMetaKey.STATUS.value] == RunStatus.RUNNING.value
    assert [row["job_id"] for row in store.list_jobs(status=RunStatus.RUNNING.value)] == [job_id]

    assert store.delete_job(job_id) is True
    assert store.get_job(job_id) is None


def test_submit_records_are_scoped_and_do_not_require_existing_job(store):
    job_id = str(uuid.uuid4())
    record = _submit_record(job_id)

    assert store.create_submit_record(record) is True
    assert store.create_submit_record(record) is False

    same_token_different_submitter = _submit_record(
        str(uuid.uuid4()), token="token-1", submitter_name="other@nvidia.com"
    )
    assert store.create_submit_record(same_token_different_submitter) is True

    fetched = store.get_submit_record(
        "study_a",
        {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"},
        "token-1",
    )
    assert fetched[SubmitRecordKey.JOB_ID.value] == job_id

    fetched[SubmitRecordKey.STATE.value] = SubmitRecordState.CREATED.value
    updated = store.update_submit_record(fetched)
    assert updated[SubmitRecordKey.STATE.value] == SubmitRecordState.CREATED.value

    deleted = store.mark_submit_records_job_deleted(
        job_id,
        {"name": "deleter@nvidia.com", "org": "nvidia", "role": "project_admin"},
    )
    assert len(deleted) == 1
    assert deleted[0][SubmitRecordKey.STATE.value] == SubmitRecordState.JOB_DELETED.value


def test_studies_and_disabled_clients(store):
    study = store.upsert_study(
        "study_a",
        {"admins": ["admin@nvidia.com"], "site_orgs": {}},
    )
    assert study["name"] == "study_a"
    assert store.get_study("study_a")["config_json"]["admins"] == ["admin@nvidia.com"]

    updated = store.upsert_study("study_a", {"admins": [], "site_orgs": {}})
    assert updated["config_json"] == {"admins": [], "site_orgs": {}}
    assert store.delete_study("study_a") is True
    assert store.get_study("study_a") is None

    disabled = store.disable_client("site-1", disabled_by="admin@nvidia.com", reason="maintenance")
    assert disabled["client_name"] == "site-1"
    assert store.list_disabled_clients()[0]["reason"] == "maintenance"
    assert store.enable_client("site-1") is True
    assert store.list_disabled_clients() == []


def test_study_membership_operations_update_relational_rows(store):
    store.upsert_study("study_a", {"admins": ["admin@nvidia.com"], "site_orgs": {"org_a": ["site-a"]}})

    store.add_study_admin("study_a", "lead@nvidia.com")
    assert store.get_study("study_a")["config_json"]["admins"] == ["admin@nvidia.com", "lead@nvidia.com"]

    store.add_study_sites("study_a", {"org_b": ["site-b"]})
    assert store.get_study("study_a")["config_json"]["site_orgs"] == {"org_a": ["site-a"], "org_b": ["site-b"]}

    store.remove_study_admin("study_a", "lead@nvidia.com")
    store.remove_study_sites("study_a", {"org_a": ["site-a"]})
    assert store.get_study("study_a")["config_json"] == {
        "admins": ["admin@nvidia.com"],
        "site_orgs": {"org_b": ["site-b"]},
    }

    assert store.delete_study("study_a") is True
    assert store.get_study("study_a") is None
