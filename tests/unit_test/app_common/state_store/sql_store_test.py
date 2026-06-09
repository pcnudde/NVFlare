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

import threading
import uuid

import pytest

pytest.importorskip("alembic")
pytest.importorskip("sqlalchemy")

from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from nvflare.apis.job_def import JobMetaKey, RunStatus, SubmitRecordKey, SubmitRecordState
from nvflare.app_common.state_store import sql_store
from nvflare.app_common.state_store.sql_store import (
    SqlStateStore,
    migrate_database,
    resolve_relative_db_url,
    sqlite_url,
    validate_database,
)
from tests.unit_test.app_common.state_store.state_store_helpers import job_meta, make_sqlite_store, submit_record


@pytest.fixture
def store(tmp_path):
    return make_sqlite_store(tmp_path)


def _without_timestamps(row: dict) -> dict:
    # SQLite round-trips DateTime columns without tzinfo; ignore them in row comparisons.
    return {k: v for k, v in row.items() if k not in ("created_at", "updated_at")}


def test_migration_creates_minimal_tables(store):
    inspector = inspect(store.engine)

    assert set(inspector.get_table_names()) >= {
        "alembic_version",
        "studies",
        "study_admins",
        "study_orgs",
        "study_sites",
        "jobs",
        "submit_records",
        "disabled_clients",
        "state_store_migrations",
    }

    submit_indexes = {index["name"] for index in inspector.get_indexes("submit_records")}
    assert "idx_submit_records_job_id" in submit_indexes

    job_columns = {column["name"] for column in inspector.get_columns("jobs")}
    assert job_columns == {
        "job_id",
        "study",
        "status",
        "content_uri",
        "content_hash",
        "content_size",
        "meta_json",
        "version",
        "created_at",
        "updated_at",
    }
    record_columns = {column["name"] for column in inspector.get_columns("submit_records")}
    assert record_columns == {
        "id",
        "study_hash",
        "submitter_hash",
        "submit_token_hash",
        "job_id",
        "record_json",
        "version",
        "created_at",
        "updated_at",
    }


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


def test_migrate_and_validate_handle_percent_in_db_url(tmp_path):
    # '%' is ConfigParser interpolation syntax in Alembic's Config and must be escaped.
    pct_dir = tmp_path / "p%40ss"
    pct_dir.mkdir()
    db_url = sqlite_url(str(pct_dir / "state_store.db"))
    assert "%" in db_url

    migrate_database(db_url)
    validate_database(db_url)
    SqlStateStore(db_url).initialize()


def test_resolve_relative_db_url(tmp_path):
    base = tmp_path / "workspace"
    resolved = resolve_relative_db_url("sqlite:///state-store.db", str(base))
    assert resolved == f"sqlite:///{(base / 'state-store.db').resolve()}"

    nested = resolve_relative_db_url("sqlite:///data/state-store.db", str(base))
    assert nested == f"sqlite:///{(base / 'data' / 'state-store.db').resolve()}"

    absolute = sqlite_url(str(tmp_path / "abs.db"))
    assert resolve_relative_db_url(absolute, str(base)) == absolute

    postgres = "postgresql+psycopg://fl:p%40ss@db/state"
    assert resolve_relative_db_url(postgres, str(base)) == postgres


def test_jobs(store):
    job_id = str(uuid.uuid4())

    job = store.create_job(job_meta(job_id), content_uri="artifact://jobs/hello.zip", content_hash="sha256:abc")
    assert job["job_id"] == job_id
    assert job["status"] == RunStatus.SUBMITTED.value
    assert job["content_uri"] == "artifact://jobs/hello.zip"
    # create_job returns the created row without a second read
    assert _without_timestamps(job) == _without_timestamps(store.get_job(job_id))

    running = store.update_job_meta(job_id, {JobMetaKey.STATUS.value: RunStatus.RUNNING.value})
    assert running["status"] == RunStatus.RUNNING.value
    assert running["version"] == 2
    assert running["meta_json"][JobMetaKey.STATUS.value] == RunStatus.RUNNING.value
    # update_job_meta returns the merged row without a second read
    assert _without_timestamps(running) == _without_timestamps(store.get_job(job_id))
    assert [row["job_id"] for row in store.list_jobs(status=RunStatus.RUNNING.value)] == [job_id]

    assert store.delete_job(job_id) is True
    assert store.get_job(job_id) is None
    assert store.update_job_meta(job_id, {"k": "v"}) is None


def test_list_jobs_accepts_status_list(store):
    submitted = str(uuid.uuid4())
    running = str(uuid.uuid4())
    finished = str(uuid.uuid4())
    store.create_job(job_meta(submitted, status=RunStatus.SUBMITTED.value), content_uri="u1")
    store.create_job(job_meta(running, status=RunStatus.RUNNING.value), content_uri="u2")
    store.create_job(job_meta(finished, status=RunStatus.FINISHED_COMPLETED.value), content_uri="u3")

    rows = store.list_jobs(status=[RunStatus.SUBMITTED.value, RunStatus.RUNNING])
    assert {row["job_id"] for row in rows} == {submitted, running}
    assert [row["job_id"] for row in store.list_jobs(status=RunStatus.RUNNING)] == [running]
    assert len(store.list_jobs(study="study_a")) == 3


def test_concurrent_update_job_meta_keeps_both_keys(store):
    job_id = str(uuid.uuid4())
    store.create_job(job_meta(job_id), content_uri="artifact://jobs/hello.zip")

    barrier = threading.Barrier(2)
    errors = []

    def update_key(key):
        try:
            barrier.wait(timeout=10)
            store.update_job_meta(job_id, {key: key})
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=update_key, args=(key,)) for key in ("left", "right")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    meta = store.get_job(job_id)["meta_json"]
    assert meta["left"] == "left"
    assert meta["right"] == "right"
    assert store.get_job(job_id)["version"] == 3


def test_update_job_meta_retries_on_version_conflict(store, monkeypatch):
    job_id = str(uuid.uuid4())
    store.create_job(job_meta(job_id), content_uri="artifact://jobs/hello.zip")

    # Simulate a stale read (a competing writer committed after our read): the first read
    # reports an old version, forcing the optimistic UPDATE's rowcount to 0 and exercising
    # the retry loop, which re-reads and succeeds.
    original_read = SqlStateStore._read_job_for_update
    state = {"stale_reads": 0}

    def stale_read(self, conn, jid):
        row = original_read(self, conn, jid)
        if row and state["stale_reads"] == 0:
            state["stale_reads"] += 1
            row = dict(row, version=row["version"] - 1, meta_json={"stale": True})
        return row

    monkeypatch.setattr(SqlStateStore, "_read_job_for_update", stale_read)

    updated = store.update_job_meta(job_id, {"k": "v"})

    assert state["stale_reads"] == 1
    assert updated["meta_json"]["k"] == "v"
    assert "stale" not in updated["meta_json"]  # merged from the fresh re-read, not the stale one
    assert updated["version"] == 2


def test_submit_records_are_scoped_and_do_not_require_existing_job(store):
    job_id = str(uuid.uuid4())
    record = submit_record(job_id)

    assert store.create_submit_record(record) is True
    assert store.create_submit_record(record) is False

    same_token_different_submitter = submit_record(str(uuid.uuid4()), token="token-1", submitter_name="o@nvidia.com")
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


def test_update_submit_record_inserts_when_missing(store):
    job_id = str(uuid.uuid4())
    record = submit_record(job_id, token="upsert-token")

    updated = store.update_submit_record(record)
    assert updated[SubmitRecordKey.JOB_ID.value] == job_id
    fetched = store.get_submit_record(
        "study_a", {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}, "upsert-token"
    )
    assert fetched[SubmitRecordKey.JOB_ID.value] == job_id


def test_update_submit_record_converges_when_insert_loses_race(store, monkeypatch):
    job_id = str(uuid.uuid4())
    record = submit_record(job_id, token="race-token")
    original_insert_ignore = sql_store._insert_ignore
    state = {"raced": False}

    def racing_insert_ignore(conn, table, values, **kwargs):
        if table is sql_store.submit_records and not state["raced"]:
            state["raced"] = True
            original_insert_ignore(conn, table, dict(values))  # concurrent writer wins
        return original_insert_ignore(conn, table, values, **kwargs)

    monkeypatch.setattr(sql_store, "_insert_ignore", racing_insert_ignore)

    updated = store.update_submit_record(record)
    assert updated[SubmitRecordKey.JOB_ID.value] == job_id
    assert state["raced"] is True


def test_studies_and_disabled_clients(store):
    study = store.upsert_study(
        "study_a",
        {"admins": ["admin@nvidia.com"], "site_orgs": {}},
    )
    assert study["name"] == "study_a"
    assert store.get_study("study_a")["config_json"]["admins"] == ["admin@nvidia.com"]

    # upsert_study is additive: existing members are never deleted by a (possibly stale) snapshot
    merged = store.upsert_study("study_a", {"admins": ["lead@nvidia.com"], "site_orgs": {}})
    assert merged["config_json"]["admins"] == ["admin@nvidia.com", "lead@nvidia.com"]

    assert store.delete_study("study_a") is True
    assert store.get_study("study_a") is None

    disabled = store.disable_client("site-1", disabled_by="admin@nvidia.com", reason="maintenance")
    assert disabled["client_name"] == "site-1"
    assert store.get_disabled_client("site-1")["reason"] == "maintenance"
    assert store.enable_client("site-1") is True
    assert store.get_disabled_client("site-1") is None


def test_disable_client_converges_on_integrity_error(store, monkeypatch):
    original_insert_ignore = sql_store._insert_ignore
    state = {"raced": False}

    def racing_insert_ignore(conn, table, values, **kwargs):
        if table is sql_store.disabled_clients and not state["raced"]:
            state["raced"] = True
            original_insert_ignore(conn, table, dict(values, disabled_by="racer"))  # concurrent writer wins
        return original_insert_ignore(conn, table, values, **kwargs)

    monkeypatch.setattr(sql_store, "_insert_ignore", racing_insert_ignore)

    disabled = store.disable_client("site-1", disabled_by="admin@nvidia.com", reason="maintenance")

    assert state["raced"] is True
    assert disabled["client_name"] == "site-1"
    assert disabled["disabled_by"] == "admin@nvidia.com"  # converged to an UPDATE
    assert disabled["version"] == 2


def test_disable_client_is_idempotent_and_updates(store):
    first = store.disable_client("site-1", disabled_by="a@nvidia.com", reason="one")
    assert first["version"] == 1
    second = store.disable_client("site-1", disabled_by="b@nvidia.com", reason="two")
    assert second["version"] == 2
    assert second["reason"] == "two"


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
        "site_orgs": {"org_a": [], "org_b": ["site-b"]},
    }

    assert store.delete_study("study_a") is True
    assert store.get_study("study_a") is None


def test_delete_study_if_no_jobs_blocks_while_jobs_exist(store):
    store.upsert_study("study_a", {"admins": ["admin@nvidia.com"], "site_orgs": {"org_a": ["site-a"]}})
    job_id = str(uuid.uuid4())
    store.create_job(job_meta(job_id, study="study_a"), content_uri="artifact://jobs/hello.zip")

    assert store.delete_study_if_no_jobs("study_a") == {"deleted": False, "job_count": 1}
    assert store.get_study("study_a") is not None

    # jobs in other studies do not block; only jobs whose study matches count
    store.upsert_study("study_b", {"admins": [], "site_orgs": {}})
    assert store.delete_study_if_no_jobs("study_b") == {"deleted": True}

    # once the blocking job is gone, the study can be deleted
    assert store.delete_job(job_id) is True
    assert store.delete_study_if_no_jobs("study_a") == {"deleted": True}
    assert store.get_study("study_a") is None


def test_delete_study_if_no_jobs_removes_member_rows(store):
    store.upsert_study("study_a", {"admins": ["admin@nvidia.com"], "site_orgs": {"org_a": ["site-a"], "org_b": []}})

    assert store.delete_study_if_no_jobs("study_a") == {"deleted": True}

    with store.engine.begin() as conn:
        for table in (sql_store.study_admins, sql_store.study_orgs, sql_store.study_sites):
            assert conn.execute(sql_store.select(table)).fetchall() == []


def test_delete_study_if_no_jobs_returns_not_found_for_missing_study(store):
    assert store.delete_study_if_no_jobs("ghost") == {"deleted": False, "not_found": True}


def test_org_with_zero_sites_stays_enrolled(store):
    store.upsert_study("study_a", {"admins": [], "site_orgs": {"org_a": ["site-a"]}})

    # removing the org's last site keeps the org enrolled with an empty site list
    store.remove_study_sites("study_a", {"org_a": ["site-a"]})
    assert store.get_study("study_a")["config_json"]["site_orgs"] == {"org_a": []}

    # the org can re-add sites without re-enrolling
    store.add_study_sites("study_a", {"org_a": ["site-a2"]})
    assert store.get_study("study_a")["config_json"]["site_orgs"] == {"org_a": ["site-a2"]}

    # an org can be enrolled with zero sites up front (round trip through upsert)
    store.upsert_study("study_b", {"admins": [], "site_orgs": {"org_empty": []}})
    assert store.get_study("study_b")["config_json"]["site_orgs"] == {"org_empty": []}
    assert [s["config_json"]["site_orgs"] for s in store.list_studies() if s["name"] == "study_b"] == [
        {"org_empty": []}
    ]


def test_list_studies_is_batched_and_grouped(store):
    store.upsert_study("study_a", {"admins": ["a@nvidia.com"], "site_orgs": {"org_a": ["site-a"], "org_b": []}})
    store.upsert_study("study_b", {"admins": [], "site_orgs": {}})

    listed = {row["name"]: row["config_json"] for row in store.list_studies()}
    assert listed == {
        "study_a": {"admins": ["a@nvidia.com"], "site_orgs": {"org_a": ["site-a"], "org_b": []}},
        "study_b": {"admins": [], "site_orgs": {}},
    }


def test_long_write_transaction_does_not_block_reads(store):
    # Regression for BEGIN IMMEDIATE firing on every transaction: a long-held WRITE
    # transaction must not make a concurrent pure read raise "database is locked".
    job_id = str(uuid.uuid4())
    store.create_job(job_meta(job_id), content_uri="artifact://jobs/hello.zip")

    write_started = threading.Event()
    release_writer = threading.Event()
    errors = []

    def long_writer():
        try:
            with store._begin_write() as conn:
                conn.execute(
                    sql_store.update(sql_store.jobs)
                    .where(sql_store.jobs.c.job_id == job_id)
                    .values(status=RunStatus.RUNNING.value)
                )
                write_started.set()
                assert release_writer.wait(timeout=10)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)
            write_started.set()

    writer = threading.Thread(target=long_writer)
    writer.start()
    try:
        assert write_started.wait(timeout=10)
        # Pure reads proceed while the write lock is held.
        assert store.get_job(job_id)["job_id"] == job_id
        assert [row["job_id"] for row in store.list_jobs()] == [job_id]
        assert store.get_disabled_client("nope") is None
    finally:
        release_writer.set()
        writer.join(timeout=10)
    assert not errors
    assert store.get_job(job_id)["status"] == RunStatus.RUNNING.value  # writer committed


def test_add_study_sites_to_missing_study_raises(store):
    with pytest.raises(IntegrityError):
        store.add_study_sites("ghost", {"org_a": ["site-a"]})

    with pytest.raises(IntegrityError):
        store.add_study_admin("ghost", "alice@nvidia.com")

    # nothing was silently written
    with store.engine.begin() as conn:
        assert conn.execute(sql_store.select(sql_store.study_orgs)).fetchall() == []
        assert conn.execute(sql_store.select(sql_store.study_sites)).fetchall() == []
        assert conn.execute(sql_store.select(sql_store.study_admins)).fetchall() == []


def test_insert_ignore_still_converges_on_genuine_duplicates(store):
    store.upsert_study("study_a", {"admins": ["admin@nvidia.com"], "site_orgs": {"org_a": ["site-a"]}})

    # duplicate adds converge instead of raising
    result = store.add_study_sites("study_a", {"org_a": ["site-a"]})
    assert result["config_json"]["site_orgs"] == {"org_a": ["site-a"]}
    result = store.add_study_admin("study_a", "admin@nvidia.com")
    assert result["config_json"]["admins"] == ["admin@nvidia.com"]


def test_update_submit_record_raises_on_retry_exhaustion(store, monkeypatch):
    # Force exhaustion: the scoped row never exists (UPDATE rowcount 0) and every INSERT
    # "loses the race" (returns False). The loop must raise, not silently return success.
    monkeypatch.setattr(sql_store, "_insert_ignore", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="update_submit_record"):
        store.update_submit_record(submit_record(str(uuid.uuid4()), token="exhausted-token"))


def test_migration_markers(store):
    assert store.get_migration_marker("m1") is None
    marker = store.set_migration_marker("m1", "fresh-install", {"status": "complete"})
    assert marker["source_format"] == "fresh-install"
    assert marker["summary_json"] == {"status": "complete"}

    # converges on an existing marker instead of failing
    again = store.set_migration_marker("m1", "other", {"status": "other"})
    assert again["source_format"] == "fresh-install"
