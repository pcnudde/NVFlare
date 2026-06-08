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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    event,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError

from nvflare.apis.job_def import DEFAULT_STUDY, JobMetaKey, SubmitRecordKey, SubmitRecordState
from nvflare.apis.state_store import StateStore
from nvflare.apis.utils.job_submit_token import submit_record_scope_hashes, submitter_to_dict

metadata = MetaData()

studies = Table(
    "studies",
    metadata,
    Column("name", String(255), primary_key=True),
)

study_admins = Table(
    "study_admins",
    metadata,
    Column("study_name", String(255), ForeignKey("studies.name", ondelete="CASCADE"), primary_key=True),
    Column("user_name", String(255), primary_key=True),
    Index("idx_study_admins_user", "user_name"),
)

study_sites = Table(
    "study_sites",
    metadata,
    Column("study_name", String(255), ForeignKey("studies.name", ondelete="CASCADE"), primary_key=True),
    Column("site_name", String(255), primary_key=True),
    Column("org", String(255), nullable=False),
    Index("idx_study_sites_org", "org"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("job_id", String(64), primary_key=True),
    Column("study", String(255), nullable=False, default=DEFAULT_STUDY),
    Column("status", String(64), nullable=False),
    Column("job_name", String(255)),
    Column("job_folder_name", String(255)),
    Column("submitter_name", String(255)),
    Column("submitter_org", String(255)),
    Column("submitter_role", String(255)),
    Column("content_uri", Text, nullable=False),
    Column("content_hash", String(255)),
    Column("content_size", BigInteger),
    Column("result_uri", Text),
    Column("submit_time", Float),
    Column("submit_time_iso", String(255)),
    Column("start_time", String(255)),
    Column("duration", String(255)),
    Column("schedule_count", Integer, nullable=False, default=0),
    Column("last_schedule_time", Float),
    Column("schedule_history", JSON),
    Column("meta_json", JSON, nullable=False),
    Column("version", Integer, nullable=False, default=1),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("idx_jobs_status", "status"),
    Index("idx_jobs_study_status", "study", "status"),
    Index("idx_jobs_submitter", "submitter_name", "submitter_org", "submitter_role"),
)

submit_records = Table(
    "submit_records",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("study", String(255), nullable=False),
    Column("study_hash", String(64), nullable=False),
    Column("submitter_hash", String(64), nullable=False),
    Column("submit_token_hash", String(64), nullable=False),
    Column("submitter_name", String(255)),
    Column("submitter_org", String(255)),
    Column("submitter_role", String(255)),
    Column("job_id", String(64), nullable=False),
    Column("job_content_hash", String(255)),
    Column("job_name", String(255)),
    Column("job_folder_name", String(255)),
    Column("state", String(64), nullable=False),
    Column("submit_time", String(255)),
    Column("deleted_time", String(255)),
    Column("deleted_by_json", JSON),
    Column("record_json", JSON, nullable=False),
    Column("version", Integer, nullable=False, default=1),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("study_hash", "submitter_hash", "submit_token_hash", name="uq_submit_record_scope"),
    Index("idx_submit_records_job_id", "job_id"),
)

disabled_clients = Table(
    "disabled_clients",
    metadata,
    Column("client_name", String(255), primary_key=True),
    Column("disabled_by", String(255)),
    Column("disabled_at", DateTime(timezone=True), nullable=False),
    Column("reason", Text),
    Column("version", Integer, nullable=False, default=1),
)

state_store_migrations = Table(
    "state_store_migrations",
    metadata,
    Column("name", String(128), primary_key=True),
    Column("source_format", String(64), nullable=False),
    Column("applied_at", DateTime(timezone=True), nullable=False),
    Column("nvflare_version", String(64)),
    Column("summary_json", JSON, nullable=False),
)


def sqlite_url(db_path: str) -> str:
    """Return a SQLAlchemy SQLite URL for a filesystem DB path."""
    if db_path == ":memory:":
        raise ValueError("Alembic-managed SQLite state DBs must be filesystem-backed")
    return f"sqlite:///{Path(db_path).expanduser().resolve()}"


def resolve_db_url(db_url: str = None, db_url_env: str = None) -> str:
    if db_url_env:
        resolved = os.environ.get(db_url_env)
        if not resolved:
            raise ValueError(f"environment variable '{db_url_env}' must be set for state store db_url")
        return resolved
    if not db_url:
        raise ValueError("state store requires db_url or db_url_env")
    return db_url


def migrate_database(db_url: str, revision: str = "head"):
    """Apply Alembic migrations to the configured state-store database."""
    _prepare_sqlite_parent(db_url)
    command.upgrade(_alembic_config(db_url), revision)


def validate_database(db_url: str):
    """Fail if the configured state-store database is not at the Alembic head revision."""
    config = _alembic_config(db_url)
    expected = ScriptDirectory.from_config(config).get_current_head()
    engine = create_engine(db_url, future=True)
    try:
        with engine.connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()
    if current != expected:
        raise RuntimeError(f"state store schema is at revision {current}; expected {expected}")


def _alembic_config(db_url: str) -> Config:
    migrations_dir = Path(__file__).with_name("migrations")
    config = Config()
    config.set_main_option("script_location", str(migrations_dir))
    config.set_main_option("sqlalchemy.url", db_url)
    return config


def _prepare_sqlite_parent(db_url: str):
    url = make_url(db_url)
    if not url.drivername.startswith("sqlite"):
        return
    if url.database == ":memory:":
        raise ValueError("Alembic-managed SQLite state DBs must be filesystem-backed")
    if not url.database:
        return
    Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _now():
    return datetime.now(timezone.utc)


def _json_safe(value):
    if value is None:
        return {}
    return json.loads(json.dumps(value))


def _get(mapping: dict, key, default=None):
    return mapping.get(key.value, mapping.get(key, default))


def _status_value(status):
    return getattr(status, "value", status)


def _row_dict(row) -> Optional[dict]:
    return dict(row._mapping) if row else None


def _study_config_from_rows(admin_rows, site_rows) -> dict:
    site_orgs = {}
    for row in site_rows:
        row = row._mapping
        site_orgs.setdefault(row["org"], []).append(row["site_name"])
    return {
        "admins": [row._mapping["user_name"] for row in admin_rows],
        "site_orgs": site_orgs,
    }


def _study_config_from_conn(conn, name: str) -> dict:
    admin_rows = conn.execute(
        select(study_admins.c.user_name).where(study_admins.c.study_name == name).order_by(study_admins.c.user_name)
    ).fetchall()
    site_rows = conn.execute(
        select(study_sites.c.org, study_sites.c.site_name)
        .where(study_sites.c.study_name == name)
        .order_by(study_sites.c.org, study_sites.c.site_name)
    ).fetchall()
    return _study_config_from_rows(admin_rows, site_rows)


def _study_member_values(name: str, config: dict):
    config = _json_safe(config)
    admin_values = [{"study_name": name, "user_name": admin} for admin in config.get("admins", []) or []]
    site_values = []
    for org, sites_for_org in (config.get("site_orgs", {}) or {}).items():
        for site in sites_for_org:
            site_values.append({"study_name": name, "org": org, "site_name": site})
    return admin_values, site_values


def _job_columns_from_meta(meta: dict) -> dict:
    status = _status_value(_get(meta, JobMetaKey.STATUS))
    return {
        "job_id": _get(meta, JobMetaKey.JOB_ID),
        "study": _get(meta, JobMetaKey.STUDY, DEFAULT_STUDY) or DEFAULT_STUDY,
        "status": status,
        "job_name": _get(meta, JobMetaKey.JOB_NAME),
        "job_folder_name": _get(meta, JobMetaKey.JOB_FOLDER_NAME),
        "submitter_name": _get(meta, JobMetaKey.SUBMITTER_NAME),
        "submitter_org": _get(meta, JobMetaKey.SUBMITTER_ORG),
        "submitter_role": _get(meta, JobMetaKey.SUBMITTER_ROLE),
        "result_uri": _get(meta, JobMetaKey.RESULT_LOCATION),
        "submit_time": _get(meta, JobMetaKey.SUBMIT_TIME),
        "submit_time_iso": _get(meta, JobMetaKey.SUBMIT_TIME_ISO),
        "start_time": _get(meta, JobMetaKey.START_TIME),
        "duration": _get(meta, JobMetaKey.DURATION),
        "schedule_count": _get(meta, JobMetaKey.SCHEDULE_COUNT, 0) or 0,
        "last_schedule_time": _get(meta, JobMetaKey.LAST_SCHEDULE_TIME),
        "schedule_history": _json_safe(_get(meta, JobMetaKey.SCHEDULE_HISTORY, [])),
    }


def _submitter_from_record(record: dict) -> dict:
    return {
        "name": record.get(SubmitRecordKey.SUBMITTER_NAME.value, ""),
        "org": record.get(SubmitRecordKey.SUBMITTER_ORG.value, ""),
        "role": record.get(SubmitRecordKey.SUBMITTER_ROLE.value, ""),
    }


def _submit_record_values(record: dict) -> dict:
    study = record.get(SubmitRecordKey.STUDY.value, DEFAULT_STUDY) or DEFAULT_STUDY
    submitter = _submitter_from_record(record)
    study_hash, submitter_hash, submit_token_hash = submit_record_scope_hashes(
        study, submitter, record.get(SubmitRecordKey.SUBMIT_TOKEN.value, "")
    )
    deleted_by = record.get(SubmitRecordKey.DELETED_BY.value)
    return {
        "study": study,
        "study_hash": study_hash,
        "submitter_hash": submitter_hash,
        "submit_token_hash": submit_token_hash,
        "submitter_name": submitter["name"],
        "submitter_org": submitter["org"],
        "submitter_role": submitter["role"],
        "job_id": record.get(SubmitRecordKey.JOB_ID.value),
        "job_content_hash": record.get(SubmitRecordKey.JOB_CONTENT_HASH.value),
        "job_name": record.get(SubmitRecordKey.JOB_NAME.value),
        "job_folder_name": record.get(SubmitRecordKey.JOB_FOLDER_NAME.value),
        "state": record.get(SubmitRecordKey.STATE.value, SubmitRecordState.CREATING.value),
        "submit_time": record.get(SubmitRecordKey.SUBMIT_TIME.value),
        "deleted_time": record.get(SubmitRecordKey.DELETED_TIME.value),
        "deleted_by_json": _json_safe(deleted_by) if deleted_by else None,
        "record_json": _json_safe(record),
    }


class SqlStateStore(StateStore):
    """SQLAlchemy/Alembic-backed implementation of the minimal state store."""

    def __init__(self, db_url: str = None, db_url_env: str = None, engine: Engine = None):
        self.db_url = resolve_db_url(db_url=db_url, db_url_env=db_url_env)
        self.db_url_env = db_url_env
        self.engine = engine or create_engine(self.db_url, future=True)
        self._configure_engine()

    def _configure_engine(self):
        if self.engine.dialect.name != "sqlite":
            return

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    @classmethod
    def sqlite(cls, db_path: str):
        return cls(sqlite_url(db_path))

    def initialize(self):
        validate_database(self.db_url)

    def upsert_study(self, name: str, config: dict) -> dict:
        with self.engine.begin() as conn:
            row = conn.execute(select(studies.c.name).where(studies.c.name == name)).first()
            if not row:
                conn.execute(insert(studies).values(name=name))
            conn.execute(delete(study_admins).where(study_admins.c.study_name == name))
            conn.execute(delete(study_sites).where(study_sites.c.study_name == name))
            admin_values, site_values = _study_member_values(name, config)
            if admin_values:
                conn.execute(insert(study_admins), admin_values)
            if site_values:
                conn.execute(insert(study_sites), site_values)
        return self.get_study(name)

    def get_study(self, name: str) -> Optional[dict]:
        with self.engine.begin() as conn:
            row = conn.execute(select(studies).where(studies.c.name == name)).first()
            if not row:
                return None
            result = _row_dict(row)
            result["config_json"] = _study_config_from_conn(conn, name)
            return result

    def list_studies(self) -> List[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(studies).order_by(studies.c.name)).fetchall()
            result = []
            for row in rows:
                study = _row_dict(row)
                study["config_json"] = _study_config_from_conn(conn, study["name"])
                result.append(study)
            return result

    def delete_study(self, name: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(delete(studies).where(studies.c.name == name))
        return result.rowcount > 0

    def add_study_sites(self, name: str, site_orgs: Dict[str, List[str]]) -> dict:
        with self.engine.begin() as conn:
            for org, sites_for_org in (site_orgs or {}).items():
                for site in sites_for_org:
                    row = conn.execute(
                        select(study_sites.c.site_name).where(
                            and_(study_sites.c.study_name == name, study_sites.c.site_name == site)
                        )
                    ).first()
                    if not row:
                        conn.execute(
                            insert(study_sites).values(
                                study_name=name,
                                org=org,
                                site_name=site,
                            )
                        )
        return self.get_study(name)

    def remove_study_sites(self, name: str, site_orgs: Dict[str, List[str]]) -> dict:
        with self.engine.begin() as conn:
            for org, sites_for_org in (site_orgs or {}).items():
                for site in sites_for_org:
                    conn.execute(
                        delete(study_sites).where(
                            and_(
                                study_sites.c.study_name == name,
                                study_sites.c.org == org,
                                study_sites.c.site_name == site,
                            )
                        )
                    )
        return self.get_study(name)

    def add_study_admin(self, name: str, user: str) -> dict:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(study_admins.c.user_name).where(
                    and_(study_admins.c.study_name == name, study_admins.c.user_name == user)
                )
            ).first()
            if not row:
                conn.execute(insert(study_admins).values(study_name=name, user_name=user))
        return self.get_study(name)

    def remove_study_admin(self, name: str, user: str) -> dict:
        with self.engine.begin() as conn:
            conn.execute(
                delete(study_admins).where(and_(study_admins.c.study_name == name, study_admins.c.user_name == user))
            )
        return self.get_study(name)

    def create_job(self, meta: dict, content_uri: str, content_hash: str = None, content_size: int = None) -> dict:
        meta_json = _json_safe(meta)
        values = _job_columns_from_meta(meta_json)
        job_id = values.get("job_id")
        if not job_id:
            raise ValueError("job metadata must contain job_id")
        if not values.get("status"):
            raise ValueError("job metadata must contain status")
        now = _now()
        values.update(
            {
                "content_uri": content_uri,
                "content_hash": content_hash,
                "content_size": content_size,
                "meta_json": meta_json,
                "version": 1,
                "created_at": now,
                "updated_at": now,
            }
        )
        with self.engine.begin() as conn:
            conn.execute(insert(jobs).values(**values))
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[dict]:
        with self.engine.begin() as conn:
            row = conn.execute(select(jobs).where(jobs.c.job_id == job_id)).first()
        return _row_dict(row)

    def delete_job(self, job_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(delete(jobs).where(jobs.c.job_id == job_id))
        return result.rowcount > 0

    def list_jobs(self, status: str = None, study: str = None) -> List[dict]:
        conditions = []
        if status is not None:
            conditions.append(jobs.c.status == _status_value(status))
        if study is not None:
            conditions.append(jobs.c.study == study)
        stmt = select(jobs).order_by(jobs.c.created_at)
        if conditions:
            stmt = stmt.where(and_(*conditions))
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_dict(row) for row in rows]

    def update_job_meta(self, job_id: str, meta: dict) -> dict:
        with self.engine.begin() as conn:
            row = conn.execute(select(jobs.c.meta_json).where(jobs.c.job_id == job_id).with_for_update()).first()
            if not row:
                return None
            merged_meta = dict(row._mapping["meta_json"] or {})
            merged_meta.update(_json_safe(meta))
            values = _job_columns_from_meta(merged_meta)
            values.pop("job_id", None)
            values["meta_json"] = merged_meta
            values["version"] = jobs.c.version + 1
            values["updated_at"] = _now()
            conn.execute(update(jobs).where(jobs.c.job_id == job_id).values(**values))
        return self.get_job(job_id)

    def set_job_status(self, job_id: str, status: str) -> dict:
        return self.update_job_meta(job_id, {JobMetaKey.STATUS.value: _status_value(status)})

    def create_submit_record(self, record: dict) -> bool:
        values = _submit_record_values(record)
        if not values.get("job_id"):
            raise ValueError("submit record must contain job_id")
        now = _now()
        values.update({"version": 1, "created_at": now, "updated_at": now})
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(submit_records).values(**values))
        except IntegrityError:
            return False
        return True

    def get_submit_record(self, study: str, submitter: Any, submit_token: str) -> Optional[dict]:
        study_hash, submitter_hash, submit_token_hash = submit_record_scope_hashes(study, submitter, submit_token)
        with self.engine.begin() as conn:
            row = conn.execute(
                select(submit_records.c.record_json).where(
                    and_(
                        submit_records.c.study_hash == study_hash,
                        submit_records.c.submitter_hash == submitter_hash,
                        submit_records.c.submit_token_hash == submit_token_hash,
                    )
                )
            ).first()
        return row._mapping["record_json"] if row else None

    def update_submit_record(self, record: dict) -> dict:
        values = _submit_record_values(record)
        now = _now()
        with self.engine.begin() as conn:
            result = conn.execute(
                update(submit_records)
                .where(
                    and_(
                        submit_records.c.study_hash == values["study_hash"],
                        submit_records.c.submitter_hash == values["submitter_hash"],
                        submit_records.c.submit_token_hash == values["submit_token_hash"],
                    )
                )
                .values(**values, version=submit_records.c.version + 1, updated_at=now)
            )
            if result.rowcount == 0:
                values.update({"version": 1, "created_at": now, "updated_at": now})
                conn.execute(insert(submit_records).values(**values))
        return _json_safe(record)

    def mark_submit_records_job_deleted(self, job_id: str, deleted_by: Any) -> List[dict]:
        deleted_by_info = submitter_to_dict(deleted_by)
        deleted_time = _now().isoformat()
        updated = []
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(submit_records.c.id, submit_records.c.record_json).where(submit_records.c.job_id == job_id)
            ).fetchall()
            for row in rows:
                record = dict(row._mapping["record_json"] or {})
                if record.get(SubmitRecordKey.STATE.value) == SubmitRecordState.JOB_DELETED.value:
                    continue
                record[SubmitRecordKey.STATE.value] = SubmitRecordState.JOB_DELETED.value
                record[SubmitRecordKey.DELETED_TIME.value] = deleted_time
                record[SubmitRecordKey.DELETED_BY.value] = deleted_by_info
                values = _submit_record_values(record)
                conn.execute(
                    update(submit_records)
                    .where(submit_records.c.id == row._mapping["id"])
                    .values(**values, version=submit_records.c.version + 1, updated_at=_now())
                )
                updated.append(record)
        return updated

    def disable_client(self, client_name: str, disabled_by: str = None, reason: str = None) -> dict:
        now = _now()
        with self.engine.begin() as conn:
            row = conn.execute(
                select(disabled_clients.c.client_name).where(disabled_clients.c.client_name == client_name)
            ).first()
            if row:
                conn.execute(
                    update(disabled_clients)
                    .where(disabled_clients.c.client_name == client_name)
                    .values(
                        disabled_by=disabled_by,
                        disabled_at=now,
                        reason=reason,
                        version=disabled_clients.c.version + 1,
                    )
                )
            else:
                conn.execute(
                    insert(disabled_clients).values(
                        client_name=client_name,
                        disabled_by=disabled_by,
                        disabled_at=now,
                        reason=reason,
                        version=1,
                    )
                )
        return self.get_disabled_client(client_name)

    def get_disabled_client(self, client_name: str) -> Optional[dict]:
        with self.engine.begin() as conn:
            row = conn.execute(select(disabled_clients).where(disabled_clients.c.client_name == client_name)).first()
        return _row_dict(row)

    def enable_client(self, client_name: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(delete(disabled_clients).where(disabled_clients.c.client_name == client_name))
        return result.rowcount > 0

    def list_disabled_clients(self) -> List[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(disabled_clients).order_by(disabled_clients.c.client_name)).fetchall()
        return [_row_dict(row) for row in rows]

    def get_migration_marker(self, name: str) -> Optional[dict]:
        with self.engine.begin() as conn:
            row = conn.execute(select(state_store_migrations).where(state_store_migrations.c.name == name)).first()
        return _row_dict(row)
