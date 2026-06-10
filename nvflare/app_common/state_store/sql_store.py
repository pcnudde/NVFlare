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

import copy
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
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
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import Select

import nvflare
from nvflare.apis.job_def import DEFAULT_STUDY, JobMetaKey, SubmitRecordKey, SubmitRecordState, run_status_value
from nvflare.apis.state_store import StateStore
from nvflare.apis.utils.job_submit_token import submit_record_scope_hashes, submitter_to_dict

# Bounded retries for optimistic-concurrency and check-then-insert convergence loops.
_MAX_WRITE_ATTEMPTS = 10
_RETRY_BACKOFF_SECS = 0.01

# SQLite busy timeout (seconds): how long a connection waits for the single write lock
# before raising SQLITE_BUSY. Explicit so we do not depend on pysqlite's 5s default.
_SQLITE_BUSY_TIMEOUT_SECS = 15

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

# Org enrollment is tracked separately from sites so an org with zero sites stays enrolled.
study_orgs = Table(
    "study_orgs",
    metadata,
    Column("study_name", String(255), ForeignKey("studies.name", ondelete="CASCADE"), primary_key=True),
    Column("org", String(255), primary_key=True),
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
    Column("content_uri", Text, nullable=False),
    Column("content_hash", String(255)),
    Column("content_size", BigInteger),
    Column("meta_json", JSON, nullable=False),
    Column("version", Integer, nullable=False, default=1),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("idx_jobs_status", "status"),
    Index("idx_jobs_study_status", "study", "status"),
)

submit_records = Table(
    "submit_records",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("study_hash", String(64), nullable=False),
    Column("submitter_hash", String(64), nullable=False),
    Column("submit_token_hash", String(64), nullable=False),
    Column("job_id", String(64), nullable=False),
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


def default_state_store_db_url(server_root: str) -> str:
    """Default state-store db_url for a server workspace without a configured state_store component.

    Shared by the server deployer and the migrate CLI so both fall back to the same SQLite
    file under the server root when the component (or its db_url) is absent.
    """
    return sqlite_url(os.path.join(server_root, "state-store.db"))


def resolve_db_url(db_url: str = None, db_url_env: str = None) -> str:
    if db_url_env:
        resolved = os.environ.get(db_url_env)
        if not resolved:
            raise ValueError(f"environment variable '{db_url_env}' must be set for state store db_url")
        return resolved
    if not db_url:
        raise ValueError("state store requires db_url or db_url_env")
    return db_url


def resolve_relative_db_url(db_url: str, base_dir: str) -> str:
    """Resolve a relative SQLite db_url path against base_dir.

    Non-SQLite URLs and absolute SQLite paths are returned unchanged. This keeps the
    migrate CLI (--server-root) and the server (workspace dir) pointing at the same file
    regardless of the process CWD.
    """
    url = make_url(db_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return db_url
    db_path = Path(url.database).expanduser()
    if db_path.is_absolute():
        return db_url
    resolved = (Path(base_dir).expanduser().resolve() / db_path).resolve()
    return url.set(database=str(resolved)).render_as_string(hide_password=False)


def migrate_database(db_url: str, revision: str = "head"):
    """Apply Alembic migrations to the configured state-store database."""
    ensure_sqlite_parent_dir(db_url)
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
    # Config is ConfigParser-backed: '%' is interpolation syntax and must be escaped.
    config.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))
    return config


def ensure_sqlite_parent_dir(db_url: str):
    """Create the parent directory of a file-backed SQLite db_url if absent.

    SQLite can create a missing DB file but not missing parent directories, and schema
    validation connects before the bootstrap/migration path gets a chance to create them.
    Non-SQLite URLs are ignored; :memory: URLs are rejected (state DBs must be file-backed).
    """
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
    return copy.deepcopy(value)


def _get(mapping: dict, key, default=None):
    return mapping.get(key.value, mapping.get(key, default))


def _row_dict(row) -> Optional[dict]:
    return dict(row._mapping) if row else None


def _insert_ignore(conn: Connection, table: Table, values: dict, verify: Optional[Select] = None) -> bool:
    """Insert a row, converging on already-present: returns False if the row already exists.

    Uses a SAVEPOINT so a unique-constraint IntegrityError does not poison the enclosing
    transaction (required for PostgreSQL; portable to SQLite).

    Convergence is VERIFIED: on IntegrityError the intended row is re-selected (by the
    table's primary key, or via the caller-supplied `verify` SELECT when the primary key
    is not part of `values`). If the row is absent, the IntegrityError was a genuine
    constraint violation (foreign key, NOT NULL, CHECK) — not a duplicate insert — and is
    re-raised instead of being silently swallowed.
    """
    try:
        with conn.begin_nested():
            conn.execute(insert(table).values(**values))
        return True
    except IntegrityError:
        if verify is None:
            verify = select(table).where(and_(*(col == values[col.name] for col in table.primary_key.columns)))
        if conn.execute(verify.limit(1)).first() is None:
            raise
        return False


def _site_orgs_from_rows(rows) -> dict:
    """Build {org: [site, ...]} from (org, site_name) pairs; site_name None marks an org-only row."""
    site_orgs = {}
    for org, site_name in rows:
        sites = site_orgs.setdefault(org, [])
        if site_name is not None:
            sites.append(site_name)
    return site_orgs


def _job_status_and_study(meta: dict):
    status = run_status_value(_get(meta, JobMetaKey.STATUS))
    study = _get(meta, JobMetaKey.STUDY, DEFAULT_STUDY) or DEFAULT_STUDY
    return status, study


def _submit_record_values(record: dict) -> dict:
    study = record.get(SubmitRecordKey.STUDY.value, DEFAULT_STUDY) or DEFAULT_STUDY
    study_hash, submitter_hash, submit_token_hash = submit_record_scope_hashes(
        study, submitter_to_dict(record), record.get(SubmitRecordKey.SUBMIT_TOKEN.value, "")
    )
    return {
        "study_hash": study_hash,
        "submitter_hash": submitter_hash,
        "submit_token_hash": submit_token_hash,
        "job_id": record.get(SubmitRecordKey.JOB_ID.value),
        "record_json": _json_safe(record),
    }


def _submit_record_scope_condition(values: dict):
    return and_(
        submit_records.c.study_hash == values["study_hash"],
        submit_records.c.submitter_hash == values["submitter_hash"],
        submit_records.c.submit_token_hash == values["submit_token_hash"],
    )


def _submit_record_scope_select(values: dict) -> Select:
    return select(submit_records.c.id).where(_submit_record_scope_condition(values))


class SqlStateStore(StateStore):
    """SQLAlchemy/Alembic-backed implementation of the minimal state store."""

    def __init__(self, db_url: str = None, db_url_env: str = None, engine: Engine = None):
        self.db_url = resolve_db_url(db_url=db_url, db_url_env=db_url_env)
        self.db_url_env = db_url_env
        self.engine = engine or self._create_engine(self.db_url)
        self._validated = False
        self._configure_engine()

    @staticmethod
    def _create_engine(db_url: str) -> Engine:
        kwargs = {"future": True, "pool_pre_ping": True}
        if make_url(db_url).drivername.startswith("sqlite"):
            # Explicit busy timeout: a connection that hits the single SQLite write lock
            # waits this long before raising SQLITE_BUSY ("database is locked").
            kwargs["connect_args"] = {"timeout": _SQLITE_BUSY_TIMEOUT_SECS}
        return create_engine(db_url, **kwargs)

    def _configure_engine(self):
        if self.engine.dialect.name != "sqlite":
            return

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    @contextmanager
    def _begin_write(self):
        """Open a WRITE transaction; every mutator must use this instead of engine.begin().

        On SQLite, BEGIN IMMEDIATE takes the single writer lock up front, avoiding
        shared-to-exclusive lock-upgrade deadlocks between concurrent read-merge-write
        transactions. pysqlite's legacy isolation handling does not emit its implicit
        BEGIN until the first DML statement, so issuing BEGIN IMMEDIATE as the first
        statement of the transaction works (the standard pysqlite-legacy recipe), and
        SAVEPOINTs (used by _insert_ignore) nest inside it normally.

        Pure reads must use plain engine.begin(): they take no write lock and therefore
        never contend with (or block) a long-running write transaction.
        """
        with self.engine.begin() as conn:
            if self.engine.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            yield conn

    @classmethod
    def sqlite(cls, db_path: str):
        return cls(sqlite_url(db_path))

    def initialize(self):
        # Memoized: server startup validates several times (marker check, study_store.configure,
        # bootstrap). The flag is only set on success, so a failed validation (pre-migration)
        # is retried; worst case under races is a redundant validation, which is harmless.
        if self._validated:
            return
        validate_database(self.db_url)
        self._validated = True

    # ------------------------------------------------------------------ studies

    def upsert_study(self, name: str, config: dict) -> dict:
        with self._begin_write() as conn:
            self._upsert_study(conn, name, config)
        return self.get_study(name)

    def _upsert_study(self, conn: Connection, name: str, config: dict):
        """Create the study row if needed and additively merge members from config.

        This never deletes member rows: a stale full-snapshot writer can no longer wipe out
        concurrently added admins/sites (the old delete-and-reinsert race). Removal is the
        job of the incremental remove_study_sites / remove_study_admin operations.
        """
        config = config or {}
        _insert_ignore(conn, studies, {"name": name})

        for user in config.get("admins", []) or []:
            _insert_ignore(conn, study_admins, {"study_name": name, "user_name": user})

        for org, org_sites in (config.get("site_orgs", {}) or {}).items():
            _insert_ignore(conn, study_orgs, {"study_name": name, "org": org})
            for site in org_sites or []:
                _insert_ignore(conn, study_sites, {"study_name": name, "org": org, "site_name": site})

    def get_study(self, name: str) -> Optional[dict]:
        with self.engine.begin() as conn:
            row = conn.execute(select(studies).where(studies.c.name == name)).first()
            if not row:
                return None
            admin_rows = conn.execute(
                select(study_admins.c.user_name)
                .where(study_admins.c.study_name == name)
                .order_by(study_admins.c.user_name)
            ).fetchall()
            org_site_rows = conn.execute(
                select(study_orgs.c.org, study_sites.c.site_name)
                .select_from(
                    study_orgs.outerjoin(
                        study_sites,
                        and_(
                            study_orgs.c.study_name == study_sites.c.study_name,
                            study_orgs.c.org == study_sites.c.org,
                        ),
                    )
                )
                .where(study_orgs.c.study_name == name)
                .order_by(study_orgs.c.org, study_sites.c.site_name)
            ).fetchall()
        result = _row_dict(row)
        result["config_json"] = {
            "admins": [user for (user,) in admin_rows],
            "site_orgs": _site_orgs_from_rows(org_site_rows),
        }
        return result

    def list_studies(self) -> List[dict]:
        with self.engine.begin() as conn:
            study_rows = conn.execute(select(studies).order_by(studies.c.name)).fetchall()
            admin_rows = conn.execute(
                select(study_admins.c.study_name, study_admins.c.user_name).order_by(study_admins.c.user_name)
            ).fetchall()
            org_site_rows = conn.execute(
                select(study_orgs.c.study_name, study_orgs.c.org, study_sites.c.site_name)
                .select_from(
                    study_orgs.outerjoin(
                        study_sites,
                        and_(
                            study_orgs.c.study_name == study_sites.c.study_name,
                            study_orgs.c.org == study_sites.c.org,
                        ),
                    )
                )
                .order_by(study_orgs.c.org, study_sites.c.site_name)
            ).fetchall()

        admins_by_study = {}
        for study_name, user in admin_rows:
            admins_by_study.setdefault(study_name, []).append(user)
        org_sites_by_study = {}
        for study_name, org, site in org_site_rows:
            org_sites_by_study.setdefault(study_name, []).append((org, site))

        result = []
        for row in study_rows:
            study = _row_dict(row)
            name = study["name"]
            study["config_json"] = {
                "admins": admins_by_study.get(name, []),
                "site_orgs": _site_orgs_from_rows(org_sites_by_study.get(name, [])),
            }
            result.append(study)
        return result

    def delete_study(self, name: str) -> bool:
        # Member rows (study_admins/study_orgs/study_sites) are removed by ON DELETE CASCADE.
        with self._begin_write() as conn:
            result = conn.execute(delete(studies).where(studies.c.name == name))
        return result.rowcount > 0

    def delete_study_if_no_jobs(self, name: str) -> dict:
        """Atomically delete the study iff no jobs reference it.

        The study row is locked, the jobs are counted, and the delete all happen in ONE
        transaction, so a delete cannot interleave with a count taken in a separate
        transaction. On SQLite, BEGIN IMMEDIATE serializes all writers, which makes the
        count-then-delete atomic against concurrent job submissions. On PostgreSQL the
        study row is locked with SELECT ... FOR UPDATE, but under READ COMMITTED a residual
        race remains: job inserts do not lock the study row, so a submit that commits
        between our count and our commit can still orphan its job. Full closure would need
        a jobs.study foreign key to studies.name (or SERIALIZABLE isolation) — out of scope.

        The job filter (jobs.study == name) matches the legacy command-level check, which
        counted get_all_jobs entries whose meta study equals the name: the study column is
        derived as meta-study-or-DEFAULT_STUDY, and the default study is never deletable,
        so the column filter is exactly equivalent. Deleted jobs have no row and never block.

        Member rows (study_admins/study_orgs/study_sites) are removed by ON DELETE CASCADE,
        matching delete_study.

        Returns:
            {"deleted": True} if the study was deleted;
            {"deleted": False, "not_found": True} if the study does not exist;
            {"deleted": False, "job_count": n} if n jobs still reference the study.
        """
        with self._begin_write() as conn:
            row = conn.execute(select(studies).where(studies.c.name == name).with_for_update()).first()
            if not row:
                return {"deleted": False, "not_found": True}
            job_count = conn.execute(select(func.count()).select_from(jobs).where(jobs.c.study == name)).scalar_one()
            if job_count:
                return {"deleted": False, "job_count": job_count}
            conn.execute(delete(studies).where(studies.c.name == name))
            return {"deleted": True}

    def add_study_sites(self, name: str, site_orgs: Dict[str, List[str]]) -> dict:
        with self._begin_write() as conn:
            for org, sites_for_org in (site_orgs or {}).items():
                _insert_ignore(conn, study_orgs, {"study_name": name, "org": org})
                for site in sites_for_org or []:
                    _insert_ignore(conn, study_sites, {"study_name": name, "org": org, "site_name": site})
        return self.get_study(name)

    def remove_study_sites(self, name: str, site_orgs: Dict[str, List[str]]) -> dict:
        # Org enrollment rows are intentionally kept: removing an org's last site must not
        # un-enroll the org (its admins would lose visibility of the study).
        with self._begin_write() as conn:
            for org, sites_for_org in (site_orgs or {}).items():
                for site in sites_for_org or []:
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
        with self._begin_write() as conn:
            _insert_ignore(conn, study_admins, {"study_name": name, "user_name": user})
        return self.get_study(name)

    def remove_study_admin(self, name: str, user: str) -> dict:
        with self._begin_write() as conn:
            conn.execute(
                delete(study_admins).where(and_(study_admins.c.study_name == name, study_admins.c.user_name == user))
            )
        return self.get_study(name)

    # ------------------------------------------------------------------ jobs

    def create_job(self, meta: dict, content_uri: str, content_hash: str = None, content_size: int = None) -> dict:
        with self._begin_write() as conn:
            return self._create_job(conn, meta, content_uri, content_hash=content_hash, content_size=content_size)

    def _create_job(
        self, conn: Connection, meta: dict, content_uri: str, content_hash: str = None, content_size: int = None
    ) -> dict:
        meta_json = _json_safe(meta)
        status, study = _job_status_and_study(meta_json)
        job_id = _get(meta_json, JobMetaKey.JOB_ID)
        if not job_id:
            raise ValueError("job metadata must contain job_id")
        if not status:
            raise ValueError("job metadata must contain status")
        now = _now()
        values = {
            "job_id": job_id,
            "study": study,
            "status": status,
            "content_uri": content_uri,
            "content_hash": content_hash,
            "content_size": content_size,
            "meta_json": meta_json,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
        conn.execute(insert(jobs).values(**values))
        return dict(values)

    def get_job(self, job_id: str) -> Optional[dict]:
        with self.engine.begin() as conn:
            row = conn.execute(select(jobs).where(jobs.c.job_id == job_id)).first()
        return _row_dict(row)

    def delete_job(self, job_id: str) -> bool:
        with self._begin_write() as conn:
            result = conn.execute(delete(jobs).where(jobs.c.job_id == job_id))
        return result.rowcount > 0

    def list_jobs(self, status: Union[str, List[str]] = None, study: str = None) -> List[dict]:
        conditions = []
        if status is not None:
            if isinstance(status, (list, tuple, set)):
                conditions.append(jobs.c.status.in_([run_status_value(s) for s in status]))
            else:
                conditions.append(jobs.c.status == run_status_value(status))
        if study is not None:
            conditions.append(jobs.c.study == study)
        stmt = select(jobs).order_by(jobs.c.created_at)
        if conditions:
            stmt = stmt.where(and_(*conditions))
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_dict(row) for row in rows]

    def update_job_meta(self, job_id: str, meta: dict) -> dict:
        """Merge meta into the job row with optimistic concurrency.

        The version column guards the read-merge-write: the UPDATE only applies if the row
        version is unchanged since the read; on conflict the whole read-merge-write retries
        (SQLite renders SELECT ... FOR UPDATE as a no-op, so locking alone is insufficient).
        """
        for attempt in range(_MAX_WRITE_ATTEMPTS):
            with self._begin_write() as conn:
                current = self._read_job_for_update(conn, job_id)
                if not current:
                    return None
                merged_meta = dict(current["meta_json"] or {})
                merged_meta.update(_json_safe(meta))
                status, study = _job_status_and_study(merged_meta)
                values = {
                    "study": study,
                    "status": status,
                    "meta_json": merged_meta,
                    "version": current["version"] + 1,
                    "updated_at": _now(),
                }
                result = conn.execute(
                    update(jobs)
                    .where(and_(jobs.c.job_id == job_id, jobs.c.version == current["version"]))
                    .values(**values)
                )
                if result.rowcount > 0:
                    current.update(values)
                    return current
            time.sleep(_RETRY_BACKOFF_SECS * (attempt + 1))
        raise RuntimeError(
            f"update_job_meta for job '{job_id}' lost {_MAX_WRITE_ATTEMPTS} optimistic-concurrency races"
        )

    def _read_job_for_update(self, conn: Connection, job_id: str) -> Optional[dict]:
        # FOR UPDATE row-locks on PostgreSQL; SQLite renders it as a no-op (the version
        # check in update_job_meta covers it there).
        return _row_dict(conn.execute(select(jobs).where(jobs.c.job_id == job_id).with_for_update()).first())

    # ------------------------------------------------------------------ submit records

    def create_submit_record(self, record: dict) -> bool:
        with self._begin_write() as conn:
            return self._create_submit_record(conn, record)

    def _create_submit_record(self, conn: Connection, record: dict) -> bool:
        """Insert a submit record; returns False if its scope already exists."""
        values = _submit_record_values(record)
        if not values.get("job_id"):
            raise ValueError("submit record must contain job_id")
        now = _now()
        values.update({"version": 1, "created_at": now, "updated_at": now})
        # The primary key (autoincrement id) is not in the values: verify convergence
        # against the unique scope instead.
        return _insert_ignore(conn, submit_records, values, verify=_submit_record_scope_select(values))

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
        scope = _submit_record_scope_condition(values)
        with self._begin_write() as conn:
            for _attempt in range(_MAX_WRITE_ATTEMPTS):
                now = _now()
                result = conn.execute(
                    update(submit_records)
                    .where(scope)
                    .values(**values, version=submit_records.c.version + 1, updated_at=now)
                )
                if result.rowcount > 0:
                    break
                insert_values = dict(values, version=1, created_at=now, updated_at=now)
                if _insert_ignore(conn, submit_records, insert_values, verify=_submit_record_scope_select(values)):
                    break
                # A concurrent writer inserted the row between our UPDATE and INSERT; retry as UPDATE.
            else:
                raise RuntimeError(
                    f"update_submit_record for job '{values.get('job_id')}' lost "
                    f"{_MAX_WRITE_ATTEMPTS} update/insert races"
                )
        # record_json is already a detached deepcopy of record (made by _submit_record_values).
        return values["record_json"]

    def mark_submit_records_job_deleted(self, job_id: str, deleted_by: Any) -> List[dict]:
        deleted_by_info = submitter_to_dict(deleted_by)
        deleted_time = _now().isoformat()
        updated = []
        with self._begin_write() as conn:
            # FOR UPDATE row-locks the records on PostgreSQL so the per-row tombstone UPDATE
            # cannot be computed from a stale read while a concurrent update_submit_record
            # commits (SQLite is already serialized by the _begin_write write lock).
            # Residual hardening for HA: carry a version guard into each UPDATE
            # (WHERE version == read version) and retry, like update_job_meta.
            rows = conn.execute(
                select(submit_records.c.id, submit_records.c.record_json)
                .where(submit_records.c.job_id == job_id)
                .with_for_update()
            ).fetchall()
            for row in rows:
                record = dict(row._mapping["record_json"] or {})
                if record.get(SubmitRecordKey.STATE.value) == SubmitRecordState.JOB_DELETED.value:
                    continue
                record[SubmitRecordKey.STATE.value] = SubmitRecordState.JOB_DELETED.value
                record[SubmitRecordKey.DELETED_TIME.value] = deleted_time
                record[SubmitRecordKey.DELETED_BY.value] = deleted_by_info
                # Only record_json changes: the scope hashes and job_id are derived from
                # fields the tombstone does not touch, so they are not recomputed/rewritten.
                conn.execute(
                    update(submit_records)
                    .where(submit_records.c.id == row._mapping["id"])
                    .values(record_json=_json_safe(record), version=submit_records.c.version + 1, updated_at=_now())
                )
                updated.append(record)
        return updated

    # ------------------------------------------------------------------ disabled clients

    def disable_client(self, client_name: str, disabled_by: str = None, reason: str = None) -> dict:
        with self._begin_write() as conn:
            self._disable_client(conn, client_name, disabled_by=disabled_by, reason=reason)
        return self.get_disabled_client(client_name)

    def _disable_client(self, conn: Connection, client_name: str, disabled_by: str = None, reason: str = None):
        now = _now()
        for _attempt in range(_MAX_WRITE_ATTEMPTS):
            try:
                inserted = _insert_ignore(
                    conn,
                    disabled_clients,
                    {
                        "client_name": client_name,
                        "disabled_by": disabled_by,
                        "disabled_at": now,
                        "reason": reason,
                        "version": 1,
                    },
                )
            except IntegrityError:
                # disabled_clients has no foreign keys: a verify-miss after a duplicate-key
                # error means the blocking row was concurrently deleted (enable_client) —
                # retry rather than fail.
                continue
            if inserted:
                return
            result = conn.execute(
                update(disabled_clients)
                .where(disabled_clients.c.client_name == client_name)
                .values(
                    disabled_by=disabled_by,
                    disabled_at=now,
                    reason=reason,
                    version=disabled_clients.c.version + 1,
                )
            )
            if result.rowcount > 0:
                return
            # The row that blocked the INSERT was concurrently deleted (enable_client); retry.
        raise RuntimeError(f"disable_client for '{client_name}' lost {_MAX_WRITE_ATTEMPTS} insert/update races")

    def get_disabled_client(self, client_name: str) -> Optional[dict]:
        with self.engine.begin() as conn:
            row = conn.execute(select(disabled_clients).where(disabled_clients.c.client_name == client_name)).first()
        return _row_dict(row)

    def enable_client(self, client_name: str) -> bool:
        with self._begin_write() as conn:
            result = conn.execute(delete(disabled_clients).where(disabled_clients.c.client_name == client_name))
        return result.rowcount > 0

    # ------------------------------------------------------------------ migration markers

    def get_migration_marker(self, name: str) -> Optional[dict]:
        with self.engine.begin() as conn:
            return self._read_migration_marker(conn, name)

    def set_migration_marker(self, name: str, source_format: str, summary: dict) -> dict:
        """Write a migration marker, converging on an existing marker of the same name."""
        with self._begin_write() as conn:
            self._insert_migration_marker(conn, name, source_format, summary)
        return self.get_migration_marker(name)

    def _read_migration_marker(self, conn: Connection, name: str) -> Optional[dict]:
        row = conn.execute(select(state_store_migrations).where(state_store_migrations.c.name == name)).first()
        return _row_dict(row)

    def _insert_migration_marker(self, conn: Connection, name: str, source_format: str, summary: dict) -> bool:
        return _insert_ignore(
            conn,
            state_store_migrations,
            {
                "name": name,
                "source_format": source_format,
                "applied_at": _now(),
                "nvflare_version": getattr(nvflare, "__version__", None),
                "summary_json": _json_safe(summary),
            },
        )

    def _update_migration_marker_summary(self, conn: Connection, name: str, summary: dict):
        conn.execute(
            update(state_store_migrations)
            .where(state_store_migrations.c.name == name)
            .values(summary_json=_json_safe(summary))
        )

    def _state_tables_with_rows(self, conn: Connection) -> List[str]:
        existing = []
        for table, name in (
            (studies, "studies"),
            (jobs, "jobs"),
            (submit_records, "submit_records"),
            (disabled_clients, "disabled_clients"),
        ):
            if conn.execute(select(table).limit(1)).first() is not None:
                existing.append(name)
        return existing
