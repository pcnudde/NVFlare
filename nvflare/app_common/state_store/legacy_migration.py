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
from pathlib import Path, PurePath
from typing import Optional

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

import nvflare
from nvflare.apis import study_store
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.state_store import StateStore
from nvflare.apis.storage import StorageException, StorageSpec
from nvflare.app_common.state_store.legacy_study_registry import LegacyStudyRegistry
from nvflare.app_common.state_store.sql_store import (
    SqlStateStore,
    _job_columns_from_meta,
    _json_safe,
    _now,
    _study_member_values,
    _submit_record_values,
    disabled_clients,
    jobs,
    state_store_migrations,
    studies,
    study_admins,
    study_sites,
    submit_records,
)

LEGACY_MIGRATION_MARKER = "legacy_filesystem_migration_v1"
LEGACY_SOURCE_FORMAT = "legacy-filesystem"
_SUBMIT_RECORD_URI_ROOT = "job_submit_records"
_SUBMIT_RECORD_JOB_INDEX_URI_ROOT = "job_submit_record_index"
_SUBMIT_RECORD_URIS_KEY = "submit_record_uris"


def validate_state_store_migrated(state_store: StateStore, marker_name: str = LEGACY_MIGRATION_MARKER) -> dict:
    state_store.initialize()
    marker = state_store.get_migration_marker(marker_name)
    if marker is None:
        raise RuntimeError(
            f"state store migration marker '{marker_name}' is missing; "
            "run nvflare-state-store-migrate before starting the server"
        )
    return marker


def migrate_legacy_state_store(
    state_store: StateStore,
    job_storage: Optional[StorageSpec] = None,
    jobs_uri_root: str = "jobs",
    study_registry_path: str = None,
    disabled_clients_path: str = None,
    marker_name: str = LEGACY_MIGRATION_MARKER,
) -> dict:
    """Import legacy filesystem state exactly once and write the migration marker."""
    state_store.initialize()
    marker = state_store.get_migration_marker(marker_name)
    if marker:
        return {"migrated": False, "marker": marker}

    if not isinstance(state_store, SqlStateStore):
        raise TypeError(f"legacy migration requires SqlStateStore but got {type(state_store)}")

    try:
        with state_store.engine.begin() as conn:
            if conn.execute(select(state_store_migrations).where(state_store_migrations.c.name == marker_name)).first():
                return {"migrated": False, "marker": state_store.get_migration_marker(marker_name)}

            now = _now()
            conn.execute(
                insert(state_store_migrations).values(
                    name=marker_name,
                    source_format=LEGACY_SOURCE_FORMAT,
                    applied_at=now,
                    nvflare_version=getattr(nvflare, "__version__", None),
                    summary_json={"status": "in_progress"},
                )
            )
            _ensure_no_existing_state_data(conn)

            summary = {
                "source_format": LEGACY_SOURCE_FORMAT,
                "study_registry_path": _path_or_none(study_registry_path),
                "disabled_clients_path": _path_or_none(disabled_clients_path),
                "jobs_uri_root": jobs_uri_root,
            }
            summary.update(_import_studies(conn, study_registry_path))
            summary.update(_import_jobs(conn, job_storage, jobs_uri_root))
            summary.update(_import_disabled_clients(conn, disabled_clients_path))
            summary["status"] = "complete"

            conn.execute(
                update(state_store_migrations)
                .where(state_store_migrations.c.name == marker_name)
                .values(summary_json=_json_safe(summary))
            )
    except IntegrityError:
        marker = state_store.get_migration_marker(marker_name)
        if marker:
            return {"migrated": False, "marker": marker}
        raise

    return {"migrated": True, "marker": state_store.get_migration_marker(marker_name)}


def load_legacy_study_registry(path: str) -> dict:
    with open(path, "rt", encoding="utf-8") as f:
        config = json.load(f)

    registry = LegacyStudyRegistry(config)
    normalized_studies = {}
    for study_name, study_def in registry.get_studies().items():
        normalized_studies[study_name] = {
            "admins": study_def.get("admins", []),
            "site_orgs": study_def.get("site_orgs", {}),
        }
    return {"format_version": LegacyStudyRegistry.FORMAT_VERSION, "studies": normalized_studies}


def load_legacy_disabled_clients(path: str) -> list:
    with open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("disabled clients file must be a JSON object")
    disabled = data.get("disabled_clients")
    if not isinstance(disabled, list):
        raise ValueError("disabled_clients must be a list")

    return sorted({str(client_name) for client_name in disabled if client_name})


def _path_or_none(path: str):
    return str(Path(path).expanduser()) if path else None


def _has_rows(conn, table) -> bool:
    return conn.execute(select(table).limit(1)).first() is not None


def _ensure_no_existing_state_data(conn):
    existing = []
    for table, name in (
        (studies, "studies"),
        (jobs, "jobs"),
        (submit_records, "submit_records"),
        (disabled_clients, "disabled_clients"),
    ):
        if _has_rows(conn, table):
            existing.append(name)
    if existing:
        raise RuntimeError(
            "state store migration marker is missing but state data already exists: " + ", ".join(existing)
        )


def _upsert_study_in_conn(conn, name: str, config: dict):
    row = conn.execute(select(studies.c.name).where(studies.c.name == name)).first()
    if not row:
        conn.execute(insert(studies).values(name=name))
    conn.execute(study_admins.delete().where(study_admins.c.study_name == name))
    conn.execute(study_sites.delete().where(study_sites.c.study_name == name))
    admin_values, site_values = _study_member_values(name, config)
    if admin_values:
        conn.execute(insert(study_admins), admin_values)
    if site_values:
        conn.execute(insert(study_sites), site_values)


def _import_studies(conn, study_registry_path: str) -> dict:
    if not study_registry_path or not os.path.exists(study_registry_path):
        return {"imported_studies": []}

    config = load_legacy_study_registry(study_registry_path)
    imported = []
    for study_name, study_def in (config.get("studies") or {}).items():
        _upsert_study_in_conn(conn, study_name, study_store.normalize_study(study_name, study_def))
        imported.append(study_name)
    return {"imported_studies": sorted(imported)}


def _insert_job_in_conn(conn, uri: str, meta: dict):
    values = _job_columns_from_meta(meta)
    job_id = values.get("job_id")
    if not job_id:
        raise ValueError(f"legacy job metadata at {uri} is missing job_id")
    if not values.get("status"):
        raise ValueError(f"legacy job metadata at {uri} is missing status")

    now = _now()
    values.update(
        {
            "content_uri": uri,
            "content_hash": None,
            "content_size": None,
            "meta_json": _json_safe(meta),
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
    )
    conn.execute(insert(jobs).values(**values))


def _insert_submit_record_in_conn(conn, record: dict):
    values = _submit_record_values(record)
    if not values.get("job_id"):
        raise ValueError("legacy submit record is missing job_id")

    now = _now()
    values.update({"version": 1, "created_at": now, "updated_at": now})
    conn.execute(insert(submit_records).values(**values))


def _import_jobs(conn, job_storage: Optional[StorageSpec], jobs_uri_root: str) -> dict:
    if not job_storage:
        return {"imported_jobs": [], "imported_submit_records": 0}

    imported_jobs = []
    job_uris, jobs_root_missing = _list_legacy_job_objects(job_storage, jobs_uri_root)
    for uri in job_uris:
        meta = dict(job_storage.get_meta(uri) or {})
        job_id = meta.get(JobMetaKey.JOB_ID.value) or PurePath(uri).name
        meta[JobMetaKey.JOB_ID.value] = job_id
        _insert_job_in_conn(conn, uri, meta)
        imported_jobs.append(job_id)

    imported_submit_records = 0
    for record in _legacy_submit_records(job_storage, jobs_uri_root):
        _insert_submit_record_in_conn(conn, record)
        imported_submit_records += 1

    return {
        "imported_jobs": sorted(imported_jobs),
        "imported_submit_records": imported_submit_records,
        "legacy_jobs_uri_root_missing": jobs_root_missing,
    }


def _import_disabled_clients(conn, disabled_clients_path: str) -> dict:
    if not disabled_clients_path or not os.path.exists(disabled_clients_path):
        return {"imported_disabled_clients": []}

    now = _now()
    imported_clients = load_legacy_disabled_clients(disabled_clients_path)
    for client_name in imported_clients:
        conn.execute(
            insert(disabled_clients).values(
                client_name=client_name,
                disabled_by=None,
                disabled_at=now,
                reason="migrated from disabled_clients.json",
                version=1,
            )
        )
    return {"imported_disabled_clients": imported_clients}


def _list_legacy_job_objects(storage: StorageSpec, uri_root: str):
    try:
        return storage.list_objects(uri_root), False
    except StorageException as e:
        if _is_missing_filesystem_uri(storage, uri_root):
            return [], True
        raise RuntimeError(f"failed to list legacy jobs under '{uri_root}': {e}") from e


def _list_optional_objects(storage: StorageSpec, uri_root: str):
    try:
        return storage.list_objects(uri_root)
    except StorageException as e:
        if _is_missing_filesystem_uri(storage, uri_root):
            return []
        raise RuntimeError(f"failed to list legacy objects under '{uri_root}': {e}") from e


def _is_missing_filesystem_uri(storage: StorageSpec, uri: str) -> bool:
    object_path = getattr(storage, "_object_path", None)
    if not callable(object_path):
        return False
    try:
        path = object_path(uri)
    except StorageException:
        return False
    return not os.path.exists(path)


def _legacy_submit_records(storage: StorageSpec, jobs_uri_root: str):
    record_uris = []
    index_root = _sidecar_root(jobs_uri_root, _SUBMIT_RECORD_JOB_INDEX_URI_ROOT)
    for index_uri in _list_optional_objects(storage, index_root):
        index_meta = storage.get_meta(index_uri) or {}
        record_uris.extend(index_meta.get(_SUBMIT_RECORD_URIS_KEY, []))

    for record_uri in dict.fromkeys(record_uris):
        yield storage.get_meta(record_uri)


def _sidecar_root(jobs_uri_root: str, sidecar_name: str):
    root = jobs_uri_root.rstrip(os.sep) or jobs_uri_root
    return os.path.join(os.path.dirname(root), sidecar_name)
