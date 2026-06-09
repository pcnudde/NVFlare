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
import sys
from pathlib import Path, PurePath
from typing import List, Optional

from nvflare.apis import study_store
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.storage import StorageException, StorageSpec
from nvflare.app_common.state_store.legacy_study_registry import LegacyStudyRegistry
from nvflare.app_common.state_store.sql_store import SqlStateStore, migrate_database

LEGACY_MIGRATION_MARKER = "legacy_filesystem_migration_v1"
LEGACY_SOURCE_FORMAT = "legacy-filesystem"
FRESH_INSTALL_SOURCE_FORMAT = "fresh-install"
_SUBMIT_RECORD_URI_ROOT = "job_submit_records"
_SUBMIT_RECORD_JOB_INDEX_URI_ROOT = "job_submit_record_index"
_SUBMIT_RECORD_URIS_KEY = "submit_record_uris"


class MigrationSkipError(RuntimeError):
    """Raised in --strict mode for conditions that are otherwise skip-with-warning."""

    pass


def validate_state_store_migrated(state_store: SqlStateStore, marker_name: str = LEGACY_MIGRATION_MARKER) -> dict:
    state_store.initialize()
    marker = state_store.get_migration_marker(marker_name)
    if marker is None:
        raise RuntimeError(
            f"state store migration marker '{marker_name}' is missing; "
            "run nvflare-state-store-migrate before starting the server"
        )
    return marker


def classify_legacy_state(server_root: str, jobs_dir: Optional[str]) -> dict:
    """Classify legacy filesystem state under a server workspace root.

    Args:
        server_root: server workspace root containing local/ and startup/.
        jobs_dir: the FULLY RESOLVED legacy jobs directory (computed by the caller from the
            job store's root_dir + uri_root, exactly as the migrate CLI resolves it). Pass
            None when it cannot be determined; that is treated as "no legacy jobs".

    Returns:
        {
            "jobs": bool,                          # jobs_dir exists and is non-empty
            "study_registry": Optional[str],       # path to local/study_registry.json, or None
            "disabled_clients": bool,              # disabled_clients.json exists under the root
        }
    """
    root = Path(server_root).expanduser()
    registry = root / "local" / "study_registry.json"
    jobs_path = Path(jobs_dir).expanduser() if jobs_dir else None
    return {
        "jobs": bool(jobs_path and jobs_path.is_dir() and any(jobs_path.iterdir())),
        "study_registry": str(registry) if registry.exists() else None,
        "disabled_clients": (root / "disabled_clients.json").exists(),
    }


def has_legacy_state(server_root: str, jobs_dir_name: str = "jobs") -> bool:
    """Detect legacy filesystem state under a server workspace root.

    Used by deployers to decide between bootstrap_fresh_state_store (fresh install) and the
    full nvflare-state-store-migrate flow (legacy upgrade).

    jobs_dir_name is resolved against server_root (an absolute jobs_dir_name is used as is).
    Callers that know the job store's actual location should prefer classify_legacy_state
    with the fully resolved jobs directory.
    """
    jobs_dir = str(Path(server_root).expanduser() / jobs_dir_name)
    state = classify_legacy_state(server_root, jobs_dir)
    return bool(state["jobs"] or state["study_registry"] or state["disabled_clients"])


def bootstrap_fresh_state_store(store: SqlStateStore, marker_name: str = LEGACY_MIGRATION_MARKER) -> dict:
    """Prepare a state store for a fresh install (no legacy data to import).

    Applies the schema if absent and writes a fresh-install migration marker (the same
    marker validate_state_store_migrated checks), so server startup can proceed without
    running nvflare-state-store-migrate.

    Refuses to stamp a populated database: if state rows already exist without a marker
    (renamed marker, partial restore, shared database), declaring it "fresh" would be a
    silent lie, so a RuntimeError is raised instead.
    """
    try:
        store.initialize()
    except RuntimeError:
        migrate_database(store.db_url)
        store.initialize()
    marker = store.get_migration_marker(marker_name)
    if marker:
        return {"bootstrapped": False, "marker": marker}
    summary = {"status": "complete", "source_format": FRESH_INSTALL_SOURCE_FORMAT}
    with store._begin_write() as conn:
        inserted = store._insert_migration_marker(conn, marker_name, FRESH_INSTALL_SOURCE_FORMAT, summary)
        if inserted:
            # Same transaction as the marker insert: the check and the stamp are atomic.
            existing = store._state_tables_with_rows(conn)
            if existing:
                raise RuntimeError(
                    f"refusing to bootstrap a fresh state store: migration marker '{marker_name}' is "
                    "missing but state data already exists in tables: " + ", ".join(existing) + ". "
                    "If this database was migrated before, restore its marker; otherwise investigate "
                    "where the data came from before stamping it as a fresh install."
                )
    return {"bootstrapped": inserted, "marker": store.get_migration_marker(marker_name)}


def migrate_legacy_state_store(
    state_store: SqlStateStore,
    job_storage: Optional[StorageSpec] = None,
    jobs_uri_root: str = "jobs",
    study_registry_path: str = None,
    disabled_clients_path: str = None,
    marker_name: str = LEGACY_MIGRATION_MARKER,
    strict: bool = False,
    warnings: Optional[List[str]] = None,
) -> dict:
    """Import legacy filesystem state exactly once and write the migration marker.

    Per-item problems (dangling submit-record index entries, unreadable or status-less job
    metas, invalid study definitions) are skipped with a warning recorded in the marker
    summary, unless strict=True which turns them into errors that abort the migration.
    """
    state_store.initialize()
    marker = state_store.get_migration_marker(marker_name)
    if marker:
        return {"migrated": False, "marker": marker}

    notes = _WarningLog(strict=strict, warnings=list(warnings or []))

    # NOTE: everything inside this transaction must go through the open `conn` (the
    # state_store._xxx(conn, ...) helpers). Calling a public state_store method here would
    # open a SECOND connection, which on SQLite blocks against our own write lock.
    # The transaction (and thus the SQLite write lock) is held for the whole import,
    # including storage I/O — acceptable for this offline CLI, which runs before the server.
    existing_marker = None
    with state_store._begin_write() as conn:
        if not state_store._insert_migration_marker(conn, marker_name, LEGACY_SOURCE_FORMAT, {"status": "in_progress"}):
            # A concurrent migration already wrote the marker: converge gracefully.
            existing_marker = state_store._read_migration_marker(conn, marker_name)
        else:
            _ensure_no_existing_state_data(state_store, conn)

            summary = {
                "source_format": LEGACY_SOURCE_FORMAT,
                "study_registry_path": _path_or_none(study_registry_path),
                "disabled_clients_path": _path_or_none(disabled_clients_path),
                "jobs_uri_root": jobs_uri_root,
            }
            summary.update(_import_studies(state_store, conn, study_registry_path, notes))
            summary.update(_import_jobs(state_store, conn, job_storage, jobs_uri_root, notes))
            summary.update(_import_disabled_clients(state_store, conn, disabled_clients_path))
            summary["warnings"] = notes.warnings
            summary["status"] = "complete"

            state_store._update_migration_marker_summary(conn, marker_name, summary)

    if existing_marker is not None:
        return {"migrated": False, "marker": existing_marker}
    return {"migrated": True, "marker": state_store.get_migration_marker(marker_name)}


def load_legacy_study_registry(path: str) -> dict:
    with open(path, "rt", encoding="utf-8") as f:
        config = json.load(f)

    registry = LegacyStudyRegistry(config)
    return {"format_version": LegacyStudyRegistry.FORMAT_VERSION, "studies": registry.get_studies()}


def load_legacy_disabled_clients(path: str) -> list:
    with open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("disabled clients file must be a JSON object")
    disabled = data.get("disabled_clients")
    if not isinstance(disabled, list):
        raise ValueError("disabled_clients must be a list")

    return sorted({str(client_name) for client_name in disabled if client_name})


class _WarningLog:
    def __init__(self, strict: bool, warnings: List[str]):
        self.strict = strict
        self.warnings = warnings

    def note(self, message: str):
        if self.strict:
            raise MigrationSkipError(message)
        print(f"WARNING: {message}", file=sys.stderr)
        self.warnings.append(message)


def _path_or_none(path: str):
    return str(Path(path).expanduser()) if path else None


def _ensure_no_existing_state_data(state_store: SqlStateStore, conn):
    existing = state_store._state_tables_with_rows(conn)
    if existing:
        raise RuntimeError(
            "state store migration marker is missing but state data already exists: " + ", ".join(existing)
        )


def _import_studies(state_store: SqlStateStore, conn, study_registry_path: str, notes: _WarningLog) -> dict:
    if not study_registry_path or not os.path.exists(study_registry_path):
        return {"imported_studies": []}

    config = load_legacy_study_registry(study_registry_path)
    imported = []
    for study_name, study_def in (config.get("studies") or {}).items():
        try:
            normalized = study_store.normalize_study(study_name, study_def)
        except ValueError as e:
            notes.note(f"skipping legacy study '{study_name}': {e}")
            continue
        state_store._upsert_study(conn, study_name, normalized)
        imported.append(study_name)
    return {"imported_studies": sorted(imported)}


def _import_jobs(
    state_store: SqlStateStore, conn, job_storage: Optional[StorageSpec], jobs_uri_root: str, notes: _WarningLog
) -> dict:
    if not job_storage:
        return {"imported_jobs": [], "imported_submit_records": 0}

    imported_jobs = []
    job_uris, jobs_root_missing = _list_legacy_objects(job_storage, jobs_uri_root, optional=False)
    for uri in job_uris:
        try:
            meta = dict(job_storage.get_meta(uri) or {})
        except StorageException as e:
            notes.note(f"skipping legacy job at '{uri}': failed to read meta: {e}")
            continue
        job_id = meta.get(JobMetaKey.JOB_ID.value) or PurePath(uri).name
        meta[JobMetaKey.JOB_ID.value] = job_id
        if not meta.get(JobMetaKey.STATUS.value):
            notes.note(f"skipping legacy job at '{uri}': metadata is missing status")
            continue
        state_store._create_job(conn, meta, content_uri=uri)
        imported_jobs.append(job_id)

    imported_submit_records = 0
    for record in _legacy_submit_records(job_storage, jobs_uri_root, notes):
        if state_store._create_submit_record(conn, record):
            imported_submit_records += 1

    return {
        "imported_jobs": sorted(imported_jobs),
        "imported_submit_records": imported_submit_records,
        "legacy_jobs_uri_root_missing": jobs_root_missing,
    }


def _import_disabled_clients(state_store: SqlStateStore, conn, disabled_clients_path: str) -> dict:
    if not disabled_clients_path or not os.path.exists(disabled_clients_path):
        return {"imported_disabled_clients": []}

    imported_clients = load_legacy_disabled_clients(disabled_clients_path)
    for client_name in imported_clients:
        state_store._disable_client(conn, client_name, reason="migrated from disabled_clients.json")
    return {"imported_disabled_clients": imported_clients}


def _list_legacy_objects(storage: StorageSpec, uri_root: str, optional: bool = True):
    """List object URIs under uri_root; returns (objects, root_missing)."""
    try:
        return storage.list_objects(uri_root), False
    except StorageException as e:
        if _is_missing_filesystem_uri(storage, uri_root):
            return [], True
        kind = "objects" if optional else "jobs"
        raise RuntimeError(f"failed to list legacy {kind} under '{uri_root}': {e}") from e


def _is_missing_filesystem_uri(storage: StorageSpec, uri: str) -> bool:
    object_path = getattr(storage, "_object_path", None)
    if not callable(object_path):
        return False
    try:
        path = object_path(uri)
    except StorageException:
        return False
    return not os.path.exists(path)


def _scan_submit_record_uris(storage: StorageSpec, records_root: str) -> Optional[List[str]]:
    """Recursively scan the records root for object URIs (FilesystemStorage layouts only).

    Returns None when the storage does not expose a filesystem layout, in which case the
    caller falls back to index-based enumeration only.
    """
    object_path = getattr(storage, "_object_path", None)
    if not callable(object_path):
        return None
    try:
        root_path = object_path(records_root)
    except StorageException:
        return None
    if not os.path.isdir(root_path):
        return []
    uris = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if "meta" in filenames and "data" in filenames:
            rel = os.path.relpath(dirpath, root_path)
            uris.append(records_root if rel == os.curdir else os.path.join(records_root, rel))
            dirnames[:] = []  # an object dir has no nested objects
    return sorted(uris)


def _legacy_submit_records(storage: StorageSpec, jobs_uri_root: str, notes: _WarningLog):
    record_uris = []

    # Primary enumeration: scan the records root directly so records whose index write was
    # lost are still migrated.
    records_root = _sidecar_root(jobs_uri_root, _SUBMIT_RECORD_URI_ROOT)
    scanned = _scan_submit_record_uris(storage, records_root)
    if scanned:
        record_uris.extend(scanned)

    # Union with the index sidecars (covers non-filesystem layouts the scan cannot see).
    index_root = _sidecar_root(jobs_uri_root, _SUBMIT_RECORD_JOB_INDEX_URI_ROOT)
    index_uris, _missing = _list_legacy_objects(storage, index_root)
    for index_uri in index_uris:
        try:
            index_meta = storage.get_meta(index_uri) or {}
        except StorageException as e:
            notes.note(f"skipping legacy submit-record index '{index_uri}': failed to read meta: {e}")
            continue
        record_uris.extend(index_meta.get(_SUBMIT_RECORD_URIS_KEY, []))

    for record_uri in dict.fromkeys(record_uris):
        try:
            record = storage.get_meta(record_uri)
        except StorageException as e:
            notes.note(f"skipping legacy submit record '{record_uri}': failed to read meta: {e}")
            continue
        if record:
            yield record


def _sidecar_root(jobs_uri_root: str, sidecar_name: str):
    root = jobs_uri_root.rstrip(os.sep) or jobs_uri_root
    return os.path.join(os.path.dirname(root), sidecar_name)
