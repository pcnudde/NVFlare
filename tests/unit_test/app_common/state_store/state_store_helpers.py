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

"""Shared fixtures-as-functions for state-store unit and integration tests."""

import json

from nvflare.apis.job_def import JobMetaKey, RunStatus, SubmitRecordKey, SubmitRecordState
from nvflare.app_common.state_store.legacy_migration import _SUBMIT_RECORD_URIS_KEY
from nvflare.app_common.state_store.sql_store import SqlStateStore, migrate_database
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage


def make_sqlite_store(tmp_path, db_name: str = "state_store.db") -> SqlStateStore:
    store = SqlStateStore.sqlite(str(tmp_path / db_name))
    migrate_database(store.db_url)
    store.initialize()
    return store


def job_meta(
    job_id: str,
    study: str = "study_a",
    status: str = RunStatus.SUBMITTED.value,
) -> dict:
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


def submit_record(
    job_id: str,
    token: str = "token-1",
    submitter_name: str = "admin@nvidia.com",
    study: str = "study_a",
) -> dict:
    return {
        SubmitRecordKey.SCHEMA_VERSION.value: 1,
        SubmitRecordKey.STATE.value: SubmitRecordState.CREATING.value,
        SubmitRecordKey.SUBMIT_TOKEN.value: token,
        SubmitRecordKey.JOB_ID.value: job_id,
        SubmitRecordKey.STUDY.value: study,
        SubmitRecordKey.SUBMITTER_NAME.value: submitter_name,
        SubmitRecordKey.SUBMITTER_ORG.value: "nvidia",
        SubmitRecordKey.SUBMITTER_ROLE.value: "project_admin",
        SubmitRecordKey.JOB_NAME.value: "hello",
        SubmitRecordKey.JOB_FOLDER_NAME.value: "hello_job",
        SubmitRecordKey.JOB_CONTENT_HASH.value: "sha256:abc",
        SubmitRecordKey.SUBMIT_TIME.value: "2026-06-08T00:00:00+00:00",
    }


def write_registry(path, studies: dict = None):
    path.write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "studies": (
                    studies
                    if studies is not None
                    else {
                        "study-a": {
                            "admins": ["admin@nvidia.com"],
                            "site_orgs": {"org-a": ["site-a"]},
                        }
                    }
                ),
            }
        ),
        encoding="utf-8",
    )


def write_disabled_clients(path, clients=("site-b",)):
    path.write_text(json.dumps({"disabled_clients": list(clients)}), encoding="utf-8")


def write_legacy_job_store(
    root_dir,
    job_id: str = "job-1",
    study: str = "study-a",
    token: str = "retry-1",
    with_index: bool = True,
) -> FilesystemStorage:
    """Create a legacy filesystem job store with one job and one indexed submit record."""
    storage = FilesystemStorage(root_dir=str(root_dir), uri_root="/")
    meta = job_meta(job_id, study=study)
    storage.create_object(f"jobs/{job_id}", b"job bytes", meta, overwrite_existing=False)

    record_uri = "job_submit_records/study/submitter/token"
    record = submit_record(job_id, token=token, submitter_name="submitter@nvidia.com", study=study)
    storage.create_object(record_uri, b"", record, overwrite_existing=False)
    if with_index:
        storage.create_object(
            f"job_submit_record_index/{job_id}",
            b"",
            {SubmitRecordKey.JOB_ID.value: job_id, _SUBMIT_RECORD_URIS_KEY: [record_uri]},
            overwrite_existing=False,
        )
    return storage
