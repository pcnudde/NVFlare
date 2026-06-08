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

import datetime
import os
from unittest import mock

from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
from nvflare.apis.job_def import SubmitRecordKey, SubmitRecordState
from nvflare.apis.utils.job_submit_token import submit_record_scope_hashes, submitter_to_dict


def _submitter():
    return {"name": "submitter@nvidia.com", "org": "nvidia", "role": "lead"}


def _record(job_id="job-1", state="created"):
    return {
        "schema_version": 1,
        "state": state,
        "submit_token": "retry-1",
        "job_id": job_id,
        "study": "study-a",
        "submitter_name": "submitter@nvidia.com",
        "submitter_org": "nvidia",
        "submitter_role": "lead",
        "job_name": "hello",
        "job_folder_name": "hello",
        "job_content_hash": "sha256:abc",
        "submit_time": "2026-04-29T10:00:00-07:00",
    }


class _FakeStateStore:
    def __init__(self):
        self.records = {}

    def _key(self, study, submitter, submit_token):
        return submit_record_scope_hashes(study, submitter, submit_token)

    def _key_from_record(self, record):
        submitter = {
            "name": record.get(SubmitRecordKey.SUBMITTER_NAME.value, ""),
            "org": record.get(SubmitRecordKey.SUBMITTER_ORG.value, ""),
            "role": record.get(SubmitRecordKey.SUBMITTER_ROLE.value, ""),
        }
        return self._key(
            record.get(SubmitRecordKey.STUDY.value, ""),
            submitter,
            record.get(SubmitRecordKey.SUBMIT_TOKEN.value),
        )

    def create_submit_record(self, record: dict) -> bool:
        key = self._key_from_record(record)
        if key in self.records:
            return False
        self.records[key] = dict(record)
        return True

    def get_submit_record(self, study: str, submitter, submit_token: str):
        record = self.records.get(self._key(study, submitter, submit_token))
        return dict(record) if record else None

    def update_submit_record(self, record: dict):
        self.records[self._key_from_record(record)] = dict(record)
        return dict(record)

    def mark_submit_records_job_deleted(self, job_id: str, deleted_by):
        deleted_by_info = submitter_to_dict(deleted_by)
        deleted_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        updated = []
        for key, record in list(self.records.items()):
            if record.get(SubmitRecordKey.JOB_ID.value) != job_id:
                continue
            if record.get(SubmitRecordKey.STATE.value) == SubmitRecordState.JOB_DELETED.value:
                continue
            record = dict(record)
            record[SubmitRecordKey.STATE.value] = SubmitRecordState.JOB_DELETED.value
            record[SubmitRecordKey.DELETED_TIME.value] = deleted_time
            record[SubmitRecordKey.DELETED_BY.value] = deleted_by_info
            self.records[key] = record
            updated.append(dict(record))
        return updated


def test_submit_record_persists_across_manager_restart(tmp_path):
    state_store = _FakeStateStore()
    fl_ctx = FLContext()

    with mock.patch.object(SimpleJobDefManager, "_get_state_store", return_value=state_store):
        manager = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        manager.create_submit_record(_record(), fl_ctx)

        restarted = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        record = restarted.get_submit_record("study-a", _submitter(), "retry-1", fl_ctx)

    assert record["job_id"] == "job-1"
    assert record["job_content_hash"] == "sha256:abc"
    assert record["submit_token"] == "retry-1"


def test_creating_record_survives_restart_for_retry_recovery(tmp_path):
    state_store = _FakeStateStore()
    fl_ctx = FLContext()

    with mock.patch.object(SimpleJobDefManager, "_get_state_store", return_value=state_store):
        manager = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        manager.create_submit_record(_record(job_id="pre-generated-job", state="creating"), fl_ctx)

        restarted = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        record = restarted.get_submit_record("study-a", _submitter(), "retry-1", fl_ctx)

    assert record["state"] == "creating"
    assert record["job_id"] == "pre-generated-job"


def test_mark_submit_record_job_deleted_preserves_record_for_audit(tmp_path):
    state_store = _FakeStateStore()
    fl_ctx = FLContext()

    with mock.patch.object(SimpleJobDefManager, "_get_state_store", return_value=state_store):
        manager = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        manager.create_submit_record(_record(), fl_ctx)

        marked = manager.mark_submit_records_job_deleted(
            "job-1", {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}, fl_ctx
        )
        record = manager.get_submit_record("study-a", _submitter(), "retry-1", fl_ctx)

    assert len(marked) == 1
    assert record["state"] == "job_deleted"
    assert record["job_id"] == "job-1"
    assert record["deleted_time"]
    assert datetime.datetime.fromisoformat(record["deleted_time"]).tzinfo == datetime.timezone.utc
    assert record["deleted_by"] == {
        "name": "admin@nvidia.com",
        "org": "nvidia",
        "role": "project_admin",
    }


def test_job_manager_defaults_to_state_store_component(tmp_path):
    manager = SimpleJobDefManager(uri_root=str(tmp_path / "jobs") + os.sep)

    assert manager.state_store_id == "state_store"
