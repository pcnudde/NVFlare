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

from unittest.mock import Mock

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
from nvflare.apis.job_def import SubmitRecordKey, SubmitRecordState
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage


def test_shared_job_index_updates_records_once_and_ignores_stale_entries(tmp_path, monkeypatch):
    monkeypatch.delenv("NVFL_JOB_STORE_ROOT", raising=False)
    manager = SimpleJobDefManager(str(tmp_path / "jobs"))
    store = FilesystemStorage(str(tmp_path / "store"))
    engine = Mock(spec=ServerEngineSpec)
    engine.get_component.return_value = store
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    alice = {"name": "alice", "org": "org", "role": "admin"}
    first = manager.new_submit_record("study", alice, "one", "hash", job_id="job-one")
    second = manager.new_submit_record("study", alice, "two", "hash", job_id="job-one")
    assert manager.create_submit_record(first, ctx)
    assert manager.create_submit_record(second, ctx)
    second["note"] = "updated"
    assert manager.update_submit_record(second, ctx) == second
    index_uri = manager._submit_record_job_index_uri("job-one")
    index = store.get_meta(index_uri)
    assert len(index["submit_record_uris"]) == 2
    # A missing sidecar and a stale cross-job entry must not change another job.
    other = manager.new_submit_record("study", alice, "other", "hash", job_id="job-two")
    assert manager.create_submit_record(other, ctx)
    other_uri = manager._submit_record_uri_from_record(other)
    index["submit_record_uris"].extend(["missing-record", other_uri])
    store.update_meta(index_uri, index, replace=True)
    deleted = manager.mark_submit_records_job_deleted("job-one", alice, ctx)
    assert {r[SubmitRecordKey.SUBMIT_TOKEN] for r in deleted} == {"one", "two"}
    for record in deleted:
        assert record[SubmitRecordKey.STATE] == SubmitRecordState.JOB_DELETED.value
        assert record[SubmitRecordKey.DELETED_BY] == alice
        assert record[SubmitRecordKey.DELETED_TIME].endswith("+00:00")
    assert manager.mark_submit_records_job_deleted("job-one", alice, ctx) == []
    assert manager.mark_submit_records_job_deleted("unknown", alice, ctx) == []
    assert store.get_meta(other_uri)[SubmitRecordKey.STATE] == other[SubmitRecordKey.STATE]
    assert manager.get_submit_record("study", alice, "two", ctx)["note"] == "updated"
