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

import io
import zipfile
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl import job_def_manager as subject
from nvflare.apis.job_def import JobDataKey, JobMetaKey, SubmitRecordKey, job_from_meta
from nvflare.apis.job_def_manager_spec import RunStatus
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.storage import StorageException, StorageSpec
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage
from nvflare.fuel.utils import fobs


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.delenv("NVFL_JOB_STORE_ROOT", raising=False)
    manager = subject.SimpleJobDefManager(str(tmp_path / "jobs"))
    store = FilesystemStorage(str(tmp_path / "store"))
    engine = Mock(spec=ServerEngineSpec)
    engine.get_component.return_value = store
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    return manager, store, ctx


def create(manager, ctx, **extra):
    return manager.create({JobMetaKey.JOB_FOLDER_NAME.value: "demo", **extra}, b"archive", ctx)


def test_scan_filters_status_review_and_previously_scheduled_jobs(storage):
    manager, store, ctx = storage
    first = create(manager, ctx)
    second = create(manager, ctx)
    jid = first[JobMetaKey.JOB_ID]
    manager.set_status(jid, RunStatus.RUNNING, ctx)
    manager.set_approval(jid, "reviewer", True, "approved", ctx)
    assert {j.job_id for j in manager.get_all_jobs(ctx)} == {jid, second[JobMetaKey.JOB_ID]}
    assert [j.job_id for j in manager.get_jobs_by_status(RunStatus.RUNNING, ctx)] == [jid]
    assert len(manager.get_jobs_by_status([RunStatus.RUNNING, RunStatus.SUBMITTED], ctx)) == 2
    assert [j.job_id for j in manager.get_jobs_waiting_for_review("reviewer", ctx)] == [second[JobMetaKey.JOB_ID]]
    assert [j.job_id for j in manager.get_jobs_to_schedule(ctx)] == [second[JobMetaKey.JOB_ID]]
    assert jid not in [j.job_id for j in manager.get_jobs_to_schedule(ctx)]
    assert store.get_meta(manager.job_uri(jid))[JobMetaKey.APPROVALS] == {"reviewer": [True, "approved"]}


def test_clone_preserves_content_resets_runtime_and_can_be_deleted(storage):
    manager, _, ctx = storage
    original = create(manager, ctx)
    meta = manager.clone(original[JobMetaKey.JOB_ID], {"name": "copy"}, ctx)
    jid = meta[JobMetaKey.JOB_ID]
    assert jid != original[JobMetaKey.JOB_ID]
    assert meta[JobMetaKey.STATUS] == RunStatus.SUBMITTED.value
    assert meta[JobMetaKey.START_TIME] == "" and meta[JobMetaKey.DURATION] == "N/A"
    assert manager.get_content({**meta, JobMetaKey.DATA_STORAGE_FORMAT.value: 2}, ctx) == b"archive"
    manager.delete(jid, ctx)
    assert manager.get_job(jid, ctx) is None


def test_component_roundtrip_download_and_missing_component(storage, tmp_path):
    manager, _, ctx = storage
    meta = create(manager, ctx)
    jid = meta[JobMetaKey.JOB_ID]
    manager.set_client_data(jid, b"trace", "site1", "LOG", ctx)
    assert manager.get_client_data(jid, "site1", "LOG", ctx) == b"trace"
    assert manager.get_client_data(jid, "absent", "LOG", ctx) is None
    assert "LOG_site1" in manager.list_components(jid, ctx)
    target = tmp_path / "downloads"
    manager.get_storage_for_download(jid, str(target), "LOG_site1", "client.log", ctx)
    assert (target / jid / "client.log").read_bytes() == b"trace"


def test_partial_refresh_and_results_uri_preserve_other_metadata(storage):
    manager, _, ctx = storage
    meta = create(manager, ctx, name="original")
    jid = meta[JobMetaKey.JOB_ID]
    job = job_from_meta({**meta, "name": "updated", "transient": "not saved"})
    manager.refresh_meta(job, ["name", "missing"], ctx)
    result = manager.set_results_uri(jid, "results/run1", ctx)
    assert result.meta["name"] == "updated"
    assert "transient" not in result.meta
    assert result.meta[JobMetaKey.RESULT_LOCATION] == "results/run1"
    manager.refresh_meta(job, None, ctx)
    assert manager.get_job(jid, ctx).meta["transient"] == "not saved"


def test_legacy_content_decodes_container_and_missing_job_is_optional(storage):
    manager, store, ctx = storage
    meta = create(manager, ctx)
    meta.pop(JobMetaKey.DATA_STORAGE_FORMAT.value)
    store.update_object(manager.job_uri(meta[JobMetaKey.JOB_ID]), fobs.dumps({JobDataKey.JOB_DATA.value: b"old"}))
    assert manager.get_content(meta, ctx) == b"old"
    manager.delete(meta[JobMetaKey.JOB_ID], ctx)
    assert manager.get_content(meta, ctx) is None
    with pytest.raises(RuntimeError, match="no Job ID"):
        manager.get_content({}, ctx)


def test_app_export_contains_only_requested_application(storage):
    manager, _, ctx = storage
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("demo/app/config.json", '{"value": 3}')
        output.writestr("demo/other/secret.txt", "not this app")
    meta = manager.create({JobMetaKey.JOB_FOLDER_NAME.value: "demo"}, archive.getvalue(), ctx)
    result = manager.get_app(job_from_meta(meta), "app", ctx)
    with zipfile.ZipFile(io.BytesIO(result)) as application:
        assert application.namelist() == ["config.json"]
        assert application.read("config.json") == b'{"value": 3}'


def test_invalid_engine_or_storage_is_rejected_before_job_access(storage):
    manager, _, ctx = storage
    ctx.set_prop(ReservedKey.ENGINE, object())
    with pytest.raises(TypeError, match="engine should be"):
        manager.get_all_jobs(ctx)
    engine = Mock(spec=ServerEngineSpec)
    engine.get_component.return_value = object()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    with pytest.raises(TypeError, match="StorageSpec"):
        manager.get_all_jobs(ctx)


def test_scan_stops_when_filter_declines_and_skips_empty_metadata(storage, monkeypatch):
    manager, _, ctx = storage
    store = Mock(spec=StorageSpec)
    store.list_objects.return_value = [manager.job_uri("job-one"), manager.job_uri("job-two")]
    store.get_meta.side_effect = [{}, {"name": "second"}]
    monkeypatch.setattr(manager, "_get_job_store", lambda _: store)
    job_filter = Mock()
    job_filter.filter_job.return_value = False
    manager._scan(job_filter, ctx)
    assert job_filter.filter_job.call_count == 1
    assert job_filter.filter_job.call_args.args[0].job_id == "job-two"
    store.list_objects.return_value = []
    manager._scan(job_filter, ctx)
    assert job_filter.filter_job.call_count == 1


def test_submit_record_conflict_propagates_when_existing_record_cannot_be_read(storage, monkeypatch):
    manager, _, ctx = storage
    store = Mock(spec=StorageSpec)
    store.create_object.side_effect = StorageException("write failed")
    store.get_meta.side_effect = StorageException("read failed")
    monkeypatch.setattr(manager, "_get_job_store", lambda _: store)
    record = manager.new_submit_record("study", {"name": "alice"}, "token", "hash")
    with pytest.raises(StorageException, match="write failed"):
        manager.create_submit_record(record, ctx)
    store.update_meta.assert_not_called()


def test_index_creation_race_merges_both_submit_record_uris(storage, monkeypatch):
    manager, _, ctx = storage
    store = Mock(spec=StorageSpec)
    store.get_meta.side_effect = [None, {"submit_record_uris": ["existing-record"]}]
    store.create_object.side_effect = StorageException("concurrent creator")
    record = manager.new_submit_record("study", {"name": "alice"}, "token", "hash", job_id="job-one")
    manager._upsert_submit_record_job_index(store, record)
    updated = store.update_meta.call_args.args[1]
    assert updated[SubmitRecordKey.JOB_ID] == "job-one"
    assert updated["submit_record_uris"] == ["existing-record", manager._submit_record_uri_from_record(record)]
