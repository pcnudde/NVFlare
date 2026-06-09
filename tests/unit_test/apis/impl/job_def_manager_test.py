# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
from nvflare.apis.job_def import JobMetaKey, RunStatus, job_from_meta
from nvflare.apis.storage import META, WORKSPACE, StorageException
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage
from nvflare.fuel.utils.zip_utils import zip_directory_to_bytes
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator


class _FakeStateStore:
    def __init__(self):
        self.jobs = {}
        self.list_jobs_calls = 0

    def create_job(self, meta: dict, content_uri: str, content_hash: str = None, content_size: int = None) -> dict:
        job_id = meta[JobMetaKey.JOB_ID.value]
        row = {
            "job_id": job_id,
            "status": meta[JobMetaKey.STATUS.value],
            "content_uri": content_uri,
            "content_hash": content_hash,
            "content_size": content_size,
            "meta_json": dict(meta),
        }
        self.jobs[job_id] = row
        return dict(row)

    def get_job(self, job_id: str):
        row = self.jobs.get(job_id)
        return dict(row) if row else None

    def delete_job(self, job_id: str):
        return self.jobs.pop(job_id, None) is not None

    def list_jobs(self, status=None, study: str = None):
        self.list_jobs_calls += 1
        rows = list(self.jobs.values())
        if status is not None:
            statuses = status if isinstance(status, list) else [status]
            rows = [row for row in rows if row["status"] in statuses]
        if study is not None:
            rows = [row for row in rows if row["meta_json"].get(JobMetaKey.STUDY.value) == study]
        return [dict(row) for row in rows]

    def update_job_meta(self, job_id: str, meta: dict):
        row = self.jobs.get(job_id)
        if not row:
            return None
        row["meta_json"].update(meta)
        row["status"] = row["meta_json"][JobMetaKey.STATUS.value]
        return dict(row)


class _RacingCreateStateStore(_FakeStateStore):
    """Simulates a concurrent winner committing its state row before our create_job fails."""

    def create_job(self, meta: dict, content_uri: str, content_hash: str = None, content_size: int = None) -> dict:
        # the concurrent winner's row lands first, then our insert hits duplicate-key
        super().create_job(meta, content_uri, content_hash, content_size)
        raise RuntimeError("duplicate key")


class _FailingCreateStateStore(_FakeStateStore):
    """create_job fails without committing any row (true orphan scenario)."""

    def create_job(self, meta: dict, content_uri: str, content_hash: str = None, content_size: int = None) -> dict:
        raise RuntimeError("db down")


class _FailingDeleteFilesystemStorage(FilesystemStorage):
    """delete_object fails transiently; the object remains intact."""

    def delete_object(self, uri: str):
        raise StorageException("delete failed")


class _DeleteRaisesObjectGoneStorage(FilesystemStorage):
    """delete_object raises even though the object ends up gone (e.g. already-deleted race)."""

    def delete_object(self, uri: str):
        super().delete_object(uri)
        raise StorageException("transient error, object already gone")


class TestJobManager(unittest.TestCase):
    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.uri_root = tempfile.mkdtemp()
        self.data_folder = os.path.join(dir_path, "../../data/jobs")
        self.state_store = _FakeStateStore()
        self.state_store_patcher = mock.patch.object(
            SimpleJobDefManager, "_get_state_store", return_value=self.state_store
        )
        self.state_store_patcher.start()
        self.job_manager = SimpleJobDefManager(uri_root=self.uri_root)
        self.fl_ctx = FLContext()

    def tearDown(self) -> None:
        self.state_store_patcher.stop()
        shutil.rmtree(self.uri_root)

    def test_create_job(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job()
            content = self.job_manager.get_content(meta, self.fl_ctx)
            assert content == data

    def test_get_app_rejects_escaping_job_folder_name(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            _, meta = self._create_job()
            meta[JobMetaKey.JOB_FOLDER_NAME.value] = "../../outside"
            with self.assertRaisesRegex(ValueError, "job folder.*escapes"):
                self.job_manager.get_app(job_from_meta(meta), "sag", self.fl_ctx)

    def _create_job(self):
        return self._create_job_with_manager(self.job_manager)

    def _create_job_with_manager(self, job_manager):
        data = zip_directory_to_bytes(self.data_folder, "valid_job")
        folder_name = "valid_job"
        job_validator = JobMetaValidator()
        valid, error, meta = job_validator.validate(folder_name, data)
        meta = job_manager.create(meta, data, self.fl_ctx)
        return data, meta

    def test_save_workspace(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job()
            job_id = meta.get(JobMetaKey.JOB_ID)
            self.job_manager.save_workspace(job_id, data, self.fl_ctx)
            result = self.job_manager.get_storage_component(job_id, WORKSPACE, self.fl_ctx)
            assert result == data

    def test_create_rejects_traversing_job_id(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            meta = {JobMetaKey.JOB_ID.value: "../outside"}
            with self.assertRaises(ValueError):
                self.job_manager.create(meta, b"data", self.fl_ctx)

    def test_create_does_not_overwrite_existing_job_id(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job()
            with self.assertRaises(StorageException):
                self.job_manager.create(dict(meta), b"replacement", self.fl_ctx)

            content = self.job_manager.get_content(meta, self.fl_ctx)
            assert content == data

    def test_state_store_backed_manager_uses_db_for_job_metadata(self):
        job_manager = SimpleJobDefManager(uri_root=self.uri_root, state_store_id="state_store")
        with mock.patch.object(job_manager, "_get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job_with_manager(job_manager)
            job_id = meta[JobMetaKey.JOB_ID.value]

            job_manager.update_meta(job_id, {"custom": "db-only"}, self.fl_ctx)
            job = job_manager.get_job(job_id, self.fl_ctx)
            assert job.meta["custom"] == "db-only"
            assert job_manager.get_content(job.meta, self.fl_ctx) == data

            assert [job.job_id for job in job_manager.get_jobs_to_schedule(self.fl_ctx)] == [job_id]

            job_manager.set_status(job_id, RunStatus.RUNNING, self.fl_ctx)
            assert job_manager.get_jobs_to_schedule(self.fl_ctx) == []
            assert [job.job_id for job in job_manager.get_jobs_by_status(RunStatus.RUNNING, self.fl_ctx)] == [job_id]

            job_manager.delete(job_id, self.fl_ctx)
            assert job_manager.get_job(job_id, self.fl_ctx) is None

    def test_meta_download_uses_current_state_store_metadata(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]
            self.job_manager.update_meta(job_id, {"custom": "db-only"}, self.fl_ctx)

            with tempfile.TemporaryDirectory() as download_dir:
                self.job_manager.get_storage_for_download(job_id, download_dir, META, "meta.json", self.fl_ctx)
                meta_path = os.path.join(download_dir, job_id, "meta.json")
                with open(meta_path, "rt", encoding="utf-8") as f:
                    downloaded_meta = json.load(f)

            assert downloaded_meta["custom"] == "db-only"

    def test_delete_reraises_and_keeps_state_row_when_object_still_exists(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            storage = _FailingDeleteFilesystemStorage()
            mock_store.return_value = storage

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]
            with self.assertRaisesRegex(StorageException, "delete failed"):
                self.job_manager.delete(job_id, self.fl_ctx)

            # transient failure with intact object: nothing was deleted
            assert self.state_store.get_job(job_id) is not None
            assert storage.get_meta(self.job_manager.job_uri(job_id)) is not None

    def test_delete_removes_state_row_when_delete_fails_but_object_is_gone(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = _DeleteRaisesObjectGoneStorage()

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]
            self.job_manager.delete(job_id, self.fl_ctx)

            assert self.state_store.get_job(job_id) is None

    def test_delete_with_missing_storage_object_removes_state_row(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            storage = FilesystemStorage()
            mock_store.return_value = storage

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]
            # simulate a previously half-completed delete: storage object gone, state row left
            storage.delete_object(self.job_manager.job_uri(job_id))

            self.job_manager.delete(job_id, self.fl_ctx)

            assert self.state_store.get_job(job_id) is None
            assert self.job_manager.get_job(job_id, self.fl_ctx) is None

    def test_set_status_writes_through_to_storage_meta(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            storage = FilesystemStorage()
            mock_store.return_value = storage

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]

            self.job_manager.set_status(job_id, RunStatus.RUNNING, self.fl_ctx)
            storage_meta = storage.get_meta(self.job_manager.job_uri(job_id))
            assert storage_meta[JobMetaKey.STATUS.value] == RunStatus.RUNNING.value
            assert storage_meta[JobMetaKey.START_TIME.value]

            self.job_manager.set_status(job_id, RunStatus.FINISHED_COMPLETED, self.fl_ctx)
            storage_meta = storage.get_meta(self.job_manager.job_uri(job_id))
            assert storage_meta[JobMetaKey.STATUS.value] == RunStatus.FINISHED_COMPLETED.value

    def test_update_meta_writes_through_to_storage_meta(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            storage = FilesystemStorage()
            mock_store.return_value = storage

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]

            self.job_manager.update_meta(job_id, {"custom": "mirrored"}, self.fl_ctx)
            storage_meta = storage.get_meta(self.job_manager.job_uri(job_id))
            assert storage_meta["custom"] == "mirrored"

    def test_update_meta_succeeds_when_storage_write_fails(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            storage = FilesystemStorage()
            mock_store.return_value = storage

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]

            with mock.patch.object(storage, "update_meta", side_effect=StorageException("storage down")):
                self.job_manager.update_meta(job_id, {"custom": "db-only"}, self.fl_ctx)

            # state store (authoritative) was still updated
            assert self.job_manager.get_job(job_id, self.fl_ctx).meta["custom"] == "db-only"

    def test_create_retries_over_orphan_storage_object(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            storage = FilesystemStorage()
            mock_store.return_value = storage

            # simulate a crash between create_object and create_job: storage
            # object exists for the reserved job_id, but there is no state row
            job_id = "orphaned-job-id"
            storage.create_object(self.job_manager.job_uri(job_id), b"stale", {"status": "SUBMITTED"})
            assert self.state_store.get_job(job_id) is None

            data = zip_directory_to_bytes(self.data_folder, "valid_job")
            valid, error, meta = JobMetaValidator().validate("valid_job", data)
            meta[JobMetaKey.JOB_ID.value] = job_id
            meta = self.job_manager.create(meta, data, self.fl_ctx)

            assert self.state_store.get_job(job_id) is not None
            assert self.job_manager.get_content(meta, self.fl_ctx) == data

    def test_create_still_rejects_genuine_duplicate(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job()
            with self.assertRaises(StorageException):
                self.job_manager.create(dict(meta), b"replacement", self.fl_ctx)

    def test_create_cleanup_keeps_object_when_concurrent_create_won(self):
        # Race: A's create_object succeeded but its create_job has not committed
        # yet; our (B's) create sees the object, finds no state row, takes the
        # orphan branch and overwrites; A then commits; B's create_job fails
        # duplicate-key. B's cleanup must NOT delete the object behind A's row.
        racing_store = _RacingCreateStateStore()
        with mock.patch.object(SimpleJobDefManager, "_get_state_store", return_value=racing_store):
            with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
                storage = FilesystemStorage()
                mock_store.return_value = storage

                job_id = "raced-job-id"
                job_uri = self.job_manager.job_uri(job_id)
                # A's in-flight storage object (no state row yet)
                storage.create_object(job_uri, b"in-flight", {"status": "SUBMITTED"})

                data = zip_directory_to_bytes(self.data_folder, "valid_job")
                valid, error, meta = JobMetaValidator().validate("valid_job", data)
                meta[JobMetaKey.JOB_ID.value] = job_id
                with self.assertRaisesRegex(RuntimeError, "duplicate key"):
                    self.job_manager.create(meta, data, self.fl_ctx)

                # the winner's state row exists and its content was not deleted
                assert racing_store.get_job(job_id) is not None
                assert storage.get_data(job_uri) == data

    def test_create_cleanup_removes_true_orphan_object(self):
        # create_job fails without any state row landing: the object is a true
        # orphan from our own failed attempt and must be cleaned up.
        failing_store = _FailingCreateStateStore()
        with mock.patch.object(SimpleJobDefManager, "_get_state_store", return_value=failing_store):
            with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
                storage = FilesystemStorage()
                mock_store.return_value = storage

                data = zip_directory_to_bytes(self.data_folder, "valid_job")
                valid, error, meta = JobMetaValidator().validate("valid_job", data)
                with self.assertRaisesRegex(RuntimeError, "db down"):
                    self.job_manager.create(meta, data, self.fl_ctx)

                job_id = meta[JobMetaKey.JOB_ID.value]
                assert failing_store.get_job(job_id) is None
                with self.assertRaises(StorageException):
                    storage.get_meta(self.job_manager.job_uri(job_id))

    def test_set_status_mirror_writes_serialize_per_job(self):
        # Two concurrent set_status calls: the per-job lock must hold across
        # [state-store update + mirror write] so the mirror cannot end up with
        # a staler status than the state store.
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            storage = FilesystemStorage()
            mock_store.return_value = storage

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]

            running_in_mirror = threading.Event()
            release = threading.Event()
            orig_update_meta = storage.update_meta

            def slow_update_meta(uri, meta, replace=False):
                if meta.get(JobMetaKey.STATUS.value) == RunStatus.RUNNING.value:
                    running_in_mirror.set()
                    assert release.wait(timeout=10)
                return orig_update_meta(uri=uri, meta=meta, replace=replace)

            with mock.patch.object(storage, "update_meta", side_effect=slow_update_meta):
                t1 = threading.Thread(target=self.job_manager.set_status, args=(job_id, RunStatus.RUNNING, self.fl_ctx))
                t1.start()
                assert running_in_mirror.wait(timeout=10)

                t2 = threading.Thread(
                    target=self.job_manager.set_status, args=(job_id, RunStatus.FINISHED_COMPLETED, self.fl_ctx)
                )
                t2.start()
                time.sleep(0.2)
                # t2 must be blocked on the per-job lock: the state store still
                # holds t1's status while t1 is inside the mirror write
                assert self.state_store.get_job(job_id)["status"] == RunStatus.RUNNING.value

                release.set()
                t1.join(timeout=10)
                t2.join(timeout=10)
                assert not t1.is_alive() and not t2.is_alive()

            # final mirror status matches final state-store status
            assert self.state_store.get_job(job_id)["status"] == RunStatus.FINISHED_COMPLETED.value
            storage_meta = storage.get_meta(self.job_manager.job_uri(job_id))
            assert storage_meta[JobMetaKey.STATUS.value] == RunStatus.FINISHED_COMPLETED.value

    def test_get_jobs_by_status_uses_single_list_jobs_query(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]

            self.state_store.list_jobs_calls = 0
            jobs = self.job_manager.get_jobs_by_status([RunStatus.SUBMITTED, RunStatus.RUNNING], self.fl_ctx)

            assert self.state_store.list_jobs_calls == 1
            assert [job.job_id for job in jobs] == [job_id]
