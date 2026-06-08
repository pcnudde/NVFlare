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

    def list_jobs(self, status: str = None, study: str = None):
        rows = list(self.jobs.values())
        if status is not None:
            rows = [row for row in rows if row["status"] == status]
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


class _FailingDeleteFilesystemStorage(FilesystemStorage):
    def delete_object(self, uri: str):
        raise StorageException("delete failed")


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

    def test_delete_keeps_state_row_when_object_delete_fails(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = _FailingDeleteFilesystemStorage()

            _, meta = self._create_job()
            job_id = meta[JobMetaKey.JOB_ID.value]
            with self.assertRaises(StorageException):
                self.job_manager.delete(job_id, self.fl_ctx)

            assert self.state_store.get_job(job_id) is not None
