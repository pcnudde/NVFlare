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

from nvflare.apis.fl_constant import SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.apis.storage import WORKSPACE
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage
from nvflare.fuel.utils.zip_utils import zip_directory_to_bytes
from nvflare.lighter.tool_consts import NVFLARE_SUBMISSION_ATTESTATION_FILE
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key, verify_folder_signature
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator


class TestJobManager(unittest.TestCase):
    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.uri_root = tempfile.mkdtemp()
        self.data_folder = os.path.join(dir_path, "../../data/jobs")
        self.job_manager = SimpleJobDefManager(uri_root=self.uri_root)
        self.fl_ctx = FLContext()

    def tearDown(self) -> None:
        shutil.rmtree(self.uri_root)

    def test_create_job(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job()
            content = self.job_manager.get_content(meta, self.fl_ctx)
            assert content == data

    def _create_job(self):
        data = zip_directory_to_bytes(self.data_folder, "valid_job")
        folder_name = "valid_job"
        job_validator = JobMetaValidator()
        valid, error, meta = job_validator.validate(folder_name, data)
        meta = self.job_manager.create(meta, data, self.fl_ctx)
        return data, meta

    def test_save_workspace(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job()
            job_id = meta.get(JobMetaKey.JOB_ID)
            self.job_manager.save_workspace(job_id, data, self.fl_ctx)
            result = self.job_manager.get_storage_component(job_id, WORKSPACE, self.fl_ctx)
            assert result == data

    def test_prepare_uploaded_content_server_attests_token_submission(self):
        workspace_root = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(workspace_root, "startup"), exist_ok=True)
            os.makedirs(os.path.join(workspace_root, "local"), exist_ok=True)
            workspace = Workspace(root_dir=workspace_root, site_name=SiteType.SERVER)
            startup_dir = workspace.get_startup_kit_dir()

            root_pri_key, root_pub_key = generate_keys()
            server_pri_key, server_pub_key = generate_keys()
            root_cert = generate_cert(Identity("root"), Identity("root"), root_pri_key, root_pub_key, ca=True)
            server_cert = generate_cert(Identity("server"), Identity("root"), root_pri_key, server_pub_key)

            with open(os.path.join(startup_dir, "rootCA.pem"), "wb") as f:
                f.write(serialize_cert(root_cert))
            with open(os.path.join(startup_dir, "server.crt"), "wb") as f:
                f.write(serialize_cert(server_cert))
            with open(os.path.join(startup_dir, "server.key"), "wb") as f:
                f.write(serialize_pri_key(server_pri_key))

            job_root = tempfile.mkdtemp()
            try:
                job_folder = os.path.join(job_root, "job_a")
                app_folder = os.path.join(job_folder, "app_server")
                os.makedirs(app_folder, exist_ok=True)
                with open(os.path.join(app_folder, "config_fed_server.json"), "w") as f:
                    f.write("{}")
                uploaded_content = zip_directory_to_bytes(job_root, "job_a")
            finally:
                shutil.rmtree(job_root)

            engine = mock.Mock()
            engine.get_workspace.return_value = workspace
            fl_ctx = mock.Mock()
            fl_ctx.get_engine.return_value = engine

            meta = {
                JobMetaKey.JOB_ID.value: "job-id",
                JobMetaKey.JOB_NAME.value: "job_a",
                JobMetaKey.JOB_FOLDER_NAME.value: "job_a",
                JobMetaKey.DEPLOY_MAP.value: {"app_server": ["server"]},
                JobMetaKey.SUBMITTER_NAME.value: "alice",
                JobMetaKey.SUBMITTER_ORG.value: "org_a",
                JobMetaKey.SUBMITTER_ROLE.value: "lead",
                JobMetaKey.SUBMITTER_AUTH_SOURCE.value: "token",
                JobMetaKey.SUBMIT_TIME.value: 1.0,
                JobMetaKey.SUBMIT_TIME_ISO.value: "2026-03-05T00:00:00+00:00",
            }

            prepared = self.job_manager._prepare_uploaded_content(meta=meta, uploaded_content=uploaded_content, fl_ctx=fl_ctx)
            unpack_dir = tempfile.mkdtemp()
            try:
                from nvflare.fuel.utils.zip_utils import unzip_all_from_bytes

                unzip_all_from_bytes(prepared, unpack_dir)
                app_path = os.path.join(unpack_dir, "job_a", "app_server")
                assert verify_folder_signature(app_path, os.path.join(startup_dir, "rootCA.pem"))
                with open(os.path.join(app_path, NVFLARE_SUBMISSION_ATTESTATION_FILE), "r") as f:
                    attestation = json.load(f)
                assert attestation["submitter_name"] == "alice"
                assert attestation["submitter_auth_source"] == "token"
                assert attestation["app_name"] == "app_server"
            finally:
                shutil.rmtree(unpack_dir)
        finally:
            shutil.rmtree(workspace_root)
