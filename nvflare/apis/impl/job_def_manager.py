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
import datetime
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.fl_constant import SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import (
    Job,
    JobDataKey,
    JobMetaKey,
    SubmitRecordKey,
    SubmitRecordState,
    job_from_meta,
    new_job_id,
)
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec, RunStatus
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.state_store import StateStore
from nvflare.apis.storage import META, WORKSPACE, StorageException, StorageSpec
from nvflare.apis.utils.format_check import check_job_app_name, check_job_id
from nvflare.apis.utils.job_submit_token import canonical_job_content_hash, submitter_to_dict
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.zip_utils import unzip_all_from_bytes, zip_directory_to_bytes


class SimpleJobDefManager(JobDefManagerSpec):
    def __init__(
        self,
        uri_root: str = "jobs",
        job_store_id: str = "job_store",
        state_store_id: str = SystemComponents.STATE_STORE,
    ):
        super().__init__()
        self.uri_root = uri_root

        # if env var is defined, use it to override uri_root!
        job_store_root = os.environ.get("NVFL_JOB_STORE_ROOT")
        if job_store_root:
            self.uri_root = job_store_root

        os.makedirs(uri_root, exist_ok=True)
        self.job_store_id = job_store_id
        self.state_store_id = state_store_id

    def _get_job_store(self, fl_ctx):
        engine = fl_ctx.get_engine()

        if not (isinstance(engine, ServerEngineSpec) or isinstance(engine, ClientEngineSpec)):
            raise TypeError(f"engine should be of type ServerEngineSpec or ClientEngineSpec, but got {type(engine)}")
        store = engine.get_component(self.job_store_id)
        if not isinstance(store, StorageSpec):
            raise TypeError(f"engine should have a job store component of type StorageSpec, but got {type(store)}")
        return store

    def _get_state_store(self, fl_ctx) -> StateStore:
        engine = fl_ctx.get_engine()
        if not engine or not hasattr(engine, "get_component"):
            raise TypeError("fl_ctx engine must provide get_component for job metadata")
        store = engine.get_component(self.state_store_id)
        if not isinstance(store, StateStore):
            raise TypeError(
                f"engine should have a state store component '{self.state_store_id}' "
                f"of type StateStore, but got {type(store)}"
            )
        return store

    def job_uri(self, jid: str):
        check_job_id(jid)
        return os.path.join(self.uri_root, jid)

    @staticmethod
    def _content_size(uploaded_content: Union[str, bytes]) -> Optional[int]:
        if isinstance(uploaded_content, bytes):
            return len(uploaded_content)
        if isinstance(uploaded_content, str) and os.path.isfile(uploaded_content):
            return os.path.getsize(uploaded_content)
        return None

    @staticmethod
    def _job_from_state_row(row: dict) -> Optional[Job]:
        if not row:
            return None
        return job_from_meta(row.get("meta_json") or {})

    @staticmethod
    def _status_value(status):
        return getattr(status, "value", status)

    @staticmethod
    def _update_state_job_meta(state_store: StateStore, jid: str, meta: dict):
        updated = state_store.update_job_meta(jid, meta)
        if updated is None:
            raise StorageException(f"job '{jid}' is missing from state store")
        return updated

    def get_job_content_hash(self, uploaded_content: Union[str, bytes]) -> str:
        return canonical_job_content_hash(uploaded_content)

    def get_submit_record(self, study: str, submitter, submit_token: str, fl_ctx: FLContext) -> Optional[dict]:
        return self._get_state_store(fl_ctx).get_submit_record(study, submitter, submit_token)

    def create_submit_record(self, record: dict, fl_ctx: FLContext) -> bool:
        return self._get_state_store(fl_ctx).create_submit_record(record)

    def update_submit_record(self, record: dict, fl_ctx: FLContext) -> dict:
        return self._get_state_store(fl_ctx).update_submit_record(record)

    def mark_submit_records_job_deleted(self, job_id: str, deleted_by, fl_ctx: FLContext) -> List[dict]:
        return self._get_state_store(fl_ctx).mark_submit_records_job_deleted(job_id, deleted_by)

    def get_job_by_submit_token(self, study: str, submitter, submit_token: str, fl_ctx: FLContext) -> Optional[Job]:
        record = self.get_submit_record(study, submitter, submit_token, fl_ctx)
        if not record:
            return None
        jid = record.get(SubmitRecordKey.JOB_ID.value)
        if not jid:
            return None
        return self.get_job(jid, fl_ctx)

    @staticmethod
    def new_submit_record(
        study: str,
        submitter,
        submit_token: str,
        job_content_hash: str,
        job_name: str = "",
        job_folder_name: str = "",
        job_id: str = None,
        state: str = SubmitRecordState.CREATING.value,
    ) -> dict:
        submitter_info = submitter_to_dict(submitter)
        return {
            SubmitRecordKey.SCHEMA_VERSION.value: 1,
            SubmitRecordKey.STATE.value: state,
            SubmitRecordKey.SUBMIT_TOKEN.value: submit_token,
            SubmitRecordKey.JOB_ID.value: job_id or new_job_id(),
            SubmitRecordKey.STUDY.value: study,
            SubmitRecordKey.SUBMITTER_NAME.value: submitter_info["name"],
            SubmitRecordKey.SUBMITTER_ORG.value: submitter_info["org"],
            SubmitRecordKey.SUBMITTER_ROLE.value: submitter_info["role"],
            SubmitRecordKey.JOB_NAME.value: job_name,
            SubmitRecordKey.JOB_FOLDER_NAME.value: job_folder_name,
            SubmitRecordKey.JOB_CONTENT_HASH.value: job_content_hash,
            SubmitRecordKey.SUBMIT_TIME.value: datetime.datetime.now().astimezone().isoformat(),
        }

    def create(self, meta: dict, uploaded_content: Union[str, bytes], fl_ctx: FLContext) -> Dict[str, Any]:
        meta.pop(SubmitRecordKey.SUBMIT_TOKEN.value, None)
        # validate meta to make sure it has:
        jid = meta.get(JobMetaKey.JOB_ID.value, None)
        if not jid:
            jid = new_job_id()
            meta[JobMetaKey.JOB_ID.value] = jid
        else:
            check_job_id(jid)

        now = time.time()
        meta[JobMetaKey.SUBMIT_TIME.value] = now
        meta[JobMetaKey.SUBMIT_TIME_ISO.value] = datetime.datetime.fromtimestamp(now).astimezone().isoformat()
        meta[JobMetaKey.START_TIME.value] = ""
        meta[JobMetaKey.DURATION.value] = "N/A"
        meta[JobMetaKey.DATA_STORAGE_FORMAT.value] = 2
        meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value

        state_store = self._get_state_store(fl_ctx)
        store = self._get_job_store(fl_ctx)
        job_uri = self.job_uri(jid)
        store.create_object(job_uri, uploaded_content, meta, overwrite_existing=False)
        try:
            state_store.create_job(meta, content_uri=job_uri, content_size=self._content_size(uploaded_content))
        except Exception:
            store.delete_object(job_uri)
            raise
        return meta

    def clone(self, from_jid: str, meta: dict, fl_ctx: FLContext) -> Dict[str, Any]:
        check_job_id(from_jid)
        jid = meta.get(JobMetaKey.JOB_ID.value, None)
        if not jid:
            jid = new_job_id()
            meta[JobMetaKey.JOB_ID.value] = jid
        else:
            check_job_id(jid)

        now = time.time()
        meta[JobMetaKey.SUBMIT_TIME.value] = now
        meta[JobMetaKey.SUBMIT_TIME_ISO.value] = datetime.datetime.fromtimestamp(now).astimezone().isoformat()
        meta[JobMetaKey.START_TIME.value] = ""
        meta[JobMetaKey.DURATION.value] = "N/A"
        meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value

        # write it to the store
        store = self._get_job_store(fl_ctx)
        state_store = self._get_state_store(fl_ctx)
        source_row = state_store.get_job(from_jid)
        if not source_row:
            raise RuntimeError(f"source job '{from_jid}' is missing from state store")

        job_uri = self.job_uri(jid)
        store.clone_object(from_uri=self.job_uri(from_jid), to_uri=job_uri, meta=meta, overwrite_existing=False)
        try:
            state_store.create_job(
                meta,
                content_uri=job_uri,
                content_hash=source_row.get("content_hash"),
                content_size=source_row.get("content_size"),
            )
        except Exception:
            store.delete_object(job_uri)
            raise
        return meta

    def delete(self, jid: str, fl_ctx: FLContext):
        state_store = self._get_state_store(fl_ctx)
        store = self._get_job_store(fl_ctx)
        store.delete_object(self.job_uri(jid))
        state_store.delete_job(jid)

    def _validate_meta(self, meta):
        """Validate meta

        Args:
            meta: meta to validate

        Returns:

        """
        pass

    def _validate_uploaded_content(self, uploaded_content) -> bool:
        """Validate uploaded content for creating a run config. (THIS NEEDS TO HAPPEN BEFORE CONTENT IS PROVIDED NOW)

        Internally used by create and update.

        1. check all sites in deployment are in resources
        2. each site in deployment need to have resources (each site in resource need to be in deployment ???)
        """
        pass

    def get_job(self, jid: str, fl_ctx: FLContext) -> Optional[Job]:
        return self._job_from_state_row(self._get_state_store(fl_ctx).get_job(jid))

    def set_results_uri(self, jid: str, result_uri: str, fl_ctx: FLContext):
        updated_meta = {JobMetaKey.RESULT_LOCATION.value: result_uri}
        self.update_meta(jid, updated_meta, fl_ctx)
        return self.get_job(jid, fl_ctx)

    def get_app(self, job: Job, app_name: str, fl_ctx: FLContext) -> bytes:
        check_job_id(job.job_id)
        check_job_app_name(app_name)
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id_dir = self._load_job_data_from_store(job, temp_dir, fl_ctx)
            job_folder = os.path.join(job_id_dir, job.meta[JobMetaKey.JOB_FOLDER_NAME.value])
            fullpath_src = os.path.join(job_folder, app_name)
            job_id_dir_real = os.path.realpath(job_id_dir)
            job_folder_real = os.path.realpath(job_folder)
            fullpath_src_real = os.path.realpath(fullpath_src)
            if os.path.commonpath([job_id_dir_real, job_folder_real]) != job_id_dir_real:
                raise ValueError(f"job folder for app '{app_name}' escapes job data folder")
            if os.path.commonpath([job_folder_real, fullpath_src_real]) != job_folder_real:
                raise ValueError(f"app '{app_name}' escapes job folder")
            result = zip_directory_to_bytes(fullpath_src_real, "")
        return result

    def _load_job_data_from_store(self, job: Job, temp_dir: str, fl_ctx: FLContext):
        check_job_id(job.job_id)
        data_bytes = self.get_content(job.meta, fl_ctx)
        job_id_dir = os.path.join(temp_dir, job.job_id)
        if os.path.exists(job_id_dir):
            shutil.rmtree(job_id_dir)
        os.mkdir(job_id_dir)
        unzip_all_from_bytes(data_bytes, job_id_dir)
        return job_id_dir

    def get_content(self, meta: dict, fl_ctx: FLContext) -> Optional[bytes]:
        store = self._get_job_store(fl_ctx)
        jid = meta.get(JobMetaKey.JOB_ID.value)
        if not jid:
            raise RuntimeError("no Job ID in meta")

        try:
            stored_data = store.get_data(self.job_uri(jid))
            storage_format = meta.get(JobMetaKey.DATA_STORAGE_FORMAT.value)
            if storage_format:
                # new format
                return stored_data
            else:
                # old format
                return fobs.loads(stored_data).get(JobDataKey.JOB_DATA.value)
        except StorageException:
            return None

    def set_client_data(self, jid: str, data: Union[bytes, str], client_name: str, data_type: str, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        data_object_type = f"{data_type}_{client_name}"
        store.update_object(self.job_uri(jid), data, data_object_type)

    def get_client_data(self, jid: str, client_name: str, data_type: str, fl_ctx: FLContext) -> Optional[bytes]:
        store = self._get_job_store(fl_ctx)
        data_object_type = f"{data_type}_{client_name}"
        try:
            data_data = store.get_data(self.job_uri(jid), data_object_type)
            return data_data
        except StorageException:
            return None

    def list_components(self, jid: str, fl_ctx: FLContext) -> List[str]:
        store = self._get_job_store(fl_ctx)
        self.log_debug(
            fl_ctx, f"list_components called for {jid}: {store.list_components_of_object(self.job_uri(jid))}"
        )
        return store.list_components_of_object(self.job_uri(jid))

    def set_status(self, jid: str, status: RunStatus, fl_ctx: FLContext):
        status_value = self._status_value(status)
        meta = {JobMetaKey.STATUS.value: status_value}
        state_store = self._get_state_store(fl_ctx)
        if status_value == RunStatus.RUNNING.value:
            meta[JobMetaKey.START_TIME.value] = str(datetime.datetime.now())
        elif status_value in [
            RunStatus.FINISHED_ABORTED.value,
            RunStatus.FINISHED_COMPLETED.value,
            RunStatus.FINISHED_EXECUTION_EXCEPTION.value,
            RunStatus.FINISHED_CANT_SCHEDULE.value,
        ]:
            row = state_store.get_job(jid)
            job_meta = (row or {}).get("meta_json") or {}
            if job_meta.get(JobMetaKey.START_TIME.value):
                start_time = datetime.datetime.strptime(
                    job_meta.get(JobMetaKey.START_TIME.value), "%Y-%m-%d %H:%M:%S.%f"
                )
                meta[JobMetaKey.DURATION.value] = str(datetime.datetime.now() - start_time)
        self._update_state_job_meta(state_store, jid, meta)

    def update_meta(self, jid: str, meta, fl_ctx: FLContext):
        self._update_state_job_meta(self._get_state_store(fl_ctx), jid, meta)

    def refresh_meta(self, job: Job, meta_keys: list, fl_ctx: FLContext):
        """Refresh meta of the job as specified in the meta keys
        Save the values of the specified keys into job store

        Args:
            job: job object
            meta_keys: meta keys need to updated
            fl_ctx: FLContext

        """
        if meta_keys:
            meta = {}
            for k in meta_keys:
                if k in job.meta:
                    meta[k] = job.meta[k]
        else:
            meta = job.meta
        if meta:
            self.update_meta(job.job_id, meta, fl_ctx)

    def get_all_jobs(self, fl_ctx: FLContext) -> List[Job]:
        return [self._job_from_state_row(row) for row in self._get_state_store(fl_ctx).list_jobs()]

    def get_jobs_to_schedule(self, fl_ctx: FLContext) -> List[Job]:
        return [
            self._job_from_state_row(row)
            for row in self._get_state_store(fl_ctx).list_jobs(status=RunStatus.SUBMITTED.value)
        ]

    def get_jobs_by_status(self, status: Union[RunStatus, List[RunStatus]], fl_ctx: FLContext) -> List[Job]:
        """Get jobs that are in the specified status

        Args:
            status: a single status value or a list of status values
            fl_ctx: the FL context

        Returns: list of jobs that are in specified status

        """
        if not isinstance(status, list):
            status = [status]
        result = []
        state_store = self._get_state_store(fl_ctx)
        for run_status in status:
            rows = state_store.list_jobs(status=self._status_value(run_status))
            result.extend(self._job_from_state_row(row) for row in rows)
        return result

    def get_jobs_waiting_for_review(self, reviewer_name: str, fl_ctx: FLContext) -> List[Job]:
        result = []
        for row in self._get_state_store(fl_ctx).list_jobs():
            meta = row.get("meta_json") or {}
            approvals = meta.get(JobMetaKey.APPROVALS.value)
            if not approvals or reviewer_name not in approvals:
                result.append(job_from_meta(meta))
        return result

    def set_approval(
        self, jid: str, reviewer_name: str, approved: bool, note: str, fl_ctx: FLContext
    ) -> Dict[str, Any]:
        meta = self.get_job(jid, fl_ctx).meta
        if meta:
            approvals = meta.get(JobMetaKey.APPROVALS.value)
            if not approvals:
                approvals = {}
                meta[JobMetaKey.APPROVALS.value] = approvals
            approvals[reviewer_name] = (approved, note)
            updated_meta = {JobMetaKey.APPROVALS.value: approvals}
            self.update_meta(jid, updated_meta, fl_ctx)
        return meta

    def save_workspace(self, jid: str, data: Union[bytes, str, List[str]], fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        return store.update_object(self.job_uri(jid), data, WORKSPACE)

    def get_storage_component(self, jid: str, component: str, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        return store.get_data(self.job_uri(jid), component)

    def get_storage_for_download(
        self, jid: str, download_dir: str, component: str, download_file: str, fl_ctx: FLContext
    ):
        """Prepares the specified component of the job for download at the specified directory

        The component is prepared for download at download_dir/jid/download_file.

        Args:
            jid: job ID
            download_dir: directory to download the component to
            component: component name
            download_file: file name to save the downloaded component
            fl_ctx: FLContext
        """
        store = self._get_job_store(fl_ctx)
        job_uri = self.job_uri(jid)
        os.makedirs(os.path.join(download_dir, jid), exist_ok=True)
        destination_file = os.path.join(download_dir, jid, download_file)
        if component == META:
            row = self._get_state_store(fl_ctx).get_job(jid)
            if not row:
                raise StorageException(f"job '{jid}' is missing from state store")
            with open(destination_file, "wt", encoding="utf-8") as f:
                json.dump(row.get("meta_json") or {}, f, indent=2, sort_keys=True)
            return
        store.get_data_for_download(job_uri, component, destination_file)
