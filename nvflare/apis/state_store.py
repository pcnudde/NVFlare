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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StateStore(ABC):
    """Minimal state-store API for durable server state records.

    This API intentionally stores large bytes elsewhere. Job packages,
    workspaces, logs, checkpoints, and snapshots should be referenced by
    artifact URIs, not stored in the state DB.
    """

    @abstractmethod
    def initialize(self):
        """Validate that the backing store is ready for runtime use."""
        pass

    @abstractmethod
    def upsert_study(self, name: str, config: dict) -> dict:
        pass

    @abstractmethod
    def get_study(self, name: str) -> Optional[dict]:
        pass

    @abstractmethod
    def list_studies(self) -> List[dict]:
        pass

    @abstractmethod
    def delete_study(self, name: str) -> bool:
        pass

    @abstractmethod
    def add_study_sites(self, name: str, site_orgs: Dict[str, List[str]]) -> dict:
        pass

    @abstractmethod
    def remove_study_sites(self, name: str, site_orgs: Dict[str, List[str]]) -> dict:
        pass

    @abstractmethod
    def add_study_admin(self, name: str, user: str) -> dict:
        pass

    @abstractmethod
    def remove_study_admin(self, name: str, user: str) -> dict:
        pass

    @abstractmethod
    def create_job(self, meta: dict, content_uri: str, content_hash: str = None, content_size: int = None) -> dict:
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[dict]:
        pass

    @abstractmethod
    def delete_job(self, job_id: str) -> bool:
        pass

    @abstractmethod
    def list_jobs(self, status: str = None, study: str = None) -> List[dict]:
        pass

    @abstractmethod
    def update_job_meta(self, job_id: str, meta: dict) -> dict:
        pass

    @abstractmethod
    def set_job_status(self, job_id: str, status: str) -> dict:
        pass

    @abstractmethod
    def create_submit_record(self, record: dict) -> bool:
        pass

    @abstractmethod
    def get_submit_record(self, study: str, submitter: Any, submit_token: str) -> Optional[dict]:
        pass

    @abstractmethod
    def update_submit_record(self, record: dict) -> dict:
        pass

    @abstractmethod
    def mark_submit_records_job_deleted(self, job_id: str, deleted_by: Any) -> List[dict]:
        pass

    @abstractmethod
    def disable_client(self, client_name: str, disabled_by: str = None, reason: str = None) -> dict:
        pass

    @abstractmethod
    def get_disabled_client(self, client_name: str) -> Optional[dict]:
        pass

    @abstractmethod
    def enable_client(self, client_name: str) -> bool:
        pass

    @abstractmethod
    def list_disabled_clients(self) -> List[dict]:
        pass

    @abstractmethod
    def get_migration_marker(self, name: str) -> Optional[dict]:
        pass
