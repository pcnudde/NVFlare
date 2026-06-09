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

from copy import deepcopy
from typing import Dict


class LegacyStudyRegistry:
    """Reader for the legacy study_registry.json format.

    Only validates the registry envelope (format_version + studies mapping) and returns the
    raw study definitions; per-study validation is delegated to study_store.normalize_study
    by the migration code.
    """

    FORMAT_VERSION = "1.0"

    def __init__(self, studies_config: dict):
        if not isinstance(studies_config, dict):
            raise ValueError(f"studies_config must be dict but got {type(studies_config)}")

        format_version = studies_config.get("format_version")
        if format_version != self.FORMAT_VERSION:
            raise ValueError(f"missing or invalid study registry format_version: must be {self.FORMAT_VERSION}")

        studies = studies_config.get("studies")
        if not isinstance(studies, dict):
            raise ValueError(f"study registry 'studies' must be dict but got {type(studies)}")

        self._studies = {study_name: dict(study_def or {}) for study_name, study_def in studies.items()}

    def get_studies(self) -> Dict[str, dict]:
        return deepcopy(self._studies)
