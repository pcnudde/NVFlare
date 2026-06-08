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

import pytest

from nvflare.apis import study_store
from nvflare.apis.job_def import DEFAULT_STUDY


class _FakeStudyStore:
    def __init__(self, studies=None):
        self.studies = studies or {}
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def get_study(self, name):
        study_def = self.studies.get(name)
        if study_def is None:
            return None
        return {"name": name, "config_json": study_def}

    def list_studies(self):
        return [{"name": name, "config_json": study_def} for name, study_def in self.studies.items()]


def setup_function():
    study_store.reset()


def teardown_function():
    study_store.reset()


def test_configure_sets_state_store_and_validates_existing_rows():
    store = _FakeStudyStore({"cancer-research": {"site_orgs": {"org_a": ["site-a"]}, "admins": ["admin@nvidia.com"]}})

    study_store.configure(store)

    assert store.initialized is True
    assert study_store.get_state_store() is store
    assert study_store.has_study("cancer-research") is True
    assert study_store.has_user("admin@nvidia.com", "cancer-research") is True
    assert study_store.get_sites("cancer-research") == {"site-a"}
    assert study_store.get_sites("missing-study") == set()
    assert study_store.get_sites(DEFAULT_STUDY) is None


def test_normalize_study_def():
    normalized = study_store.normalize_study(
        "cancer-research",
        {
            "site_orgs": {"org_a": ["site-a"]},
            "admins": ["admin@nvidia.com", "admin@nvidia.com"],
        },
    )

    assert normalized == {
        "site_orgs": {"org_a": ["site-a"]},
        "admins": ["admin@nvidia.com"],
    }


def test_list_studies_returns_normalized_rows():
    store = _FakeStudyStore({"study-a": {"site_orgs": {"org_a": ["site-a"]}, "admins": []}})
    study_store.configure(store)

    assert study_store.list_studies() == [
        {"name": "study-a", "config_json": {"site_orgs": {"org_a": ["site-a"]}, "admins": []}}
    ]


def test_study_row_requires_name():
    with pytest.raises(ValueError, match="missing name"):
        study_store.study_from_row({"config_json": {"site_orgs": {}, "admins": []}})


def test_reset_clears_state_store():
    study_store.set_state_store(_FakeStudyStore())

    study_store.reset()

    assert study_store.get_state_store() is None
