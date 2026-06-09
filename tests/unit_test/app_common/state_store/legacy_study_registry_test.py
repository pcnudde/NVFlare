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

from nvflare.app_common.state_store.legacy_study_registry import LegacyStudyRegistry


def test_legacy_study_registry_returns_raw_studies():
    registry = LegacyStudyRegistry(
        {
            "format_version": "1.0",
            "studies": {
                "cancer-research": {
                    "site_orgs": {"org_a": ["site-a", "site-b"], "org_empty": []},
                    "admins": ["admin@nvidia.com"],
                },
                "empty-study": None,
            },
        }
    )

    studies = registry.get_studies()
    assert studies == {
        "cancer-research": {
            "site_orgs": {"org_a": ["site-a", "site-b"], "org_empty": []},
            "admins": ["admin@nvidia.com"],
        },
        "empty-study": {},
    }

    # returned dict is a copy
    studies["cancer-research"]["admins"].append("other@nvidia.com")
    assert registry.get_studies()["cancer-research"]["admins"] == ["admin@nvidia.com"]


def test_legacy_study_registry_rejects_missing_or_invalid_format_version():
    with pytest.raises(ValueError, match="format_version"):
        LegacyStudyRegistry({"studies": {"cancer-research": {}}})

    with pytest.raises(ValueError, match="format_version"):
        LegacyStudyRegistry({"format_version": "2.0", "studies": {"cancer-research": {}}})


def test_legacy_study_registry_rejects_missing_studies_mapping():
    with pytest.raises(ValueError, match="studies"):
        LegacyStudyRegistry({"format_version": "1.0"})

    with pytest.raises(ValueError, match="dict"):
        LegacyStudyRegistry("not a dict")
