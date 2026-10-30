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

from nvflare.security.study_registry import StudyRegistry


@pytest.mark.parametrize(
    "definition, message",
    [
        ({"admins": "alice"}, "admins must be list"),
        ({"admins": [1]}, "admin entries must be str"),
        ({"site_orgs": []}, "site_orgs must be dict"),
        ({"site_orgs": {"org": "site"}}, "must be list"),
        ({"site_orgs": {"org": [1]}}, "site entry.*must be str"),
        ({"site_orgs": {"org-one": ["site"], "org-two": ["site"]}}, "duplicate site"),
    ],
)
def test_malformed_membership_is_rejected_during_registry_construction(definition, message):
    with pytest.raises(ValueError, match=message):
        StudyRegistry({"format_version": "1.0", "studies": {"study": definition}})


def test_null_membership_is_empty_and_repeated_admins_do_not_duplicate_permissions():
    registry = StudyRegistry(
        {
            "format_version": "1.0",
            "studies": {
                "empty": {"admins": None, "site_orgs": None},
                "study": {"admins": ["alice", "alice"], "site_orgs": {"org": ["site"]}},
            },
        }
    )
    assert registry.get_study("empty") == {"admins": [], "site_orgs": {}, "sites": []}
    assert registry.get_study("study")["admins"] == ["alice"]
    assert registry.has_user("alice", "study")
    assert not registry.has_user("alice", "empty")
    assert registry.has_org("study", "org")
    assert not registry.has_org("unknown", "org")
    assert registry.get_site_orgs("unknown") is None
    assert registry.get_study("unknown") is None
