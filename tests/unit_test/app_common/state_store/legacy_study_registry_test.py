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

from nvflare.app_common.state_store.legacy_study_registry import LegacyStudyRegistry


def _make_registry_config(studies):
    return {"format_version": "1.0", "studies": studies}


def test_legacy_study_registry_tracks_user_membership_for_study():
    registry = LegacyStudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {"org_a": ["site-a", "site-b"]},
                    "admins": ["admin@nvidia.com"],
                }
            }
        )
    )

    assert registry.has_user("admin@nvidia.com", "cancer-research") is True


def test_legacy_study_registry_rejects_missing_or_invalid_format_version():
    try:
        LegacyStudyRegistry({"studies": {"cancer-research": {}}})
        assert False, "expected ValueError for missing format_version"
    except ValueError as e:
        assert "format_version" in str(e)

    try:
        LegacyStudyRegistry({"format_version": "2.0", "studies": {"cancer-research": {}}})
        assert False, "expected ValueError for invalid format_version"
    except ValueError as e:
        assert "format_version" in str(e)


def test_legacy_study_registry_rejects_missing_studies_mapping():
    try:
        LegacyStudyRegistry({"format_version": "1.0"})
        assert False, "expected ValueError for missing studies mapping"
    except ValueError as e:
        assert "studies" in str(e)


def test_legacy_study_registry_returns_false_for_missing_user_or_study():
    registry = LegacyStudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {"org_a": ["site-a"]},
                    "admins": ["admin@nvidia.com"],
                }
            }
        )
    )

    assert registry.has_user("other@nvidia.com", "cancer-research") is False
    assert registry.has_user("admin@nvidia.com", "unknown-study") is False
    assert registry.get_sites("unknown-study") is None
    assert registry.has_study("unknown-study") is False


def test_legacy_study_registry_returns_enrolled_sites_as_a_set():
    registry = LegacyStudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {"org_a": ["site-a"], "org_b": ["site-b"]},
                    "admins": ["admin@nvidia.com"],
                }
            }
        )
    )

    assert registry.get_sites("cancer-research") == {"site-a", "site-b"}
    assert registry.has_study("cancer-research") is True


def test_legacy_study_registry_rejects_duplicate_site_across_org_groups():
    try:
        LegacyStudyRegistry(
            _make_registry_config(
                {
                    "cancer-research": {
                        "site_orgs": {
                            "org_a": ["site-shared"],
                            "org_b": ["site-shared"],
                        },
                        "admins": [],
                    }
                }
            )
        )
        assert False, "expected ValueError for duplicate site across org groups"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_legacy_study_registry_derived_flat_sites_union_of_all_org_groups():
    registry = LegacyStudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {
                        "org_a": ["site-a", "site-b"],
                        "org_b": ["site-c"],
                    },
                    "admins": [],
                }
            }
        )
    )

    sites = registry.get_sites("cancer-research")
    assert sites == {"site-a", "site-b", "site-c"}


def test_legacy_study_registry_flat_sites_excludes_sites_from_removed_org():
    registry = LegacyStudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {
                        "org_a": ["site-a"],
                        "org_b": [],
                    },
                    "admins": [],
                }
            }
        )
    )

    sites = registry.get_sites("cancer-research")
    assert sites == {"site-a"}
