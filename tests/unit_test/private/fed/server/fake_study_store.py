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

"""Shared class-level fake of the nvflare.apis.study_store module API for server unit tests.

The store content is the class attribute ``sites``: a {study: enrolled-sites-or-None}
mapping. A study mapped to None exists but has no site restriction (get_sites/
sites_from_study_def return None, so callers skip enrolled-site filtering).
"""


class FakeStudyStore:
    sites = {}

    @classmethod
    def has_study(cls, study):
        return study in cls.sites

    @classmethod
    def get_sites(cls, study):
        return cls.sites.get(study)

    @classmethod
    def get_study(cls, study):
        if study not in cls.sites:
            return None
        return {"_sites": cls.sites[study]}

    @staticmethod
    def sites_from_study_def(study_def):
        return (study_def or {}).get("_sites")


def install_fake_study_store(monkeypatch, module, studies):
    """Patch the module's study_store with FakeStudyStore holding the given mapping."""
    monkeypatch.setattr(module, "study_store", FakeStudyStore, raising=False)
    monkeypatch.setattr(FakeStudyStore, "sites", studies, raising=False)
