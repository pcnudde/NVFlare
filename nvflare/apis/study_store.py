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
from typing import Optional

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.apis.state_store import StateStore

_state_store: Optional[StateStore] = None


def configure(store: StateStore):
    store.initialize()
    for row in store.list_studies():
        study_from_row(row)
    set_state_store(store)


def set_state_store(store: Optional[StateStore]):
    global _state_store
    _state_store = store


def get_state_store() -> Optional[StateStore]:
    return _state_store


def _require_state_store() -> StateStore:
    assert _state_store is not None, "state_store must be configured before study mutations are used"
    return _state_store


def reset():
    set_state_store(None)


def normalize_study(study: str, study_def: dict) -> dict:
    study_def = deepcopy(study_def or {})
    admins = study_def.get("admins", [])
    if admins is None:
        admins = []
    if not isinstance(admins, list):
        raise ValueError(f"study '{study}' admins must be list but got {type(admins)}")

    admin_list = []
    seen_admins = set()
    for admin in admins:
        if not isinstance(admin, str):
            raise ValueError(f"study '{study}' admin entries must be str but got {type(admin)}")
        if admin in seen_admins:
            continue
        seen_admins.add(admin)
        admin_list.append(admin)

    site_orgs = study_def.get("site_orgs", {})
    if site_orgs is None:
        site_orgs = {}
    if not isinstance(site_orgs, dict):
        raise ValueError(f"study '{study}' site_orgs must be dict but got {type(site_orgs)}")

    normalized_site_orgs = {}
    seen_sites = set()
    for org, org_sites in site_orgs.items():
        if not isinstance(org_sites, list):
            raise ValueError(f"study '{study}' site_orgs[{org}] must be list but got {type(org_sites)}")
        normalized_sites = []
        for site in org_sites:
            if not isinstance(site, str):
                raise ValueError(f"study '{study}' site entry for org '{org}' must be str but got {type(site)}")
            if site in seen_sites:
                raise ValueError(f"study '{study}' contains duplicate site '{site}' across org groups")
            seen_sites.add(site)
            normalized_sites.append(site)
        normalized_site_orgs[org] = normalized_sites

    return {"admins": admin_list, "site_orgs": normalized_site_orgs}


def _normalize_study_row(row: dict):
    if row is None:
        return None
    name = row.get("name")
    if not name:
        raise ValueError("study row is missing name")
    return {"name": name, "config_json": normalize_study(name, deepcopy(row.get("config_json") or {}))}


def study_from_row(row: dict):
    row = _normalize_study_row(row)
    return row["config_json"] if row else None


def get_study(study: str):
    store = get_state_store()
    return study_from_row(store.get_study(study)) if store else None


def list_studies():
    store = get_state_store()
    if not store:
        return []
    return [_normalize_study_row(row) for row in store.list_studies()]


def upsert_study(study: str, study_def: dict):
    return study_from_row(_require_state_store().upsert_study(study, normalize_study(study, study_def)))


def delete_study(study: str):
    return _require_state_store().delete_study(study)


def delete_study_if_no_jobs(study: str) -> dict:
    """Atomically delete the study iff no jobs reference it (see StateStore.delete_study_if_no_jobs)."""
    return _require_state_store().delete_study_if_no_jobs(study)


def add_sites(study: str, site_orgs: dict):
    return study_from_row(
        _require_state_store().add_study_sites(study, normalize_study(study, {"site_orgs": site_orgs})["site_orgs"])
    )


def remove_sites(study: str, site_orgs: dict):
    return study_from_row(
        _require_state_store().remove_study_sites(study, normalize_study(study, {"site_orgs": site_orgs})["site_orgs"])
    )


def add_user(study: str, user: str):
    return study_from_row(_require_state_store().add_study_admin(study, user))


def remove_user(study: str, user: str):
    return study_from_row(_require_state_store().remove_study_admin(study, user))


def sites_from_study_def(study_def: dict) -> set:
    """Returns the flat set of sites enrolled in the study, across all orgs."""
    sites = set()
    for org_sites in (study_def or {}).get("site_orgs", {}).values():
        sites.update(org_sites)
    return sites


def get_sites(study: str):
    if study == DEFAULT_STUDY:
        return None
    study_def = get_study(study)
    if study_def is None:
        return set()
    return sites_from_study_def(study_def)


def has_study(study: str) -> bool:
    return get_study(study) is not None


def has_user(user_name: str, study: str) -> bool:
    study_def = get_study(study)
    return user_name in set((study_def or {}).get("admins", []))
