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

from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import MagicMock, patch

from nvflare.apis import study_store
from nvflare.apis.client import Client, ClientPropKey
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server.study_cmds import StudyCommandModule

_EMPTY_REGISTRY = {"format_version": "1.0", "studies": {}}

_REGISTRY_WITH_STUDY = {
    "format_version": "1.0",
    "studies": {
        "study1": {
            "site_orgs": {"org_a": ["site-existing"]},
            "admins": ["admin@example.com"],
        }
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConnection:
    def __init__(self, role, org, engine=None, user="admin@example.com"):
        self._props = {
            ConnProps.USER_NAME: user,
            ConnProps.USER_ROLE: role,
            ConnProps.USER_ORG: org,
        }
        self.app_ctx = engine
        self.replies = []

    def get_prop(self, key, default=None):
        return self._props.get(key, default)

    def append_dict(self, data, meta=None):
        self.replies.append(data)

    def append_error(self, message, meta=None):
        self.replies.append({"error": message})

    @property
    def last_reply(self):
        return self.replies[-1] if self.replies else None


def _make_client(org: str) -> Client:
    client = MagicMock(spec=Client)
    client.get_prop.side_effect = lambda key, default="": org if key == ClientPropKey.ORG else default
    return client


def _make_engine(site_map: dict):
    """
    site_map: {site_name: org_str}
    Omitting a site name → disconnected (get_client_from_name returns None).
    """
    engine = MagicMock()

    def _get_client(name):
        if name not in site_map:
            return None
        return _make_client(site_map[name])

    engine.client_manager.get_client_from_name.side_effect = _get_client
    return engine


class _FakeStateStore:
    def __init__(self, config=None):
        if config is None:
            config = _EMPTY_REGISTRY
        self.studies = deepcopy(config.get("studies", {}))
        self.jobs_by_study = {}  # study name => job count, consulted by delete_study_if_no_jobs
        self.write_count = 0
        self.calls = []

    def initialize(self):
        pass

    def list_studies(self):
        return [{"name": name, "config_json": deepcopy(study_def)} for name, study_def in self.studies.items()]

    def get_study(self, name: str):
        study_def = self.studies.get(name)
        if study_def is None:
            return None
        return {"name": name, "config_json": deepcopy(study_def)}

    def upsert_study(self, name: str, config: dict):
        self.calls.append("upsert_study")
        self.studies[name] = deepcopy(config)
        self.write_count += 1
        return {"name": name, "config_json": deepcopy(config), "version": self.write_count}

    def delete_study(self, name: str):
        self.calls.append("delete_study")
        existed = name in self.studies
        self.studies.pop(name, None)
        self.write_count += 1
        return existed

    def delete_study_if_no_jobs(self, name: str):
        self.calls.append("delete_study_if_no_jobs")
        if name not in self.studies:
            return {"deleted": False, "not_found": True}
        job_count = self.jobs_by_study.get(name, 0)
        if job_count:
            return {"deleted": False, "job_count": job_count}
        self.studies.pop(name)
        self.write_count += 1
        return {"deleted": True}

    def add_study_sites(self, name: str, site_orgs: dict):
        self.calls.append("add_study_sites")
        study = self.studies.setdefault(name, {"site_orgs": {}, "admins": []})
        existing = {site for org_sites in study.get("site_orgs", {}).values() for site in org_sites}
        for org, sites in site_orgs.items():
            current = study.setdefault("site_orgs", {}).setdefault(org, [])
            for site in sites:
                if site not in existing:
                    current.append(site)
                    existing.add(site)
        self.write_count += 1
        return self.get_study(name)

    def remove_study_sites(self, name: str, site_orgs: dict):
        self.calls.append("remove_study_sites")
        study = self.studies.get(name)
        if not study:
            return None
        for org, sites in site_orgs.items():
            if org not in study.get("site_orgs", {}):
                continue
            current = study["site_orgs"][org]
            study["site_orgs"][org] = [site for site in current if site not in sites]
        self.write_count += 1
        return self.get_study(name)

    def add_study_admin(self, name: str, user: str):
        self.calls.append("add_study_admin")
        study = self.studies.setdefault(name, {"site_orgs": {}, "admins": []})
        if user not in study.setdefault("admins", []):
            study["admins"].append(user)
        self.write_count += 1
        return self.get_study(name)

    def remove_study_admin(self, name: str, user: str):
        self.calls.append("remove_study_admin")
        study = self.studies.get(name)
        if not study:
            return None
        if user in study.setdefault("admins", []):
            study["admins"].remove(user)
        self.write_count += 1
        return self.get_study(name)


@contextmanager
def _state_store_ctx(store):
    study_store.set_state_store(store)
    try:
        yield
    finally:
        study_store.reset()


@contextmanager
def _mutation_ctx(initial_config=None):
    """Provides an in-memory StateStore."""
    if initial_config is None:
        initial_config = _EMPTY_REGISTRY
    store = _FakeStateStore(initial_config)
    with (
        _state_store_ctx(store),
        # Make isinstance(engine, ServerEngine) pass for MagicMock engines
        patch("nvflare.private.fed.server.study_cmds.ServerEngine", MagicMock),
    ):
        yield store


# ---------------------------------------------------------------------------
# Section 1: _validate_sites_for_org (direct unit tests)
# ---------------------------------------------------------------------------


class TestValidateSitesForOrg:
    def test_all_valid_returns_empty(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_a"})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-a", "site-b"], "org_a")
        assert result == []

    def test_wrong_org_is_rejected(self):
        engine = _make_engine({"site-a": "org_b"})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "org_a")
        assert result == ["site-a"]

    def test_disconnected_site_is_rejected(self):
        engine = _make_engine({})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-unknown"], "org_a")
        assert result == ["site-unknown"]

    def test_empty_org_on_client_is_rejected(self):
        engine = _make_engine({"site-a": ""})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "org_a")
        assert result == ["site-a"]

    def test_empty_sites_list_returns_empty(self):
        engine = _make_engine({})
        result = StudyCommandModule._validate_sites_for_org(engine, [], "org_a")
        assert result == []

    def test_mixed_returns_only_bad_sites(self):
        engine = _make_engine(
            {
                "site-ok": "org_a",  # valid
                "site-wrong": "org_b",  # wrong org
                # site-offline → disconnected
            }
        )
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-ok", "site-wrong", "site-offline"], "org_a")
        assert set(result) == {"site-wrong", "site-offline"}

    def test_multiple_orgs_validated_independently(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_b"})
        assert StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "org_a") == []
        assert StudyCommandModule._validate_sites_for_org(engine, ["site-b"], "org_b") == []
        assert StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "org_b") == ["site-a"]

    def test_empty_expected_org_rejects_all_sites(self):
        # Empty caller-cert org must never pass — reject every site regardless of what
        # the site cert carries, including a site that also has an empty org.
        engine = _make_engine({"site-a": ""})
        result = StudyCommandModule._validate_sites_for_org(engine, ["site-a"], "")
        assert result == ["site-a"]

    def test_synthetic_admin_client_name_is_rejected(self):
        # get_client_from_name can return a synthetic Client for admin-style names
        # (e.g. "admin@example.com"). That client has no ORG prop, so it must be
        # rejected when a real org is expected.
        engine = _make_engine({"admin@example.com": ""})
        result = StudyCommandModule._validate_sites_for_org(engine, ["admin@example.com"], "org_a")
        assert result == ["admin@example.com"]


# ---------------------------------------------------------------------------
# Section 2: cmd_register_study — site->org validation
# ---------------------------------------------------------------------------


class TestRegisterStudySiteOrgValidation:
    def _module(self):
        return StudyCommandModule()

    def test_org_admin_valid_connected_site_succeeds(self):
        engine = _make_engine({"site-a": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply is not None
        assert "error_code" not in conn.last_reply
        assert conn.last_reply.get("name") == "study1"

    def test_org_admin_wrong_org_returns_invalid_site(self):
        engine = _make_engine({"site-a": "org_b"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_org_admin_disconnected_site_returns_invalid_site(self):
        engine = _make_engine({})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-offline"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_valid_site_org_succeeds(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_a"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--site-org", "org_a:site-a,site-b"])
        assert "error_code" not in conn.last_reply

    def test_project_admin_site_with_wrong_org_returns_invalid_site(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(conn, ["register_study", "study1", "--site-org", "org_a:site-a,site-b"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_disconnected_site_returns_invalid_site(self):
        engine = _make_engine({"site-a": "org_a"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(
                conn, ["register_study", "study1", "--site-org", "org_a:site-a,site-offline"]
            )
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_multiple_site_org_groups_all_valid_succeeds(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(
                conn,
                ["register_study", "study1", "--site-org", "org_a:site-a", "--site-org", "org_b:site-b"],
            )
        assert "error_code" not in conn.last_reply

    def test_project_admin_multiple_groups_one_bad_returns_invalid_site(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_c"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx():
            self._module().cmd_register_study(
                conn,
                ["register_study", "study1", "--site-org", "org_a:site-a", "--site-org", "org_b:site-b"],
            )
        assert conn.last_reply["error_code"] == "INVALID_SITE"


# ---------------------------------------------------------------------------
# Section 3: cmd_add_study_site — site->org validation
# ---------------------------------------------------------------------------


class TestAddStudySiteOrgValidation:
    def _module(self):
        return StudyCommandModule()

    def test_org_admin_valid_site_succeeds(self):
        engine = _make_engine({"site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-new"])
        assert "error_code" not in conn.last_reply
        assert "site-new" in conn.last_reply.get("added", [])

    def test_org_admin_wrong_org_returns_invalid_site(self):
        engine = _make_engine({"site-new": "org_b"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-new"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_org_admin_disconnected_site_returns_invalid_site(self):
        engine = _make_engine({})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-offline"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_valid_site_org_succeeds(self):
        engine = _make_engine({"site-new": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--site-org", "org_b:site-new"])
        assert "error_code" not in conn.last_reply

    def test_project_admin_wrong_org_returns_invalid_site(self):
        engine = _make_engine({"site-new": "org_c"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--site-org", "org_b:site-new"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_project_admin_disconnected_site_returns_invalid_site(self):
        engine = _make_engine({})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--site-org", "org_b:site-offline"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"

    def test_mixed_valid_and_invalid_returns_invalid_site(self):
        engine = _make_engine({"site-ok": "org_b", "site-bad": "org_c"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_site(
                conn, ["add_study_site", "study1", "--site-org", "org_b:site-ok,site-bad"]
            )
        assert conn.last_reply["error_code"] == "INVALID_SITE"


# ---------------------------------------------------------------------------
# Section 4: cmd_remove_study_site — site->org validation
# ---------------------------------------------------------------------------


class TestRemoveStudySiteOrgValidation:
    def _module(self):
        return StudyCommandModule()

    def test_org_admin_valid_site_succeeds(self):
        engine = _make_engine({"site-existing": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_org_admin_site_in_different_engine_org_still_succeeds(self):
        # Engine reports site-existing under org_b, but the study registry has it under org_a.
        # For removal the registry is the source of truth; engine org is irrelevant.
        engine = _make_engine({"site-existing": "org_b"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_org_admin_disconnected_site_succeeds(self):
        # Site is offline (not in engine) but is enrolled in the study — removal must still work.
        engine = _make_engine({})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_project_admin_valid_site_org_succeeds(self):
        engine = _make_engine({"site-existing": "org_a"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(
                conn, ["remove_study_site", "study1", "--site-org", "org_a:site-existing"]
            )
        assert "error_code" not in conn.last_reply

    def test_project_admin_site_in_different_engine_org_still_succeeds(self):
        # Engine reports site-existing under org_b, but admin requests removal from org_a.
        # Study registry has it under org_a; engine org is irrelevant for removal.
        engine = _make_engine({"site-existing": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(
                conn, ["remove_study_site", "study1", "--site-org", "org_a:site-existing"]
            )
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_project_admin_disconnected_site_succeeds(self):
        # Site is offline but enrolled — removal must succeed.
        engine = _make_engine({})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(
                conn, ["remove_study_site", "study1", "--site-org", "org_a:site-existing"]
            )
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])

    def test_project_admin_mixed_sites_enrolled_and_not_enrolled(self):
        # site-existing is enrolled under org_a; site-b is specified under org_b but not enrolled.
        # Enrolled site is removed; unenrolled site lands in not_enrolled (no error).
        engine = _make_engine({"site-existing": "org_a", "site-b": "org_c"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_site(
                conn,
                [
                    "remove_study_site",
                    "study1",
                    "--site-org",
                    "org_a:site-existing",
                    "--site-org",
                    "org_b:site-b",
                ],
            )
        assert "error_code" not in conn.last_reply
        assert "site-existing" in conn.last_reply.get("removed", [])
        assert "site-b" in conn.last_reply.get("not_enrolled", [])

    def test_unenrolled_org_does_not_get_phantom_registry_entry(self):
        # org_b has no sites in study1. Requesting removal of org_b:site-b must not
        # create a phantom {"org_b": []} entry that would grant org_b visibility.
        engine = _make_engine({})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--site-org", "org_b:site-b"])
        assert "org_b" not in store.studies.get("study1", {}).get("site_orgs", {})


# ---------------------------------------------------------------------------
# Section 5: INVALID_ARGS — input-shape enforcement (server-side authoritative)
# ---------------------------------------------------------------------------


class TestInvalidArgsInputShape:
    """
    Verifies that the server rejects the three forbidden input shapes with
    INVALID_ARGS regardless of which site-mutation command is used.
    Rules under test:
      1. mixed --sites + --site-org
      2. org_admin using --site-org
      3. project_admin using --sites
    """

    def _module(self):
        return StudyCommandModule()

    # --- register_study ---

    def test_register_mixed_sites_and_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_register_study(
            conn, ["register_study", "study1", "--sites", "site-a", "--site-org", "org_a:site-b"]
        )
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_register_org_admin_with_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_register_study(conn, ["register_study", "study1", "--site-org", "org_a:site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_register_project_admin_with_sites_returns_invalid_args(self):
        conn = _FakeConnection(role="project_admin", org="project")
        self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    # --- add_study_site ---

    def test_add_site_mixed_sites_and_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_add_study_site(
            conn, ["add_study_site", "study1", "--sites", "site-a", "--site-org", "org_a:site-b"]
        )
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_add_site_org_admin_with_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--site-org", "org_a:site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_add_site_project_admin_with_sites_returns_invalid_args(self):
        conn = _FakeConnection(role="project_admin", org="project")
        self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    # --- remove_study_site ---

    def test_remove_site_mixed_sites_and_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_remove_study_site(
            conn, ["remove_study_site", "study1", "--sites", "site-a", "--site-org", "org_a:site-b"]
        )
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_remove_site_org_admin_with_site_org_returns_invalid_args(self):
        conn = _FakeConnection(role="org_admin", org="org_a")
        self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--site-org", "org_a:site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"

    def test_remove_site_project_admin_with_sites_returns_invalid_args(self):
        conn = _FakeConnection(role="project_admin", org="project")
        self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_ARGS"


# ---------------------------------------------------------------------------
# Section 6: STUDY_ALREADY_EXISTS — register when org not enrolled
# ---------------------------------------------------------------------------


class TestStudyAlreadyExists:
    def _module(self):
        return StudyCommandModule()

    def test_org_admin_register_existing_study_with_no_enrollment_returns_already_exists(self):
        engine = _make_engine({"site-new": "org_b"})
        conn = _FakeConnection(role="org_admin", org="org_b", engine=engine)
        # study1 exists but org_b is not in site_orgs
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-new"])
        assert conn.last_reply["error_code"] == "STUDY_ALREADY_EXISTS"

    def test_org_admin_register_existing_study_already_enrolled_merges(self):
        engine = _make_engine({"site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        # org_a is already in study1 site_orgs — register should merge not reject
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-new"])
        assert "error_code" not in conn.last_reply
        assert "site-new" in conn.last_reply.get("site_orgs", {}).get("org_a", [])

    def test_project_admin_register_existing_study_succeeds(self):
        engine = _make_engine({"site-new": "org_b"})
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_register_study(conn, ["register_study", "study1", "--site-org", "org_b:site-new"])
        assert "error_code" not in conn.last_reply


# ---------------------------------------------------------------------------
# Section 7: cmd_remove_study
# ---------------------------------------------------------------------------


class TestRemoveStudy:
    def _module(self):
        return StudyCommandModule()

    def _engine(self):
        return MagicMock()

    def test_project_admin_removes_existing_study(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=self._engine())
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            self._module().cmd_remove_study(conn, ["remove_study", "study1"])
            assert "study1" not in store.studies
            # the check-and-delete is a single store call, not a separate count + delete
            assert store.calls == ["delete_study_if_no_jobs"]
        assert conn.last_reply.get("removed") is True

    def test_remove_nonexistent_study_returns_not_found(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=self._engine())
        with _mutation_ctx(_EMPTY_REGISTRY):
            self._module().cmd_remove_study(conn, ["remove_study", "ghost"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"

    def test_remove_study_with_associated_jobs_returns_study_has_jobs(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=self._engine())
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            store.jobs_by_study["study1"] = 2
            self._module().cmd_remove_study(conn, ["remove_study", "study1"])
            assert "study1" in store.studies
        assert conn.last_reply["error_code"] == "STUDY_HAS_JOBS"
        assert "2 associated job(s)" in conn.last_reply["message"]


# ---------------------------------------------------------------------------
# Section 8: cmd_list_studies — visibility filtering
# ---------------------------------------------------------------------------


def _make_store(studies_dict):
    return _FakeStateStore(
        {
            "format_version": "1.0",
            "studies": {
                name: {"site_orgs": def_["site_orgs"], "admins": def_.get("admins", [])}
                for name, def_ in studies_dict.items()
            },
        }
    )


class TestListStudiesVisibility:
    def _module(self):
        return StudyCommandModule()

    @staticmethod
    def _authorize_submit_for_roles(*allowed_roles):
        def _authorize(ctx):
            if ctx.right == "submit_job" and ctx.user.role in allowed_roles:
                return True, ""
            return False, f"user '{ctx.user.name}' is not authorized for '{ctx.right}'"

        return _authorize

    def test_lead_role_is_authorized_to_list_studies(self):
        conn = _FakeConnection(role="lead", org="org_a", user="lead@example.com")

        assert self._module().authorize_list_studies(conn, ["list_studies"]) == PreAuthzReturnCode.OK

    def test_project_admin_sees_all_studies(self):
        store = _make_store(
            {
                "study-alpha": {"site_orgs": {"org_a": ["site-a"]}},
                "study-beta": {"site_orgs": {"org_b": ["site-b"]}},
            }
        )
        conn = _FakeConnection(role="project_admin", org="project")
        with (
            _state_store_ctx(store),
            patch(
                "nvflare.private.fed.server.study_cmds.AuthorizationService.authorize",
                side_effect=self._authorize_submit_for_roles("project_admin", "lead"),
            ),
        ):
            self._module().cmd_list_studies(conn, ["list_studies"])
        assert set(conn.last_reply["studies"]) == {"study-alpha", "study-beta"}
        assert conn.last_reply["identity"] == {
            "name": "admin@example.com",
            "org": "project",
            "role": "project_admin",
        }
        assert conn.last_reply["study_details"] == [
            {
                "name": "study-alpha",
                "role": "project_admin",
                "capabilities": {"submit_job": True},
                "can_submit_job": True,
            },
            {
                "name": "study-beta",
                "role": "project_admin",
                "capabilities": {"submit_job": True},
                "can_submit_job": True,
            },
        ]

    def test_org_admin_sees_only_enrolled_studies(self):
        store = _make_store(
            {
                "study-alpha": {"site_orgs": {"org_a": ["site-a"]}},
                "study-beta": {"site_orgs": {"org_b": ["site-b"]}},
            }
        )
        conn = _FakeConnection(role="org_admin", org="org_a")
        with (
            _state_store_ctx(store),
            patch(
                "nvflare.private.fed.server.study_cmds.AuthorizationService.authorize",
                side_effect=self._authorize_submit_for_roles("project_admin", "lead"),
            ),
        ):
            self._module().cmd_list_studies(conn, ["list_studies"])
        assert conn.last_reply["studies"] == ["study-alpha"]
        assert conn.last_reply["study_details"][0]["name"] == "study-alpha"
        assert conn.last_reply["study_details"][0]["capabilities"] == {"submit_job": False}
        assert conn.last_reply["study_details"][0]["can_submit_job"] is False
        assert (
            conn.last_reply["study_details"][0]["reason"]
            == "user 'admin@example.com' is not authorized for 'submit_job'"
        )

    def test_org_admin_with_no_enrollment_sees_empty_list(self):
        store = _make_store({"study-alpha": {"site_orgs": {"org_b": ["site-b"]}}})
        conn = _FakeConnection(role="org_admin", org="org_a")
        with _state_store_ctx(store):
            self._module().cmd_list_studies(conn, ["list_studies"])
        assert conn.last_reply["studies"] == []
        assert conn.last_reply["study_details"] == []

    def test_lead_sees_only_mapped_studies(self):
        store = _make_store(
            {
                "study-alpha": {
                    "site_orgs": {"org_a": ["site-a"]},
                    "admins": ["lead@example.com"],
                },
                "study-beta": {
                    "site_orgs": {"org_a": ["site-b"]},
                    "admins": ["other@example.com"],
                },
            }
        )
        conn = _FakeConnection(role="lead", org="org_a", user="lead@example.com")
        with (
            _state_store_ctx(store),
            patch(
                "nvflare.private.fed.server.study_cmds.AuthorizationService.authorize",
                side_effect=self._authorize_submit_for_roles("project_admin", "lead"),
            ),
        ):
            self._module().cmd_list_studies(conn, ["list_studies"])

        assert conn.last_reply["studies"] == ["study-alpha"]
        assert conn.last_reply["study_details"] == [
            {
                "name": "study-alpha",
                "role": "lead",
                "capabilities": {"submit_job": True},
                "can_submit_job": True,
            }
        ]

    def test_member_visible_study_cannot_submit(self):
        store = _make_store(
            {
                "study-alpha": {
                    "site_orgs": {"org_a": ["site-a"]},
                    "admins": ["member@example.com"],
                },
            }
        )
        conn = _FakeConnection(role="member", org="org_a", user="member@example.com")
        with (
            _state_store_ctx(store),
            patch(
                "nvflare.private.fed.server.study_cmds.AuthorizationService.authorize",
                side_effect=self._authorize_submit_for_roles("project_admin", "lead"),
            ),
        ):
            self._module().cmd_list_studies(conn, ["list_studies"])

        assert conn.last_reply["studies"] == ["study-alpha"]
        assert conn.last_reply["study_details"] == [
            {
                "name": "study-alpha",
                "role": "member",
                "capabilities": {"submit_job": False},
                "can_submit_job": False,
                "reason": "user 'member@example.com' is not authorized for 'submit_job'",
            }
        ]


# ---------------------------------------------------------------------------
# Section 9: cmd_show_study
# ---------------------------------------------------------------------------


class TestShowStudy:
    def _module(self):
        return StudyCommandModule()

    def test_project_admin_can_show_any_study(self):
        store = _make_store({"study1": {"site_orgs": {"org_a": ["site-a"]}, "admins": ["u@x.com"]}})
        conn = _FakeConnection(role="project_admin", org="project")
        with _state_store_ctx(store):
            self._module().cmd_show_study(conn, ["show_study", "study1"])
        assert conn.last_reply.get("name") == "study1"
        assert "error_code" not in conn.last_reply

    def test_org_admin_can_show_enrolled_study(self):
        store = _make_store({"study1": {"site_orgs": {"org_a": ["site-a"]}}})
        conn = _FakeConnection(role="org_admin", org="org_a")
        with _state_store_ctx(store):
            self._module().cmd_show_study(conn, ["show_study", "study1"])
        assert conn.last_reply.get("name") == "study1"

    def test_org_admin_cannot_show_hidden_study(self):
        store = _make_store({"study1": {"site_orgs": {"org_b": ["site-b"]}}})
        conn = _FakeConnection(role="org_admin", org="org_a")
        with _state_store_ctx(store):
            self._module().cmd_show_study(conn, ["show_study", "study1"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"

    def test_lead_cannot_show_org_study_without_user_mapping(self):
        store = _make_store({"study1": {"site_orgs": {"org_a": ["site-a"]}, "admins": ["other@example.com"]}})
        conn = _FakeConnection(role="lead", org="org_a", user="lead@example.com")
        with _state_store_ctx(store):
            self._module().cmd_show_study(conn, ["show_study", "study1"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"

    def test_show_nonexistent_study_returns_not_found(self):
        store = _make_store({})
        conn = _FakeConnection(role="project_admin", org="project")
        with _state_store_ctx(store):
            self._module().cmd_show_study(conn, ["show_study", "ghost"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"


# ---------------------------------------------------------------------------
# Section 10: user membership commands
# ---------------------------------------------------------------------------


class TestUserMembership:
    def _module(self):
        return StudyCommandModule()

    def test_add_user_succeeds(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_user(conn, ["add_study_user", "study1", "newuser@x.com"])
        assert conn.last_reply.get("user") == "newuser@x.com"
        assert "error_code" not in conn.last_reply

    def test_add_user_duplicate_returns_user_already_in_study(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_user(conn, ["add_study_user", "study1", "admin@example.com"])
        assert conn.last_reply["error_code"] == "USER_ALREADY_IN_STUDY"

    def test_add_user_to_hidden_study_returns_not_found(self):
        conn = _FakeConnection(role="org_admin", org="org_b", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_add_study_user(conn, ["add_study_user", "study1", "newuser@x.com"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"

    def test_remove_user_succeeds(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_user(conn, ["remove_study_user", "study1", "admin@example.com"])
        assert conn.last_reply.get("removed") is True
        assert "error_code" not in conn.last_reply

    def test_remove_user_not_in_study_returns_user_not_in_study(self):
        conn = _FakeConnection(role="project_admin", org="project", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_user(conn, ["remove_study_user", "study1", "ghost@x.com"])
        assert conn.last_reply["error_code"] == "USER_NOT_IN_STUDY"

    def test_remove_user_from_hidden_study_returns_not_found(self):
        conn = _FakeConnection(role="org_admin", org="org_b", engine=MagicMock())
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            self._module().cmd_remove_study_user(conn, ["remove_study_user", "study1", "admin@example.com"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"


# ---------------------------------------------------------------------------
# Section 11: atomicity — no write on validation failure
# ---------------------------------------------------------------------------


@contextmanager
def _mutation_ctx_with_write_tracker(initial_config=None):
    if initial_config is None:
        initial_config = _EMPTY_REGISTRY
    store = _FakeStateStore(initial_config)
    with (
        _state_store_ctx(store),
        patch("nvflare.private.fed.server.study_cmds.ServerEngine", MagicMock),
    ):
        yield store


class TestAtomicityGuarantee:
    def _module(self):
        return StudyCommandModule()

    def test_invalid_site_prevents_registry_write(self):
        engine = _make_engine({"site-a": "org_b"})  # wrong org
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx_with_write_tracker() as store:
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert conn.last_reply["error_code"] == "INVALID_SITE"
        assert store.write_count == 0

    def test_partial_invalid_site_org_group_prevents_registry_write(self):
        engine = _make_engine({"site-a": "org_a", "site-b": "org_c"})  # org_b:site-b is wrong
        conn = _FakeConnection(role="project_admin", org="project", engine=engine)
        with _mutation_ctx_with_write_tracker() as store:
            self._module().cmd_register_study(
                conn,
                ["register_study", "study1", "--site-org", "org_a:site-a", "--site-org", "org_b:site-b"],
            )
        assert conn.last_reply["error_code"] == "INVALID_SITE"
        assert store.write_count == 0

    def test_study_not_found_prevents_registry_write_on_add_site(self):
        engine = _make_engine({"site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx_with_write_tracker(_EMPTY_REGISTRY) as store:
            self._module().cmd_add_study_site(conn, ["add_study_site", "ghost", "--sites", "site-new"])
        assert conn.last_reply["error_code"] == "STUDY_NOT_FOUND"
        assert store.write_count == 0


# ---------------------------------------------------------------------------
# Section 12: register-study merge uses incremental store calls (A4)
# ---------------------------------------------------------------------------


class TestRegisterStudyIncrementalMerge:
    def _module(self):
        return StudyCommandModule()

    def test_register_existing_study_does_not_rewrite_full_snapshot(self):
        # The existing-study branch must use incremental add calls; a full upsert of a
        # stale snapshot would clobber concurrently added admins/sites.
        engine = _make_engine({"site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine, user="other-admin@example.com")
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-new"])
        assert "error_code" not in conn.last_reply
        assert "upsert_study" not in store.calls
        assert store.calls == ["add_study_sites", "add_study_admin"]
        assert "site-new" in store.studies["study1"]["site_orgs"]["org_a"]
        assert "other-admin@example.com" in store.studies["study1"]["admins"]

    def test_register_existing_study_skips_admin_write_when_caller_already_admin(self):
        engine = _make_engine({"site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine, user="admin@example.com")
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-new"])
        assert "error_code" not in conn.last_reply
        assert store.calls == ["add_study_sites"]
        assert store.studies["study1"]["admins"] == ["admin@example.com"]

    def test_register_new_study_uses_single_upsert(self):
        engine = _make_engine({"site-a": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_EMPTY_REGISTRY) as store:
            self._module().cmd_register_study(conn, ["register_study", "study1", "--sites", "site-a"])
        assert "error_code" not in conn.last_reply
        assert store.calls == ["upsert_study"]
        assert store.studies["study1"]["admins"] == ["admin@example.com"]

    def test_add_site_reports_added_and_already_enrolled_without_extra_writes(self):
        engine = _make_engine({"site-existing": "org_a", "site-new": "org_a"})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            self._module().cmd_add_study_site(conn, ["add_study_site", "study1", "--sites", "site-existing,site-new"])
        assert "error_code" not in conn.last_reply
        assert conn.last_reply["added"] == ["site-new"]
        assert conn.last_reply["already_enrolled"] == ["site-existing"]
        assert store.calls == ["add_study_sites"]

    def test_remove_site_reports_removed_and_not_enrolled(self):
        engine = _make_engine({})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            self._module().cmd_remove_study_site(
                conn, ["remove_study_site", "study1", "--sites", "site-existing,site-ghost"]
            )
        assert "error_code" not in conn.last_reply
        assert conn.last_reply["removed"] == ["site-existing"]
        assert conn.last_reply["not_enrolled"] == ["site-ghost"]
        assert store.calls == ["remove_study_sites"]


# ---------------------------------------------------------------------------
# Section 13: org with zero sites stays enrolled — no org_admin lockout (A6)
# ---------------------------------------------------------------------------


class TestOrgZeroSitesNoLockout:
    """Regression for the org_admin self-lockout: after an org_admin removes their org's
    only site, the org stays enrolled (site_orgs keeps {"org": []}), so the study remains
    visible and recoverable by that org's admin without project_admin help."""

    def _module(self):
        return StudyCommandModule()

    def test_remove_last_site_keeps_org_enrolled_in_store(self):
        engine = _make_engine({})
        conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            self._module().cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
            assert "site-existing" in conn.last_reply.get("removed", [])
            assert store.studies["study1"]["site_orgs"] == {"org_a": []}
            # the zero-site org is still reported in the payload
            assert conn.last_reply["site_orgs"] == {"org_a": []}

    def test_show_study_still_visible_after_removing_last_site(self):
        engine = _make_engine({})
        module = self._module()
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
            module.cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
            show_conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
            module.cmd_show_study(show_conn, ["show_study", "study1"])
        assert show_conn.last_reply.get("name") == "study1"
        assert "error_code" not in show_conn.last_reply

    def test_add_site_works_after_removing_last_site(self):
        engine = _make_engine({"site-new": "org_a"})
        module = self._module()
        with _mutation_ctx(_REGISTRY_WITH_STUDY) as store:
            conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
            module.cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
            add_conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
            module.cmd_add_study_site(add_conn, ["add_study_site", "study1", "--sites", "site-new"])
        assert "error_code" not in add_conn.last_reply
        assert "site-new" in add_conn.last_reply.get("added", [])
        assert store.studies["study1"]["site_orgs"]["org_a"] == ["site-new"]

    def test_register_merges_for_enrolled_org_with_zero_sites(self):
        # register on the existing study must take the merge path (the org is still
        # enrolled), not return STUDY_ALREADY_EXISTS.
        engine = _make_engine({"site-new": "org_a"})
        module = self._module()
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
            module.cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
            reg_conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
            module.cmd_register_study(reg_conn, ["register_study", "study1", "--sites", "site-new"])
        assert "error_code" not in reg_conn.last_reply
        assert "site-new" in reg_conn.last_reply.get("site_orgs", {}).get("org_a", [])

    def test_register_by_unenrolled_org_still_reports_already_exists(self):
        engine = _make_engine({"site-b": "org_b"})
        module = self._module()
        with _mutation_ctx(_REGISTRY_WITH_STUDY):
            conn = _FakeConnection(role="org_admin", org="org_a", engine=engine)
            module.cmd_remove_study_site(conn, ["remove_study_site", "study1", "--sites", "site-existing"])
            other_conn = _FakeConnection(role="org_admin", org="org_b", engine=engine)
            module.cmd_register_study(other_conn, ["register_study", "study1", "--sites", "site-b"])
        assert other_conn.last_reply["error_code"] == "STUDY_ALREADY_EXISTS"
