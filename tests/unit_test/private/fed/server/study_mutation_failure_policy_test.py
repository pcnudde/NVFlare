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

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server import study_cmds as subject
from nvflare.private.fed.server.server_engine import ServerEngine


@pytest.fixture
def env(monkeypatch, tmp_path):
    service = SimpleNamespace(acquire_lock=Mock(return_value=True), release_lock=Mock(), initialize=Mock())
    monkeypatch.setattr(subject, "StudyRegistryService", service)
    engine = Mock(spec=ServerEngine)
    workspace = engine.get_workspace.return_value
    workspace.get_study_registry_file_path.return_value = str(tmp_path / "registry.json")
    workspace.get_study_registry_write_path.return_value = str(tmp_path / "registry.json")
    connection = Mock(app_ctx=engine)
    return SimpleNamespace(
        module=subject.StudyCommandModule(),
        conn=connection,
        engine=engine,
        service=service,
        path=tmp_path / "registry.json",
    )


@pytest.mark.parametrize("failure", [None, "busy", "callback", "empty", "denied", "write"])
def test_study_mutation_releases_lock_and_publishes_registry_only_after_success(env, monkeypatch, failure):
    if failure == "busy":
        env.service.acquire_lock.return_value = False

    def mutate(engine, config):
        if failure == "callback":
            raise ValueError("invalid mutation")
        config["studies"]["study-a"] = {"site_orgs": {"org": ["site-a"]}, "admins": []}
        return None if failure == "empty" else {"error_code": "DENIED"} if failure == "denied" else {"study": "study-a"}

    callback = Mock(side_effect=mutate)
    if failure == "write":
        monkeypatch.setattr(env.module, "_write_registry_config", Mock(side_effect=OSError("disk full")))
    env.module._with_mutation(env.conn, callback)
    env.service.acquire_lock.assert_called_once_with(30.0)
    assert env.service.release_lock.call_count == int(failure != "busy")
    assert callback.call_count == int(failure != "busy")
    assert env.service.initialize.call_count == int(failure is None)
    if failure is None:
        assert json.loads(env.path.read_text())["studies"]["study-a"]["site_orgs"] == {"org": ["site-a"]}
        assert env.conn.append_dict.call_args.args[0] == {"study": "study-a"}
    else:
        assert not env.path.exists()
        reply = env.conn.append_dict.call_args.args[0]
        assert reply["error_code"] == (
            "LOCK_TIMEOUT" if failure == "busy" else "DENIED" if failure == "denied" else "INTERNAL_ERROR"
        )


@pytest.mark.parametrize(
    "role,allowed", [("project_admin", True), ("org_admin", True), ("member", False), ("unknown", False)]
)
def test_study_admin_authorization_uses_certificate_role(env, role, allowed):
    env.conn.get_prop.side_effect = lambda key, default=None: role if key == ConnProps.USER_ROLE else default
    result = env.module.authorize_study_admin(env.conn, ["command"])
    assert result == (PreAuthzReturnCode.OK if allowed else PreAuthzReturnCode.ERROR)
    assert env.conn.append_error.call_count == int(not allowed)


@pytest.mark.parametrize("sites", [None, " , "])
def test_study_site_lists_require_a_non_empty_enrollment(sites):
    with pytest.raises(subject._InvalidArgsError):
        subject.StudyCommandModule._parse_sites(sites)


@pytest.mark.parametrize("groups", [None, ["org-without-colon"], ["org:"], ["a:site-a", "b:site-a"]])
def test_study_org_groups_reject_missing_sites_and_ambiguous_ownership(groups):
    with pytest.raises(subject._InvalidArgsError):
        subject.StudyCommandModule._parse_site_orgs(groups)


@pytest.mark.parametrize("key,bad", [("admins", {}), ("site_orgs", [])])
def test_registry_normalization_preserves_schema_and_rejects_wrong_container_type(key, bad):
    method = (
        subject.StudyCommandModule._normalize_admins
        if key == "admins"
        else subject.StudyCommandModule._normalize_site_orgs
    )
    config = {key: None}
    normalized = method(config)
    assert config[key] is normalized
    assert normalized == ([] if key == "admins" else {})
    with pytest.raises(ValueError):
        method({key: bad})
