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

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server import study_cmds as module


@pytest.fixture
def command():
    subject = module.StudyCommandModule()
    subject._with_mutation = Mock()
    return subject


@pytest.fixture
def connection():
    properties = {ConnProps.USER_ROLE: "org_admin", ConnProps.USER_ORG: "org"}
    result = Mock()
    result.get_prop.side_effect = lambda key, default=None: properties.get(key, default)
    result.properties = properties
    return result


@pytest.mark.parametrize("method", ["cmd_register_study", "cmd_add_study_site", "cmd_remove_study_site"])
@pytest.mark.parametrize(
    "case,code",
    [
        ("study", "INVALID_STUDY_NAME"),
        ("site", "INVALID_SITE"),
        ("org", "INVALID_ARGS"),
        ("missing_org", "INVALID_ARGS"),
    ],
)
def test_study_site_mutations_reject_bad_names_before_acquiring_registry(command, connection, method, case, code):
    args = [method, "default" if case == "study" else "study", "--sites", "bad name" if case == "site" else "site"]
    if case == "org":
        connection.properties[ConnProps.USER_ROLE] = "project_admin"
        args = [method, "study", "--site-org", "bad org:site"]
    elif case == "missing_org":
        connection.properties[ConnProps.USER_ORG] = ""
    getattr(command, method)(connection, args)
    result = connection.append_dict.call_args.args[0]
    assert result["error_code"] == code and result["exit_code"] == 4
    command._with_mutation.assert_not_called()


@pytest.mark.parametrize(
    "method", ["cmd_remove_study", "cmd_show_study", "cmd_add_study_user", "cmd_remove_study_user"]
)
def test_study_lookup_and_user_mutation_reject_reserved_study_without_registry_access(command, connection, method):
    args = [method, "default"] + (["alice"] if "user" in method else [])
    getattr(command, method)(connection, args)
    assert connection.append_dict.call_args.args[0]["error_code"] == "INVALID_STUDY_NAME"
    command._with_mutation.assert_not_called()


def test_study_authorization_requires_known_role_and_visibility_requires_org(command, connection):
    connection.properties[ConnProps.USER_ROLE] = "unknown"
    assert command.authorize_list_studies(connection, ["list_studies"]) == PreAuthzReturnCode.ERROR
    connection.append_error.assert_called_once()
    connection.properties[ConnProps.USER_ROLE] = "org_admin"
    connection.properties[ConnProps.USER_ORG] = ""
    assert not command._is_visible_to_caller(connection, "study", {"site_orgs": {"org": ["site"]}})
    command.cmd_list_studies(connection, ["list_studies", "unexpected"])
    assert connection.append_dict.call_args.args[0]["error_code"] == "INVALID_ARGS"
    with pytest.raises(module._InvalidSiteError, match="sites are required"):
        command._validate_site_names([])


def test_atomic_registry_replace_failure_removes_partial_and_preserves_original(monkeypatch, tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("original")
    replace = Mock(side_effect=OSError("rename failed"))
    proxy = SimpleNamespace(path=os.path, makedirs=os.makedirs, fdopen=os.fdopen, replace=replace, remove=os.remove)
    monkeypatch.setattr(module, "os", proxy)
    with pytest.raises(OSError, match="rename failed"):
        module.StudyCommandModule._write_registry_config(str(path), {"studies": {}})
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]
    assert replace.call_count == 1
