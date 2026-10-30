# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.fuel.flare_api.api_spec import AuthenticationError, AuthorizationError, InternalError, NoConnection
from nvflare.tool import cli_output
from nvflare.tool.study import study_cli as study
from nvflare.tool.system import system_cli as system


def test_study_and_system_dispatch_reject_unknown_commands_and_forward_known_arguments(monkeypatch):
    for module, key, handlers in [
        (study, "study_sub_cmd", "_study_handlers"),
        (system, "system_sub_cmd", "_system_handlers"),
    ]:
        handler = Mock()
        monkeypatch.setattr(module, handlers, {"known": handler})
        args = SimpleNamespace(**{key: "known"})
        getattr(module, "handle_study_cmd" if module is study else "handle_system_cmd")(args)
        handler.assert_called_once_with(args)
        with pytest.raises(CLIUnknownCmdException, match="Unknown"):
            getattr(module, "handle_study_cmd" if module is study else "handle_system_cmd")(
                SimpleNamespace(**{key: "invalid"})
            )
    with pytest.raises(CLIUnknownCmdException, match="required"):
        system.handle_system_cmd(SimpleNamespace())


@pytest.mark.parametrize(
    "error,code",
    [
        (AuthorizationError("denied"), "NOT_AUTHORIZED"),
        (AuthenticationError("login"), "AUTH_FAILED"),
        (NoConnection("offline"), "CONNECTION_FAILED"),
        (InternalError("failure"), "INTERNAL_ERROR"),
    ],
)
def test_study_failure_classification_emits_machine_readable_error_contract(capsys, error, code):
    cli_output.set_output_format("json")
    with pytest.raises(SystemExit):
        study._handle_command_error(error)
    assert json.loads(capsys.readouterr().out)["error_code"] == code
    unexpected = RuntimeError("unknown")
    with pytest.raises(RuntimeError, match="unknown") as raised:
        study._handle_command_error(unexpected)
    assert raised.value is unexpected


@pytest.mark.parametrize(
    "role,sites,site_org,expected",
    [
        (None, ["site-1,site-2"], None, (["site-1", "site-2"], None)),
        (None, None, ["org:site-1", " "], (None, ["org:site-1"])),
        ("org_admin", ["site-1"], None, (["site-1"], None)),
    ],
)
def test_study_lifecycle_resolves_site_lists_without_guessing_unavailable_role(
    monkeypatch, role, sites, site_org, expected
):
    monkeypatch.setattr(study, "_try_get_caller_role", Mock(return_value=role))
    assert (
        study._resolve_lifecycle_inputs(SimpleNamespace(name="study", sites=sites, site_org=site_org), "add-site")
        == expected
    )


@pytest.mark.parametrize("sites", [None, [" "]])
def test_study_requires_nonempty_sites_and_org_admin_gets_usage_guidance(monkeypatch, capsys, sites):
    with pytest.raises(ValueError):
        study._parse_sites_arg(sites)
    monkeypatch.setattr(study, "_try_get_caller_role", Mock(return_value="org_admin"))
    cli_output.set_output_format("json")
    with pytest.raises(SystemExit):
        study._resolve_lifecycle_inputs(SimpleNamespace(name="study", sites=sites, site_org=None), "add-site")
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "INVALID_ARGS"
    assert "--sites" in result["hint"]
