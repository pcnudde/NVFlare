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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
from nvflare.tool import cli_output
from nvflare.tool.system import system_cli as subject


@pytest.mark.parametrize("operation", ["enable", "disable"])
@pytest.mark.parametrize("failure", ["authentication", "connection", "unexpected"])
def test_client_state_command_preserves_auth_failures_and_reports_unexpected_connection_errors(
    monkeypatch, capsys, operation, failure
):
    monkeypatch.setattr(cli_output, "_output_format", "json")
    session = Mock()
    error = {
        "authentication": AuthenticationError("denied"),
        "connection": NoConnection("offline"),
        "unexpected": ValueError("bad reply"),
    }[failure]
    getattr(session, operation + "_client").side_effect = error
    monkeypatch.setattr(subject, "_system_session", lambda args: nullcontext(session))
    args = SimpleNamespace(client_name="site", force=True)
    if failure != "unexpected":
        with pytest.raises(type(error)) as raised:
            getattr(subject, "cmd_system_" + operation + "_client")(args)
        assert raised.value is error
    else:
        with pytest.raises(SystemExit) as raised:
            getattr(subject, "cmd_system_" + operation + "_client")(args)
        assert raised.value.code == 2
        reply = json.loads(capsys.readouterr().out)
        assert reply["error_code"] == "CONNECTION_FAILED" and "bad reply" in reply["message"]
    getattr(session, operation + "_client").assert_called_once_with("site")


def test_system_presentation_handles_absent_client_state_and_unknown_timestamps(monkeypatch, capsys):
    assert subject._first_client_result({}, "site", "disabled", False) == {
        "client_name": "site",
        "state": "disabled",
        "credential_revoked": False,
        "rejoin_allowed": False,
    }
    assert subject._fmt_ts(None) == "unknown"
    assert subject._fmt_ts("invalid") == "invalid"
    assert (
        subject._get_system_state_change_timeout(SimpleNamespace(timeout=True))
        == subject._DEFAULT_SYSTEM_STATE_CHANGE_TIMEOUT
    )
    monkeypatch.setattr(cli_output, "_output_format", "human")
    subject._render_version_human({"sites": [], "compatible": True, "admin_version": "1"})
    assert "Mismatched sites: none" in capsys.readouterr().out
