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

import errno
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import yaml

from nvflare.cli_exception import CLIException
from nvflare.tool import cli_schema
from nvflare.tool.deploy import deploy_common, docker_deploy
from nvflare.tool.poc import poc_commands as subject
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("role", [deploy_common.ROLE_CLIENT, deploy_common.ROLE_SERVER])
def test_poc_docker_kit_preparation_delegates_deploy_boundary_and_writes_local_study_mapping(
    tmp_path, monkeypatch, role
):
    (tmp_path / "local").mkdir()
    (tmp_path / "startup").mkdir()
    path = tmp_path / "startup" / "fed_client.json"
    path.write_text(json.dumps({"servers": [{"service": {"target": "localhost:8002"}}]}))
    validation, preparation = Mock(), Mock()
    monkeypatch.setattr(docker_deploy, "validate_config", validation)
    monkeypatch.setattr(deploy_common, "validate_kit", Mock(return_value=SimpleNamespace(role=role)))
    monkeypatch.setattr(docker_deploy, "prepare", preparation)
    subject._prepare_poc_docker_kit(str(tmp_path), {"image": "training"}, str(tmp_path / "workspace"))
    validation.assert_called_once_with({"image": "training"})
    preparation.assert_called_once()
    runtime = yaml.safe_load((tmp_path / "local" / "study_runtime.yaml").read_text())
    assert runtime["studies"][subject.POC_DOCKER_DATA_STUDY]["datasets"][subject.POC_DOCKER_DATA_DATASET][
        "source"
    ] == str(tmp_path / "workspace" / "data")
    endpoint = json.loads(path.read_text())["servers"][0]["service"]["target"]
    assert endpoint == (
        f"{subject.POC_DOCKER_SERVER_ALIAS}:8002" if role == deploy_common.ROLE_CLIENT else "localhost:8002"
    )


@pytest.mark.parametrize("error", [SystemExit(4), RuntimeError("deployment")])
def test_docker_kit_preparation_wraps_external_deployer_failure_in_cli_error(tmp_path, monkeypatch, error):
    monkeypatch.setattr(docker_deploy, "validate_config", Mock(side_effect=error))
    with pytest.raises(CLIException, match="failed to prepare Docker"):
        subject._prepare_poc_docker_kit(str(tmp_path), {}, str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "fault,expected",
    [
        (None, (True, None)),
        (OSError(errno.EADDRINUSE, "in use"), (False, "in_use")),
        (OSError(errno.EACCES, "denied"), (False, "denied")),
    ],
)
def test_port_preflight_uses_fake_socket_and_classifies_bind_failure(monkeypatch, fault, expected):
    socket = MagicMock()
    socket.__enter__.return_value.bind.side_effect = fault
    factory = Mock(return_value=socket)
    monkeypatch.setattr(
        subject, "socket", SimpleNamespace(socket=factory, AF_INET=1, SOCK_STREAM=2, SOL_SOCKET=3, SO_REUSEADDR=4)
    )
    assert subject._is_local_port_available(8002, "127.0.0.1") == expected
    socket.__exit__.assert_called_once()
    assert subject._is_local_port_available("invalid") == (False, "invalid_port")
    assert subject._is_local_port_available(70000) == (False, "invalid_port")
    assert factory.call_count == 1


@pytest.mark.parametrize("stopped", [True, False])
def test_ensure_existing_poc_shutdown_obeys_finite_poll_budget_and_reports_timeout(monkeypatch, stopped):
    stop = Mock()
    monkeypatch.setattr(subject, "_stop_poc", stop)
    monkeypatch.setattr(subject, "is_poc_running", Mock(return_value=not stopped))
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(side_effect=[0, 1, 31])
    clock.sleep = Mock()
    if stopped:
        subject._ensure_poc_stopped(
            "workspace", project_config={"name": "project"}, service_config={"server": "server"}
        )
    else:
        with pytest.raises(CLIException, match="still running"):
            subject._ensure_poc_stopped(
                "workspace", project_config={"name": "project"}, service_config={"server": "server"}
            )
    stop.assert_called_once()
    assert clock.sleep.call_count == (0 if stopped else 1)


@pytest.mark.parametrize(
    "already_stopped,no_wait,message",
    [(True, False, "already stopped"), (False, True, "shutdown requested"), (False, False, "system stopped")],
)
def test_human_shutdown_result_distinguishes_stopped_requested_and_already_stopped(
    monkeypatch, capsys, already_stopped, no_wait, message
):
    monkeypatch.setattr(cli_schema, "handle_schema_flag", Mock())
    monkeypatch.setattr(subject, "get_poc_workspace", Mock(return_value="/fake/workspace"))
    stop = Mock(return_value={"already_stopped": already_stopped})
    monkeypatch.setattr(subject, "_stop_poc", stop)
    subject.stop_poc(SimpleNamespace(service=None, ex=None, no_wait=no_wait))
    assert message in capsys.readouterr().out
    assert stop.call_args.kwargs["wait"] is (not no_wait)
