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

import importlib
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.dashboard.utils import EnvVar
from tests.timing_utils import isolate_time

pytest.importorskip("docker", reason="dashboard launch policies require the optional Docker client")
subject = importlib.import_module("nvflare.dashboard.cli")


@pytest.fixture
def args(tmp_path):
    return Namespace(
        local=False,
        folder=str(tmp_path),
        env=["MODE=demo"],
        passphrase="passphrase",
        cred="alice@example.com:password:org",
        image="example/dashboard:latest",
        port=8443,
    )


@pytest.fixture
def docker(monkeypatch):
    # Other suites replace Docker SDK modules during collection. Own the exception
    # hierarchy together with the client boundary so collection order cannot leak in.
    class DockerException(Exception):
        pass

    class APIError(DockerException):
        pass

    class NotFound(APIError):
        pass

    client = Mock()
    errors = SimpleNamespace(DockerException=DockerException, APIError=APIError, NotFound=NotFound)
    proxy = SimpleNamespace(from_env=Mock(return_value=client), errors=errors)
    monkeypatch.setattr(subject, "docker", proxy)
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "sleep", Mock())
    return client


def test_docker_launch_passes_volume_ports_and_explicit_credentials_without_real_daemon(args, docker):
    container = docker.containers.run.return_value
    container.status = "running"
    subject.start(args)
    docker.images.pull.assert_called_once_with(args.image)
    call = docker.containers.run.call_args
    assert call.args == (args.image,)
    assert call.kwargs["entrypoint"] == ["python3", "-m", "nvflare.dashboard.wsgi"]
    assert call.kwargs["ports"] == {8443: 8443}
    assert call.kwargs["volumes"] == {args.folder: {"bind": "/var/tmp/nvflare/dashboard", "mode": "rw"}}
    assert call.kwargs["environment"] == {
        "MODE": "demo",
        EnvVar.DASHBOARD_PP: "passphrase",
        EnvVar.CREDENTIAL: args.cred,
    }
    assert call.kwargs["detach"] and call.kwargs["auto_remove"]
    container.reload.assert_called_once_with()


@pytest.mark.parametrize("failure,code", [("daemon", 0), ("pull", 1), ("run", 1)])
def test_docker_launch_reports_boundary_failure_without_starting_unowned_container(args, docker, failure, code):
    if failure == "daemon":
        subject.docker.from_env.side_effect = subject.docker.errors.DockerException("offline")
    elif failure == "pull":
        docker.images.pull.side_effect = subject.docker.errors.APIError("pull failed")
    else:
        docker.containers.run.side_effect = subject.docker.errors.APIError("run failed")
    with pytest.raises(SystemExit) as raised:
        subject.start(args)
    assert raised.value.code == code
    assert docker.containers.run.call_count == int(failure == "run")


@pytest.mark.parametrize("failure", ["removed", "exited", "logs-unavailable"])
def test_early_container_exit_is_reported_with_logs_when_available(docker, failure, capsys):
    container = Mock()
    container.status = "exited"
    container.logs.return_value = b"startup failed\n"
    if failure == "removed":
        container.reload.side_effect = subject.docker.errors.NotFound("removed")
    elif failure == "logs-unavailable":
        container.logs.side_effect = subject.docker.errors.APIError("unavailable")
    with pytest.raises(SystemExit) as raised:
        subject._fail_if_container_exited(container)
    assert raised.value.code == 1
    output = capsys.readouterr().out
    assert ("removed" if failure == "removed" else "exited immediately") in output
    if failure == "exited":
        assert "startup failed" in output


def test_local_launch_prompts_until_admin_identity_valid_then_returns_local_process_status(args, monkeypatch, docker):
    args.local = True
    args.cred = None
    args.passphrase = None
    args.env = []
    responses = Mock(side_effect=["not-an-email", "admin@example.com", "bad org name", "org"])
    monkeypatch.setattr("builtins.input", responses)
    monkeypatch.setattr(subject.utils, "generate_password", Mock(return_value="generated"))
    local = Mock(return_value="process-status")
    monkeypatch.setattr(subject, "start_local", local)
    assert subject.start(args) == "process-status"
    assert responses.call_count == 4
    assert local.call_args.args[0][EnvVar.CREDENTIAL] == "admin@example.com:generated:org"
    subject.docker.from_env.assert_not_called()


def test_local_process_invocation_uses_current_interpreter_and_sets_web_root(monkeypatch):
    process = Mock(return_value="status")
    monkeypatch.setattr(subject, "subprocess", SimpleNamespace(run=process))
    environment = {"MODE": "demo"}
    assert subject.start_local(environment) == "status"
    kwargs = process.call_args.kwargs
    assert kwargs["args"][0] == subject.sys.executable
    assert kwargs["args"][1].endswith("/dashboard/wsgi.py")
    assert kwargs["env"] is environment
    assert environment[EnvVar.WEB_ROOT].endswith("/dashboard")


@pytest.mark.parametrize("failure", [None, "daemon", "missing"])
def test_dashboard_stop_signals_only_named_container_and_contains_missing_runtime(docker, failure):
    if failure == "daemon":
        subject.docker.from_env.side_effect = subject.docker.errors.DockerException("offline")
    elif failure == "missing":
        docker.containers.get.side_effect = subject.docker.errors.NotFound("missing")
    if failure:
        with pytest.raises(SystemExit) as raised:
            subject.stop()
        assert raised.value.code == 0
    else:
        subject.stop()
        docker.containers.get.assert_called_once_with("nvflare-dashboard")
        docker.containers.get.return_value.kill.assert_called_once_with(signal=subject.signal.SIGINT)
