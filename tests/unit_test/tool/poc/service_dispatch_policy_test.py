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

from nvflare.cli_exception import CLIException
from nvflare.tool.poc import poc_commands as subject
from tests.timing_utils import isolate_time


@pytest.fixture
def services():
    return {
        subject.SC.FLARE_SERVER: "server",
        subject.SC.FLARE_PROJ_ADMIN: "admin@example.com",
        subject.SC.FLARE_OTHER_ADMINS: ["user@example.com"],
        subject.SC.FLARE_CLIENTS: ["site-1", "site-2"],
    }


def test_service_command_discovery_filters_participants_and_orders_start_and_stop(tmp_path, monkeypatch, services):
    for name in ["site-1", "server", "admin@example.com", "user@example.com", "site-2", "unrelated"]:
        (tmp_path / name).mkdir()
    monkeypatch.setattr(subject, "get_prod_dir", Mock(return_value=str(tmp_path)))
    command = Mock(side_effect=lambda kind, prod, name, config, **kwargs: [kind, name])
    monkeypatch.setattr(subject, "get_service_command", command)
    monkeypatch.setattr(subject, "random", SimpleNamespace(randint=lambda *_: 2))
    started = subject._build_commands(
        subject.SC.CMD_START, "workspace", services, {"name": "project"}, ["site-2"], study="study"
    )
    assert started[0][0] == "server"
    assert started[-1][0] == "admin@example.com"
    assert {name for name, _ in started} == {"server", "site-1", "admin@example.com", "user@example.com"}
    stopped = subject._build_commands(
        subject.SC.CMD_STOP, "workspace", services, {"name": "project"}, [], ["server", "admin@example.com"]
    )
    assert [name for name, _ in stopped] == ["admin@example.com", "server"]
    assert subject._sort_service_cmds(subject.SC.CMD_START, [("site-1", ["start"])], services) == [
        ("site-1", ["start"])
    ]
    assert all(call.kwargs["study"] == "study" for call in command.call_args_list[:4])


def test_run_start_dispatches_server_clients_and_admin_through_mocked_process_boundaries(monkeypatch, services):
    commands = [("server", ["start-server"]), ("site-1", ["start-client"]), ("admin@example.com", ["start-admin"])]
    monkeypatch.setattr(subject, "_build_commands", Mock(return_value=commands))
    asynchronous, synchronous = Mock(), Mock()
    monkeypatch.setattr(subject, "async_process", asynchronous)
    monkeypatch.setattr(subject, "sync_process", synchronous)
    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock()
    subject._run_poc(
        subject.SC.CMD_START,
        "workspace",
        [2, 5],
        services,
        {"name": "project"},
        [],
        docker_container_names={"server": "server-container", "site-1": "client-container"},
        docker_network_name="network",
    )
    assert asynchronous.call_args_list[0].args[:3] == ("server", ["start-server"], None)
    assert asynchronous.call_args_list[1].args[:3] == ("site-1", ["start-client"], [2, 5])
    assert asynchronous.call_args_list[1].kwargs == {
        "docker_container_name": "client-container",
        "docker_network_name": "network",
    }
    synchronous.assert_called_once_with("admin@example.com", ["start-admin"])
    assert [c.args[0] for c in clock.sleep.call_args_list] == [1, 2]


@pytest.mark.parametrize("docker", [False, True])
def test_stop_wait_passes_bounded_waits_to_stop_boundary_and_observes_managed_services(monkeypatch, services, docker):
    commands = [("admin@example.com", ["stop-admin"]), ("site-1", ["stop-client"]), ("server", ["stop-server"])]
    monkeypatch.setattr(subject, "_build_commands", Mock(return_value=commands))
    stop, wait = Mock(), Mock()
    monkeypatch.setattr(subject, "_run_stop_command", stop)
    monkeypatch.setattr(subject, "_wait_for_poc_services_stopped", wait)
    monkeypatch.setattr(subject, "get_prod_dir", Mock(return_value="/fake/prod"))
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 1, 2, 3])
    services[subject.SC.IS_DOCKER_RUN] = docker
    subject._run_poc(subject.SC.CMD_STOP, "workspace", [], services, {"name": "project"}, [], wait=True)
    assert stop.call_count == 3
    assert all(0 < c.kwargs["timeout"] <= subject.POC_STOP_TIMEOUT for c in stop.call_args_list)
    if docker:
        wait.assert_not_called()
    else:
        wait.assert_called_once_with("/fake/prod", ["site-1", "server"], deadline=subject.POC_STOP_TIMEOUT)


def test_stop_wait_deadline_stops_before_launching_later_stop_command(monkeypatch, services):
    monkeypatch.setattr(subject, "_build_commands", Mock(return_value=[("server", ["stop"])]))
    stop = Mock()
    monkeypatch.setattr(subject, "_run_stop_command", stop)
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, subject.POC_STOP_TIMEOUT + 1])
    with pytest.raises(TimeoutError, match="timeout"):
        subject._run_poc(subject.SC.CMD_STOP, "workspace", [], services, {"name": "project"}, [], wait=True)
    stop.assert_not_called()


def test_jobs_link_replaces_owned_transfer_directory_and_preserves_source_contents(tmp_path, monkeypatch, services):
    source = tmp_path / "jobs"
    source.mkdir()
    (source / "job.py").write_text("job")
    prod = tmp_path / "prod"
    admin = prod / services[subject.SC.FLARE_PROJ_ADMIN]
    (admin / "startup").mkdir(parents=True)
    transfer = admin / "transfer"
    transfer.mkdir()
    (admin / "startup" / "fed_admin.json").write_text(json.dumps({"admin": {"upload_dir": "transfer"}}))
    monkeypatch.setattr(subject, "get_prod_dir", Mock(return_value=str(prod)))
    assert subject._link_jobs_dir_to_admin_transfer(str(source), "workspace", ({"name": "project"}, services))
    assert transfer.is_symlink()
    assert (transfer / "job.py").read_text() == "job"
    assert subject._link_jobs_dir_to_admin_transfer(
        str(source), "workspace", ({"name": "project"}, services), force=True
    )
    assert (source / "job.py").read_text() == "job"
    assert transfer.resolve() == source


@pytest.mark.parametrize("content", [None, "{invalid"])
def test_upload_configuration_errors_report_missing_or_invalid_admin_json(tmp_path, content):
    if content is not None:
        (tmp_path / "fed_admin.json").write_text(content)
    with pytest.raises(CLIException, match="failed to load"):
        subject.get_upload_dir(str(tmp_path))


def test_nvflare_home_normalizes_trailing_separator_and_host_validation_uses_fake_resolver(monkeypatch, capsys):
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    assert subject.get_nvflare_home() is None
    monkeypatch.setenv("NVFLARE_HOME", "/nvflare/")
    assert subject.get_nvflare_home() == "/nvflare"
    monkeypatch.setattr(
        subject,
        "socket",
        SimpleNamespace(gethostbyname=Mock(side_effect=["127.0.0.1", OSError("dns"), OSError("dns")])),
    )
    assert subject.verify_host("known")
    assert not subject.verify_host("unknown")
    monkeypatch.setattr(subject, "get_project_hosts", Mock(return_value=["unknown"]))
    subject.verify_hosts({})
    assert "unknown" in capsys.readouterr().out
