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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.app.client import client_train
from nvflare.private.fed.client.client_status import ClientStatus


@pytest.fixture
def startup(monkeypatch, tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    args = SimpleNamespace(workspace=str(tmp_path), set=["uid=site-1", "heart_beat_interval=2.5"], local_rank=2)
    monkeypatch.chdir(tmp_path)
    client = Mock(token="registered", status=ClientStatus.STOPPED, sp_established=False, cell=None)
    client._get_project_name.return_value = "project"
    client.servers = {"project": {"target": "host:8002", "scheme": "tcp"}}
    cell = object()

    def connect(*args):
        client.sp_established = True
        client.cell = cell

    client.connect_to_server.side_effect = connect
    deployer = Mock(
        secure_train=True, client_config={"client": True}, server_config={"server": True}, req_processors=[]
    )
    deployer.create_fed_client.return_value = client
    config = Mock(base_deployer=deployer, site_org="org", app_validator=object())
    monkeypatch.setattr(client_train, "FLClientStarterConfiger", Mock(return_value=config))
    context = FLContext()
    engine = Mock()
    engine.new_context.side_effect = lambda: context
    engine_factory = Mock(return_value=engine)
    monkeypatch.setattr(client_train, "ClientEngine", engine_factory)

    def make_thread(target, daemon):
        thread = Mock()
        thread.start.side_effect = target
        thread.is_alive.return_value = False
        return thread

    thread_factory = Mock(side_effect=make_thread)
    monkeypatch.setattr(client_train, "threading", SimpleNamespace(Thread=thread_factory))
    for name in (
        "fobs_initialize",
        "configure_logging",
        "security_init",
        "component_security_check",
        "create_admin_agent",
    ):
        monkeypatch.setattr(client_train, name, Mock())
    monkeypatch.setattr(client_train, "create_privacy_manager", Mock(return_value=None))
    monkeypatch.setattr(client_train.PrivacyService, "initialize", Mock())
    sleep = Mock(side_effect=AssertionError("startup must not poll with a completed fake connection"))
    monkeypatch.setattr(client_train, "time", SimpleNamespace(sleep=sleep))
    return SimpleNamespace(
        args=args,
        client=client,
        engine=engine,
        context=context,
        engine_factory=engine_factory,
        deployer=deployer,
        config=config,
        cell=cell,
        thread_factory=thread_factory,
    )


def test_client_startup_applies_identity_security_registration_and_shutdown_order(startup):
    args = startup.args
    client_train.main(args)
    assert args.client_config == JobConstants.CLIENT_JOB_CONFIG and args.job_id is None
    assert args.log_config is None and args.env.startswith("config/") and args.train_config.startswith("config/")
    startup.config.configure.assert_called_once_with()
    startup.client.connect_to_server.assert_called_once_with("project", "host:8002", "tcp")
    startup.engine_factory.assert_called_once_with(startup.client, args, 2)
    startup.engine.initialize_comm.assert_called_once_with(startup.cell)
    assert startup.client.use_gpu is False and startup.client.config_folder == ""
    assert startup.context.get_prop(FLContextKey.CLIENT_CONFIG) == {"client": True}
    assert startup.context.get_prop(FLContextKey.SERVER_CONFIG) == {"server": True}
    assert startup.context.get_prop(FLContextKey.SECURE_MODE) is True
    assert startup.context.get_prop(FLContextKey.CLIENT_TOKEN) == "registered"
    assert startup.context.get_prop(FLContextKey.SITE_OBJ) is startup.client
    events = [call.args[0] for call in startup.engine.fire_event.call_args_list]
    assert events == [EventType.SYSTEM_BOOTSTRAP, EventType.BEFORE_CLIENT_REGISTER, EventType.AFTER_CLIENT_REGISTER]
    startup.client.register.assert_called_once_with(startup.context)
    startup.client.start_heartbeat.assert_called_once_with(interval=2.5)
    startup.deployer.close.assert_called_once_with()
    client_train.create_admin_agent.assert_called_once_with([], startup.client, startup.engine)


def test_custom_config_folder_and_stale_restart_markers_are_handled_before_start(startup, tmp_path):
    startup.args.set.append("config_folder=custom-config")
    for name in ("restart.fl", "shutdown.fl"):
        (tmp_path / name).write_text("stale")
    client_train.main(startup.args)
    assert startup.args.client_config == "custom-config/" + JobConstants.CLIENT_JOB_CONFIG
    assert startup.client.config_folder == "custom-config"
    assert not (tmp_path / "restart.fl").exists() and not (tmp_path / "shutdown.fl").exists()


def test_missing_server_target_fails_before_connection_or_engine_start(startup):
    startup.client.servers = {"project": {"scheme": "tcp"}}
    with pytest.raises(RuntimeError, match="missing 'target'"):
        client_train.main(startup.args)
    startup.client.connect_to_server.assert_not_called()
    startup.engine_factory.assert_not_called()


def test_connection_exception_is_propagated_before_creating_engine(startup):
    startup.client.connect_to_server.side_effect = RuntimeError("unreachable")
    with pytest.raises(RuntimeError, match="failed to connect") as failure:
        client_train.main(startup.args)
    assert str(failure.value.__cause__) == "unreachable"
    startup.engine_factory.assert_not_called()


@pytest.mark.parametrize(
    "established,expected", [(False, "before establishing connection"), (True, "before creating the client cell")]
)
def test_connection_thread_exit_is_detected_in_both_readiness_phases(startup, established, expected):
    def incomplete_connect(*args):
        startup.client.sp_established = established
        startup.client.cell = None

    startup.client.connect_to_server.side_effect = incomplete_connect
    with pytest.raises(RuntimeError, match=expected):
        client_train.main(startup.args)
    startup.client.register.assert_not_called()


def test_missing_registration_token_prevents_heartbeat_and_admin_start(startup):
    startup.client.token = None
    with pytest.raises(RuntimeError, match="Login failed"):
        client_train.main(startup.args)
    startup.client.start_heartbeat.assert_not_called()
    client_train.create_admin_agent.assert_not_called()


def test_configuration_errors_are_rendered_without_starting_client(startup, capsys):
    startup.config.configure.side_effect = client_train.ConfigError("invalid configuration")
    client_train.main(startup.args)
    assert "ConfigError:" in capsys.readouterr().out
    startup.deployer.create_fed_client.assert_not_called()


def test_stale_marker_removal_error_exits_before_configuration(startup, tmp_path, monkeypatch, capsys):
    (tmp_path / "restart.fl").write_text("stale")
    monkeypatch.setattr(
        client_train, "os", SimpleNamespace(path=client_train.os.path, remove=Mock(side_effect=OSError("denied")))
    )
    with pytest.raises(SystemExit) as error:
        client_train.main(startup.args)
    assert error.value.code == -1 and "Could not remove" in capsys.readouterr().out
    startup.config.configure.assert_not_called()


def test_admin_agent_registration_connects_all_processors_and_starts_system(monkeypatch):
    agent = Mock()
    factory = Mock(return_value=agent)
    monkeypatch.setattr(client_train, "FedAdminAgent", factory)
    client, engine, context = Mock(cell=object()), Mock(), FLContext()
    engine.new_context.return_value = context
    processors = [object(), object()]
    assert client_train.create_admin_agent(processors, client, engine) is agent
    factory.assert_called_once_with(client_name="admin_agent", cell=client.cell, app_ctx=engine)
    engine.set_agent.assert_called_once_with(agent)
    client.set_client_engine.assert_called_once_with(engine)
    assert [c.args[0] for c in agent.register_processor.call_args_list] == processors
    engine.fire_event.assert_called_once_with(EventType.SYSTEM_START, context)


def test_client_startup_argument_parser_requires_kit_and_supports_local_rank(monkeypatch):
    monkeypatch.setattr(
        client_train.sys,
        "argv",
        [
            "client",
            "--workspace",
            "/workspace",
            "--fed_client",
            "client.json",
            "--local_rank",
            "3",
            "--set",
            "uid=site-1",
        ],
    )
    args = client_train.parse_arguments()
    assert args.local_rank == 3 and args.fed_client == "client.json" and args.set == ["uid=site-1"]
    monkeypatch.setattr(client_train.sys, "argv", ["client", "--workspace", "/workspace"])
    with pytest.raises(SystemExit):
        client_train.parse_arguments()
