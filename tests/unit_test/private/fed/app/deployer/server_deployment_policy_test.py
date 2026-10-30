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

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.app.deployer import server_deployer as subject
from nvflare.private.fed.server.server_status import ServerStatus


@pytest.fixture
def deployer():
    value = subject.ServerDeployer()
    value.build(
        {
            "server_config": [{"name": "project", "service": {"target": "old:1234"}, "heart_beat_timeout": 15}],
            "secure_train": True,
            "app_validator": None,
            "server_host": "new",
            "snapshot_persistor": "persistor",
            "server_components": {SystemComponents.JOB_MANAGER: "manager"},
            "server_handlers": [],
        }
    )
    return value


def test_create_server_applies_configured_host_and_lifecycle_limits(deployer, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(subject, "FederatedServer", factory)
    args = Namespace(workspace="/workspace")
    config, server = deployer.create_fl_server(args, secure_train=True)
    assert config["service"]["target"] == "new:1234"
    assert server is factory.return_value
    factory.assert_called_once_with(
        project_name="project",
        min_num_clients=1,
        max_num_clients=100,
        cmd_modules=deployer.cmd_modules,
        heart_beat_timeout=15,
        args=args,
        secure_train=True,
        snapshot_persistor="persistor",
        shutdown_period=30.0,
        check_engine_frequency=3.0,
    )


def test_deploy_sets_run_manager_security_context_and_bootstrap_before_start(deployer, monkeypatch):
    args = Namespace(workspace="/workspace", config_folder="config")
    service = Mock()
    context = FLContext()
    service.engine.new_context.return_value = context
    monkeypatch.setattr(deployer, "create_fl_server", Mock(return_value=({"config": 1}, service)))
    run_manager = Mock()
    monkeypatch.setattr(subject, "RunManager", Mock(return_value=run_manager))
    workspace = Mock()
    monkeypatch.setattr(subject, "Workspace", Mock(return_value=workspace))
    job_runner = Mock()
    monkeypatch.setattr(subject, "JobRunner", Mock(return_value=job_runner))
    events = []
    service.engine.fire_event.side_effect = lambda event, ctx: events.append(event)
    monkeypatch.setattr(subject, "component_security_check", lambda ctx: events.append("security"))
    thread = Mock()
    thread.start.side_effect = lambda: events.append("thread-start")
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=Mock(return_value=thread)))
    result = deployer.deploy(args)
    assert result is service
    assert events == [EventType.SYSTEM_BOOTSTRAP, "security", "thread-start", EventType.SYSTEM_START]
    service.deploy.assert_called_once_with(args, grpc_args={"config": 1}, secure_train=True)
    service.engine.set_run_manager.assert_called_once_with(run_manager)
    service.engine.set_job_runner.assert_called_once_with(job_runner, "manager")
    run_manager.add_component.assert_called_once_with(SystemComponents.JOB_RUNNER, job_runner)
    assert context.get_prop(FLContextKey.WORKSPACE_OBJECT) is workspace
    assert context.get_prop(FLContextKey.ARGS) is args
    assert context.get_prop(FLContextKey.SITE_OBJ) is service
    assert context.get_prop(ReservedKey.RUN_ABORT_SIGNAL).triggered is False
    assert service.status == ServerStatus.STARTED
    deployer._start_job_runner(job_runner, context)
    job_runner.run.assert_called_once_with(context)


def test_deploy_rejects_insecure_components_before_launching_worker(deployer, monkeypatch):
    args = Namespace(workspace="/workspace", config_folder="config")
    service = Mock()
    service.engine.new_context.return_value = FLContext()
    monkeypatch.setattr(deployer, "create_fl_server", Mock(return_value=({}, service)))
    for name in ("Workspace", "RunManager", "JobRunner"):
        monkeypatch.setattr(subject, name, Mock())
    monkeypatch.setattr(subject, "component_security_check", Mock(side_effect=ValueError("unsafe component")))
    factory = Mock()
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=factory))
    with pytest.raises(ValueError, match="unsafe component"):
        deployer.deploy(args)
    factory.assert_not_called()
