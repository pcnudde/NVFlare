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

from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.client import client_app_runner as module


@pytest.mark.parametrize("privacy", ["undefined", "empty", "components"])
def test_client_runner_configures_authorization_before_runtime_and_privacy_components(monkeypatch, privacy):
    args = Namespace(workspace="/workspace", client_name="site", client_config="client.json", job_id="job", set=[])
    workspace = object()
    workspace_factory = Mock(return_value=workspace)
    monkeypatch.setattr(module, "Workspace", workspace_factory)
    config = SimpleNamespace(components={}, handlers=[], add_component=Mock())
    conf = Mock(runner_config=config, config_data={"executors": []})
    conf_factory = Mock(return_value=conf)
    monkeypatch.setattr(module, "ClientJsonConfigurator", conf_factory)
    metadata = {"job": "meta"}
    monkeypatch.setattr(module, "get_job_meta_from_workspace", Mock(return_value=metadata))
    monkeypatch.setattr(module, "get_client_execution_lifetime", Mock(return_value="task"))
    validate = Mock()
    monkeypatch.setattr(module, "validate_cj_execution_lifetime", validate)
    component = object()
    manager = Mock(components={"privacy": component} if privacy == "components" else {})
    manager.is_policy_defined.return_value = privacy != "undefined"
    monkeypatch.setattr(module, "create_privacy_manager", Mock(return_value=manager))
    service = Mock()
    monkeypatch.setattr(module, "PrivacyService", SimpleNamespace(initialize=service))
    run_manager = Mock()
    context = FLContext()
    run_manager.new_context.return_value = context
    runner = Mock()
    factory = Mock(return_value=runner)
    monkeypatch.setattr(module, "ClientRunner", factory)
    subject = module.ClientAppRunner()
    subject.create_run_manager = Mock(return_value=run_manager)
    client = SimpleNamespace(client_args={"fqsn": "site"})
    handlers = [object()]
    assert subject.create_client_runner("/app", args, "config", client, True, handlers) is runner
    assert args.set == ["secure_train=True"]
    validate.assert_called_once_with(metadata, "site", "task")
    conf.set_component_build_authorizer.assert_called_once()
    assert conf.set_component_build_authorizer.call_args.kwargs["event_handlers"] is handlers
    conf.configure.assert_called_once_with()
    assert config.add_component.call_count == int(privacy == "components")
    if privacy == "components":
        config.add_component.assert_called_once_with("privacy", component)
    service.assert_called_once_with(manager)
    assert client.run_manager is run_manager and client.runner_config is config and client.handlers is config.handlers
    assert factory.call_args.kwargs == dict(
        config=config, client_config={"fqsn": "site"}, job_id="job", engine=run_manager
    )
    run_manager.add_handler.assert_called_once_with(runner)
    workspace_factory.assert_called_once_with("/workspace", "site", "config")


def test_client_run_manager_factory_and_parent_sync_preserve_job_identity(monkeypatch):
    args = Namespace(client_name="site", job_id="job")
    conf = SimpleNamespace(runner_config=SimpleNamespace(components={"component": object()}, handlers=[]))
    manager = Mock()
    context = FLContext()
    manager.new_context.return_value = context
    factory = Mock(return_value=manager)
    monkeypatch.setattr(module, "ClientRunManager", factory)
    subject = module.ClientAppRunner()
    client = SimpleNamespace(run_manager=manager)
    assert subject.create_run_manager(args, conf, client, "workspace") is manager
    factory.assert_called_once_with(
        client_name="site",
        job_id="job",
        workspace="workspace",
        client=client,
        components=conf.runner_config.components,
        handlers=[],
        conf=conf,
    )
    subject.sync_up_parents_process(client)
    manager.get_job_clients.assert_called_once_with(context)
    command = Mock()
    monkeypatch.setattr(module, "CommandAgent", Mock(return_value=command))
    subject.start_command_agent(args, client, context)
    command.start.assert_called_once_with(context)
    subject.close()
    command.shutdown.assert_called_once_with()
