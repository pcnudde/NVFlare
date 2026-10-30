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

from nvflare.apis.impl.controller import Controller
from nvflare.private.fed.server import server_json_config as module


@pytest.fixture
def configurator(monkeypatch):
    monkeypatch.setattr(module.FedJsonConfigurator, "process_config_element", lambda self, context, node: None)
    subject = module.ServerJsonConfigurator.__new__(module.ServerJsonConfigurator)
    subject.workflows = []
    subject.components = {}
    subject.handlers = []
    subject.authorize_and_build_component = Mock()
    return subject


@pytest.mark.parametrize("field", ["heart_beat_timeout", "task_request_interval"])
@pytest.mark.parametrize("value", ["bad", 0, -1])
def test_server_deadline_configuration_rejects_non_numeric_or_nonpositive_intervals(configurator, field, value):
    node = SimpleNamespace(element=value, path=lambda: "server." + field)
    with pytest.raises(module.ConfigError, match=field):
        configurator.process_config_element(object(), node)


@pytest.mark.parametrize("case", ["wrong_controller", "wrong_id", "duplicate_workflow", "duplicate_component", "valid"])
def test_workflow_configuration_rejects_type_and_id_collisions_before_registering_communicator(
    monkeypatch, configurator, case
):
    controller = Mock(spec=Controller)
    configurator.authorize_and_build_component.return_value = object() if case == "wrong_controller" else controller
    workflow_id = 7 if case == "wrong_id" else "workflow"
    node = SimpleNamespace(element={"id": workflow_id}, path=lambda: "workflows.#0")
    if case == "duplicate_workflow":
        configurator.workflows = [module.WorkFlow("workflow", controller)]
    if case == "duplicate_component":
        configurator.components["workflow"] = object()
    communicator = object()
    factory = Mock(return_value=communicator)
    monkeypatch.setattr(module, "WFCommServer", factory)
    if case == "valid":
        configurator.process_config_element(object(), node)
        assert configurator.workflows[0].controller is controller
        assert configurator.components["workflow"] is controller
        assert configurator.handlers == [communicator]
        controller.set_communicator.assert_called_once_with(communicator)
    else:
        with pytest.raises(module.ConfigError, match="controller|id|duplicate"):
            configurator.process_config_element(object(), node)
        factory.assert_not_called()
        controller.set_communicator.assert_not_called()
