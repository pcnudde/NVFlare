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

import json
from unittest.mock import Mock

import pytest

import nvflare
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.private.admin_defs import Message
from nvflare.private.defs import SysCommandTopic, TrainingTopic
from nvflare.private.fed.client import scheduler_cmds as module
from nvflare.private.scheduler_constants import ShareableHeader


@pytest.fixture
def resources():
    engine = Mock(spec=module.ClientEngineInternalSpec)
    manager = Mock(spec=module.ResourceManagerSpec)
    engine.get_component.return_value = manager
    context = FLContext()
    engine.new_context.return_value = context
    return engine, manager, context


@pytest.mark.parametrize("failure", [True, False])
def test_cancel_reserved_resources_returns_explicit_acknowledgment_on_success_or_manager_failure(resources, failure):
    engine, manager, context = resources
    manager.cancel_resources.side_effect = ValueError("release failed") if failure else None
    processor = module.CancelResourceProcessor()
    assert processor.get_topics() == [TrainingTopic.CANCEL_RESOURCE]
    request = Message(TrainingTopic.CANCEL_RESOURCE, {"gpu": 1})
    request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, "reserve")
    response = processor.process(request, engine)
    assert response.topic == "reply_" + TrainingTopic.CANCEL_RESOURCE
    assert response.body.get_return_code() == (ReturnCode.EXECUTION_EXCEPTION if failure else ReturnCode.OK)
    manager.cancel_resources.assert_called_once_with(resource_requirement={"gpu": 1}, token="reserve", fl_ctx=context)


def test_start_job_rejects_missing_wire_information_before_resource_allocation(resources):
    engine, manager, _ = resources
    request = Mock(topic=TrainingTopic.START_JOB, body={}, get_header=Mock(side_effect=ValueError("broken headers")))
    processor = module.StartJobProcessor()
    assert processor.get_topics() == [TrainingTopic.START_JOB]
    assert "missing required information" in processor.process(request, engine).body
    manager.allocate_resources.assert_not_called()
    engine.start_app.assert_not_called()


def test_resource_report_routes_topic_and_serializes_manager_inventory(resources):
    engine, manager, context = resources
    manager.report_resources.return_value = {"gpu": 2}
    processor = module.ReportResourcesProcessor()
    assert processor.get_topics() == [SysCommandTopic.REPORT_RESOURCES]
    response = processor.process(Message(SysCommandTopic.REPORT_RESOURCES, ""), engine)
    assert json.loads(response.body) == {"gpu": 2}
    manager.report_resources.assert_called_once_with(context)
    assert module.CheckResourceProcessor().get_topics() == [TrainingTopic.CHECK_RESOURCE]


def test_version_report_survives_missing_installed_version_attribute(monkeypatch):
    monkeypatch.delattr(nvflare, "__version__")
    processor = module.ReportVersionProcessor()
    response = processor.process(Message(SysCommandTopic.REPORT_VERSION, ""), object())
    assert json.loads(response.body) == {"version": "unknown"}
