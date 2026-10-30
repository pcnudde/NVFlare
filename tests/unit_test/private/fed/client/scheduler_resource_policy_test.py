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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceConsumerSpec, ResourceManagerSpec
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader
from nvflare.private.fed.client import scheduler_cmds
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.scheduler_constants import ShareableHeader


@pytest.fixture
def resources():
    engine = Mock(spec=ClientEngineInternalSpec)
    manager, consumer, context = Mock(spec=ResourceManagerSpec), Mock(spec=ResourceConsumerSpec), FLContext()
    engine.new_context.side_effect = lambda: context
    engine.get_component.side_effect = lambda name: {
        SystemComponents.RESOURCE_MANAGER: manager,
        SystemComponents.RESOURCE_CONSUMER: consumer,
    }[name]
    request = Message("resources", {"gpu": 1})
    request.set_headers(
        {
            RequestHeader.JOB_ID: "job",
            RequestHeader.JOB_META: {"name": "model"},
            ShareableHeader.RESOURCE_RESERVE_TOKEN: "reservation",
        }
    )
    return engine, manager, consumer, context, request


@pytest.mark.parametrize("helper", [scheduler_cmds._get_resource_manager, scheduler_cmds._get_resource_consumer])
def test_resource_components_require_typed_engine_and_component(helper):
    with pytest.raises(ValueError, match="ClientEngineInternalSpec"):
        helper(object())
    engine = Mock(spec=ClientEngineInternalSpec)
    engine.get_component.return_value = object()
    with pytest.raises(ValueError, match="should be of type"):
        helper(engine)


@pytest.mark.parametrize("case", ["accepted", "blocked", "exception"])
def test_resource_check_publishes_context_and_honors_policy_before_reservation(resources, case):
    engine, manager, _, context, request = resources
    manager.check_resources.return_value = (True, "reservation")
    if case == "blocked":
        engine.fire_event.side_effect = lambda event, ctx: ctx.set_prop(FLContextKey.JOB_BLOCK_REASON, "privacy policy")
    elif case == "exception":
        manager.check_resources.side_effect = RuntimeError("private-sensitive-detail")
    reply = scheduler_cmds.CheckResourceProcessor().process(request, engine)
    assert context.get_prop(FLContextKey.CLIENT_RESOURCE_SPECS) == {"gpu": 1}
    assert context.get_prop(FLContextKey.CURRENT_JOB_ID) == "job"
    assert context.get_prop(FLContextKey.JOB_META) == {"name": "model"}
    engine.fire_event.assert_called_once_with(EventType.BEFORE_CHECK_RESOURCE_MANAGER, context)
    assert reply.body.get_header(ShareableHeader.IS_RESOURCE_ENOUGH) is (case == "accepted")
    token = reply.body.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
    if case == "blocked":
        assert token == "privacy policy"
        manager.check_resources.assert_not_called()
    elif case == "exception":
        assert "private-sensitive-detail" not in token and "exception" in token
    else:
        assert token == "reservation"
        manager.check_resources.assert_called_once_with(resource_requirement={"gpu": 1}, fl_ctx=context)


@pytest.mark.parametrize("failure", [None, "consumer", "start"])
def test_job_start_allocates_consumes_and_reclaims_resources_on_failed_launch(resources, failure):
    engine, manager, consumer, context, request = resources
    allocation = {"gpu_ids": [0]}
    manager.allocate_resources.return_value = allocation
    engine.start_app.return_value = ""
    if failure == "consumer":
        consumer.consume.side_effect = RuntimeError("consume failed")
    elif failure == "start":
        engine.start_app.side_effect = RuntimeError("start failed")
    reply = scheduler_cmds.StartJobProcessor().process(request, engine)
    manager.allocate_resources.assert_called_once_with(
        resource_requirement={"gpu": 1}, token="reservation", fl_ctx=context
    )
    consumer.consume.assert_called_once_with(allocation)
    if failure:
        assert "Start job execution exception" in reply.body
        manager.free_resources.assert_called_once_with(resources=allocation, token="reservation", fl_ctx=context)
    else:
        assert reply.body == "OK"
        engine.start_app.assert_called_once_with(
            "job",
            job_meta={"name": "model"},
            allocated_resource=allocation,
            token="reservation",
            resource_manager=manager,
        )
        manager.free_resources.assert_not_called()


def test_resource_report_is_json_and_zero_allocation_does_not_consume(resources):
    engine, manager, consumer, _, request = resources
    manager.allocate_resources.return_value = {}
    engine.start_app.return_value = "running"
    assert scheduler_cmds.StartJobProcessor().process(request, engine).body == "running"
    consumer.consume.assert_not_called()
    manager.report_resources.return_value = {"gpu": 2, "memory": 1024}
    report = scheduler_cmds.ReportResourcesProcessor().process(request, engine)
    assert json.loads(report.body) == {"gpu": 2, "memory": 1024}
