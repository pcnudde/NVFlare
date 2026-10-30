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

from nvflare.private.admin_defs import Message
from nvflare.private.defs import ComponentCallerTopic, RequestHeader
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.client.comp_caller_cmd import ComponentCallerProcessor
from nvflare.widgets.comp_caller import ComponentCaller
from nvflare.widgets.widget import WidgetID


@pytest.mark.parametrize("failure", ["engine", "widget"])
def test_component_command_requires_client_engine_and_registered_caller(failure):
    engine = Mock(spec=ClientEngineInternalSpec) if failure == "widget" else object()
    if failure == "widget":
        engine.get_widget = Mock(return_value=object())
    with pytest.raises(TypeError, match="caller must" if failure == "widget" else "engine must"):
        ComponentCallerProcessor().process(Message("call", {}), engine)


@pytest.mark.parametrize(
    "run_id,answer,expected",
    [
        (None, {}, {"error": "app not running"}),
        (-1, {}, {"error": "app not running"}),
        (1, {"result": 2}, {"result": 2}),
        (1, "bad", {}),
    ],
)
def test_component_command_dispatches_only_when_app_is_running(run_id, answer, expected):
    engine = Mock(spec=ClientEngineInternalSpec)
    caller = Mock(spec=ComponentCaller)
    caller.call_components.return_value = answer
    engine.get_widget = Mock(return_value=caller)
    engine.get_current_run_info = Mock(return_value=SimpleNamespace(job_id=run_id) if run_id is not None else None)
    request = Message("call", {"x": 1})
    request.set_header(RequestHeader.COMPONENT_TARGET, "trainer")
    request.set_header(RequestHeader.CALL_NAME, "inspect")
    processor = ComponentCallerProcessor()
    assert processor.get_topics() == [ComponentCallerTopic.CALL_COMPONENT]
    reply = processor.process(request, engine)
    assert reply.topic == "call" and reply.body == expected
    engine.get_widget.assert_called_once_with(WidgetID.COMPONENT_CALLER)
    if run_id == 1:
        caller.call_components.assert_called_once_with(target="trainer", call_name="inspect", params={"x": 1})
    else:
        caller.call_components.assert_not_called()
