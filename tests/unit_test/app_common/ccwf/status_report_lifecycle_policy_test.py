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

from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.ccwf.client_controller_executor import ClientControllerExecutor
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController


@pytest.fixture(params=["client-controller", "swarm"])
def controller(request):
    return ClientControllerExecutor([]) if request.param == "client-controller" else SwarmClientController()


def test_status_round_never_moves_backwards_and_completion_stays_latched(controller):
    controller.update_status(last_round=3, action="train", all_done=True)
    controller.update_status(last_round=2, action="save", all_done=False)
    status = controller.current_status
    assert status.last_round == 3 and status.all_done
    assert status.action == "save" and status.timestamp > 0


@pytest.mark.parametrize("state", ["unconfigured", "completed", "nothing-new"])
def test_pull_task_status_removes_only_its_own_stale_report(controller, monkeypatch, state):
    context = FLContext()
    reports = {"workflow": {"old": True}, "other": {"keep": True}}
    context.set_prop(Constant.STATUS_REPORTS, reports)
    if state != "unconfigured":
        controller.workflow_id = "workflow"
    controller.workflow_done = state == "completed"
    fresh = Mock(return_value=None)
    monkeypatch.setattr(controller, "_get_status_report", fresh)
    publish = Mock()
    monkeypatch.setattr(controller, "_add_status_report", publish)
    controller.handle_event(EventType.BEFORE_PULL_TASK, context)
    assert reports == (
        {"workflow": {"old": True}, "other": {"keep": True}} if state == "unconfigured" else {"other": {"keep": True}}
    )
    publish.assert_not_called()
    assert fresh.call_count == int(state == "nothing-new")
