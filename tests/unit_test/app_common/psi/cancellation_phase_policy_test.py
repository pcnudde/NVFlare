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

from unittest.mock import Mock, call

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.psi.psi_controller import PSIController
from nvflare.app_common.psi.psi_workflow_spec import PSIWorkflow


@pytest.mark.parametrize("phase", ["before", "pre_process", "run", "post_process", None])
def test_psi_controller_stops_at_first_cancelled_phase(phase):
    abort = Signal()
    workflow = Mock(spec=PSIWorkflow)
    if phase == "before":
        abort.trigger(True)
    elif phase:
        getattr(workflow, phase).side_effect = lambda signal: signal.trigger(True)
    context = FLContext()
    engine = Mock()
    engine.get_clients.return_value = ["one", "two"]
    engine.get_component.return_value = workflow
    context.set_prop(ReservedKey.ENGINE, engine)
    controller = PSIController("psi")
    controller.start_controller(context)
    assert controller.fl_ctx is context and controller.psi_workflow is workflow
    workflow.initialize.assert_called_once_with(context, controller=controller)
    result = controller.control_flow(abort, context)
    names = ["pre_process", "run", "post_process"]
    expected = [] if phase == "before" else names[: names.index(phase) + 1] if phase else names
    assert workflow.mock_calls[1:] == [getattr(call, name)(abort) for name in expected]
    assert result is (False if phase else None)
    controller.stop_controller(context)
    workflow.finalize.assert_called_once_with(context)
