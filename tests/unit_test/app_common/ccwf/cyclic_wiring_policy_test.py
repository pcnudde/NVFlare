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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf import ccwf_job as job
from nvflare.app_common.ccwf import cyclic_client_ctl as subject
from nvflare.app_common.ccwf.common import Constant, CyclicOrder


def test_cyclic_job_wiring_adds_evaluation_workflow_and_one_shared_training_executor():
    executor = Mock(spec=Executor)
    client = job.CyclicClientConfig(
        executor, Mock(spec=ModelPersistor), Mock(spec=ShareableGenerator), learn_task_ack_timeout=7
    )
    server = job.CyclicServerConfig(num_rounds=3, participating_clients=["one", "two"], cyclic_order=CyclicOrder.RANDOM)
    workflow = job.CCWFJob()
    workflow.to_server = Mock()
    workflow.to_clients = Mock(side_effect=lambda component, **kwargs: kwargs.get("id", "auto"))
    workflow.add_cyclic(server, client, job.CrossSiteEvalConfig())
    workflow.add_cyclic(server, client)
    assert workflow.executor is executor
    executor_calls = [c for c in workflow.to_clients.call_args_list if c.args[0] is executor]
    assert len(executor_calls) == 1
    assert executor_calls[0].kwargs["tasks"] == ["train", "validate", "submit_model"]
    components = [c.args[0] for c in workflow.to_clients.call_args_list]
    cyclic = [c for c in components if isinstance(c, subject.CyclicClientController)]
    assert len(cyclic) == 2
    assert cyclic[0].learn_task_ack_timeout == 7
    assert cyclic[0].persistor_id == "persistor"
    server_controllers = [c.args[0] for c in workflow.to_server.call_args_list]
    assert len(server_controllers) == 3
    assert server_controllers[0].num_rounds == 3
    assert server_controllers[0].cyclic_order == CyclicOrder.RANDOM


@pytest.mark.parametrize("leg", ["first", "last", "random", "error"])
def test_cyclic_client_hands_updated_model_to_next_leg_and_rotates_random_round_boundaries(monkeypatch, leg):
    controller = subject.CyclicClientController()
    controller.me = "one" if leg == "first" else "two"
    controller.config = {Constant.ORDER: CyclicOrder.RANDOM if leg == "random" else CyclicOrder.FIXED}
    controller.update_status = Mock()
    controller.send_learn_task = Mock(return_value=True)
    controller.record_last_result = Mock()
    controller.broadcast_final_result = Mock()
    before, after = {"w": [1]}, {"w": [2]}
    controller.shareable_generator = Mock()
    controller.shareable_generator.shareable_to_learnable.side_effect = [before, after]
    controller.shareable_generator.learnable_to_shareable.return_value = Shareable()
    result = make_reply(ReturnCode.EXECUTION_EXCEPTION if leg == "error" else ReturnCode.OK)
    controller.execute_learn_task = Mock(return_value=result)
    data = Shareable()
    controller._set_task_headers(data, 2, 0, ["one", "two"])
    if leg == "random":

        def shuffle(values):
            values[:] = ["two", "one"]

        monkeypatch.setattr(subject, "random", SimpleNamespace(shuffle=shuffle))
    ctx = FLContext()
    controller.do_learn_task("train", data, ctx, Signal())
    assert ctx.get_prop(AppConstants.GLOBAL_MODEL) is before
    if leg == "error":
        controller.send_learn_task.assert_not_called()
        assert controller.update_status.call_args.kwargs["error"] == ReturnCode.EXECUTION_EXCEPTION
    else:
        call = controller.send_learn_task.call_args.kwargs
        assert call["targets"] == (["two"] if leg == "first" else ["one"])
        assert call["request"].get_header(AppConstants.CURRENT_ROUND) == (0 if leg == "first" else 1)
        assert call["request"].get_header(Constant.CLIENT_ORDER) == ["one", "two"]
        controller.shareable_generator.learnable_to_shareable.assert_called_once_with(after, ctx)
        controller.broadcast_final_result.assert_not_called()
