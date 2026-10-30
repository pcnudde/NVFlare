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
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.executors.splitnn_learner_executor import SplitNNLearnerExecutor


@pytest.fixture
def runtime():
    learner = Mock(spec=Learner)
    learner.init_model = Mock(return_value=Shareable())
    learner.train.return_value = Shareable()
    executor = SplitNNLearnerExecutor("learner", init_model_task_name="initialize", train_task_name="fit")
    executor.log_exception = Mock()
    engine = Mock()
    engine.get_component.return_value = learner
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    return executor, learner, engine, ctx


@pytest.mark.parametrize("task,method", [("initialize", "init_model"), ("fit", "train")])
def test_splitnn_delegates_named_task_and_lifecycle_with_original_context(runtime, task, method):
    executor, learner, engine, ctx = runtime
    executor.handle_event(EventType.START_RUN, ctx)
    learner.initialize.assert_called_once_with(engine.get_all_components.return_value, ctx)
    data, signal = Shareable(), Signal()
    result = executor.execute(task, data, ctx, signal)
    getattr(learner, method).assert_called_once_with(shareable=data, fl_ctx=ctx, abort_signal=signal)
    assert result is getattr(learner, method).return_value
    executor.handle_event(EventType.ABORT_TASK, ctx)
    learner.abort.assert_called_once_with(ctx)
    executor.handle_event(EventType.END_RUN, ctx)
    learner.finalize.assert_called_once_with(ctx)


@pytest.mark.parametrize("failure", ["wrong-component", "initialize", "train", "abort", "finalize"])
def test_splitnn_contains_user_extension_failures_and_unknown_tasks(runtime, failure):
    executor, learner, engine, ctx = runtime
    if failure == "wrong-component":
        engine.get_component.return_value = object()
    elif failure == "initialize":
        learner.initialize.side_effect = RuntimeError("initialize")
    executor.initialize(ctx)
    if failure in ["train", "abort", "finalize"]:
        getattr(learner, failure).side_effect = RuntimeError(failure)
        if failure == "train":
            assert (
                executor.execute("fit", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            )
        elif failure == "abort":
            executor.handle_event(EventType.ABORT_TASK, ctx)
        else:
            executor.finalize(ctx)
    executor.log_exception.assert_called_once()
    assert executor.execute("unknown", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN


def test_splitnn_teardown_before_initialization_is_safe():
    executor = SplitNNLearnerExecutor("learner")
    executor.handle_event(EventType.ABORT_TASK, FLContext())
    executor.handle_event(EventType.END_RUN, FLContext())
    assert executor.learner is None
