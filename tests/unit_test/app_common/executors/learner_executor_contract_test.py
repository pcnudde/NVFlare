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

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_common.executors.learner_executor import LearnerExecutor


@pytest.fixture
def runtime():
    executor = LearnerExecutor("learner")
    executor.learner = Mock(spec=Learner)
    engine = Mock()
    engine.get_component.return_value = executor.learner
    engine.get_all_components.return_value = {"learner": executor.learner}
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    return executor, ctx


def test_initialization_resolves_typed_learner_once_across_tasks(runtime):
    executor, ctx = runtime
    executor.learner.get_model_for_validation.return_value = DXO(DataKind.WEIGHTS, {"w": 1}).to_shareable()
    request = Shareable()
    request.set_header(AppConstants.SUBMIT_MODEL_NAME, "best")
    assert executor.execute(executor.submit_model_task, request, ctx, Signal()).get_return_code() == ReturnCode.OK
    assert executor.execute("unknown", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN
    executor.learner.initialize.assert_called_once_with({"learner": executor.learner}, ctx)
    executor.learner.get_model_for_validation.assert_called_once_with("best", ctx)


@pytest.mark.parametrize("invalid", [None, object()])
def test_initialization_refuses_non_learner_components(runtime, invalid):
    executor, ctx = runtime
    ctx.get_engine().get_component.return_value = invalid
    with pytest.raises(TypeError, match="Learner"):
        executor.initialize(ctx)


@pytest.mark.parametrize("validation", ["valid", "none", "wrong-type", "malformed", "error"])
def test_pretrain_validation_sets_initial_metrics_only_for_valid_dxo(runtime, validation):
    executor, ctx = runtime
    metrics = DXO(DataKind.METRICS, {MetaKey.INITIAL_METRICS: 0.7}).to_shareable()
    executor.learner.validate.return_value = {
        "valid": metrics,
        "none": None,
        "wrong-type": 5,
        "malformed": Shareable(),
        "error": make_reply(ReturnCode.ERROR),
    }[validation]
    trained = DXO(DataKind.WEIGHTS, {"w": 4}).to_shareable()
    executor.learner.train.return_value = trained
    request = DXO(DataKind.WEIGHTS, {"w": 1}).to_shareable()
    reply = executor.train(request, ctx, Signal())
    assert from_shareable(reply).data == {"w": 4}
    assert request.get_header(AppConstants.VALIDATE_TYPE) == ValidateType.BEFORE_TRAIN_VALIDATE
    if validation == "valid":
        assert from_shareable(reply).meta[MetaKey.INITIAL_METRICS] == 0.7
    else:
        assert reply is trained
        assert MetaKey.INITIAL_METRICS not in from_shareable(reply).meta


@pytest.mark.parametrize("operation", ["train", "validate", "submit_model"])
@pytest.mark.parametrize("invalid", [None, "not-shareable"])
def test_learner_results_must_be_shareables(runtime, operation, invalid):
    executor, ctx = runtime
    executor.learner.validate.return_value = invalid
    executor.learner.train.return_value = invalid
    executor.learner.get_model_for_validation.return_value = invalid
    if operation == "submit_model":
        result = executor.submit_model(Shareable(), ctx)
    else:
        result = getattr(executor, operation)(Shareable(), ctx, Signal())
    assert result.get_return_code() == ReturnCode.EMPTY_RESULT


def test_validation_dispatch_sets_model_validate_header(runtime):
    executor, ctx = runtime
    executor.is_initialized = True
    reply = DXO(DataKind.METRICS, {"loss": 0.2}).to_shareable()
    executor.learner.validate.return_value = reply
    request = Shareable()
    assert executor.execute(executor.validate_task, request, ctx, Signal()) is reply
    assert request.get_header(AppConstants.VALIDATE_TYPE) == ValidateType.MODEL_VALIDATE


@pytest.mark.parametrize("unsafe", [False, True])
def test_abort_finalize_respect_unsafe_policy_and_contain_learner_errors(runtime, unsafe):
    executor, ctx = runtime
    executor.unsafe = unsafe
    executor.learner.abort.side_effect = RuntimeError("abort failed")
    executor.learner.finalize.side_effect = RuntimeError("finalize failed")
    executor.handle_event(EventType.ABORT_TASK, ctx)
    executor.handle_event(EventType.END_RUN, ctx)
    assert executor.learner.abort.call_count == (0 if unsafe else 1)
    assert executor.learner.finalize.call_count == (0 if unsafe else 1)
