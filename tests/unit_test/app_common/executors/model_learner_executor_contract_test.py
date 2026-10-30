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

from nvflare.apis.dxo import MetaKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_common.executors.model_learner_executor import ModelLearnerExecutor
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


@pytest.fixture
def executor():
    ex = ModelLearnerExecutor("learner")
    ex.learner = Mock(spec=ModelLearner)
    ex.learner_name = "Learner"
    return ex


def model_request():
    return FLModelUtils.to_shareable(FLModel(params={"weight": [1, 2]}))


@pytest.mark.parametrize(
    "result", [None, "ignored", object(), FLModel(params={"w": 3}), FLModel(metrics={"loss": 0.5})]
)
def test_train_retains_only_valid_pretrain_metrics(executor, result):
    executor.learner.validate.return_value = result
    executor.learner.train.return_value = FLModel(params={"w": 4})
    request = model_request()
    reply = executor.train(request, FLContext())
    assert reply.get_return_code() == ReturnCode.OK
    trained = FLModelUtils.from_shareable(reply)
    assert trained.params == {"w": 4}
    expected = {"loss": 0.5} if isinstance(result, FLModel) and result.metrics else None
    assert trained.meta.get(MetaKey.INITIAL_METRICS) == expected
    assert request.get_header(AppConstants.VALIDATE_TYPE) == ValidateType.BEFORE_TRAIN_VALIDATE


@pytest.mark.parametrize("method", ["train", "validate", "configure"])
def test_tasks_reject_payload_without_model_before_invoking_learner(executor, method):
    assert getattr(executor, method)(Shareable(), FLContext()).get_return_code() == ReturnCode.BAD_REQUEST_DATA
    getattr(executor.learner, method).assert_not_called()


@pytest.mark.parametrize("method,learner_method", [("validate", "validate"), ("submit_model", "get_model")])
@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("code", ReturnCode.TASK_ABORTED),
        ("empty", ReturnCode.EMPTY_RESULT),
        ("error", ReturnCode.EXECUTION_EXCEPTION),
        ("model", ReturnCode.OK),
    ],
)
def test_result_tasks_translate_learner_outcomes(executor, method, learner_method, outcome, expected):
    operation = getattr(executor.learner, learner_method)
    if outcome == "error":
        operation.side_effect = RuntimeError("failed")
    else:
        operation.return_value = {"code": ReturnCode.TASK_ABORTED, "empty": None, "model": FLModel(params={"w": 5})}[
            outcome
        ]
    request = model_request()
    request.set_header(AppConstants.SUBMIT_MODEL_NAME, "best")
    reply = getattr(executor, method)(request, FLContext())
    assert reply.get_return_code() == expected
    if outcome == "model":
        assert FLModelUtils.from_shareable(reply).params == {"w": 5}
    if learner_method == "get_model":
        operation.assert_called_once_with("best")


@pytest.mark.parametrize("stage", ["validate", "train"])
def test_training_exceptions_become_execution_exception(executor, stage):
    executor.learner.validate.return_value = None
    getattr(executor.learner, stage).side_effect = RuntimeError("failed")
    assert executor.train(model_request(), FLContext()).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    if stage == "validate":
        executor.learner.train.assert_not_called()


@pytest.mark.parametrize(
    "result,expected", [(ReturnCode.TASK_ABORTED, ReturnCode.TASK_ABORTED), (None, ReturnCode.EMPTY_RESULT)]
)
def test_training_invalid_results_are_reported(executor, result, expected):
    executor.learner.validate.return_value = None
    executor.learner.train.return_value = result
    assert executor.train(model_request(), FLContext()).get_return_code() == expected


@pytest.mark.parametrize("fails", [False, True])
def test_configuration_translates_failure_without_returning_arbitrary_learner_data(executor, fails):
    if fails:
        executor.learner.configure.side_effect = RuntimeError("invalid config")
    reply = executor.configure(model_request(), FLContext())
    assert reply.get_return_code() == (ReturnCode.EXECUTION_EXCEPTION if fails else ReturnCode.OK)


def test_setup_learner_sets_execution_context_and_preserves_cached_job_fields():
    learner = ModelLearner()
    ctx = Mock(spec=FLContext)
    values = {FLContextKey.ARGS: {"epochs": 2}, FLContextKey.CURRENT_RUN: "job-1"}
    ctx.get_prop.side_effect = lambda key: values.get(key)
    ctx.get_identity_name.return_value = "site-1"
    engine = ctx.get_engine.return_value
    workspace = engine.get_workspace.return_value
    workspace.get_root_dir.return_value = "/root"
    workspace.get_run_dir.return_value = "/root/job-1"
    workspace.get_app_dir.return_value = "/root/job-1/app"
    request = model_request()
    request.set_header(AppConstants.CURRENT_ROUND, 3)
    request.set_header(AppConstants.NUM_ROUNDS, 7)
    signal = Signal()
    ModelLearnerExecutor._setup_learner(learner, request, ctx, signal)
    assert (learner.site_name, learner.job_id, learner.current_round, learner.total_rounds) == ("site-1", "job-1", 3, 7)
    assert (learner.workspace_root, learner.job_root, learner.app_root) == ("/root", "/root/job-1", "/root/job-1/app")
    assert learner.abort_signal is signal
    assert learner.shareable is request and learner.fl_ctx is ctx
    ModelLearnerExecutor._setup_learner(learner, None, ctx, signal)
    engine.get_workspace.assert_called_once()


def test_dispatch_initializes_once_and_rejects_unknown_tasks(executor, monkeypatch):
    ctx = FLContext()
    setup = Mock()
    monkeypatch.setattr(executor, "_setup_learner", setup)
    initialize = Mock()
    monkeypatch.setattr(executor, "initialize", initialize)
    for _ in range(2):
        assert executor.execute("unrecognized", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN
    initialize.assert_called_once_with(ctx)
    assert setup.call_count == 2


def test_start_run_resolves_and_checks_component_type(executor):
    ctx = Mock(spec=FLContext)
    ctx.get_peer_context.return_value = None
    ctx.get_engine.return_value.get_component.return_value = executor.learner
    executor.handle_event(EventType.START_RUN, ctx)
    ctx.get_engine.return_value.get_component.assert_called_once_with("learner")
    ctx.get_engine.return_value.get_component.return_value = object()
    with pytest.raises(TypeError, match="ModelLearner"):
        executor.handle_event(EventType.START_RUN, ctx)


@pytest.mark.parametrize("unsafe", [False, True])
def test_abort_and_finalize_respect_unsafe_learner(executor, unsafe):
    executor.unsafe = unsafe
    ctx = FLContext()
    executor.handle_event(EventType.ABORT_TASK, ctx)
    executor.handle_event(EventType.END_RUN, ctx)
    assert executor.learner.abort.call_count == (0 if unsafe else 1)
    assert executor.learner.finalize.call_count == (0 if unsafe else 1)


def test_lifecycle_failures_are_contained_except_initialization(executor):
    ctx = Mock(spec=FLContext)
    ctx.get_peer_context.return_value = None
    executor.learner.initialize.side_effect = RuntimeError("initialization")
    with pytest.raises(RuntimeError, match="initialization"):
        executor.initialize(ctx)
    assert executor.learner.fl_ctx is ctx
    assert executor.learner.engine is ctx.get_engine.return_value
    executor.learner.abort.side_effect = RuntimeError("abort")
    executor.learner.finalize.side_effect = RuntimeError("finalize")
    executor.handle_event(EventType.ABORT_TASK, ctx)
    executor.finalize(ctx)
    executor.learner.abort.assert_called_once()
    executor.learner.finalize.assert_called_once()
