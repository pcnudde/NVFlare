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

from unittest.mock import MagicMock

import joblib
import pytest

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.sklearn import sklearn_executor as subject


@pytest.fixture
def executor(tmp_path, monkeypatch):
    learner = MagicMock(spec=Learner)
    learner.n_samples = 7
    learner.train.return_value = ({"weight": [3.0]}, {"local": True})
    learner.validate.return_value = ({"accuracy": 0.75}, {"global": True})
    engine = MagicMock()
    engine.get_component.return_value = learner
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path))
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site")
    writer = MagicMock()
    monkeypatch.setattr(subject.tensorboard.summary, "Writer", MagicMock(return_value=writer), raising=False)
    executor = subject.SKLearnExecutor("learner")
    executor.handle_event(EventType.START_RUN, ctx)
    return executor, learner, ctx, writer


@pytest.mark.parametrize("round_number", [0, 2])
def test_training_saves_models_metrics_and_sample_count(executor, round_number):
    executor, learner, ctx, writer = executor
    data = DXO(DataKind.WEIGHTS, {"weight": [1.0]}).to_shareable()
    data.set_header(AppConstants.CURRENT_ROUND, round_number)
    result = executor.execute(AppConstants.TASK_TRAIN, data, ctx, Signal())
    assert result.get_return_code() == ReturnCode.OK
    assert from_shareable(result).data == {"weight": [3.0]}
    assert from_shareable(result).get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND) == 7
    assert ctx.get_prop(AppConstants.CURRENT_ROUND) == round_number
    assert joblib.load(executor.local_model_path) == {"local": True}
    learner.initialize.assert_called_once()
    if round_number:
        assert joblib.load(executor.global_model_path) == {"global": True}
        writer.add_scalar.assert_called_once_with("accuracy", 0.75, round_number)
        writer.flush.assert_called_once_with()
    else:
        learner.validate.assert_not_called()
    executor.handle_event(EventType.ABORT_TASK, ctx)
    learner.abort.assert_called_once_with(ctx)
    executor.handle_event(EventType.END_RUN, ctx)
    learner.finalize.assert_called_once_with(ctx)


def test_unknown_aborted_and_failed_tasks_have_distinct_return_codes(executor):
    executor, learner, ctx, _ = executor
    data = DXO(DataKind.WEIGHTS, {}).to_shareable()
    assert executor.execute("unknown", data, ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN
    signal = Signal()
    signal.trigger(True)
    assert executor.execute("train", data, ctx, signal).get_return_code() == ReturnCode.TASK_ABORTED
    learner.finalize.assert_called_once_with(ctx)
    data.set_header(AppConstants.CURRENT_ROUND, 1)
    learner.validate.side_effect = ValueError("invalid model")
    assert executor.execute("train", data, ctx, Signal()).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    learner.train.assert_not_called()


def test_learner_boundary_failures_are_logged_without_escaping(executor):
    executor, learner, ctx, _ = executor
    learner.abort.side_effect = RuntimeError("abort")
    learner.finalize.side_effect = RuntimeError("finalize")
    executor.handle_event(EventType.ABORT_TASK, ctx)
    executor.finalize(ctx)
    learner.abort.assert_called_once_with(ctx)
    learner.finalize.assert_called_once_with(ctx)
    executor.writer = None
    executor.log_value("metric", 1, 0)
    ctx.get_engine().get_component.return_value = object()
    executor.initialize(ctx)
    assert executor.local_model_path.endswith("model_local.joblib")
