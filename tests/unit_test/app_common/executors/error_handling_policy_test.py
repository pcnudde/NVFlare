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

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.error_handling_executor import ErrorHandlingExecutor


class DelegatingExecutor(ErrorHandlingExecutor):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def get_task_handler(self, fl_ctx):
        return self.handler()

    def get_data_kind(self):
        return DataKind.METRICS


@pytest.mark.parametrize(
    "exception,code,panic",
    [
        (TypeError("configuration"), ReturnCode.EXECUTION_EXCEPTION, False),
        (RuntimeError("unavailable"), ReturnCode.EXECUTION_RESULT_ERROR, True),
    ],
)
def test_initialization_failure_preserves_error_policy(exception, code, panic):
    factory = Mock(side_effect=exception)
    executor = DelegatingExecutor(factory)
    executor.system_panic = Mock()
    ctx = FLContext()
    executor.handle_event(EventType.START_RUN, ctx)
    assert not executor.init_status_ok
    result = executor.execute("calculate", Shareable(), ctx, Signal())
    assert result.get_return_code() == code
    assert executor.system_panic.called is panic
    if panic:
        executor.system_panic.assert_called_once_with(exception, ctx)


@pytest.mark.parametrize("outcome", ["success", "empty", "exception", "abort"])
def test_executor_converts_results_and_contains_task_failures(outcome):
    handler = Mock()
    handler.execute_task.return_value = {"loss": 0.25} if outcome == "success" else None
    if outcome == "exception":
        handler.execute_task.side_effect = RuntimeError("calculation")
    executor = DelegatingExecutor(lambda: handler)
    ctx, signal, data = FLContext(), Signal(), Shareable()
    executor.handle_event(EventType.START_RUN, ctx)
    assert executor.init_status_ok
    if outcome == "abort":
        signal.trigger(True)
    result = executor.execute("calculate", data, ctx, signal)
    if outcome == "success":
        dxo = from_shareable(result)
        assert dxo.data_kind == DataKind.METRICS
        assert dxo.data == {"loss": 0.25}
        handler.execute_task.assert_called_once_with("calculate", data, ctx, signal)
    else:
        assert result.get_return_code() == (
            ReturnCode.TASK_ABORTED if outcome == "abort" else ReturnCode.EXECUTION_RESULT_ERROR
        )
    if outcome == "abort":
        handler.execute_task.assert_not_called()
    executor.handle_event(EventType.END_RUN, ctx)
    handler.finalize.assert_called_once_with(ctx)


def test_finalize_contains_user_handler_exception_and_ignores_unrelated_event():
    executor = DelegatingExecutor(Mock())
    executor.task_handler = Mock()
    executor.task_handler.finalize.side_effect = RuntimeError("cleanup")
    executor.log_exception = Mock()
    ctx = FLContext()
    executor.handle_event("unrelated", ctx)
    executor.task_handler.finalize.assert_not_called()
    executor.finalize(ctx)
    executor.log_exception.assert_called_once()
    executor.task_handler = None
    executor.finalize(ctx)
