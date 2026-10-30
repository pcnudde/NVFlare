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

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors import multi_process_executor as subject
from tests.timing_utils import isolate_time


class ProcessExecutor(subject.MultiProcessExecutor):
    def get_multi_process_command(self):
        return "python"


def test_execute_waits_only_until_completion_callback_publishes_result(monkeypatch):
    executor = ProcessExecutor("learner", num_of_processes=1, components=[])
    executor.executor = Mock(spec=Executor)
    message, context, abort = Shareable(), FLContext(), Signal()
    result = Shareable({"weights": 3})
    dispatch = Mock()
    monkeypatch.setattr(executor, "_execute_multi_process", dispatch)

    def sleep(delay):
        executor.execute_result, executor.execute_complete = result, True

    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock(side_effect=[None])

    def one_sleep(delay):
        clock.sleep.side_effect = []
        sleep(delay)

    clock.sleep.side_effect = one_sleep
    assert executor.execute("train", message, context, abort) is result
    dispatch.assert_called_once_with(task_name="train", shareable=message, fl_ctx=context, abort_signal=abort)
    clock.sleep.assert_called_once_with(0.2)


def test_dispatch_transport_exception_returns_execution_error_without_spawning_process(monkeypatch):
    executor = ProcessExecutor("learner", num_of_processes=1, components=[])
    context = FLContext()
    executor.engine = Mock()
    executor.targets = ["site.job.0"]
    context.set_prop(ReservedKey.ENGINE, executor.engine)
    executor.engine.client.cell.fire_and_forget.side_effect = OSError("disconnected")
    error = Mock()
    monkeypatch.setattr(executor, "log_error", error)
    result = executor._execute_multi_process("train", Shareable(), context, Signal())
    assert result.get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR
    error.assert_called_once_with(context, "Multi-Process Execution error.")
