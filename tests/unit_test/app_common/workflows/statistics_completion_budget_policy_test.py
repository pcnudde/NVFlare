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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.workflows.statistics_controller import StatisticsController


def controller():
    return StatisticsController({"count": {}}, "writer", min_clients=2, enable_pre_run_task=False)


def test_statistics_completion_refuses_dataset_with_insufficient_nonempty_clients(monkeypatch):
    workflow, context = controller(), FLContext()
    workflow.client_statistics = {"count": {"one": {"train": {"age": 3}}, "two": {"train": {}}}}
    assert not workflow._validate_min_clients(2, workflow.client_statistics)
    panic = Mock()
    monkeypatch.setattr(workflow, "system_panic", panic)
    assert workflow.post_fn("stats", context) is False
    panic.assert_called_once()
    assert "Not all required 2" in panic.call_args.args[0]


@pytest.mark.parametrize("aborted_after_wait", [False, True])
def test_statistics_timeout_or_abort_prevents_post_processing(monkeypatch, aborted_after_wait):
    workflow, abort = controller(), Signal()
    run = Mock()
    monkeypatch.setattr(workflow, "statistics_task_flow", run)
    post = Mock()
    monkeypatch.setattr(workflow, "post_fn", post)

    def wait(*args):
        if aborted_after_wait:
            abort.trigger(True)
        return aborted_after_wait

    monkeypatch.setattr(StatisticsController, "_wait_for_all_results", staticmethod(wait))
    assert workflow.control_flow(abort, FLContext()) is False
    assert run.call_count == 2
    post.assert_not_called()


def test_statistics_result_wait_checks_preexisting_abort_before_sleep():
    abort = Signal()
    abort.trigger(True)
    assert StatisticsController._wait_for_all_results(Mock(), 1, 2, {"count": {"one": {}}}, 0.1, abort) is False
