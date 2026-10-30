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
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.workflows import broadcast_operator as subject
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("aborted", [False, True])
def test_multicast_sends_each_assignment_before_bounded_completion_or_abort(monkeypatch, aborted):
    ctx = FLContext()
    controller = Mock()
    controller.get_num_standing_tasks.side_effect = [2] if aborted else [2, 0]
    helper = subject.BroadcastAndWait(ctx, controller)
    data = {"one": Shareable({"w": 1}), "two": Shareable({"w": 2})}
    abort = Signal()
    if aborted:
        abort.trigger(True)
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None]))
    result = helper.multicasts_and_wait("train", data, ctx, abort, task_check_period=0.2)
    assert [c.kwargs["targets"] for c in controller.send.call_args_list] == [["one"], ["two"]]
    assert [c.kwargs["task"].data for c in controller.send.call_args_list] == list(data.values())
    assert result == (None if aborted else {})
    assert clock.sleep.call_count == int(not aborted)
    if not aborted:
        clock.sleep.assert_called_once_with(0.2)
