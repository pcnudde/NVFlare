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
from nvflare.app_common.ccwf import SwarmClientController
from nvflare.app_common.ccwf.common import Constant, rotate_to_front
from nvflare.app_common.ccwf.comps.cwe_result_printer import CWEResultPrinter


@pytest.mark.parametrize(
    "selected,expected",
    [
        ("first", ["first", "second", "third"]),
        ("second", ["second", "third", "first"]),
        ("third", ["third", "first", "second"]),
    ],
)
def test_rotation_preserves_membership_and_relative_cyclic_order(selected, expected):
    members = ["first", "second", "third"]
    rotate_to_front(selected, members)
    assert members == expected
    with pytest.raises(ValueError):
        rotate_to_front("missing", members)
    assert members == expected


@pytest.mark.parametrize("state", ["best", "none", "wrong-executor", "other-event"])
def test_result_printer_reports_only_finalized_swarm_results(monkeypatch, state):
    printer = CWEResultPrinter()
    info = Mock()
    monkeypatch.setattr(printer, "log_info", info)
    executor = Mock(spec=SwarmClientController)
    executor.best_result = {"accuracy": 0.9} if state == "best" else None
    executor.best_round, executor.best_metric = 4, 0.9
    ctx = FLContext()
    ctx.set_prop(Constant.EXECUTOR, object() if state == "wrong-executor" else executor)
    printer.handle_event("other" if state == "other-event" else Constant.EXECUTOR_FINALIZED, ctx)
    if state == "best":
        assert [c.args[1] for c in info.call_args_list] == [
            "My Best Round: 4",
            "My Best Metric: 0.9",
            "My Best Result: {'accuracy': 0.9}",
        ]
    elif state == "none":
        info.assert_called_once_with(ctx, "I have no best result")
    else:
        info.assert_not_called()
