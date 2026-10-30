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

from unittest.mock import Mock

import pytest

from nvflare.fuel.utils.fsm import FSM, State


@pytest.mark.parametrize("name,error", [(None, TypeError), (3, TypeError), ("", ValueError), ("  ", ValueError)])
def test_invalid_state_name_is_rejected(name, error):
    with pytest.raises(error, match="name"):
        State(name)


def test_state_registration_binds_owner_and_rejects_duplicates_or_unknown_states():
    machine, state = FSM("workflow"), State("  ready  ")
    assert state.name == "ready"
    machine.add_state(state)
    assert state.fsm is machine
    with pytest.raises(RuntimeError, match="duplicate"):
        machine.add_state(State("ready"))
    with pytest.raises(TypeError, match="State"):
        machine.add_state(object())
    with pytest.raises(RuntimeError, match="no such state"):
        machine.set_current_state("missing")
    machine.set_prop("count", 3)
    assert machine.props == {"count": 3}
    assert machine.states == {"ready": state}


def test_transition_order_and_stay_then_exit_are_driven_by_current_state():
    machine, first, second = FSM("workflow"), State("first"), State("second")
    events = []
    first.execute = Mock(side_effect=[None, "second"])
    first.leave = lambda: events.append("leave-first")
    second.enter = lambda: events.append("enter-second")
    second.execute = Mock(return_value=FSM.STATE_NAME_EXIT)
    for state in (first, second):
        machine.add_state(state)
    machine.set_current_state("first")
    assert machine.execute(value=3) is first
    assert events == []
    assert machine.execute(value=4) is second
    assert events == ["leave-first", "enter-second"]
    assert first.execute.call_args_list == [((), {"value": 3}), ((), {"value": 4})]
    assert machine.execute() is None and machine.get_current_state() is None
    assert machine.error is None


@pytest.mark.parametrize("stage", ["execute", "leave", "enter", "missing", "not_started"])
def test_invalid_transition_and_callback_failure_stop_machine_with_error(stage):
    machine, first, second = FSM("workflow"), State("first"), State("second")
    first.execute = Mock(return_value="missing" if stage == "missing" else "second")
    machine.add_state(first)
    machine.add_state(second)
    if stage in ("execute", "leave", "enter"):
        setattr(second if stage == "enter" else first, stage, Mock(side_effect=ValueError("callback failed")))
    if stage != "not_started":
        machine.set_current_state("first")
    assert machine.execute() is None and machine.get_current_state() is None
    assert "exception occurred" in machine.error
    assert ("callback failed" if stage in ("execute", "leave", "enter") else "no") in machine.error
