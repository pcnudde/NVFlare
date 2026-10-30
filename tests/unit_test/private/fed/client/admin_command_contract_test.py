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

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.private.fed.client import admin_commands
from nvflare.private.fed.client.client_status import ClientStatus, get_status_message
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import WidgetID


def _context(engine):
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, engine)
    return context


def test_admin_command_registry_resolves_commands_and_requires_processor_type(monkeypatch):
    registry = admin_commands.AdminCommands
    monkeypatch.setattr(registry, "commands", list(registry.commands))
    for processor in registry.commands:
        assert registry.get_command(processor.get_command_name()) is processor
    assert registry.get_command("unknown") is None
    with pytest.raises(TypeError, match="CommandProcessor"):
        registry.register_command(object())
    processor = Mock(spec=admin_commands.CommandProcessor)
    processor.get_command_name.return_value = "custom"
    registry.register_command(processor)
    assert registry.get_command("custom") is processor


def test_status_command_uses_live_client_state_and_abort_routes_to_runner():
    engine = Mock()
    engine.client.status = ClientStatus.STARTED
    context = _context(engine)
    assert admin_commands.CheckStatusCommand().process(Shareable(), context) == get_status_message(ClientStatus.STARTED)
    runner = Mock()
    context.set_prop(FLContextKey.RUNNER, runner)
    admin_commands.AbortCommand().process(Shareable(), context)
    runner.abort.assert_called_once_with(msg="Received command to abort job")
    assert admin_commands.AbortTaskCommand().process(Shareable(), context) is None
    runner.abort_task.assert_called_once_with()
    context.remove_prop(FLContextKey.RUNNER)
    assert admin_commands.AbortTaskCommand().process(Shareable(), context) is None


@pytest.mark.parametrize(
    "processor,operation,empty",
    [
        (admin_commands.ShowStatsCommand, "get_run_stats", "No stats info"),
        (admin_commands.ShowErrorsCommand, "get_errors", "No Errors"),
    ],
)
def test_admin_telemetry_handles_missing_invalid_and_empty_collectors(processor, operation, empty):
    engine = Mock()
    context = _context(engine)
    engine.get_widget.return_value = None
    assert processor().process(Shareable(), context) == {"error": "no info collector"}
    engine.get_widget.return_value = object()
    with pytest.raises(TypeError, match="InfoCollector"):
        processor().process(Shareable(), context)
    collector = Mock(spec=InfoCollector)
    engine.get_widget.return_value = collector
    getattr(collector, operation).return_value = None
    assert processor().process(Shareable(), context) == empty
    getattr(collector, operation).return_value = {"recorded": 7}
    assert processor().process(Shareable(), context) == {"recorded": 7}
    assert engine.get_widget.call_args.args == (WidgetID.INFO_COLLECTOR,)


def test_reset_errors_uses_engine_boundary_and_bye_is_no_op():
    engine = Mock()
    assert admin_commands.ResetErrorsCommand().process(Shareable(), _context(engine)) is None
    engine.reset_errors.assert_called_once_with()
    assert admin_commands.ByeCommand().process(Shareable(), FLContext()) is None
