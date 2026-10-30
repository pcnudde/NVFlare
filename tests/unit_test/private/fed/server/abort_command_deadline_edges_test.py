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

from types import SimpleNamespace
from unittest.mock import Mock

from nvflare.apis.fl_constant import FLContextKey, MachineStatus, ReservedKey, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.private.fed.server import server_commands as module


def test_abort_command_preserves_cold_request_and_stops_waiting_after_grace_period(monkeypatch):
    runner = Mock()
    engine = SimpleNamespace(engine_info=SimpleNamespace(status=MachineStatus.STARTED))
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, engine)
    context.set_prop(FLContextKey.RUNNER, runner)
    data = Shareable()
    data.set_header(ServerCommandKey.TURN_TO_COLD, True)
    clock = SimpleNamespace(time=Mock(side_effect=[0, 31]), sleep=Mock(side_effect=[None]))
    monkeypatch.setattr(module, "time", clock)
    assert module.AbortCommand().process(data, context) == "Aborted the run"
    runner.abort.assert_called_once_with(fl_ctx=context, turn_to_cold=True)
    clock.sleep.assert_called_once_with(1.0)
