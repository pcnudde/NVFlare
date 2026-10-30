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

from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

from nvflare.app_common.tie import py_applet as subject
from tests.timing_utils import isolate_time


class LocalApplet(subject.PyApplet):
    def get_runner(self, app_ctx):
        return self.runner


def test_applet_graceful_stop_observes_exit_before_issuing_kill(monkeypatch):
    process = Mock()
    monkeypatch.setattr(type(process), "exitcode", PropertyMock(side_effect=[None, None, 5, 5, 5]), raising=False)
    monkeypatch.setattr(subject, "multiprocessing", SimpleNamespace(Process=type(process)))
    applet = LocalApplet(False)
    applet.runner, applet.process = Mock(), process
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(side_effect=[0, 0.1, 0.2, 0.3])
    clock.sleep = Mock(side_effect=[None])
    assert applet.stop(1) == 5
    assert applet.process is None
    process.kill.assert_not_called()
    clock.sleep.assert_called_once_with(0.1)
