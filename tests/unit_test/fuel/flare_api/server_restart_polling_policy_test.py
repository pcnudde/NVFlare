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
from unittest.mock import Mock

import pytest

from nvflare.fuel.flare_api import flare_api as subject
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("outcome", ["changed-start", "went-down", "same-start", "remote-error"])
def test_restart_polling_requires_changed_identity_or_observed_downtime_and_expires_with_reason(monkeypatch, outcome):
    session = object.__new__(subject.Session)
    info = SimpleNamespace(server_info=SimpleNamespace(start_time="new" if outcome == "changed-start" else "old"))
    if outcome == "went-down":
        replies = [subject.NoConnection("down"), info]
    elif outcome == "remote-error":
        replies = [ValueError("bad response")]
    else:
        replies = [info]
    session._poll_system_info = Mock(side_effect=replies)
    clock = isolate_time(monkeypatch, subject)
    now, sleeps, reads = [0.0], [], []

    def time():
        reads.append(True)
        assert len(reads) <= 12, "restart monitor exceeded polling budget"
        return now[0]

    def sleep(delay):
        sleeps.append(delay)
        assert len(sleeps) <= 2, "restart monitor failed to stop at its deadline"
        now[0] += 1

    monkeypatch.setattr(clock, "time", time)
    monkeypatch.setattr(clock, "sleep", sleep)
    timeout = 2 if outcome == "went-down" else 1
    if outcome in ("changed-start", "went-down"):
        assert session._wait_for_server_restart("old", timeout) is info
    else:
        with pytest.raises(TimeoutError, match="previous start time" if outcome == "same-start" else "bad response"):
            session._wait_for_server_restart("old", timeout)
    assert session._poll_system_info.call_count == (2 if outcome == "went-down" else 1)


def test_shutdown_polling_keeps_last_transport_error_in_timeout_diagnostic(monkeypatch):
    session = object.__new__(subject.Session)
    session._poll_system_info = Mock(side_effect=ValueError("bad response"))
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 0, 0, 1, 2]))
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None]))
    with pytest.raises(TimeoutError, match="bad response"):
        session._wait_for_server_down(1)


@pytest.mark.parametrize("timeout", [None, "invalid", 0, -1])
def test_waiting_state_changes_require_a_positive_numeric_timeout(timeout):
    with pytest.raises(ValueError, match="positive number"):
        subject.Session._validate_state_change_timeout(timeout)
