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

from nvflare.fuel.flare_api.api_spec import JobNotFound, NoConnection, TargetType
from nvflare.tool import api_utils as subject
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("connected", [True, False])
def test_shutdown_owned_session_closes_on_success_or_already_stopped_and_aborts_only_running_jobs(
    monkeypatch, connected
):
    session = Mock()
    session.list_jobs.return_value = [
        {"job_id": "running", "status": "RUNNING"},
        {"id": "lost", "status": "RUNNING"},
        {"job_id": "done", "status": "FINISHED"},
        {},
        None,
    ]
    session.abort_job.side_effect = [None, JobNotFound("gone")]
    if not connected:
        session.try_connect.side_effect = NoConnection("stopped")
    monkeypatch.setattr(subject, "Session", Mock(return_value=session))
    result = subject.shutdown_system("/fake/prod", username="admin", wait=True, timeout_in_sec=3)
    session.close.assert_called_once()
    if connected:
        assert result["active_job_ids"] == ["running", "lost"]
        assert result["active_jobs_aborted"]
        session.shutdown.assert_called_once_with(TargetType.ALL, wait=True, timeout=3)
        assert [c.args[0] for c in session.abort_job.call_args_list] == ["running", "lost"]
    else:
        assert result == {"server_reachable": False, "already_stopped": True, "active_job_ids": [], "wait": True}
        session.shutdown.assert_not_called()


class ImmediateThread:
    def __init__(self, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


class ProbeFailure(BaseException):
    pass


@pytest.mark.parametrize("failure", ["close", "probe"])
def test_readiness_worker_contains_cleanup_failure_and_propagates_nonstandard_failure(monkeypatch, capsys, failure):
    session = Mock()
    info = SimpleNamespace(client_info=[SimpleNamespace(name="site-1")])
    session.get_system_info.return_value = info
    if failure == "probe":
        session.get_system_info.side_effect = ProbeFailure("probe failed")
    else:
        session.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(subject, "Session", Mock(return_value=session))
    # Execute the owned worker at an explicit synchronous boundary, without starting a thread.
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Event=subject.threading.Event, Thread=ImmediateThread))
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0.1 * n for n in range(24)])
    clock.sleep = Mock()
    kwargs = dict(num_clients=1, prod_dir="/fake/prod", timeout_in_sec=10, second_to_wait=1)
    if failure == "probe":
        with pytest.raises(ProbeFailure, match="probe failed"):
            subject.wait_for_system_start(**kwargs)
    else:
        assert subject.wait_for_system_start(**kwargs) is info
        assert "could not close the admin session" in capsys.readouterr().out
    session.close.assert_called_once()
    clock.sleep.assert_called_once_with(1)
    assert clock.monotonic.call_count <= 24
