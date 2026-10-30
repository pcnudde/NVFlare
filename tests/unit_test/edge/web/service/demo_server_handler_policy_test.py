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

from nvflare.edge.constants import CookieKey, EdgeApiStatus
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.service import test_run_server as subject
from tests.timing_utils import isolate_time


def test_demo_handler_returns_job_retry_result_and_selection_protocol(monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    sleep = Mock()
    monkeypatch.setattr(clock, "sleep", sleep)
    monkeypatch.setattr(subject, "uuid", SimpleNamespace(uuid4=lambda: "job-uuid"))
    handler = subject.TestQueryHandler()
    device = DeviceInfo("device")
    job = handler.handle_job_request(JobRequest("train", device, None))
    assert job.status == EdgeApiStatus.OK
    assert (job.job_id, job.job_name) == ("job-uuid", "test")
    sleep.assert_called_once_with(5.0)
    task = handler.handle_task_request(TaskRequest(device, None, "job", {}))
    assert task.status == EdgeApiStatus.RETRY
    assert task.job_id == "job"
    assert task.cookie == {CookieKey.MODEL_VERSION: 1, CookieKey.DEVICE_SELECTION_ID: 12}
    result = handler.handle_result_report(ResultReport(device, None, "job", "task", "train"))
    assert result.status == EdgeApiStatus.OK
    assert (result.task_id, result.task_name, result.message) == ("task", "train", "got it")
    selected = handler.handle_selection_request(SelectionRequest(device, "job"))
    assert selected.job_id == "job"
    assert selected.selection == {"aaaaa": 2, "bbbbb": 10}


def test_demo_shutdown_and_startup_use_owned_external_boundaries(monkeypatch, capsys):
    clock = isolate_time(monkeypatch, subject)
    order = []
    monkeypatch.setattr(clock, "sleep", lambda seconds: order.append(("sleep", seconds)))
    server = Mock()
    server.shutdown.side_effect = lambda: order.append("shutdown")
    subject.shutdown_server(server)
    assert order == [("sleep", 60), "shutdown"]
    assert "stopped server" in capsys.readouterr().out
    factory = Mock(return_value=server)
    monkeypatch.setattr(subject, "EdgeApiServer", factory)
    log = Mock()
    monkeypatch.setattr(subject, "logging", SimpleNamespace(INFO=20, basicConfig=Mock(), getLogger=lambda: log))
    thread = Mock()
    threads = Mock(return_value=thread)
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=threads))
    subject.main()
    factory.assert_called_once()
    assert factory.call_args.kwargs["address"] == "127.0.0.1:8009"
    assert factory.call_args.kwargs["max_workers"] == 100
    assert isinstance(factory.call_args.kwargs["handler"], subject.TestQueryHandler)
    threads.assert_called_once_with(target=subject.shutdown_server, daemon=True, args=(server,))
    thread.start.assert_called_once_with()
    server.start.assert_called_once_with()
    log.setLevel.assert_called_once_with(20)
