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

from nvflare.edge.constants import CookieKey, EdgeApiStatus
from nvflare.edge.simulation import simulator
from nvflare.edge.simulation.devices.num import NumDevice
from nvflare.edge.simulation.simulated_device import DeviceFactory, DeviceState
from nvflare.edge.simulation.simulator import Simulator
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from tests.timing_utils import isolate_time


@pytest.fixture
def sim(monkeypatch):
    pool = Mock()
    monkeypatch.setattr(simulator, "ThreadPoolExecutor", Mock(return_value=pool))
    factory = Mock(spec=DeviceFactory)
    factory.make_device.side_effect = NumDevice
    instance = Simulator("edge", factory, num_devices=5, num_workers=2, get_job_timeout=10)
    instance.device_id_prefix = "ours-"
    instance.set_send_func(Mock(return_value=ResultResponse(EdgeApiStatus.OK)), token="auth")
    return instance


def test_selected_devices_exclude_foreign_busy_and_already_reported_selections(sim):
    sim.busy_devices = {"ours-busy": "s1"}
    sim.used_devices = {"ours-used": "s2", "ours-reselected": "old"}
    selected = {"foreign": "s", "ours-busy": "s2", "ours-used": "s2", "ours-reselected": "new", "ours-new": "s3"}
    assert sim._determine_my_devices(selected) == {"ours-reselected": "new", "ours-new": "s3"}
    assert sim.used_devices == {"ours-used": "s2"}
    assert sim.busy_devices == {"ours-busy": "s1"}


@pytest.mark.parametrize(
    "status", [EdgeApiStatus.OK, EdgeApiStatus.RETRY, EdgeApiStatus.DONE, EdgeApiStatus.ERROR, EdgeApiStatus.NO_JOB]
)
def test_control_flow_dispatches_only_successful_tasks_and_honors_cookie_selection(monkeypatch, sim, status):
    clock = isolate_time(monkeypatch, simulator)
    clock.sleep = Mock()
    sim._ask_for_job = Mock(return_value=JobResponse(EdgeApiStatus.OK, "job", "edge", "number", {"a": 1}))
    sim._ask_for_selection = Mock(
        side_effect=[
            SelectionResponse(EdgeApiStatus.OK, selection={"ours-1": "selected"}),
            SelectionResponse(EdgeApiStatus.NO_JOB),
        ]
    )
    cookie = {CookieKey.DEVICE_SELECTION_ID: "cookie-selection"}
    task = TaskResponse(status, "job", task_id="task", task_name="train", task_data={}, cookie=cookie)
    sim._ask_for_task = Mock(return_value=task)

    sim._control_flow()

    device = sim.all_devices["ours-1"]
    assert device.job_name == "edge" and device.job_data == {"a": 1}
    if status == EdgeApiStatus.OK:
        assert device.state == DeviceState.LEARNING
        assert sim.busy_devices == {"ours-1": "cookie-selection"}
        assert device.cookie is cookie
        kwargs = sim.worker_pool.submit.call_args.kwargs
        assert kwargs == {"task_data": task, "device": device, "selection_id": "cookie-selection"}
    else:
        sim.worker_pool.submit.assert_not_called()
        assert sim.busy_devices == {}
        if status == EdgeApiStatus.DONE:
            assert device.job_id is None


@pytest.mark.parametrize("response", [None, JobResponse(EdgeApiStatus.RETRY), JobResponse(EdgeApiStatus.ERROR)])
def test_job_discovery_stops_at_deadline_or_unrecoverable_status(monkeypatch, sim, response):
    clock = isolate_time(monkeypatch, simulator)
    clock.time = Mock(side_effect=[0, 11])
    clock.sleep = Mock()
    sim._ask_for_job = Mock(return_value=response)
    sim._control_flow()
    sim._ask_for_job.assert_called_once()
    sim.worker_pool.submit.assert_not_called()


@pytest.mark.parametrize("outcome", ["valid", "exception", "wrong_type"])
def test_learning_report_records_result_or_error_and_releases_busy_device(sim, outcome):
    device = NumDevice("ours-1")
    device.cookie = {"selection": "s1"}
    device.state = DeviceState.LEARNING
    device.do_task = Mock(return_value={"w": 2} if outcome == "valid" else [2])
    if outcome == "exception":
        device.do_task.side_effect = ValueError("invalid model")
    sim.busy_devices[device.device_id] = "s1"
    task = TaskResponse(EdgeApiStatus.OK, "job", task_id="t1", task_name="train")

    sim._do_learn(task, device, "s1")

    report, sent_device = sim.send_f.call_args.args
    assert isinstance(report, ResultReport) and sent_device is device
    assert (report.job_id, report.task_id, report.task_name, report.cookie) == ("job", "t1", "train", device.cookie)
    assert report.status == (EdgeApiStatus.OK if outcome == "valid" else EdgeApiStatus.ERROR)
    assert report.result == ({"w": 2} if outcome == "valid" else {})
    assert device.state == DeviceState.IDLE
    assert sim.busy_devices == {} and sim.used_devices == {"ours-1": "s1"}


def test_transport_failure_returns_retry_and_requests_preserve_device_identity(sim):
    device = NumDevice("ours-1")
    device.set_job("job", "edge", "number", {})
    device.cookie = {"round": 1}
    sim.send_f.side_effect = RuntimeError("offline")
    assert sim._ask_for_task(device).status == EdgeApiStatus.RETRY
    assert sim._ask_for_job(device).status == EdgeApiStatus.RETRY
    assert sim._ask_for_selection("job", "ours-1").status == EdgeApiStatus.RETRY
    requests = [c.args[0] for c in sim.send_f.call_args_list]
    assert [type(r) for r in requests] == [TaskRequest, JobRequest, SelectionRequest]
    assert all(r.get_device_id() == "ours-1" for r in requests)
    assert requests[0].cookie == {"round": 1}
    assert all(c.kwargs == {"token": "auth"} for c in sim.send_f.call_args_list)


def test_start_requires_transport_and_shuts_down_every_owned_device(sim):
    with pytest.raises(ValueError, match="not callable"):
        sim.set_send_func(None)
    sim.send_f = None
    with pytest.raises(ValueError, match="send_f has not been set"):
        sim.start()
    sim.set_send_func(Mock())
    sim._control_flow = Mock()
    device = sim._make_new_device("ours-1")
    device.shutdown = Mock()
    sim.start()
    sim._control_flow.assert_called_once()
    sim.worker_pool.shutdown.assert_called_once_with()
    device.shutdown.assert_called_once_with()
    sim.device_factory.shutdown.assert_called_once_with()
    sim.stop()
    assert sim.done is True
