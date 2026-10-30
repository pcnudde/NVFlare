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

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_dict, from_shareable
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.edge.constants import CookieKey, EdgeApiStatus, MsgKey, SpecialDeviceId
from nvflare.edge.executors.edge_model_executor import EdgeModelExecutor
from nvflare.edge.executors.hug import TaskInfo
from nvflare.edge.executors.simple_edge_executor import SimpleEdgeExecutor
from nvflare.edge.mud import BaseState, StateUpdateReport
from nvflare.edge.updaters.emd import AggregatorFactory, EdgeModelUpdater
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.task_request import TaskRequest


def current_task(data):
    task = Shareable({"weights": [1, 2]})
    task.set_header(ReservedKey.TASK_ID, "assignment")
    task.set_header(ReservedKey.TASK_NAME, "train")
    info = TaskInfo(task)
    info.task = data
    return info


@pytest.fixture
def context():
    ctx = FLContext()
    ctx.set_prop(ReservedKey.RUN_NUM, "job")
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site")
    return ctx


def model_state():
    return BaseState(2, DXO(DataKind.WEIGHTS, {"w": [1.0, 2.0]}), 7, {"device": 9})


def test_simple_executor_assigns_once_per_device_and_resets_between_tasks(context):
    executor = SimpleEdgeExecutor("updater")
    info = current_task(Shareable({"weights": [1, 2]}))
    request = TaskRequest(DeviceInfo("device"), None, "job", {})
    executor.task_started(info, context)
    first = executor.process_edge_task_request(request, info, context)
    assert first.status == EdgeApiStatus.OK and first.task_id == "assignment"
    assert first.task_data == {"weights": [1, 2], MsgKey.TASK_ID: "assignment"}
    repeated = executor.process_edge_task_request(request, info, context)
    assert repeated.status == EdgeApiStatus.RETRY and repeated.retry_wait == 30
    assert executor.process_edge_selection_request(None, info, context) is None
    executor.task_ended(info, context)
    assert executor.devices is None
    executor.task_started(info, context)
    assert executor.process_edge_task_request(request, info, context).status == EdgeApiStatus.OK


def test_simple_executor_routes_result_identity_and_reports_acceptance_errors(context, monkeypatch):
    executor = SimpleEdgeExecutor("updater")
    accept = Mock()
    monkeypatch.setattr(executor, "accept_update", accept)
    report = ResultReport(DeviceInfo("device"), None, "job", "old-assignment", "train", result={"delta": 3})
    info = current_task(Shareable())
    response = executor.process_edge_result_report(report, info, context)
    assert response.status == EdgeApiStatus.OK and response.task_id == "old-assignment"
    update = accept.call_args.kwargs["update"]
    assert update["delta"] == 3
    assert update.get_header(ReservedHeaderKey.TASK_ID) == "assignment"
    assert accept.call_args.kwargs["task_id"] == "old-assignment"
    accept.side_effect = ValueError("rejected")
    error = executor.process_edge_result_report(report, info, context)
    assert error.status == EdgeApiStatus.ERROR and "rejected" in error.message


def test_model_executor_uses_aggregator_factory_and_rejects_wrong_component(context, monkeypatch):
    executor = EdgeModelExecutor("factory", max_model_versions=4)
    engine = Mock()
    monkeypatch.setattr(context, "get_engine", lambda: engine)
    engine.get_component.return_value = Mock(spec=AggregatorFactory)
    assert isinstance(executor.get_updater(context), EdgeModelUpdater)
    engine.get_component.return_value = object()
    panic = Mock()
    monkeypatch.setattr(executor, "system_panic", panic)
    assert executor.get_updater(context) is None
    assert "AggregatorFactory" in panic.call_args.args[0]


def test_model_task_selection_cookie_prevents_duplicate_training_and_caches_conversion(context, monkeypatch):
    executor = EdgeModelExecutor("factory")
    alive = Mock()
    monkeypatch.setattr(executor, "accept_alive_device", alive)
    info = current_task(model_state())
    request = TaskRequest(DeviceInfo("device"), None, "job", {})
    first = executor.process_edge_task_request(request, info, context)
    assert first.status == EdgeApiStatus.OK
    assert first.cookie == {CookieKey.MODEL_VERSION: 2, CookieKey.DEVICE_SELECTION_ID: 9}
    assert from_dict(first.task_data).get_meta_prop(MsgKey.MODEL_VERSION) == 2
    second = executor.process_edge_task_request(request, info, context)
    assert second.task_data is first.task_data
    request.cookie = first.cookie
    duplicate = executor.process_edge_task_request(request, info, context)
    assert duplicate.status == EdgeApiStatus.RETRY and duplicate.message == "Device not selected"
    info.task = BaseState(0, None, 0, None)
    assert executor.process_edge_task_request(request, info, context).message == "Model not ready"
    assert alive.call_count == 4


def test_selection_request_updates_liveness_except_dummy_probe(context, monkeypatch):
    executor = EdgeModelExecutor("factory")
    alive = Mock()
    monkeypatch.setattr(executor, "accept_alive_device", alive)
    info = current_task(model_state())
    result = executor.process_edge_selection_request(SelectionRequest(DeviceInfo("device"), "job"), info, context)
    assert result.selection == {"device": 9} and result.task_id == "assignment"
    info.task = BaseState(0, None, 0, None)
    idle = executor.process_edge_selection_request(
        SelectionRequest(DeviceInfo(SpecialDeviceId.DUMMY), "job"), info, context
    )
    assert idle.status == EdgeApiStatus.OK and idle.task_id is None
    alive.assert_called_once_with("device", context)


@pytest.mark.parametrize(
    "device_id, expected",
    [
        ("device", ["device"]),
        (f"group{SpecialDeviceId.MAX_INDICATOR}3", [f"group{SpecialDeviceId.NUM_INDICATOR}{i}" for i in range(3)]),
    ],
)
def test_alive_report_expands_range_and_preserves_site_identity(context, monkeypatch, device_id, expected):
    executor = EdgeModelExecutor("factory")
    accept = Mock(return_value="ack")
    monkeypatch.setattr(executor, "accept_update", accept)
    assert executor.accept_alive_device(device_id, context) == "ack"
    report = StateUpdateReport.from_shareable(accept.call_args.args[1])
    assert sorted(report.available_devices) == expected
    assert all(device.client_name == "site" for device in report.available_devices.values())


@pytest.mark.parametrize("cookie", [None, {}, {"unrelated": 2}])
def test_result_without_model_version_cookie_is_rejected(context, cookie):
    executor = EdgeModelExecutor("factory")
    report = ResultReport(
        DeviceInfo("device"),
        None,
        "job",
        "assignment",
        "train",
        result=DXO(DataKind.WEIGHT_DIFF, {"w": [0.5]}).to_dict(),
        cookie=cookie,
    )
    response = executor.process_edge_result_report(report, current_task(model_state()), context)
    assert response.status == EdgeApiStatus.ERROR and "cookie" in response.message


def test_result_converts_weight_delta_and_tracks_model_version_and_device(context, monkeypatch):
    executor = EdgeModelExecutor("factory")
    accept = Mock()
    monkeypatch.setattr(executor, "accept_update", accept)
    report = ResultReport(
        DeviceInfo("device"),
        None,
        "job",
        "assignment",
        "train",
        result=DXO(DataKind.WEIGHT_DIFF, {"w": [0.5]}).to_dict(),
        cookie={CookieKey.MODEL_VERSION: 2},
    )
    info = current_task(model_state())
    assert executor.process_edge_result_report(report, info, context).status == EdgeApiStatus.OK
    payload = StateUpdateReport.from_shareable(accept.call_args.args[1])
    assert set(payload.model_updates) == {2}
    update = payload.model_updates[2]
    assert from_shareable(update.update).data == {"w": [0.5]}
    assert from_shareable(update.update).get_meta_prop(ReservedHeaderKey.TASK_ID) == "assignment"
    assert set(update.devices) == {"device"} and set(payload.available_devices) == {"device"}
    report.status = EdgeApiStatus.ERROR
    assert executor.process_edge_result_report(report, info, context).status == EdgeApiStatus.ERROR
    executor.task_started(info, context)
    executor.task_ended(info, context)


def test_executorch_result_reconstructs_tensor_shapes_and_strips_wrapper_prefix(context):
    pytest.importorskip("torch")
    from nvflare.edge.executors.et_edge_model_executor import ETEdgeModelExecutor

    executor = ETEdgeModelExecutor(Mock(), (1, 2), (1,), "factory", 3)
    report = ResultReport(
        DeviceInfo("device"),
        None,
        "job",
        "assignment",
        "train",
        cookie={CookieKey.MODEL_VERSION: 2},
        result={"meta": {}, "data": {"net.weight": {"data": [1, 2, 3, 4], "sizes": [2, 2]}}},
    )
    update = executor._convert_device_result_to_model_update(report, current_task(model_state()), context)
    dxo = from_shareable(update.update)
    assert dxo.data_kind == DataKind.WEIGHT_DIFF
    np.testing.assert_array_equal(dxo.data["dict"]["weight"], [[1, 2], [3, 4]])
    assert update.model_version == 2 and set(update.devices) == {"device"}
