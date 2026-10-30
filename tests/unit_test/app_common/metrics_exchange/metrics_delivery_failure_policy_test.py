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

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_dict
from nvflare.app_common.metrics_exchange import metrics_sender as subject
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from tests.timing_utils import isolate_time


@pytest.fixture
def config():
    return {"connect_url": "tcp://localhost:1234", "receiver_fqcn": "site", "client_fqcn": "site.metrics"}


@pytest.mark.parametrize("failure", ["start", "timeout"])
def test_failed_initialization_stops_owned_cell_and_does_not_publish_initialized_state(config, monkeypatch, failure):
    cell = Mock()
    cell.is_cell_connected.return_value = False
    if failure == "start":
        cell.start.side_effect = OSError("start failed")
    monkeypatch.setattr(subject, "Cell", Mock(return_value=cell))
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "monotonic", Mock(side_effect=[0, subject.CONNECT_TIMEOUT]))
    sender = subject.MetricsSender(config=config)
    with pytest.raises(OSError if failure == "start" else RuntimeError, match="start failed|did not connect"):
        sender.init(rank=0)
    cell.stop.assert_called_once_with()
    assert sender.cell is None and not sender.initialized


def test_nonzero_and_invalid_rank_never_construct_transport_and_closed_sender_rejects_init(config, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(subject, "Cell", factory)
    sender = subject.MetricsSender(config=config)
    with pytest.raises(ValueError, match="rank must"):
        sender.init(rank=[])
    sender.init(rank=2)
    assert sender.rank == "2" and sender.initialized
    assert sender.add("loss", 1, AnalyticsDataType.SCALAR) is False
    sender.init(rank=0)
    factory.assert_not_called()
    sender.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        sender.init()


@pytest.mark.parametrize("outcome", ["success", "missing", "rejected", "exception"])
def test_metric_delivery_preserves_analytic_payload_and_reports_transport_outcome(config, monkeypatch, outcome):
    cell = Mock()
    cell.is_cell_connected.return_value = True
    cell.send_request.return_value = (
        None
        if outcome == "missing"
        else Message(
            {MessageHeaderKey.RETURN_CODE: ReturnCode.OK if outcome == "success" else ReturnCode.INVALID_REQUEST}, None
        )
    )
    if outcome == "exception":
        cell.send_request.side_effect = OSError("offline")
    monkeypatch.setattr(subject, "Cell", Mock(return_value=cell))
    sender = subject.MetricsSender(config=config)
    sender.init(rank="0")
    try:
        assert sender.add("loss", 0.5, AnalyticsDataType.SCALAR, global_step=7) is (outcome == "success")
        channel, topic, target, message, timeout = cell.send_request.call_args.args
        assert (channel, topic, target, timeout) == (
            subject.CHANNEL,
            subject.TOPIC_LOG,
            "site",
            subject.REQUEST_TIMEOUT,
        )
        analytic = AnalyticsData.from_dxo(from_dict(message.payload))
        assert analytic.tag == "loss" and analytic.value == 0.5
        assert analytic.kwargs["global_step"] == 7
    finally:
        sender.shutdown()
    cell.stop.assert_called_once_with()


def test_bootstrap_write_failure_removes_tempfile_and_preserves_existing_configuration(config, monkeypatch, tmp_path):
    path = tmp_path / "analytics.json"
    path.write_text("original")
    replace = Mock(side_effect=PermissionError("read only"))
    monkeypatch.setattr(subject.os, "replace", replace)
    with pytest.raises(PermissionError, match="read only"):
        subject.write_bootstrap(str(path), config)
    assert path.read_text() == "original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["analytics.json"]


@pytest.mark.parametrize(
    "value", [None, [], {"connect_url": "", "receiver_fqcn": "site", "client_fqcn": "other.child"}]
)
def test_invalid_bootstrap_shapes_are_rejected_before_transport_initialization(value, monkeypatch):
    monkeypatch.delenv(subject.ANALYTICS_BOOTSTRAP_ENV, raising=False)
    transport = Mock()
    monkeypatch.setattr(subject, "Cell", transport)
    with pytest.raises((RuntimeError, ValueError)):
        subject.MetricsSender(config=value)
    transport.assert_not_called()
