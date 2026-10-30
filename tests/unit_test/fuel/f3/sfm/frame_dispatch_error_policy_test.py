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

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm import conn_manager as module
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN, Prefix


@pytest.fixture
def manager():
    subject = module.ConnManager.__new__(module.ConnManager)
    subject.lock = threading.Lock()
    subject.local_endpoint = Endpoint("local")
    subject.stopped, subject.started = False, False
    subject.connectors = {}
    subject.receivers = {}
    subject.sfm_conns = {}
    subject.frame_mgr_executor = Mock()
    return subject


def frame(kind, payload=b"payload", headers=None):
    encoded = msgpack.packb(headers) if headers is not None else b""
    output = bytearray(PREFIX_LEN + len(encoded) + len(payload))
    Prefix(length=len(output), header_len=len(encoded), type=kind, app_id=7).to_buffer(output, 0)
    output[PREFIX_LEN:] = encoded + payload
    return output


@pytest.mark.parametrize("mode", ["empty", "unregistered", "unsupported", "runtime", "shutdown", "generic"])
def test_frame_dispatch_controls_empty_payloads_missing_receivers_and_processing_errors(manager, mode, caplog):
    caplog.set_level("DEBUG", logger=module.log.name)
    connection = Mock()
    connection.sfm_endpoint = SimpleNamespace(endpoint=Endpoint("peer"))
    receiver = Mock()
    if mode != "unregistered":
        manager.receivers[7] = receiver
    if mode in ["runtime", "shutdown", "generic"]:

        def fail(*args):
            if mode == "shutdown":
                manager.stopped = True
            raise (ValueError if mode == "generic" else RuntimeError)("receiver failed")

        receiver.process_message.side_effect = fail
    data = frame(255 if mode == "unsupported" else module.Types.DATA, b"" if mode == "empty" else b"payload")
    manager.process_frame_task(connection, data)
    if mode == "empty":
        assert receiver.process_message.call_args.args[3].payload is None
    else:
        expected = {
            "unregistered": "No receiver",
            "unsupported": "unsupported frame type",
            "runtime": "Error processing frame",
            "shutdown": "interrupted by shutdown",
            "generic": "Error processing frame",
        }[mode]
        assert expected in caplog.text


def test_connector_requires_ssl_capability_and_rejects_post_shutdown_registration(monkeypatch, manager):
    driver = Mock()
    driver.capabilities.return_value = {}
    monkeypatch.setattr(module, "ssl_required", lambda params: True)
    with pytest.raises(CommError, match="doesn't support"):
        manager.add_connector(driver, {DriverParams.SCHEME.value: "custom"}, module.Mode.ACTIVE)
    driver.register_conn_monitor.assert_not_called()
    monkeypatch.setattr(module, "ssl_required", lambda params: False)
    monkeypatch.setattr(module, "get_handle", Mock(return_value="connector"))
    manager.stopped = True
    with pytest.raises(CommError, match="stopped"):
        manager.add_connector(driver, {}, module.Mode.ACTIVE)
    assert manager.connectors == {}


def test_unknown_connector_and_closed_connection_do_not_change_registry(manager, caplog):
    caplog.set_level("DEBUG", logger=module.log.name)
    manager.remove_connector("missing")
    manager.close_connection(SimpleNamespace(name="missing"))
    assert manager.connectors == manager.sfm_conns == {}
    assert "Unknown connector handle" in caplog.text and "closed with no endpoint" in caplog.text


def test_sfm_receiver_records_activity_and_contains_dispatch_exception(monkeypatch, caplog):
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 10))
    manager, connection = Mock(), Mock()
    manager.process_frame.side_effect = ValueError("decoder failed")
    receiver = module.SfmFrameReceiver(manager, connection)
    receiver.process_frame(b"frame")
    assert connection.last_activity == 10
    manager.process_frame.assert_called_once_with(connection, b"frame")
    assert "decoder failed" in caplog.text


def test_null_connection_exposes_loopback_addresses_and_rejects_transport_send():
    connection = module.NullConnection()
    properties = connection.get_conn_properties()
    assert properties[DriverParams.LOCAL_ADDR.value] == properties[DriverParams.PEER_ADDR.value] == module.SELF_ADDR
    with pytest.raises(CommError, match="Can't send data"):
        connection.send_frame(b"data")
