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

import pytest

from nvflare.fuel.f3.cellnet import core_cell as module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message


@pytest.fixture
def cell():
    subject = module.CoreCell.__new__(module.CoreCell)
    subject.my_info = FqcnInfo("site.job")
    subject.ALL_CELLS = {}
    subject.root_url = "tcp://root:8002"
    subject.logger = Mock()
    subject.log_error = Mock()
    subject.log_warning = Mock()
    subject.cert_ex = Mock()
    subject.credential_manager = Mock()
    subject.connector_manager = Mock()
    subject.agents = {}
    return subject


@pytest.mark.parametrize("certificate", [DriverParams.SERVER_CERT, DriverParams.CLIENT_CERT])
def test_auth_identity_uses_available_credential_certificate_with_client_fallback(monkeypatch, certificate):
    resolve = Mock(return_value="site-identity")
    monkeypatch.setattr(module, "get_cert_common_name_from_file", resolve)
    assert module.CoreCell._get_auth_identity_from_credentials({certificate: "identity.pem"}) == "site-identity"
    resolve.assert_called_once_with("identity.pem")


@pytest.mark.parametrize("mode", ["already", "no_target", "empty"])
def test_secure_payload_guard_does_not_encrypt_twice_and_preserves_empty_content_length(cell, mode):
    message = Message(headers={MessageHeaderKey.SECURE: True, MessageHeaderKey.DESTINATION: "peer"})
    if mode == "already":
        message.set_header(MessageHeaderKey.ENCRYPTED, True)
        cell.encrypt_payload(message)
        cell.credential_manager.encrypt.assert_not_called()
    elif mode == "no_target":
        message.remove_header(MessageHeaderKey.DESTINATION)
        with pytest.raises(RuntimeError, match="destination missing"):
            cell.encrypt_payload(message)
        cell.cert_ex.get_certificate.assert_not_called()
    else:
        cell.credential_manager.encrypt.return_value = b"encrypted"
        cell.encrypt_payload(message)
        cell.credential_manager.encrypt.assert_called_once_with(cell.cert_ex.get_certificate.return_value, b"")
        assert message.payload == b"encrypted"
        assert message.get_header(MessageHeaderKey.CLEAR_PAYLOAD_LEN) == 0
        assert message.get_header(MessageHeaderKey.ENCRYPTED)


@pytest.mark.parametrize("mode", ["plain", "no_origin", "bad_length"])
def test_secure_decryption_rejects_missing_origin_and_tampered_length(cell, mode):
    message = Message(
        headers={
            MessageHeaderKey.SECURE: True,
            MessageHeaderKey.ENCRYPTED: mode != "plain",
            MessageHeaderKey.ORIGIN: "peer",
            MessageHeaderKey.CLEAR_PAYLOAD_LEN: 2,
        },
        payload=b"cipher",
    )
    if mode == "plain":
        cell.decrypt_payload(message)
        cell.credential_manager.decrypt.assert_not_called()
    elif mode == "no_origin":
        message.remove_header(MessageHeaderKey.ORIGIN)
        with pytest.raises(RuntimeError, match="origin missing"):
            cell.decrypt_payload(message)
    else:
        cell.credential_manager.decrypt.return_value = b"wrong-size"
        with pytest.raises(RuntimeError, match="Payload size changed"):
            cell.decrypt_payload(message)


def test_backbone_setup_respects_local_parent_and_listener_choices(cell):
    cell._create_bb_external_connector = Mock()
    cell._create_internal_connector = Mock()
    cell._create_internal_listener = Mock()
    cell._create_external_listener = Mock()
    cell.create_internal_listener = True
    cell.parent_resources = {"secure": True}
    cell._set_bb_for_client_root()
    cell._create_internal_listener.assert_called_once_with()
    cell._set_bb_for_client_child("tcp://parent:8002", True)
    cell._create_internal_connector.assert_called_once_with("tcp://parent:8002", {"secure": True})
    cell._set_bb_for_server_root()
    cell._create_external_listener.assert_called_once_with("tcp://root:8002")
    cell._create_internal_connector.reset_mock()
    cell.ALL_CELLS = {module.FQCN.ROOT_SERVER: object()}
    cell._set_bb_for_server_child("tcp://parent:8002", True)
    cell._create_internal_connector.assert_not_called()
    cell.ALL_CELLS = {}
    cell._set_bb_for_server_child("tcp://parent:8002", True)
    cell._create_internal_connector.assert_called_once_with("tcp://parent:8002")
    cell.running = True
    cell.ALL_CELLS = {"site": object()}
    assert cell.is_backbone_ready()
    cell.root_url = ["tcp://first:8002", "tcp://second:8002"]
    assert cell.get_root_url_for_child() == "tcp://first:8002"
    cell.root_url = "tcp://root:8002"
    assert cell.get_root_url_for_child() == "tcp://root:8002"


def test_connector_failure_and_refused_adhoc_listener_do_not_create_network_io(cell):
    cell.connector_manager.get_internal_connector.return_value = None
    with pytest.raises(RuntimeError, match="cannot create backbone internal connector"):
        cell._create_internal_connector("tcp://parent:8002")
    cell.connector_manager.adhoc_allowed = False
    assert cell._create_external_listener("") is None
    cell.connector_manager.get_external_listener.assert_not_called()


@pytest.mark.parametrize("failure", [True, False])
def test_stop_joins_owned_bulk_workers_even_when_communicator_stop_fails(cell, failure):
    cell.running, cell.stopping = True, False
    cell.agent_lock = threading.Lock()
    cell.bulk_checker, cell.bulk_processor = Mock(), Mock()
    cell.bulk_checker.is_alive.return_value = cell.bulk_processor.is_alive.return_value = True
    cell.communicator = Mock()
    if failure:
        cell.communicator.stop.side_effect = ValueError("stop failed")
    cell.stop()
    assert not cell.running and cell.asked_to_stop and cell.stopping
    cell.bulk_checker.join.assert_called_once_with()
    cell.bulk_processor.join.assert_called_once_with()
    cell.communicator.stop.assert_called_once_with()
    assert cell.log_error.call_count == int(failure)
    cell.running = True
    cell.stop()
    cell.communicator.stop.assert_called_once_with()


def test_bulk_queue_and_receive_allocate_only_owned_fake_workers(monkeypatch, cell):
    workers = []

    def worker(**kwargs):
        result = Mock()
        result.kwargs = kwargs
        workers.append(result)
        return result

    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=worker))
    sender = Mock()
    monkeypatch.setattr(module, "_BulkSender", Mock(return_value=sender))
    cell.bulk_lock = threading.Lock()
    cell.bulk_checker = None
    cell.bulk_senders = {}
    cell.max_bulk_size = 10
    request = Message(payload=b"payload")
    cell.queue_message("channel", "topic", "peer", request, optional=True)
    cell.queue_message("channel", "topic", "peer", request)
    assert len(workers) == 1 and workers[0].kwargs["target"] == cell._check_bulk
    assert sender.queue_message.call_count == 2
    workers[0].start.assert_called_once_with()
    cell.bulk_msg_lock = threading.Lock()
    cell.bulk_processor = None
    cell.bulk_messages = []
    incoming = Message(payload=[])
    cell._receive_bulk_message(incoming)
    assert cell.bulk_messages == [incoming] and len(workers) == 2
    assert workers[1].kwargs["target"] == cell._process_bulk_messages
    workers[1].start.assert_called_once_with()


@pytest.mark.parametrize("kind", ["send", "process"])
def test_bulk_worker_loops_stop_after_bounded_policy_poll_and_flush_once(monkeypatch, cell, kind):
    cell.asked_to_stop = False
    cell.bulk_check_interval = cell.bulk_process_interval = 0.1
    cell.bulk_lock = threading.Lock()
    sender = Mock()
    cell.bulk_senders = {"peer": sender}
    cell._process_pending_bulks = Mock()
    pauses = []

    def sleep(seconds):
        pauses.append(seconds)
        assert len(pauses) == 1, "bulk worker ignored shutdown"
        cell.asked_to_stop = True

    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=sleep))
    if kind == "send":
        cell._check_bulk()
        assert sender.send.call_count == 2
    else:
        cell._process_bulk_messages()
        assert cell._process_pending_bulks.call_count == 2
    assert pauses == [0.1]


@pytest.mark.parametrize(
    "error, code",
    [
        (module.InvalidRequest, ReturnCode.INVALID_REQUEST),
        (module.AuthenticationError, ReturnCode.AUTHENTICATION_ERROR),
        (module.AbortRun, ReturnCode.ABORT_RUN),
        (RuntimeError, ReturnCode.PROCESS_EXCEPTION),
    ],
)
def test_callback_policy_exceptions_map_to_protocol_error_instead_of_escaping(cell, error, code):
    def callback(message):
        raise error("controlled failure")

    result = cell._try_cb(Message(), callback)
    assert result.get_header(MessageHeaderKey.RETURN_CODE) == code
    assert cell.log_error.call_count == int(error is RuntimeError)


def test_endpoint_resolution_rejects_invalid_target_and_contains_lookup_error(cell):
    assert cell._find_endpoint("bad target", Message()) == (ReturnCode.INVALID_TARGET, None)
    cell._try_find_ep = Mock(side_effect=ValueError("lookup failed"))
    assert cell._find_endpoint("peer", Message()) == (ReturnCode.TARGET_UNREACHABLE, None)
    assert cell.log_error.call_count == 2
    assert not cell._already_visited("peer", None)
