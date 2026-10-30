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

from nvflare.fuel.f3.cellnet import core_cell as subject
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from tests.timing_utils import isolate_time


@pytest.fixture
def cell(monkeypatch):
    value = object.__new__(subject.CoreCell)
    value.my_info = FqcnInfo("site-a")
    value.logger = Mock()
    value.log_error = Mock()
    value.root_change_lock = threading.Lock()
    value.adhoc_connector_lock = threading.Lock()
    value.ext_listener_lock = threading.Lock()
    value.agent_lock = threading.Lock()
    value.bulk_lock = threading.Lock()
    value.bulk_msg_lock = threading.Lock()
    value.root_url = "tcp://old:1234"
    value.bb_ext_connector = None
    value.bb_int_connector = None
    value.adhoc_connectors = {}
    value.ext_listeners = {}
    value.ext_listener_impossible = False
    value.int_listener = None
    value.agents = {}
    value.bulk_senders = {}
    value.bulk_messages = []
    value.bulk_checker = None
    value.bulk_processor = None
    value.max_bulk_size = 2
    value.connector_manager = Mock()
    value.communicator = Mock()
    monkeypatch.setattr(value, "ALL_CELLS", {})
    return value


@pytest.mark.parametrize("failure", [False, True])
def test_bulk_sender_batches_in_order_and_reports_delivery_failure(monkeypatch, failure):
    owner = Mock()
    owner.get_fqcn.return_value = "site-a"
    owner.get_fobs_context.return_value = {"site": "site-a"}
    owner.fire_and_forget.return_value = {"site-b": "unreachable" if failure else ""}

    # The transport stamps envelope routing headers before returning its delivery status.
    def send_bulk(**kwargs):
        kwargs["message"].add_headers({MessageHeaderKey.ORIGIN: "site-a", MessageHeaderKey.DESTINATION: "site-b"})
        return {"site-b": "unreachable" if failure else ""}

    owner.fire_and_forget.side_effect = send_bulk
    encode = Mock()
    monkeypatch.setattr(subject, "encode_payload", encode)
    sender = subject._BulkSender(owner, "site-b", 2, secure=True)
    messages = [Message({}, i) for i in range(3)]
    for message in messages:
        sender.queue_message("channel", "topic", message)
    assert all(message.get_header(MessageHeaderKey.SECURE) for message in messages)
    assert encode.call_count == owner.encrypt_payload.call_count == 3
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", lambda: 123)
    sender.send()
    sent = owner.fire_and_forget.call_args.kwargs
    assert [item["message"]["payload"] for item in sent["message"].payload] == [0, 1]
    assert sent["targets"] == ["site-b"]
    assert sender.last_send_time == 123
    assert len(sender.messages) == 1
    sender.send()
    assert owner.fire_and_forget.call_count == 2
    assert not sender.messages
    sender.send()
    assert owner.fire_and_forget.call_count == 2


def test_root_change_rebuilds_server_connectors_and_drops_only_server_agents(cell, monkeypatch):
    old = SimpleNamespace(handle="backbone")
    cell.bb_ext_connector = old
    cell.adhoc_connectors = {"server.job": SimpleNamespace(handle="adhoc"), "site-b": SimpleNamespace(handle="peer")}
    cell.agents = {name: subject.CellAgent(name, Endpoint(name)) for name in ("server", "server.job", "site-b")}
    create = Mock()
    monkeypatch.setattr(cell, "_create_bb_external_connector", create)
    cell.change_server_root("tcp://new:1234")
    assert cell.root_url == "tcp://new:1234"
    assert list(cell.adhoc_connectors) == ["site-b"]
    assert list(cell.agents) == ["site-b"]
    assert cell.bb_ext_connector is None
    assert [c.args[0] for c in cell.communicator.remove_connector.call_args_list] == ["backbone", "adhoc"]
    create.assert_called_once_with()
    cell.change_server_root("tcp://new:1234")
    assert create.call_count == 1


def test_connector_removal_failures_do_not_leave_stale_backbone_or_adhoc_entries(cell):
    cell.bb_ext_connector = SimpleNamespace(handle="backbone")
    cell.adhoc_connectors = {"server.job": SimpleNamespace(handle="adhoc")}
    cell.communicator.remove_connector.side_effect = OSError("already closed")
    cell.drop_connectors()
    assert cell.bb_ext_connector is None and not cell.adhoc_connectors
    assert cell.log_error.call_count == 2


@pytest.mark.parametrize("result", [True, False])
def test_adhoc_connectors_are_cached_and_backbone_url_is_never_duplicated(cell, result):
    cell.bb_ext_connector = object()
    assert cell._add_adhoc_connector("server", cell.root_url) is None
    connector = SimpleNamespace(handle="connector") if result else None
    cell.connector_manager.get_external_connector.return_value = connector
    assert cell._add_adhoc_connector("site-b", "tcp://peer:1") is connector
    assert cell._add_adhoc_connector("site-b", "tcp://peer:1") is connector
    cell.connector_manager.get_external_connector.assert_called_once_with("tcp://peer:1", adhoc=True)


@pytest.mark.parametrize("kind", ["internal", "external", "backbone"])
def test_required_backbone_connector_or_listener_failure_is_reported(cell, kind):
    if kind == "internal":
        cell.connector_manager.get_internal_listener.return_value = None
        with pytest.raises(RuntimeError, match="cannot create backbone internal listener"):
            cell._create_internal_listener()
    elif kind == "external":
        cell.connector_manager.get_external_listener.return_value = None
        with pytest.raises(RuntimeError, match="cannot create backbone external listener"):
            cell._create_external_listener("tcp://host:1")
    else:
        cell.connector_manager.get_external_connector.return_value = None
        with pytest.raises(RuntimeError, match="cannot create backbone external connector"):
            cell._create_bb_external_connector()


def test_external_listener_is_reused_and_unavailable_adhoc_listener_is_not_retried(cell):
    listener = Mock()
    listener.get_connection_url.return_value = "tcp://host:1"
    cell.connector_manager.get_external_listener.return_value = listener
    assert cell._create_external_listener("tcp://host:1") is listener
    assert cell._create_external_listener("tcp://host:1") is listener
    assert cell._create_external_listener("") is listener
    assert cell.connector_manager.get_external_listener.call_count == 1
    cell.ext_listeners = {}
    cell.connector_manager.get_external_listener.return_value = None
    assert cell._create_external_listener("") is None
    assert cell.ext_listener_impossible
    assert cell._create_external_listener("") is None
    assert cell.connector_manager.get_external_listener.call_count == 2


def test_bulk_receive_reconstructs_headers_and_preserves_fifo_order(cell):
    requests = [Message({}, "first"), Message({}, "second")]
    target_messages = [
        subject.TargetMessage("site-a", "channel", str(i), request).to_dict() for i, request in enumerate(requests)
    ]
    first = Message({MessageHeaderKey.ORIGIN: "site-b", "custom": "value"}, target_messages)
    second = Message({MessageHeaderKey.ORIGIN: "site-c"}, [])
    cell.bulk_messages = [first, second]
    cell._process_request = Mock()
    cell._process_pending_bulks()
    assert not cell.bulk_messages
    assert [c.kwargs["message"].payload for c in cell._process_request.call_args_list] == ["first", "second"]
    delivered = [c.kwargs["message"] for c in cell._process_request.call_args_list]
    for i, request in enumerate(delivered):
        assert request.get_header(MessageHeaderKey.CHANNEL) == "channel"
        assert request.get_header(MessageHeaderKey.TOPIC) == str(i)
        assert request.get_header("custom") == "value"


def test_bulk_queue_and_receiver_start_only_one_controlled_worker_each(cell, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=factory))
    sender = Mock()
    monkeypatch.setattr(subject, "_BulkSender", Mock(return_value=sender))
    first = Message({}, "one")
    cell.queue_message("channel", "topic", "site-b", first, optional=True)
    cell.queue_message("channel", "topic", ["site-b"], Message({}, "two"))
    assert factory.call_count == 1
    assert sender.queue_message.call_count == 2
    assert first.get_header(MessageHeaderKey.OPTIONAL) is True
    cell._receive_bulk_message(Message({}, []))
    cell._receive_bulk_message(Message({}, []))
    assert factory.call_count == 2
    assert len(cell.bulk_messages) == 2
