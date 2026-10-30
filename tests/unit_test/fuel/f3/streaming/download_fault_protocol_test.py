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

import pytest

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming import download_service as module
from tests.unit_test.fuel.f3.streaming.download_test_utils import (
    MockDownloadable,
    make_service_no_monitor,
    pull_request,
)


@pytest.fixture
def service():
    subject = make_service_no_monitor()
    yield subject
    subject.shutdown()


def test_receiver_confirmation_defaults_to_enabled_when_configuration_is_unavailable(monkeypatch):
    monkeypatch.setattr(module, "_receiver_confirm_cached", None)
    lookup = Mock(side_effect=RuntimeError("not configured"))
    monkeypatch.setattr(module, "ConfigService", SimpleNamespace(get_bool_var=lookup))
    assert module._receiver_confirm_enabled()
    assert module._receiver_confirm_enabled()
    lookup.assert_called_once()


def test_source_failure_notification_retries_raised_transport_error_and_validates_request_size(monkeypatch, service):
    cell = Mock()
    cell.send_request.side_effect = [
        OSError("disconnected"),
        Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}),
    ]
    pause = Mock(side_effect=[None])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 0, sleep=pause))
    assert service.notify_source_failure(cell, ["receiver"], "owner.job", ["ref"], "failed") == {"receiver": None}
    assert cell.send_request.call_count == 2
    assert cell.send_request.call_args.kwargs["request"].payload["ref_ids"] == ("ref",)
    pause.assert_called_once()
    assert service.notify_source_failure(cell, [], "owner.job", ["ref"], "failed") == {}
    with pytest.raises(ValueError, match="too many"):
        service.notify_source_failure(
            cell, ["receiver"], "owner.job", [str(n) for n in range(module._MAX_SOURCE_FAILURE_REFS + 1)], "failed"
        )


@pytest.mark.parametrize("payload", [None, {"ref_ids": [7], "reason": "failed"}, {"ref_ids": ["ref"], "reason": ""}])
def test_source_failure_handler_rejects_malformed_payload_without_tombstones(service, payload):
    if payload is not None:
        payload["source_fqcn"] = "owner.job"
    result = service._handle_source_failure(Message(headers={MessageHeaderKey.ORIGIN: "owner"}, payload=payload))
    assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert service._source_failures == {}


def test_source_failure_tombstones_evict_oldest_preserve_active_abort_and_expire(monkeypatch, service):
    monkeypatch.setattr(module, "_MAX_SOURCE_FAILURE_TOMBSTONES", 2)
    signal = Signal()
    service._source_failures[("older.job", "old")] = ("previous failure", float("inf"))
    service._active_source_downloads[("owner.job", "a")] = {signal}
    result = service._handle_source_failure(
        Message(
            headers={MessageHeaderKey.ORIGIN: "owner"},
            payload={"source_fqcn": "owner.job", "ref_ids": ["a", "b", "c"], "reason": "failed"},
        )
    )
    assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert len(service._source_failures) == 2 and signal.triggered
    assert ("older.job", "old") not in service._source_failures
    for key in service._source_failures:
        service._source_failures[key] = ("failed", 0)
    service._purge_source_failures_locked(1)
    assert service._source_failures == {}


def test_download_lookup_and_object_validation_do_not_mutate_missing_transactions(service):
    with pytest.raises(ValueError, match="obj must be"):
        service.add_object("missing", object())
    with pytest.raises(ValueError, match="no such transaction"):
        service.add_object("missing", MockDownloadable([]))
    assert service.get_acquired_receivers("missing") == set()
    assert service.get_transaction_info("missing") is None
    response = service._handle_download(Message(headers={MessageHeaderKey.ORIGIN: "receiver"}, payload={}))
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST


@pytest.mark.parametrize("operation", ["pull", "confirm", "cancel"])
def test_settling_transaction_blocks_work_and_acks_late_control_messages(service, operation):
    transaction = service.new_transaction(Mock(), timeout=10, num_receivers=1)
    obj = MockDownloadable([b"chunk"])
    obj.produce = Mock()
    ref = service.add_object(transaction, obj)
    tx = service._tx_table[transaction]
    tx._ops_closed = True
    if operation == "pull":
        response = service._handle_download(pull_request(ref, "receiver"))
    elif operation == "confirm":
        response = service._handle_confirm(ref, "receiver", module.DownloadStatus.SUCCESS, None)
    else:
        response = service._handle_cancel(ref, "receiver")
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == (
        ReturnCode.INVALID_REQUEST if operation == "pull" else ReturnCode.OK
    )
    obj.produce.assert_not_called()
    assert tx._active_ops == 0
    assert obj.downloaded_to_one_calls == obj.transaction_done_calls == []


def test_producer_exception_returns_protocol_error_and_releases_operation_gate(service):
    transaction = service.new_transaction(Mock(), timeout=10, num_receivers=1)
    obj = MockDownloadable([b"chunk"])
    obj.produce = Mock(side_effect=ValueError("source unreadable"))
    ref = service.add_object(transaction, obj)
    response = service._handle_download(pull_request(ref, "receiver"))
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert service._tx_table[transaction]._active_ops == 0


def test_transaction_stats_progress_failure_and_closed_operations_are_explicit(monkeypatch):
    with pytest.raises(ValueError, match="progress_interval"):
        module._Transaction(10, 1, progress_interval=-1)
    tx = module._Transaction(10, 1)
    tx.logger = Mock()
    assert not tx.is_finished()
    tx.num_receivers = 0
    assert not tx.is_finished()
    tx.add_total_bytes(2)
    tx.add_total_bytes(-1)
    assert tx.get_total_bytes() == 2
    tx.receiver_acquire_timeout = tx.receiver_idle_timeout = None
    assert tx.enforce_receiver_budgets(1) is None
    tx._ops_closed = True
    assert not tx.begin_op()
    assert tx.emit_progress_event({}) is None
    tx.progress_cb = Mock(side_effect=ValueError("progress failed"))
    tx.emit_progress_event({"ref_id": "ref"})
    assert "progress failed" in tx.logger.warning.call_args.args[0]
    monkeypatch.setattr(module, "TransferOutcome", Mock(side_effect=ValueError("outcome failed")))
    assert tx._fail_closed_outcome(module.TransactionDoneStatus.DELETED) is None
    assert "outcome failed" in tx.logger.error.call_args.args[0]


def test_transfer_waiter_uses_own_service_for_acquisition_snapshot():
    service = Mock()
    service.get_acquired_receivers.return_value = {"receiver"}
    waiter = module.TransferWaiter("transaction", service)
    assert waiter.acquired_receivers() == {"receiver"}
    service.get_acquired_receivers.assert_called_once_with("transaction")
