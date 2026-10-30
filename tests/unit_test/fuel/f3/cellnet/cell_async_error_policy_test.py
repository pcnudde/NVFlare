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
from unittest.mock import MagicMock, Mock

import pytest

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet import cell as module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message


@pytest.fixture
def cell():
    subject = module.Cell.__new__(module.Cell)
    subject.logger = Mock()
    subject.requests_dict = {}
    subject.core_cell = SimpleNamespace(
        get_fobs_context=Mock(return_value={}),
        update_fobs_context=Mock(),
        regular=Mock(return_value="core"),
        register_request_cb=Mock(),
    )
    return subject


def test_channel_routing_selects_stream_adapter_or_core_and_rejects_unknown_core_method(cell):
    excluded = module.CHANNELS_TO_EXCLUDE[0]
    cell._regular = Mock(return_value="stream")
    assert not module._is_stream_channel(excluded)
    assert not module._is_server_job_cell(SimpleNamespace())
    assert cell.regular(channel=excluded) == "core"
    assert cell.regular(channel="application") == "stream"
    with pytest.raises(AttributeError, match="not in core_cell"):
        cell.nonexistent(channel=excluded)
    cell.update_fobs_context({"custom": "value"})
    cell.core_cell.update_fobs_context.assert_called_once_with({"custom": "value"})
    with pytest.raises(ValueError, match="not callable"):
        cell._register_request_cb("application", "topic", 7)
    cell.core_cell.register_request_cb.assert_not_called()


def test_broadcast_normalizes_single_target_and_maps_failed_fake_worker_to_timeout(monkeypatch, cell):
    cell._encode_message = Mock()
    future = Mock()
    future.result.side_effect = ValueError("send failed")
    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.submit.return_value = future
    factory = Mock(return_value=executor)
    monkeypatch.setattr(
        module,
        "concurrent",
        SimpleNamespace(
            futures=SimpleNamespace(ThreadPoolExecutor=factory, as_completed=lambda futures: list(futures))
        ),
    )
    request = Message(payload=b"request")
    result = cell._broadcast_request("application", "topic", "peer", request, timeout=1)
    assert list(result) == ["peer"] and result["peer"].get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.TIMEOUT
    assert cell._encode_message.call_args.kwargs["receiver_ids"] == ["peer"]
    assert executor.submit.call_args.kwargs["target"] == "peer"
    factory.assert_called_once_with(max_workers=1)
    executor.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "condition, error, expected",
    [
        (module.WaiterRC.TIMEOUT, None, False),
        (module.WaiterRC.ABORTED, None, False),
        (module.WaiterRC.IS_SET, "failed", False),
        (module.WaiterRC.IS_SET, None, True),
    ],
)
def test_future_wait_terminates_on_no_progress_abort_or_completed_error(monkeypatch, cell, condition, error, expected):
    wait = Mock(side_effect=[condition])
    monkeypatch.setattr(module, "conditional_wait", wait)
    future = Mock(error=error)
    future.get_progress.return_value = 0
    assert cell._future_wait(future, 1, Signal()) is expected
    wait.assert_called_once()
    assert cell._check_error(future) == (-1 if error else module.WaiterRC.OK)


def test_encode_failure_propagates_original_error_without_mutating_call_properties(monkeypatch, cell):
    error = ValueError("cannot encode")
    encode = Mock(side_effect=error)
    monkeypatch.setattr(module, "encode_payload", encode)
    props = {"custom": "preserved"}
    with pytest.raises(ValueError, match="cannot encode") as caught:
        cell._encode_message(Message(), Signal(), fobs_ctx_props=props)
    assert caught.value is error and props == {"custom": "preserved"}
    assert "cannot encode" in cell.logger.error.call_args.args[0]


@pytest.mark.parametrize("async_error", [True, False])
def test_adapter_converts_invalid_or_failed_async_callback_response_to_protocol_error(monkeypatch, async_error):
    core = Mock()
    core.get_fobs_context.return_value = {}
    future = Mock()
    core.send_blob.return_value = future
    adapter = module.Adapter(Mock(), SimpleNamespace(fqcn="site"), core)
    adapter.logger = Mock()
    monkeypatch.setattr(module, "encode_payload", Mock(return_value=0))
    args = ("stream-id", "request-id", "application", "topic", "peer", True)
    if async_error:
        response = Mock()
        response.result.side_effect = ValueError("callback failed")
        adapter._send_async_response(response, *args)
    else:
        adapter._send_response(object(), *args)
    message = core.send_blob.call_args.args[3]
    assert message.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert message.get_header(MessageHeaderKey.REQ_ID) == "request-id"
    assert message.get_header(module.StreamHeaderKey.STREAM_REQ_ID) == "stream-id"
    assert core.send_blob.call_args.kwargs == {"optional": True, "reliable": True}
    future.add_done_callback.assert_called_once_with(adapter._handle_reply_stream_done, future, True)


def test_stream_error_from_unrelated_origin_does_not_resolve_waiter(cell):
    waiter = SimpleNamespace(target="peer", in_receiving=Mock(), result=None)
    cell.requests_dict["request"] = waiter
    message = Message(headers={module.StreamHeaderKey.STREAM_REQ_ID: "request", MessageHeaderKey.ORIGIN: "other"})
    cell._process_stream_error(message)
    waiter.in_receiving.set.assert_not_called()
    assert waiter.result is None
    assert "unexpected failed destination" in cell.logger.warning.call_args.args[0]
