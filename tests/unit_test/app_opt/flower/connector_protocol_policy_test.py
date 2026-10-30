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
from unittest.mock import MagicMock

import pytest

pytest.importorskip("flwr")
from flwr.proto import grpcadapter_pb2 as pb
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterServicer

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.flower import grpc_client as wire_client
from nvflare.app_opt.flower import grpc_server as wire_server
from nvflare.app_opt.flower.connectors import grpc_client_connector as client_module
from nvflare.app_opt.flower.connectors import grpc_server_connector as server_module
from nvflare.app_opt.flower.defs import GRPC_DEFAULT_OPTIONS, Constant
from nvflare.app_opt.flower.utils import msg_container_to_shareable, shareable_to_msg_container


def _message(content=b"body"):
    return pb.MessageContainer(grpc_message_name="task", grpc_message_content=content, metadata={"rank": "2"})


@pytest.mark.parametrize("subject_class", [client_module.GrpcClientConnector, server_module.GrpcServerConnector])
def test_flower_round_configuration_rejects_missing_and_nonpositive_values(subject_class):
    subject = subject_class()
    with pytest.raises(RuntimeError, match="num_rounds"):
        subject.configure({}, FLContext())
    with pytest.raises(ValueError, match="num_rounds"):
        subject.configure({Constant.CONF_KEY_NUM_ROUNDS: 0}, FLContext())
    subject.configure({Constant.CONF_KEY_NUM_ROUNDS: 3}, FLContext())
    assert subject.num_rounds == 3


@pytest.mark.parametrize("bad_field", ["per_msg_timeout", "tx_timeout"])
def test_client_connector_rejects_nonpositive_request_timeouts(bad_field):
    with pytest.raises(ValueError, match=bad_field):
        client_module.GrpcClientConnector(**{bad_field: 0})


def test_flower_wire_client_owns_channel_and_validates_typed_replies(monkeypatch):
    channel = MagicMock()
    create = MagicMock(return_value=channel)
    stub = MagicMock()
    stub_factory = MagicMock(return_value=stub)
    monkeypatch.setattr(wire_client, "create_channel", create)
    monkeypatch.setattr(wire_client, "GrpcAdapterStub", stub_factory)
    client = wire_client.GrpcClient("local:123")
    client.start(ready_timeout=4)
    client.start(ready_timeout=5)
    create.assert_called_once_with(
        server_addr="local:123", grpc_options=GRPC_DEFAULT_OPTIONS, ready_timeout=4, test_only=False
    )
    stub_factory.assert_called_once_with(channel)
    message = _message()
    stub.SendReceive.return_value = _message(b"reply")
    assert client.send_request(message).grpc_message_content == b"reply"
    stub.SendReceive.assert_called_once_with(message)
    stub.SendReceive.return_value = "bad reply"
    assert client.send_request(message) is None
    stub.SendReceive.side_effect = RuntimeError("disconnected")
    assert client.send_request(message).metadata["should-exit"] == "true"
    channel.close.side_effect = RuntimeError("close failed")
    client.stop()
    client.stop()
    channel.close.assert_called_once_with()
    assert client.channel is None


@pytest.mark.parametrize("bind_error", [True, False])
def test_flower_wire_server_uses_factories_and_never_opens_live_transport(monkeypatch, bind_error):
    server = MagicMock()
    if bind_error:
        server.add_insecure_port.side_effect = RuntimeError("bind failed")
    pool = object()
    executor = MagicMock(return_value=pool)
    create = MagicMock(return_value=server)
    register = MagicMock()
    monkeypatch.setattr(wire_server, "futures", SimpleNamespace(ThreadPoolExecutor=executor))
    monkeypatch.setattr(wire_server, "grpc", SimpleNamespace(server=create))
    monkeypatch.setattr(wire_server, "add_GrpcAdapterServicer_to_server", register)
    servicer = GrpcAdapterServicer()
    subject = wire_server.GrpcServer("local:123", 2, None, servicer)
    executor.assert_called_once_with(max_workers=2)
    create.assert_called_once_with(pool, options=GRPC_DEFAULT_OPTIONS)
    register.assert_called_once_with(servicer, server)
    server.add_insecure_port.assert_called_once_with("local:123")
    subject.start(no_blocking=True)
    server.wait_for_termination.assert_not_called()
    subject.start(no_blocking=False)
    server.wait_for_termination.assert_called_once_with()
    subject.shutdown()
    subject.shutdown()
    server.stop.assert_called_once_with(grace=0.5)
    assert subject.grpc_server is None


@pytest.mark.parametrize("field,value", [("servicer", object()), ("max_workers", 0)])
def test_flower_wire_server_validates_before_allocating_resources(field, value):
    kwargs = dict(addr="local:123", max_workers=1, grpc_options=None, servicer=GrpcAdapterServicer())
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        wire_server.GrpcServer(**kwargs)


def test_client_connector_start_wires_internal_server_and_applet_addresses(monkeypatch):
    subject = client_module.GrpcClientConnector(int_server_grpc_options=[("option", 1)])
    context = FLContext()
    context.get_identity_name = lambda: "site-2"
    context.get_engine = lambda: object()
    subject.initialize(context)
    assert subject._client_name == "site-2"
    with pytest.raises(RuntimeError, match="num_rounds"):
        subject.start(context)
    subject.configure({Constant.CONF_KEY_NUM_ROUNDS: 4}, context)
    server = MagicMock()
    factory = MagicMock(return_value=server)
    monkeypatch.setattr(client_module, "GrpcServer", factory)
    monkeypatch.setattr(client_module, "get_local_addresses", lambda count: ["link:1", "clientapp:2"])
    subject.start_applet = MagicMock()
    subject.start(context)
    factory.assert_called_once_with("link:1", 10, [("option", 1)], subject)
    server.start.assert_called_once_with(no_blocking=True)
    subject.start_applet.assert_called_once_with(
        {
            Constant.APP_CTX_CLIENT_NAME: "site-2",
            Constant.APP_CTX_SUPERLINK_ADDR: "link:1",
            Constant.APP_CTX_CLIENTAPP_API_ADDR: "clientapp:2",
            Constant.APP_CTX_NUM_ROUNDS: 4,
        },
        context,
    )


@pytest.mark.parametrize("exit_sent", [True, False])
def test_client_shutdown_waits_for_exit_then_grace_before_releasing_applet(monkeypatch, exit_sent):
    subject = client_module.GrpcClientConnector(client_shutdown_timeout=0.25)
    calls = []
    subject._exit_waiter = SimpleNamespace(wait=lambda timeout: calls.append(("wait", timeout)) or exit_sent)
    monkeypatch.setattr(
        client_module, "time", SimpleNamespace(sleep=lambda duration: calls.append(("sleep", duration)))
    )
    subject.stop_applet = lambda timeout: calls.append(("stop", timeout))
    subject.internal_grpc_server = MagicMock()
    subject.stop(FLContext())
    subject.stop(FLContext())
    assert calls == [("wait", 2.0), ("sleep", 1.0), ("stop", 0.25)]
    assert subject.stopped and subject._training_stopped
    subject.internal_grpc_server.shutdown.assert_called_once_with()


@pytest.mark.parametrize(
    "state,expected", [("applet", (True, 7)), ("training", (True, 0)), ("stopping", (True, 0)), ("running", (False, 0))]
)
def test_client_stopped_status_prioritizes_applet_and_coordinated_shutdown(state, expected):
    subject = client_module.GrpcClientConnector()
    subject.is_applet_stopped = MagicMock(return_value=(state == "applet", 7 if state == "applet" else 0))
    subject._training_stopped = state == "training"
    subject._stopping = state == "stopping"
    subject.stop = MagicMock()
    assert subject._is_stopped() == expected
    assert subject.stop.call_count == int(state == "stopping")


@pytest.mark.parametrize("mode", ["ok", "server_end", "stopped", "error"])
def test_send_receive_marshals_wire_data_or_exit_and_abort_policy(mode):
    subject = client_module.GrpcClientConnector(per_msg_timeout=3, tx_timeout=8)
    subject.set_abort_signal(Signal())
    context = FLContext()
    subject.engine = MagicMock()
    subject.engine.new_context.return_value = context
    subject.system_panic = MagicMock()
    response = (
        msg_container_to_shareable(_message(b"reply")) if mode == "ok" else make_reply(ReturnCode.SERVICE_UNAVAILABLE)
    )
    subject.send_request = MagicMock(return_value=response)
    if mode == "error":
        subject.send_request.side_effect = ValueError("transport failed")
    subject.stopped = mode == "stopped"
    request = _message()
    result = subject.SendReceive(request, None)
    if mode == "ok":
        assert result.grpc_message_content == b"reply"
        assert dict(result.metadata) == {"rank": "2"}
        kwargs = subject.send_request.call_args.kwargs
        assert kwargs["op"] == "request" and kwargs["target"] is None
        assert kwargs["per_msg_timeout"] == 3 and kwargs["tx_timeout"] == 8
        assert kwargs["request"][Constant.PARAM_KEY_CONTENT] == b"body"
        assert not subject._exit_waiter.is_set()
    else:
        assert result.metadata["should-exit"] == "true"
        assert subject._exit_waiter.is_set() and subject._stopping
        assert subject.abort_signal.triggered is (mode == "error")
        assert subject.system_panic.call_count == int(mode == "error")
        assert subject.send_request.call_count == int(mode != "stopped")


def test_client_request_rejects_untyped_reliable_message_reply():
    subject = client_module.GrpcClientConnector()
    subject.send_request = MagicMock(return_value="bad")
    with pytest.raises(RuntimeError, match="expect Shareable"):
        subject._send_flower_request(Shareable())


def test_server_connector_start_stop_marshals_all_three_addresses(monkeypatch):
    subject = server_module.GrpcServerConnector(flower_server_ready_timeout=6)
    subject.configure({Constant.CONF_KEY_NUM_ROUNDS: 5}, FLContext())
    subject.start_applet = MagicMock()
    subject.stop_applet = MagicMock(return_value=9)
    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(server_module, "GrpcClient", factory)
    monkeypatch.setattr(server_module, "get_local_addresses", lambda count: ["server:1", "fleet:2", "exec:3"])
    context = FLContext()
    subject.start(context)
    factory.assert_called_once_with("fleet:2", None)
    client.start.assert_called_once_with(ready_timeout=6)
    subject.start_applet.assert_called_once_with(
        {
            Constant.APP_CTX_SERVERAPP_API_ADDR: "server:1",
            Constant.APP_CTX_FLEET_API_ADDR: "fleet:2",
            Constant.APP_CTX_EXEC_API_ADDR: "exec:3",
            Constant.APP_CTX_NUM_ROUNDS: 5,
        },
        context,
    )
    subject.is_applet_stopped = MagicMock(return_value=(False, 0))
    assert subject._is_stopped() == (False, 0)
    subject.stop(context)
    client.stop.assert_called_once_with()
    assert subject.internal_grpc_client is None
    assert subject._is_stopped() == (True, 9)
    subject.is_applet_stopped.return_value = (True, 4)
    assert subject._is_stopped() == (True, 4)


def test_server_connector_wraps_local_address_allocation_failure(monkeypatch):
    monkeypatch.setattr(server_module, "get_local_addresses", MagicMock(side_effect=ValueError("unavailable")))
    with pytest.raises(RuntimeError, match="failed to get addresses"):
        server_module.GrpcServerConnector().start(FLContext())


@pytest.mark.parametrize("state", ["running", "stopped", "no_client", "bad_reply"])
def test_server_request_converts_protobuf_payload_and_rejects_unavailable_backend(state):
    subject = server_module.GrpcServerConnector()
    subject.is_applet_stopped = MagicMock(return_value=(state == "stopped", 0))
    client = MagicMock()
    client.send_request.return_value = "bad" if state == "bad_reply" else _message(b"reply")
    subject.internal_grpc_client = None if state == "no_client" else client
    request = msg_container_to_shareable(_message())
    if state == "bad_reply":
        with pytest.raises(RuntimeError, match="expect MessageContainer"):
            subject.send_request_to_flower(request, FLContext())
    else:
        result = subject.send_request_to_flower(request, FLContext())
        if state == "running":
            assert result[Constant.PARAM_KEY_CONTENT] == b"reply"
            assert result[Constant.PARAM_KEY_HEADERS] == {"rank": "2"}
            sent = client.send_request.call_args.args[0]
            assert sent == shareable_to_msg_container(request)
        else:
            assert result.get_return_code() == ReturnCode.SERVICE_UNAVAILABLE
            client.send_request.assert_not_called()


def test_server_app_request_is_dropped_when_connector_has_already_stopped():
    subject = server_module.GrpcServerConnector()
    subject._is_stopped = MagicMock(return_value=(True, 5))
    subject.send_request_to_flower = MagicMock()
    request = Shareable()
    context = FLContext()
    assert (
        subject.process_app_request("train", request, context, Signal()).get_return_code()
        == ReturnCode.SERVICE_UNAVAILABLE
    )
    subject.send_request_to_flower.assert_not_called()
    subject._is_stopped.return_value = (False, 0)
    reply = Shareable({"result": "ready"})
    subject.send_request_to_flower.return_value = reply
    assert subject.process_app_request("train", request, context, Signal()) is reply
    subject.send_request_to_flower.assert_called_once_with(request, context)
