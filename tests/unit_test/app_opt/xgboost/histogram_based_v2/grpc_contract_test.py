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

import grpc
import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2 import grpc_client as client_module
from nvflare.app_opt.xgboost.histogram_based_v2 import grpc_server as server_module
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors import grpc_client_adaptor as client_adaptor_module
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors import grpc_server_adaptor as server_adaptor_module
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.proto import federated_pb2 as pb2


@pytest.fixture
def context(tmp_path):
    ctx = FLContext()
    engine = MagicMock()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site")
    workspace = MagicMock()
    workspace.get_run_dir.return_value = str(tmp_path)
    ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    ctx.set_prop(FLContextKey.CURRENT_RUN, "job")
    return ctx


OPERATIONS = [
    ("all_gather", "send_allgather", "Allgather", pb2.AllgatherRequest, pb2.AllgatherReply, {}),
    ("all_gather_v", "send_allgatherv", "AllgatherV", pb2.AllgatherVRequest, pb2.AllgatherVReply, {}),
    (
        "all_reduce",
        "send_allreduce",
        "Allreduce",
        pb2.AllreduceRequest,
        pb2.AllreduceReply,
        {"data_type": 0, "reduce_op": 0},
    ),
    ("broadcast", "send_broadcast", "Broadcast", pb2.BroadcastRequest, pb2.BroadcastReply, {"root": 0}),
]


@pytest.mark.parametrize("operation,send_method,rpc,request_type,reply_type,extra", OPERATIONS)
def test_server_adaptor_forwards_collective_arguments_and_rejects_bad_reply_types(
    context, operation, send_method, rpc, request_type, reply_type, extra
):
    adaptor = server_adaptor_module.GrpcServerAdaptor()
    adaptor.internal_xgb_client = MagicMock()
    sender = getattr(adaptor.internal_xgb_client, send_method)
    sender.return_value = reply_type(receive_buffer=b"reply")
    result = getattr(adaptor, operation)(rank=1, seq=2, send_buf=b"input", fl_ctx=context, **extra)
    assert result == b"reply"
    assert sender.call_args.kwargs == {"seq_num": 2, "rank": 1, "data": b"input", **extra}
    sender.return_value = object()
    with pytest.raises(RuntimeError, match="bad result"):
        getattr(adaptor, operation)(rank=1, seq=2, send_buf=b"input", fl_ctx=context, **extra)
    adaptor._stopping = True
    assert getattr(adaptor, operation)(rank=1, seq=2, send_buf=b"input", fl_ctx=context, **extra) == b""


def test_server_adaptor_start_marshals_runner_context_and_stop_releases_client(monkeypatch, context):
    adaptor = server_adaptor_module.GrpcServerAdaptor()
    adaptor.world_size = 2
    adaptor.start_runner = MagicMock()
    adaptor.stop_runner = MagicMock()
    adaptor.is_runner_stopped = MagicMock(return_value=(False, 0))
    client = MagicMock()
    monkeypatch.setattr(server_adaptor_module, "GrpcClient", MagicMock(return_value=client))
    monkeypatch.setattr(server_adaptor_module, "get_open_tcp_port", lambda resources: 12000)
    adaptor.start(context)
    runner_ctx = adaptor.start_runner.call_args.args[0]
    assert runner_ctx == {
        Constant.RUNNER_CTX_SERVER_ADDR: "127.0.0.1:12000",
        Constant.RUNNER_CTX_WORLD_SIZE: 2,
        Constant.RUNNER_CTX_PORT: 12000,
    }
    client.start.assert_called_once_with(ready_timeout=adaptor.xgb_server_ready_timeout)
    assert adaptor._is_stopped() == (False, 0)
    adaptor.stop(context)
    client.stop.assert_called_once_with()
    adaptor.stop_runner.assert_called_once_with()
    assert adaptor.internal_xgb_client is None and adaptor._is_stopped() == (True, 0)
    monkeypatch.setattr(server_adaptor_module, "get_open_tcp_port", lambda resources: None)
    with pytest.raises(RuntimeError, match="failed to get a port"):
        adaptor.start(context)


@pytest.mark.parametrize("operation,send_method,rpc,request_type,reply_type,extra", OPERATIONS)
def test_client_stub_builds_real_protobuf_request_and_checks_reply_type(
    operation, send_method, rpc, request_type, reply_type, extra
):
    client = client_module.GrpcClient("localhost:1")
    client.stub = MagicMock()
    getattr(client.stub, rpc).return_value = reply_type(receive_buffer=b"reply")
    response = getattr(client, send_method)(seq_num=2, rank=1, data=b"input", **extra)
    assert response.receive_buffer == b"reply"
    request = getattr(client.stub, rpc).call_args.args[0]
    assert isinstance(request, request_type)
    assert request.sequence_number == 2 and request.rank == 1 and request.send_buffer == b"input"
    for key, value in extra.items():
        assert getattr(request, "reduce_operation" if key == "reduce_op" else key) == value
    getattr(client.stub, rpc).return_value = object()
    assert getattr(client, send_method)(seq_num=2, rank=1, data=b"input", **extra) is None


def test_client_channel_readiness_timeout_and_idempotent_cleanup(monkeypatch):
    channel = MagicMock()
    factory = MagicMock(return_value=channel)
    future = MagicMock()
    monkeypatch.setattr(client_module.grpc, "insecure_channel", factory)
    monkeypatch.setattr(client_module.grpc, "channel_ready_future", MagicMock(return_value=future))
    monkeypatch.setattr(client_module, "FederatedStub", MagicMock())
    client = client_module.GrpcClient("localhost:1", [("x", 1)])
    client.start(ready_timeout=4)
    client.start(ready_timeout=4)
    factory.assert_called_once_with("localhost:1", options=[("x", 1)])
    future.result.assert_called_once_with(timeout=4)
    channel.close.side_effect = RuntimeError("already closed")
    client.stop()
    client.stop()
    channel.close.assert_called_once_with()
    assert client.channel is None
    future.result.side_effect = grpc.FutureTimeoutError()
    with pytest.raises(RuntimeError, match="after 3 seconds"):
        client_module.GrpcClient("localhost:1").start(ready_timeout=3)


@pytest.mark.parametrize("operation,send_method,rpc,request_type,reply_type,extra", OPERATIONS)
@pytest.mark.parametrize("fails", [False, True])
def test_client_adaptor_returns_typed_reply_or_grpc_error_and_clears_pending_request(
    monkeypatch, operation, send_method, rpc, request_type, reply_type, extra, fails
):
    adaptor = client_adaptor_module.GrpcClientAdaptor()
    method = {
        "all_gather": "_send_all_gather",
        "all_gather_v": "_do_all_gather_v",
        "all_reduce": "_send_all_reduce",
        "broadcast": "_do_broadcast",
    }[operation]
    boundary = MagicMock(return_value=(b"reply", {}) if operation in ("all_gather", "all_reduce") else b"reply")
    if fails:
        boundary.side_effect = ValueError("bad request")
    monkeypatch.setattr(adaptor, method, boundary)
    adaptor._abort = MagicMock()
    fields = {"rank": 1, "sequence_number": 2, "send_buffer": b"input"}
    fields.update({"reduce_operation" if key == "reduce_op" else key: value for key, value in extra.items()})
    grpc_context = MagicMock()
    result = getattr(adaptor, rpc)(request_type(**fields), grpc_context)
    assert isinstance(result, reply_type)
    assert result.receive_buffer == (b"" if fails else b"reply")
    assert adaptor._pending_req == {}
    if fails:
        grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        assert "bad request" in adaptor._abort.call_args.kwargs["reason"]
    else:
        kwargs = boundary.call_args.kwargs
        assert kwargs["rank"] == 1 and kwargs["seq"] == 2 and kwargs["send_buf"] == b"input"
        adaptor._abort.assert_not_called()


def test_duplicate_request_wait_is_controlled_and_cleanup_sets_waiter(monkeypatch):
    adaptor = client_adaptor_module.GrpcClientAdaptor()
    assert adaptor._check_duplicate_seq("allgather", 1, 2) is False
    waiter = MagicMock()
    adaptor._pending_req[(1, 2)] = waiter
    sleep = MagicMock()
    monkeypatch.setattr(client_adaptor_module, "time", SimpleNamespace(sleep=sleep))
    assert adaptor._check_duplicate_seq("allgather", 1, 2) is True
    waiter.wait.assert_called_once_with(client_adaptor_module.DUPLICATE_REQ_MAX_HOLD_TIME)
    sleep.assert_called_once_with(1)
    adaptor._finish_pending_req("allgather", 1, 2)
    waiter.set.assert_called_once_with()
    assert adaptor._pending_req == {}
    adaptor._finish_pending_req("allgather", 1, 2)


def test_client_adaptor_start_stop_use_mocked_server_and_runner_boundaries(monkeypatch, context):
    adaptor = client_adaptor_module.GrpcClientAdaptor()
    adaptor.initialize(context)
    adaptor.start_runner = MagicMock()
    adaptor.stop_runner = MagicMock()
    adaptor.is_runner_stopped = MagicMock(return_value=(False, 0))
    server = MagicMock()
    monkeypatch.setattr(client_adaptor_module, "GrpcServer", MagicMock(return_value=server))
    monkeypatch.setattr(client_adaptor_module, "get_open_tcp_port", lambda resources: 12345)
    with pytest.raises(RuntimeError, match="rank is not set"):
        adaptor.start(context)
    adaptor.rank, adaptor.world_size, adaptor.num_rounds = 0, 2, 3
    adaptor.start(context)
    runner_ctx = adaptor.start_runner.call_args.args[0]
    assert runner_ctx[Constant.RUNNER_CTX_CLIENT_NAME] == "site" and runner_ctx[Constant.RUNNER_CTX_RANK] == 0
    assert runner_ctx[Constant.RUNNER_CTX_SERVER_ADDR] == "127.0.0.1:12345"
    server.start.assert_called_once_with(no_blocking=True)
    assert adaptor._is_stopped() == (False, 0)
    adaptor.stop(context)
    adaptor.stop(context)
    server.shutdown.assert_called_once_with()
    adaptor.stop_runner.assert_called_once_with()
    assert adaptor._is_stopped() == (True, 0)


def test_grpc_server_configuration_and_lifecycle_never_create_live_transport(monkeypatch):
    server = MagicMock()
    pool = MagicMock()
    pool_factory = MagicMock(return_value=pool)
    factory = MagicMock(return_value=server)
    monkeypatch.setattr(server_module, "futures", SimpleNamespace(ThreadPoolExecutor=pool_factory))
    monkeypatch.setattr(server_module.grpc, "server", factory)
    register = MagicMock()
    monkeypatch.setattr(server_module, "add_FederatedServicer_to_server", register)
    servicer = server_module.FederatedServicer()
    wrapper = server_module.GrpcServer("localhost:1", 2, servicer)
    pool_factory.assert_called_once_with(max_workers=2)
    register.assert_called_once_with(servicer, server)
    server.add_insecure_port.assert_called_once_with("localhost:1")
    wrapper.start(no_blocking=True)
    server.wait_for_termination.assert_not_called()
    wrapper.start()
    server.wait_for_termination.assert_called_once_with()
    wrapper.shutdown()
    wrapper.shutdown()
    server.stop.assert_called_once_with(grace=0.5)
    with pytest.raises(TypeError):
        server_module.GrpcServer("localhost:1", 2, object())
