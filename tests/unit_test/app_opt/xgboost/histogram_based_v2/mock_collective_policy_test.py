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

pytest.importorskip("xgboost")

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.mock import mock_client_runner as runner_module
from nvflare.app_opt.xgboost.histogram_based_v2.mock import mock_data_converter as data_module
from nvflare.app_opt.xgboost.histogram_based_v2.mock.aggr_servicer import AggrServicer, ReqWaiter
from nvflare.app_opt.xgboost.histogram_based_v2.proto import federated_pb2 as pb
from nvflare.app_opt.xgboost.histogram_based_v2.sec.data_converter import FeatureAggregationResult


@pytest.mark.parametrize(
    "op,request_type,reply",
    [
        ("Allgather", pb.AllgatherRequest, pb.AllgatherReply),
        ("AllgatherV", pb.AllgatherVRequest, pb.AllgatherVReply),
        ("Allreduce", pb.AllreduceRequest, pb.AllreduceReply),
        ("Broadcast", pb.BroadcastRequest, pb.BroadcastReply),
    ],
)
def test_mock_collectives_echo_real_protobuf_payload_and_reset_waiter(op, request_type, reply):
    servicer = AggrServicer(num_clients=1, aggr_timeout=0)
    message = request_type(rank=2, sequence_number=7, send_buffer=b"data")
    result = getattr(servicer, op)(message, None)
    assert isinstance(result, reply)
    assert result.receive_buffer == b"data"
    assert servicer.req_waiter is None


def test_waiter_enforces_sequence_operation_uniqueness_and_quorum():
    waiter = ReqWaiter(2, 3, "allgather")
    with pytest.raises(RuntimeError, match="expecting seq"):
        waiter.add_request("allgather", 0, 4, pb.AllgatherRequest())
    with pytest.raises(RuntimeError, match="expecting op"):
        waiter.add_request("broadcast", 0, 3, pb.AllgatherRequest())
    waiter.add_request("allgather", 0, 3, pb.AllgatherRequest(send_buffer=b"a"))
    assert not waiter.wait(0)
    with pytest.raises(RuntimeError, match="duplicate request"):
        waiter.add_request("allgather", 0, 3, pb.AllgatherRequest())
    waiter.add_request("allgather", 1, 3, pb.AllgatherRequest(send_buffer=b"b"))
    assert waiter.wait(0)
    assert {rank: reply.receive_buffer for rank, reply in waiter.result.items()} == {0: b"a", 1: b"b"}
    with pytest.raises(RuntimeError, match="unknown request"):
        ReqWaiter(1, 1, "bad").add_request("bad", 0, 1, object())


def test_mock_converter_assigns_features_generates_label_pairs_and_validates_histograms(monkeypatch, capsys):
    monkeypatch.setattr(data_module, "SAMPLE_SIZE", 4)
    monkeypatch.setattr(data_module, "NUM_FEATURES", 2)
    monkeypatch.setattr(data_module, "RANK_FEATURES", [(0, 1), (1, 2)])
    monkeypatch.setattr(data_module, "random", SimpleNamespace(randint=lambda low, high: 2))
    converter = data_module.MockDataConverter()
    context = FLContext()
    context.set_prop(Constant.PARAM_KEY_RANK, 1)
    assert converter.decode_gh_pairs(b"not-json", context) is None
    context.set_prop(Constant.PARAM_KEY_RANK, 0)
    assert converter.decode_gh_pairs(b'{"op":"other"}', context) is None
    assert converter.decode_gh_pairs(b'{"op":"gh"}', context) == [(2, 2)] * 4
    aggregation = converter.decode_aggregation_context(b"ignored", context)
    assert [feature.feature_id for feature in aggregation.features] == [0]
    assert converter.features[1].sample_bin_assignment == [1, 2, 3, 4]
    expected = converter._aggregate_feature(converter.features[0], None)
    assert expected[:4] == [(2, 2)] * 4
    monkeypatch.setattr(data_module, "os", SimpleNamespace(urandom=lambda length: b"wire"))
    feature = FeatureAggregationResult(0, expected)
    assert converter.encode_aggregation_result({0: [feature]}, context) == b"wire"
    assert "Result OK" in capsys.readouterr().out
    feature.aggregated_hist = []
    converter.encode_aggregation_result({0: [feature]}, context)
    assert "expected aggr != received" in capsys.readouterr().out
    converter.decode_aggregation_context(b"ignored", context)
    assert converter.groups == {1: [1, 3, 4, 101], 4: [2, 7, 9, 23, 50]}


@pytest.mark.parametrize("mode", ["match", "wrong_type", "wrong_data", "abort"])
def test_mock_runner_builds_monotonic_collectives_and_reports_validation_failures(monkeypatch, mode):
    client = MagicMock()
    replies = [pb.AllgatherReply, pb.AllgatherVReply, pb.AllreduceReply, pb.BroadcastReply]
    for name, reply in zip(["send_allgather", "send_allgatherv", "send_allreduce", "send_broadcast"], replies):
        getattr(client, name).return_value = (
            object() if mode == "wrong_type" else reply(receive_buffer=b"bad" if mode == "wrong_data" else b"data")
        )
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(runner_module, "GrpcClient", factory)
    monkeypatch.setattr(runner_module, "os", SimpleNamespace(urandom=lambda size: b"data"))
    times = iter(range(20))
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(time=lambda: next(times), sleep=lambda duration: None))
    runner = runner_module.MockClientRunner()
    runner.logger = MagicMock()
    if mode == "abort":
        runner.stop()
    runner.run(
        {
            Constant.RUNNER_CTX_SERVER_ADDR: "localhost:123",
            Constant.RUNNER_CTX_RANK: 2,
            Constant.RUNNER_CTX_NUM_ROUNDS: 2,
        }
    )
    assert runner.is_stopped() == (True, 0)
    factory.assert_called_once_with(server_addr="localhost:123")
    client.start.assert_called_once_with()
    if mode == "abort":
        client.send_allgather.assert_not_called()
        return
    assert [call.kwargs["seq_num"] for call in client.send_allgather.call_args_list] == [1, 5]
    assert [call.kwargs["seq_num"] for call in client.send_broadcast.call_args_list] == [4, 8]
    assert client.send_allreduce.call_args.kwargs == {
        "seq_num": 7,
        "rank": 2,
        "data": b"data",
        "reduce_op": 2,
        "data_type": 2,
    }
    assert runner.logger.error.call_count == (0 if mode == "match" else 8)
