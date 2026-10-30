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

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("xgboost")

from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.mock import mock_secure_client_runner as secure
from nvflare.app_opt.xgboost.histogram_based_v2.mock import mock_server_runner as server_runner
from nvflare.app_opt.xgboost.histogram_based_v2.mock import run_client, run_server
from nvflare.app_opt.xgboost.histogram_based_v2.proto import federated_pb2 as pb


@pytest.mark.parametrize("mode", ["match", "wrong_type", "wrong_content", "abort"])
def test_secure_mock_runner_emits_label_and_group_protocol_in_sequence(monkeypatch, mode):
    subject = secure.MockSecureClientRunner(sample_size=4)
    client = MagicMock()

    def broadcast(**kwargs):
        return (
            object()
            if mode == "wrong_type"
            else pb.BroadcastReply(receive_buffer=b"wrong" if mode == "wrong_content" else kwargs["data"])
        )

    client.send_broadcast.side_effect = broadcast
    client.send_allgatherv.return_value = (
        object() if mode == "wrong_type" else pb.AllgatherVReply(receive_buffer=b"result")
    )
    monkeypatch.setattr(secure, "GrpcClient", MagicMock(return_value=client))
    ticks = iter(range(20))
    monkeypatch.setattr(secure, "time", SimpleNamespace(time=lambda: next(ticks), sleep=lambda seconds: None))
    subject.logger = MagicMock()
    if mode == "abort":
        subject.stop()
    subject.run(
        {Constant.RUNNER_CTX_SERVER_ADDR: "local:1", Constant.RUNNER_CTX_RANK: 2, Constant.RUNNER_CTX_NUM_ROUNDS: 1}
    )
    assert subject.is_stopped() == (True, 0)
    if mode == "abort":
        client.send_broadcast.assert_not_called()
        return
    calls = client.send_broadcast.call_args_list
    assert [call.kwargs["seq_num"] for call in calls] == [1, 2]
    assert json.loads(calls[0].kwargs["data"]) == {"op": "none"}
    assert json.loads(calls[1].kwargs["data"]) == {"op": "gh", "size": 4}
    request = client.send_allgatherv.call_args.kwargs
    assert request["seq_num"] == 3 and request["rank"] == 2
    assert json.loads(request["data"]) == {"op": "aggr", "groups": None}
    assert subject.logger.error.call_count == {"match": 0, "wrong_type": 3, "wrong_content": 1}[mode]


def test_mock_server_runner_wires_quorum_and_releases_server_once(monkeypatch):
    server = MagicMock()
    factory = MagicMock(return_value=server)
    monkeypatch.setattr(server_runner, "GrpcServer", factory)
    subject = server_runner.MockServerRunner(server_max_workers=2, aggr_timeout=3)
    assert subject.is_stopped() == (False, 0)
    subject.run({Constant.RUNNER_CTX_WORLD_SIZE: 4, Constant.RUNNER_CTX_SERVER_ADDR: "local:1"})
    assert factory.call_args.args == ("local:1",)
    kwargs = factory.call_args.kwargs
    assert kwargs["max_workers"] == 2
    assert kwargs["servicer"].num_clients == 4
    assert kwargs["servicer"].aggr_timeout == 3
    server.start.assert_called_once_with(no_blocking=False)
    subject.stop()
    subject.stop()
    server.shutdown.assert_called_once_with()
    assert subject.is_stopped() == (True, 0)


@pytest.mark.parametrize("mode", ["match", "wrong_type", "wrong_content"])
def test_client_cli_decodes_rank_and_generates_all_four_collectives(monkeypatch, capsys, mode):
    monkeypatch.setattr(sys, "argv", ["client", "-a", "local:1", "-r", "2", "-n", "1"])
    client = MagicMock()
    for name, reply in [
        ("send_allgather", pb.AllgatherReply),
        ("send_allgatherv", pb.AllgatherVReply),
        ("send_allreduce", pb.AllreduceReply),
        ("send_broadcast", pb.BroadcastReply),
    ]:
        getattr(client, name).return_value = (
            object()
            if mode == "wrong_type"
            else reply(receive_buffer=b"wrong" if mode == "wrong_content" else b"payload")
        )
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(run_client, "GrpcClient", factory)
    monkeypatch.setattr(run_client, "os", SimpleNamespace(urandom=lambda size: b"payload"))
    ticks = iter(range(20))
    monkeypatch.setattr(run_client, "time", SimpleNamespace(time=lambda: next(ticks), sleep=lambda seconds: None))
    run_client.main()
    factory.assert_called_once_with(server_addr="local:1")
    assert client.send_allgather.call_args.kwargs == {"seq_num": 1, "rank": 2, "data": b"payload"}
    assert client.send_allgatherv.call_args.kwargs["seq_num"] == 2
    assert client.send_allreduce.call_args.kwargs["seq_num"] == 3
    assert client.send_broadcast.call_args.kwargs == {"seq_num": 4, "rank": 2, "data": b"payload", "root": 3}
    output = capsys.readouterr().out
    assert "DONE:" in output
    assert output.count({"match": "OK:", "wrong_type": "expect reply", "wrong_content": "ERROR:"}[mode]) == 4


def test_server_cli_parses_worker_count_and_constructs_expected_aggregator(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server", "-a", "local:1", "-c", "4", "-w", "2"])
    monkeypatch.setattr(
        run_server, "logging", SimpleNamespace(basicConfig=MagicMock(), getLogger=lambda: MagicMock(), INFO=20)
    )
    server = MagicMock()
    factory = MagicMock(return_value=server)
    monkeypatch.setattr(run_server, "GrpcServer", factory)
    run_server.main()
    assert factory.call_args.args == ("local:1",)
    assert factory.call_args.kwargs["max_workers"] == 2
    assert factory.call_args.kwargs["servicer"].num_clients == 4
    server.start.assert_called_once_with()
