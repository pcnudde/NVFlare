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

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("flwr")
from flwr.proto import grpcadapter_pb2 as pb

from nvflare.app_opt.flower.mock import flower_client, flower_server
from nvflare.app_opt.flower.mock.echo_servicer import EchoServicer


def test_echo_servicer_advances_round_and_shuts_down_at_configured_limit():
    subject = EchoServicer(num_rounds=1)
    server = MagicMock()
    subject.set_server(server)
    first = subject.SendReceive(
        pb.MessageContainer(grpc_message_name="task", grpc_message_content=b"data", metadata={"round": "0"}), None
    )
    assert first.grpc_message_content == b"data" and first.grpc_message_name == "task"
    assert first.metadata["round"] == "1"
    assert not subject.stopped
    server.shutdown.assert_not_called()
    second = subject.SendReceive(first, None)
    assert second.metadata["round"] == "2"
    assert subject.stopped
    server.shutdown.assert_called_once_with()


@pytest.mark.parametrize("mode", ["match", "name", "content"])
def test_demo_client_preserves_origin_round_and_payload_and_exits_on_server_instruction(monkeypatch, mode):
    client = MagicMock()
    replies = []

    def reply(request):
        replies.append(request)
        return pb.MessageContainer(
            grpc_message_name="other" if mode == "name" else request.grpc_message_name,
            grpc_message_content=b"other" if mode == "content" else request.grpc_message_content,
            metadata={"round": "1"} if len(replies) == 1 else {"should-exit": "true"},
        )

    client.send_request.side_effect = reply
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(flower_client, "GrpcClient", factory)
    monkeypatch.setattr(flower_client, "os", SimpleNamespace(urandom=lambda length: b"data"))
    ticks = iter(range(20))
    sleeps = []
    monkeypatch.setattr(
        flower_client, "time", SimpleNamespace(time=lambda: next(ticks), sleep=lambda seconds: sleeps.append(seconds))
    )
    log = MagicMock()
    monkeypatch.setattr(flower_client, "log", log)
    flower_client.train("local:1", "site-2")
    factory.assert_called_once_with(server_addr="local:1")
    client.start.assert_called_once_with()
    assert [request.metadata["round"] for request in replies] == ["0", "1"]
    assert all(request.metadata["origin"] == "site-2" and request.metadata["target"] == "server" for request in replies)
    assert all(request.grpc_message_content == b"data" for request in replies)
    assert sleeps == [1.0]
    messages = [call.args[0] for call in log.call_args_list]
    assert "got should-exit!" in messages
    expected = {
        "match": "OK: result matches request!",
        "name": "ERROR: msg_name does not match request",
        "content": "ERROR: result does not match request",
    }[mode]
    assert messages.count(expected) == 2
    assert messages[-1].startswith("DONE:")


def test_demo_log_emits_timestamp_and_message(monkeypatch, capsys):
    monkeypatch.setattr(flower_client, "time", SimpleNamespace(time=lambda: 123))
    monkeypatch.setattr(flower_client, "time_to_string", lambda stamp: "timestamp")
    flower_client.log("status")
    assert "timestamp: status" in capsys.readouterr().out


def _logging():
    return SimpleNamespace(basicConfig=MagicMock(), getLogger=lambda: MagicMock(), INFO=20)


@pytest.mark.parametrize("rounds", ["2", "0", "-1"])
def test_demo_client_cli_validates_budget_before_delegating_to_training(monkeypatch, rounds):
    monkeypatch.setattr(sys, "argv", ["client", "-a", "local:1", "-c", "site-2", "-n", rounds])
    monkeypatch.setattr(flower_client, "logging", _logging())
    train = MagicMock()
    monkeypatch.setattr(flower_client, "train", train)
    if rounds == "2":
        flower_client.main()
        train.assert_called_once_with("local:1", "site-2")
    else:
        with pytest.raises(RuntimeError, match="num rounds"):
            flower_client.main()
        train.assert_not_called()


def test_demo_server_cli_wires_servicer_round_limit_and_worker_count(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server", "-a", "local:1", "-n", "3", "-w", "2"])
    monkeypatch.setattr(flower_server, "logging", _logging())
    server = MagicMock()
    factory = MagicMock(return_value=server)
    monkeypatch.setattr(flower_server, "GrpcServer", factory)
    flower_server.main()
    kwargs = factory.call_args.kwargs
    assert factory.call_args.args == ("local:1",)
    assert kwargs["max_workers"] == 2 and kwargs["grpc_options"] is None
    assert kwargs["servicer"].num_rounds == 3
    assert kwargs["servicer"].server is server
    server.start.assert_called_once_with()
