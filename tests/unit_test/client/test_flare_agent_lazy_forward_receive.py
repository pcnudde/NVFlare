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
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_constant import ReturnCode as RC
from nvflare.apis.shareable import Shareable
from nvflare.client.flare_agent import FlareAgent
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message


def _make_shareable(with_lazy_ref=True, task_id="task-1", task_name="train", ref_id="ref-1", payload="small"):
    shareable = Shareable()
    shareable.set_header(FLContextKey.TASK_ID, task_id)
    shareable.set_header(FLContextKey.TASK_NAME, task_name)
    shareable["payload"] = LazyDownloadRef("server", ref_id, "T0") if with_lazy_ref else payload
    return shareable


def _make_agent(req):
    pipe = MagicMock(spec=CellPipe)
    agent = FlareAgent.__new__(FlareAgent)
    agent.logger = MagicMock()
    agent.pipe = pipe
    agent.submit_result_timeout = 30.0
    agent.task_lock = threading.Lock()
    agent.asked_to_stop = False
    agent.current_task = None
    agent.pipe_handler = MagicMock()
    agent.pipe_handler.get_next.return_value = req
    agent.pipe_handler.send_to_peer.return_value = True
    return agent


def test_get_task_resolves_lazy_refs_before_returning_task_data():
    original = _make_shareable(with_lazy_ref=True)
    resolved = _make_shareable(with_lazy_ref=False)
    req = Message.new_request("train", original, msg_id="msg-1")
    agent = _make_agent(req)

    def _resolve(shareable):
        assert agent.current_task is not None
        assert agent.current_task.task_id == "task-1"
        return resolved

    agent._resolve_lazy_refs = MagicMock(side_effect=_resolve)
    agent.shareable_to_task_data = MagicMock(return_value="materialized")

    task = agent.get_task(timeout=1.0)

    assert task.task_name == "train"
    assert task.task_id == "task-1"
    assert task.data == "materialized"
    agent._resolve_lazy_refs.assert_called_once_with(original)
    agent.shareable_to_task_data.assert_called_once_with(resolved)


def test_get_task_materialization_failure_reports_failed_result_and_clears_current_task():
    original = _make_shareable(with_lazy_ref=True)
    req = Message.new_request("train", original, msg_id="msg-1")
    agent = _make_agent(req)
    agent._resolve_lazy_refs = MagicMock(side_effect=RuntimeError("download failed"))
    agent.shareable_to_task_data = MagicMock()

    with pytest.raises(RuntimeError, match="download failed"):
        agent.get_task(timeout=1.0)

    agent.pipe_handler.send_to_peer.assert_called_once()
    reply = agent.pipe_handler.send_to_peer.call_args[0][0]
    assert reply.topic == "train"
    assert reply.req_id == "msg-1"
    assert reply.data.get_return_code() == RC.EXECUTION_EXCEPTION
    assert agent.current_task is None


def test_get_task_materializes_many_lazy_forward_tasks_once_each_without_failure_replies():
    requests = []
    originals = []
    expected_task_ids = []
    for index in range(16):
        task_id = f"task-{index}"
        ref_id = f"ref-{index}"
        expected_task_ids.append(task_id)
        original = _make_shareable(with_lazy_ref=True, task_id=task_id, ref_id=ref_id)
        originals.append(original)
        requests.append(Message.new_request("train", original, msg_id=f"msg-{index}"))

    agent = _make_agent(requests[0])
    agent.pipe_handler.get_next.side_effect = requests

    def _resolve(shareable):
        task_id = shareable.get_header(FLContextKey.TASK_ID)
        ref_id = shareable["payload"].ref_id
        return _make_shareable(
            with_lazy_ref=False,
            task_id=task_id,
            ref_id=ref_id,
            payload=f"materialized-{task_id}",
        )

    agent._resolve_lazy_refs = MagicMock(side_effect=_resolve)
    agent.shareable_to_task_data = MagicMock(side_effect=lambda shareable: shareable["payload"])

    tasks = []
    for _ in requests:
        tasks.append(agent.get_task(timeout=1.0))
        agent.current_task = None

    assert [task.task_id for task in tasks] == expected_task_ids
    assert [task.data for task in tasks] == [f"materialized-{task_id}" for task_id in expected_task_ids]
    assert agent._resolve_lazy_refs.call_count == len(requests)
    assert [call.args[0] for call in agent._resolve_lazy_refs.call_args_list] == originals
    agent.pipe_handler.send_to_peer.assert_not_called()
