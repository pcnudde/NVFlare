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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import AdminCommandNames, RunProcessKey
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.private.defs import CellChannel
from nvflare.private.fed.client import client_executor as subject
from nvflare.private.fed.client.client_status import ClientStatus


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setattr(subject, "secure_log_traceback", Mock())
    client = Mock()
    client.cell.get_fqcn.return_value = "root.site-a"
    return subject.JobExecutor(client, "startup")


@pytest.mark.parametrize(
    "method,topic,body",
    [
        ("get_run_info", AdminCommandNames.SHOW_STATS, {}),
        ("get_errors", AdminCommandNames.SHOW_ERRORS, {"command": AdminCommandNames.SHOW_ERRORS, "data": {}}),
        ("configure_job_log", AdminCommandNames.CONFIGURE_JOB_LOG, {"level": "INFO"}),
    ],
)
@pytest.mark.parametrize("outcome", ["ok", "rejected", "error"])
def test_queries_target_job_cell_and_translate_failure_without_escaping(executor, method, topic, body, outcome):
    payload = {"worker": "ready"}
    if outcome == "error":
        executor.client.cell.send_request.side_effect = OSError("transport unavailable")
    else:
        executor.client.cell.send_request.return_value = new_cell_message(
            {MessageHeaderKey.RETURN_CODE: ReturnCode.OK if outcome == "ok" else ReturnCode.INVALID_REQUEST}, payload
        )
    result = (
        getattr(executor, method)("job-a", body)
        if method == "configure_job_log"
        else getattr(executor, method)("job-a")
    )
    call = executor.client.cell.send_request.call_args.kwargs
    assert call["target"] == "root.site-a.job-a"
    assert call["channel"] == CellChannel.CLIENT_COMMAND
    assert call["topic"] == topic
    assert call["request"].payload == body
    assert call["optional"] is True
    assert call["timeout"] == executor.job_query_timeout
    if outcome == "ok":
        assert result is payload
    elif method == "configure_job_log":
        assert ("transport unavailable" if outcome == "error" else "invalid_request") in result
    elif method == "get_run_info":
        assert result == ({"error": "no info collector. Please try again."} if outcome == "error" else {})
    else:
        assert result is None


@pytest.mark.parametrize("failure", [False, True])
def test_reset_errors_uses_optional_fire_and_forget(executor, failure):
    if failure:
        executor.client.cell.fire_and_forget.side_effect = OSError("closed")
    executor.reset_errors("job-a")
    call = executor.client.cell.fire_and_forget.call_args.kwargs
    assert call["targets"] == "root.site-a.job-a"
    assert call["channel"] == CellChannel.CLIENT_COMMAND
    assert call["topic"] == AdminCommandNames.RESET_ERRORS
    assert call["message"].payload == {"command": AdminCommandNames.RESET_ERRORS, "data": {}}
    assert call["optional"] is True
    executor.client.cell.send_request.assert_not_called()


def test_send_to_job_preserves_channel_timeout_and_optional_flag(executor):
    message = new_cell_message({}, {"data": 1})
    reply = executor.send_to_job("job-a", "channel", "topic", message, 13, optional=True)
    assert reply is executor.client.cell.send_request.return_value
    executor.client.cell.send_request.assert_called_once_with(
        target="root.site-a.job-a", channel="channel", topic="topic", request=message, timeout=13, optional=True
    )


def test_status_query_contains_invalid_state_failures(executor, monkeypatch):
    executor.run_processes["job-a"] = {RunProcessKey.STATUS: ClientStatus.STARTED}
    monkeypatch.setattr(subject, "get_status_message", Mock(side_effect=ValueError("bad state")))
    assert executor.check_status("job-a") == "execution exception. Please try again."
    subject.secure_log_traceback.assert_called_once_with()
