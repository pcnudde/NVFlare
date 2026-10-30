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

from nvflare.apis.fl_constant import ProcessType, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.private import aux_runner as module


@pytest.fixture
def runner():
    subject = module.AuxRunner(Mock())
    subject.logger = Mock()
    return subject


@pytest.mark.parametrize("optional", [True, False])
def test_aux_send_contains_transport_failure_and_uses_optional_log_level(runner, optional):
    runner._send_to_cell = Mock(side_effect=ValueError("transport failed"))
    assert runner.send_aux_request([], "topic", Shareable(), 1, FLContext(), optional=optional) == {}
    assert (runner.logger.debug if optional else runner.logger.error).call_count == 2


def test_aux_dispatch_rejects_nonshareable_request_after_resolving_handler(runner):
    runner.register_aux_message_handler("topic", Mock())
    assert runner._process_request("topic", {}, FLContext()).get_return_code() == ReturnCode.BAD_REQUEST_DATA
    runner.topic_table["topic"].assert_not_called()


@pytest.mark.parametrize(
    "process", [ProcessType.CLIENT_PARENT, ProcessType.SERVER_PARENT, ProcessType.CLIENT_JOB, "unknown"]
)
def test_job_scoped_target_requires_process_identity_and_job_id(process):
    context = FLContext()
    context.set_prop(ReservedKey.PROCESS_TYPE, process)
    target = module.AuxMsgTarget("site", "relay.site")
    if process in [ProcessType.CLIENT_PARENT, ProcessType.SERVER_PARENT]:
        assert module.AuxRunner._get_target_fqcn(target, context) == "relay.site"
    else:
        with pytest.raises(RuntimeError, match="no job ID|invalid process_type"):
            module.AuxRunner._get_target_fqcn(target, context)


def test_aux_wire_request_preserves_message_root_headers_and_waits_for_cell_with_finite_poll(monkeypatch, runner):
    context = FLContext()
    context.set_prop(ReservedKey.PROCESS_TYPE, ProcessType.CLIENT_PARENT)
    target = module.AuxMsgTarget("site", "site")
    request = Shareable()
    request.set_header(ReservedHeaderKey.MSG_ROOT_ID, "root")
    request.set_header(ReservedHeaderKey.MSG_ROOT_TTL, 3)
    cell = Mock()
    cell.broadcast_request.return_value = {}
    runner.engine.get_cell.side_effect = [None, cell]
    runner.cell_wait_timeout = 10
    stamps = iter([0, 0, 0])
    pause = Mock(side_effect=[None])
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: next(stamps), sleep=pause))
    assert runner._send_to_cell([target], "channel", "topic", request, 1, context) == {}
    pause.assert_called_once_with(0.1)
    wire = cell.broadcast_request.call_args.kwargs["request"]
    assert wire.get_header(MessageHeaderKey.MSG_ROOT_ID) == "root"
    assert wire.get_header(MessageHeaderKey.MSG_ROOT_TTL) == 3
    assert "relay.site" in str(module.AuxMsgTarget("site", "relay.site"))
    runner._wait_for_cell = Mock(return_value=None)
    assert runner._send_to_cell([target], "channel", "topic", request, 1, context) == {}
