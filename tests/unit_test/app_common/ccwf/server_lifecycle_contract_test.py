# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf import server_ctl as subject
from nvflare.app_common.ccwf.common import Constant, StatusReport, topic_for_end_workflow
from nvflare.fuel.utils.validation_utils import DefaultValuePolicy
from tests.timing_utils import isolate_time


def runtime(start=True, minimum=0):
    ctl = subject.ServerSideController(
        participating_clients=["site-1", "site-2"],
        starting_client="site-1" if start else "",
        starting_client_policy=DefaultValuePolicy.EMPTY,
        min_clients=minimum,
    )
    engine = Mock()
    engine.get_clients.return_value = [SimpleNamespace(name="site-1"), SimpleNamespace(name="site-2")]
    ctl._engine = engine
    ctl.system_panic = Mock()
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctx.set_prop(FLContextKey.WORKFLOW, "wf-1", private=True, sticky=True)
    ctl.start_controller(ctx)
    return ctl, ctx, engine


@pytest.mark.parametrize(
    "configured,minimum,start,start_ok,ended,panic",
    [
        (2, 0, True, True, 2, False),
        (1, 1, True, True, 1, False),
        (0, 0, True, True, 0, True),
        (2, 0, True, False, 2, True),
        (2, 0, True, True, 0, True),
        (2, 0, False, True, 2, False),
    ],
)
def test_workflow_configuration_start_and_end_require_expected_quorum(
    monkeypatch, configured, minimum, start, start_ok, ended, panic
):
    ctl, ctx, engine = runtime(start, minimum)
    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock()
    ctl.prepare_config = Mock(return_value={"learning_rate": 0.2})
    ctl.sub_flow = Mock()
    ctl._check_job_status = Mock(return_value=True)
    signal = Signal()

    def configure(**kwargs):
        task = kwargs["task"]
        config = task.data[Constant.CONFIG]
        assert config[AppConstants.NUM_ROUNDS] == 1
        assert config[FLContextKey.WORKFLOW] == "wf-1"
        assert config[Constant.CLIENTS] == ["site-1", "site-2"]
        assert config["learning_rate"] == 0.2
        assert kwargs["min_responses"] == (minimum or 2)
        for index, name in enumerate(kwargs["targets"]):
            task.result_received_cb(
                SimpleNamespace(
                    client=SimpleNamespace(name=name),
                    result=make_reply(ReturnCode.OK if index < configured else ReturnCode.ERROR),
                ),
                ctx,
            )

    def begin(**kwargs):
        task = kwargs["task"]
        assert kwargs["targets"] == ["site-1"]
        task.result_received_cb(
            SimpleNamespace(
                client=SimpleNamespace(name="site-1"),
                result=make_reply(ReturnCode.OK if start_ok else ReturnCode.ERROR),
            ),
            ctx,
        )

    ctl.broadcast_and_wait = Mock(side_effect=configure)
    ctl.send_and_wait = Mock(side_effect=begin)
    engine.send_aux_request.return_value = {name: make_reply(ReturnCode.OK) for name in ["site-1", "site-2"][:ended]}
    ctl.control_flow(signal, ctx)
    assert ctl.system_panic.called is panic
    if configured >= (minimum or 2) and (not start or start_ok):
        ctl.sub_flow.assert_called_once_with(signal, ctx)
        call = engine.send_aux_request.call_args.kwargs
        assert call["targets"] == ["site-1", "site-2"]
        assert call["topic"] == topic_for_end_workflow("wf-1")
        assert call["secure"] is False
    else:
        ctl.sub_flow.assert_not_called()
        engine.send_aux_request.assert_not_called()


@pytest.mark.parametrize("stage", ["config", "start"])
@pytest.mark.parametrize("outcome", ["ok", "rejected", "exception", "error"])
def test_client_callbacks_record_readiness_only_after_hook_accepts(stage, outcome):
    ctl, ctx, _ = runtime()
    hook = Mock(
        return_value=outcome != "rejected", side_effect=RuntimeError("hook failed") if outcome == "exception" else None
    )
    if stage == "config":
        ctl.process_config_reply = hook
        callback = ctl._process_configure_reply
    else:
        ctl.client_started = hook
        callback = ctl._process_start_reply
    task = SimpleNamespace(
        client=SimpleNamespace(name="site-1"),
        result=make_reply(ReturnCode.ERROR if outcome == "error" else ReturnCode.OK),
    )
    callback(task, ctx)
    recorded = bool(ctl.client_statuses["site-1"].ready_time) if stage == "config" else ctl.cw_started
    assert recorded is (outcome == "ok")
    if outcome == "error":
        hook.assert_not_called()


def test_status_reports_track_heartbeat_without_fabricating_progress_and_abort_on_error(monkeypatch):
    ctl, ctx, _ = runtime()
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=True, sticky=True)
    ctx.set_peer_context(peer)
    ctl.handle_event(EventType.BEFORE_PROCESS_TASK_REQUEST, ctx)
    assert ctl.client_statuses["site-1"].num_reports == 0
    report = StatusReport(timestamp=10, action="training", last_round=2)
    peer.set_prop(Constant.STATUS_REPORTS, {"other-wf": report.to_dict()}, private=False, sticky=False)
    ctl._update_client_status(ctx)
    assert ctl.client_statuses["site-1"].num_reports == 0
    peer.set_prop(Constant.STATUS_REPORTS, {"wf-1": report.to_dict()}, private=False, sticky=False)
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(side_effect=[100, 101, 102])
    ctl.handle_event(EventType.BEFORE_PROCESS_TASK_REQUEST, ctx)
    status = ctl.client_statuses["site-1"]
    assert status.status.last_round == 2 and status.last_progress_time == 100
    ctl._update_client_status(ctx)
    assert status.last_report_time == 101 and status.last_progress_time == 100
    report.error = ReturnCode.EXECUTION_EXCEPTION
    peer.set_prop(Constant.STATUS_REPORTS, {"wf-1": report.to_dict()}, private=False, sticky=False)
    ctl._update_client_status(ctx)
    assert ctl.asked_to_stop and status.num_reports == 3
    ctl.system_panic.assert_called_once()


def test_inactive_client_reports_do_not_create_status_entries():
    ctl, ctx, _ = runtime()
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "unknown", private=True, sticky=True)
    peer.set_prop(
        Constant.STATUS_REPORTS, {"wf-1": StatusReport(action="train").to_dict()}, private=False, sticky=False
    )
    ctx.set_peer_context(peer)
    ctl._update_client_status(ctx)
    assert set(ctl.client_statuses) == {"site-1", "site-2"}
