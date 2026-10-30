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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey, ServerCommandKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.private.fed.client import communicator as subject
from tests.timing_utils import isolate_time


@pytest.fixture
def communicator(monkeypatch):
    monkeypatch.setattr(subject, "set_scope_property", Mock())
    monkeypatch.setattr(subject, "set_add_auth_headers_filters", Mock())
    monkeypatch.setattr(subject, "get_scope_property", Mock(return_value=None))
    return subject.Communicator({"client_name": "site-a"}, cell=Mock())


def test_auth_installation_updates_session_headers_and_shared_auth_properties(communicator):
    cell = communicator.cell
    communicator.set_auth("site-a", "token", "signature", "session")
    assert (communicator.client_name, communicator.token, communicator.token_signature, communicator.ssid) == (
        "site-a",
        "token",
        "signature",
        "session",
    )
    subject.set_add_auth_headers_filters.assert_called_once_with(cell, "site-a", "token", "signature", "session")
    subject.set_scope_property.assert_any_call(scope_name="site-a", key=FLMetaKey.AUTH_TOKEN, value="token")
    subject.set_scope_property.assert_any_call(
        scope_name="site-a", key=FLMetaKey.AUTH_TOKEN_SIGNATURE, value="signature"
    )
    next_cell = Mock()
    communicator.set_cell(next_cell)
    subject.set_add_auth_headers_filters.assert_called_with(next_cell, "site-a", "token", "signature", "session")
    assert next_cell.register_request_cb.call_count == 2


@pytest.mark.parametrize("failure", ["no-cell", "unknown-host", "missing-config", "no-token"])
def test_registration_rejects_unavailable_transport_and_incomplete_secure_configuration(
    communicator, monkeypatch, failure
):
    ctx = FLContext()
    if failure == "no-cell":
        communicator.cell = None
        clock = isolate_time(monkeypatch, subject)
        monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 16]))
    elif failure in ("unknown-host", "missing-config"):
        ctx.set_prop(FLContextKey.SECURE_MODE, True)
        if failure == "missing-config":
            ctx.set_prop(FLContextKey.SERVER_CONFIG, [{"identity": "server"}])
    else:
        auth = Mock()
        auth.authenticate.return_value = (None, None, None, None)
        monkeypatch.setattr(subject, "Authenticator", Mock(return_value=auth))
    with pytest.raises((RuntimeError, FLCommunicationError)):
        communicator.client_registration("site-a", "project", ctx)
    subject.set_add_auth_headers_filters.assert_not_called()


def test_registration_copies_server_capability_info_and_binds_new_session(communicator, monkeypatch):
    ctx = FLContext()
    auth = Mock()
    verifier = Mock()

    def authenticate(peer, signal):
        peer.set_prop("_cc_info", {"policy": "enabled"}, private=False)
        return "token", "signature", "session", verifier

    auth.authenticate.side_effect = authenticate
    factory = Mock(return_value=auth)
    monkeypatch.setattr(subject, "Authenticator", factory)
    assert communicator.client_registration("site-a", "project", ctx, timeout=8) == ("token", "signature", "session")
    assert ctx.get_prop("_cc_info") == {"policy": "enabled"}
    assert communicator.token_verifier is verifier
    assert communicator._authenticator is auth
    assert factory.call_args.kwargs["timeout"] == 8
    assert factory.call_args.kwargs["secure_mode"] is False


@pytest.mark.parametrize("failure", ["missing-verifier", "reregister", "valid"])
def test_reconnected_server_session_requires_trusted_challenge_and_valid_token(communicator, failure):
    verifier = Mock()
    verifier.verify.return_value = failure == "valid"
    communicator._authenticator = Mock()
    communicator._authenticator.challenge_server.return_value = (
        "challenge",
        None if failure == "missing-verifier" else verifier,
    )
    communicator.token, communicator.token_signature = "old-token", "old-signature"
    registration = Mock(return_value=("new-token", "new-signature", "new-session"))
    communicator.client_registration = registration
    ctx = FLContext()
    client = Mock()
    ctx.set_prop(SystemComponents.FED_CLIENT, client)
    if failure == "missing-verifier":
        with pytest.raises(FLCommunicationError, match="Cannot authenticate"):
            communicator._check_server_session("site-a", "project", ctx)
    else:
        communicator._check_server_session("site-a", "project", ctx)
    assert registration.call_count == int(failure == "reregister")
    if failure == "reregister":
        assert (client.token, client.token_signature, client.ssid) == ("new-token", "new-signature", "new-session")
        assert ctx.get_prop(FLContextKey.CLIENT_TOKEN) == "new-token"


@pytest.mark.parametrize("outcome", ["transport", "authentication", "success", "invalid-minimum"])
def test_task_pull_handles_failures_and_adopts_only_numeric_server_timeout(communicator, monkeypatch, outcome):
    ctx = FLContext()
    ctx.set_prop(ReservedKey.RUN_NUM, "job")
    communicator.engine = Mock()
    monkeypatch.setattr(subject, "determine_parent_fqcn", Mock(return_value="server"))
    data = Shareable()
    data.set_header(ServerCommandKey.TASK_NAME, "train")
    data.set_header(FLContextKey.TASK_ID, "task")
    data.set_header(ServerCommandKey.MIN_GET_TASK_TIMEOUT, "bad" if outcome == "invalid-minimum" else "30")
    reply = new_cell_message(
        {
            MessageHeaderKey.RETURN_CODE: (
                ReturnCode.AUTHENTICATION_ERROR if outcome == "authentication" else ReturnCode.OK
            ),
            MessageHeaderKey.PAYLOAD_LEN: 100,
        },
        data,
    )
    if outcome == "transport":
        communicator.cell.send_request.side_effect = OSError("offline")
    else:
        communicator.cell.send_request.return_value = reply
    result = communicator.pull_task("project", "token", "session", ctx, timeout=1)
    assert result is (None if outcome in ("transport", "authentication") else reply)
    assert communicator.cell.send_request.call_args.kwargs["timeout"] == 5
    if outcome in ("success", "invalid-minimum"):
        assert communicator.pending_task is data
        assert communicator.last_task_id == "task"
        assert communicator.timeout == (30 if outcome == "success" else 5)
        assert ctx.get_prop(FLContextKey.SSID) == "session"


@pytest.mark.parametrize("failure", [False, True])
def test_remote_quit_triggers_abort_and_translates_transport_failure(communicator, failure):
    if failure:
        communicator.cell.send_request.side_effect = OSError("closed")
    else:
        communicator.cell.send_request.return_value = new_cell_message(
            {MessageHeaderKey.RETURN_CODE: ReturnCode.UNAUTHENTICATED, CellMessageHeaderKeys.MESSAGE: "removed"}, None
        )
    if failure:
        with pytest.raises(FLCommunicationError, match="client_quit"):
            communicator.quit_remote({}, "project", "token", "session", FLContext())
    else:
        assert communicator.quit_remote({}, "project", "token", "session", FLContext()) == "removed"
    assert communicator.abort_signal.triggered
    assert communicator.cell.send_request.call_args.kwargs["timeout"] == communicator.maint_msg_timeout


@pytest.mark.parametrize("outcome", ["success", "transport", "simulate-error"])
def test_heartbeat_reports_jobs_cleans_aborts_and_handles_one_failed_attempt(communicator, monkeypatch, outcome):
    ctx = FLContext()
    if outcome == "simulate-error":
        ctx.set_prop(FLContextKey.SIMULATE_MODE, True)
    engine = Mock()
    engine.new_context.return_value = ctx
    engine.get_all_job_ids.return_value = ["job-a"]

    def send(**kwargs):
        communicator.heartbeat_done = True
        if outcome == "transport":
            raise OSError("offline")
        return new_cell_message(
            {
                MessageHeaderKey.RETURN_CODE: (
                    ReturnCode.INVALID_REQUEST if outcome == "simulate-error" else ReturnCode.OK
                ),
                CellMessageHeaderKeys.ABORT_JOBS: ["job-a"],
            },
            None,
        )

    communicator.cell.send_request.side_effect = send
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None]))
    communicator.send_heartbeat({}, "project", "token", "session", "site-a", engine, 2)
    request = communicator.cell.send_request.call_args.kwargs["request"]
    assert request.get_header(CellMessageHeaderKeys.JOB_IDS) == ["job-a"]
    assert engine.fire_event.call_args_list[0].args == (EventType.BEFORE_CLIENT_HEARTBEAT, ctx)
    assert engine.abort_app.call_count == int(outcome == "success")
    if outcome == "success":
        engine.abort_app.assert_called_once_with("job-a", heartbeat_cleanup=True)


def test_run_cleanup_contains_abort_failure(communicator):
    engine = Mock()
    engine.abort_app.side_effect = ValueError("already stopped")
    communicator._clean_up_runs(engine, ["job-a"])
    engine.abort_app.assert_called_once_with("job-a", heartbeat_cleanup=True)
