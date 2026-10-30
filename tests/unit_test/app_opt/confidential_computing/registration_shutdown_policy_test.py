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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_opt.confidential_computing import cc_manager as module
from nvflare.app_opt.confidential_computing.cc_authorizer import CCTokenVerifyError


def _manager():
    subject = module.CCManager([], [], cc_enabled_sites=["server", "site"])
    subject.site_name = "server"
    return subject


def test_bootstrap_failure_requests_shutdown_and_admin_registration_bypasses_attestation():
    subject = _manager()
    context = FLContext()
    subject._setup_cc_authorizers = MagicMock(side_effect=ValueError("missing verifier"))
    subject._shutdown_system = MagicMock()
    subject.handle_event(EventType.SYSTEM_BOOTSTRAP, context)
    assert "attestation preparation" in subject._shutdown_system.call_args.args[0]
    context.set_prop(FLContextKey.CLIENT_TYPE, module.ClientType.ADMIN)
    subject._validate_client_tokens = MagicMock()
    subject._generate_and_attach_tokens = MagicMock()
    subject.handle_event(EventType.CLIENT_REGISTER_RECEIVED, context)
    subject._validate_client_tokens.assert_not_called()
    subject._generate_and_attach_tokens.assert_not_called()
    with pytest.raises(ValueError, match="verify_frequency"):
        module.CCManager([], [], verify_frequency="bad")


@pytest.mark.parametrize("side", ["server", "client"])
@pytest.mark.parametrize("present", [True, False])
def test_registration_requires_peer_tokens_and_delegates_validation(side, present):
    subject = _manager()
    context = FLContext()
    info = {"peer": [{"token": "signed"}]}
    if side == "server":
        context.set_prop(module.CC_INFO, info if present else None)
    else:
        peer = FLContext()
        peer.set_prop(module.CC_INFO, info if present else None)
        context.get_peer_context = lambda: peer
    subject._shutdown_system = MagicMock()
    subject._validate_cc_infos = MagicMock()
    (subject._validate_server_tokens if side == "server" else subject._validate_client_tokens)(context)
    if present:
        subject._validate_cc_infos.assert_called_once_with(info, context)
        subject._shutdown_system.assert_not_called()
    else:
        assert subject._shutdown_system.call_count == 1
        subject._validate_cc_infos.assert_not_called()


def test_missing_peer_context_and_verifier_failure_reject_registration():
    subject = _manager()
    subject._shutdown_system = MagicMock()
    context = FLContext()
    subject._validate_client_tokens(context)
    assert subject._shutdown_system.call_args.args[0] == "No peer context!"
    verifier = MagicMock()
    verifier.verify.side_effect = CCTokenVerifyError("invalid evidence")
    subject.cc_verifiers = {"gpu": verifier}
    _, invalid = subject._verify_participants_tokens(
        {"site": [{module.CC_TOKEN: "bad", module.CC_NAMESPACE: "gpu"}], "server": []}
    )
    assert len(invalid) == 2
    assert any("site" in item for item in invalid)
    assert any("server" in item for item in invalid)
    subject.engine = object()
    with pytest.raises(RuntimeError, match="only be called by server"):
        subject._get_all_sites()


@pytest.mark.parametrize("stop_initially", [True, False])
def test_periodic_validation_stops_during_jitter_or_skips_busy_cycle_without_spawning_thread(
    monkeypatch, stop_initially
):
    subject = _manager()
    subject.cross_validation_thread = threading.current_thread()
    stop = MagicMock()
    stop.wait.side_effect = [True] if stop_initially else [False, True]
    stop.is_set.return_value = False
    subject.cross_validation_stop_event = stop
    subject.lock = MagicMock()
    subject.lock.acquire.return_value = False
    subject._perform_cross_site_validation = MagicMock()
    monkeypatch.setattr(module, "random", SimpleNamespace(uniform=lambda start, end: 0.5))
    subject._cross_site_validation_loop(FLContext())
    assert stop.wait.call_count == (1 if stop_initially else 2)
    subject._perform_cross_site_validation.assert_not_called()
    subject.lock.release.assert_not_called()
    if not stop_initially:
        subject.lock.acquire.assert_called_once_with(blocking=False)


@pytest.mark.parametrize("fail", [True, False])
def test_server_shutdown_stops_jobs_before_sending_authenticated_shutdown_command(monkeypatch, fail):
    subject = _manager()
    context = FLContext()
    engine = MagicMock(spec=ServerEngineSpec)
    engine.run_processes = {"job-1": object(), "job-2": object()}
    admin = object()
    engine.server = SimpleNamespace(admin_server=admin)
    events = []
    engine.job_runner = SimpleNamespace(stop_run=lambda job_id, ctx: events.append(("stop", job_id, ctx)))
    context.get_engine = lambda: engine
    connection = object()
    factory = MagicMock(return_value=connection)
    monkeypatch.setattr(module, "Connection", factory)
    command = MagicMock()
    command.validate_command_targets.side_effect = lambda conn, args: events.append(("validate", args))

    def shutdown(conn, args):
        events.append(("shutdown", args))
        if fail:
            raise ValueError("shutdown unavailable")

    command.shutdown.side_effect = shutdown
    monkeypatch.setattr(module, "TrainingCommandModule", lambda: command)
    if fail:
        with pytest.raises(SystemExit) as caught:
            subject._shutdown_system("failed verification", context)
        assert caught.value.code == 1
    else:
        subject._shutdown_system("failed verification", context)
    factory.assert_called_once_with(app_ctx=engine, props={module.ConnProps.ADMIN_SERVER: admin})
    assert events == [
        ("stop", "job-1", context),
        ("stop", "job-2", context),
        ("validate", ["all"]),
        ("shutdown", ["shutdown", "all"]),
    ]
    command.validate_command_targets.assert_called_once_with(connection, ["all"])
    command.shutdown.assert_called_once_with(connection, ["shutdown", "all"])


def test_client_shutdown_uses_exit_boundary_without_terminating_test_process(monkeypatch):
    subject = _manager()
    context = FLContext()
    context.get_engine = lambda: object()
    exit_boundary = MagicMock()
    monkeypatch.setattr(module, "os", SimpleNamespace(_exit=exit_boundary))
    subject._shutdown_system("failed verification", context)
    exit_boundary.assert_called_once_with(1)
