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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_opt.confidential_computing import cc_manager as subject
from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer, CCTokenGenerateError
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey


@pytest.fixture
def manager():
    manager = subject.CCManager([], [], cc_enabled_sites=["server", "site"])
    manager.site_name = "server"
    return manager


def test_setup_rejects_invalid_components_and_duplicate_namespaces(manager):
    ctx = MagicMock()
    manager.cc_issuers_conf = [{"issuer_id": "issuer", "token_expiration": 10}]
    ctx.get_engine.return_value.get_component.return_value = object()
    with pytest.raises(RuntimeError, match="must be a CCAuthorizer"):
        manager._setup_cc_authorizers(ctx)
    issuer = MagicMock(spec=CCAuthorizer)
    issuer.get_namespace.return_value = "namespace"
    ctx.get_engine.return_value.get_component.return_value = issuer
    manager.cc_verifier_ids = ["issuer", "issuer"]
    with pytest.raises(RuntimeError, match="already exist"):
        manager._setup_cc_authorizers(ctx)
    assert manager.cc_issuers == {issuer: 10}
    assert manager.cc_verifiers == {"namespace": issuer}


def test_token_refresh_returns_only_successful_fresh_tokens(manager):
    good, empty, failed = [MagicMock(spec=CCAuthorizer) for _ in range(3)]
    good.get_namespace.return_value = "good"
    good.generate.side_effect = ["first", "second"]
    empty.generate.return_value = ""
    failed.generate.side_effect = CCTokenGenerateError("no evidence")
    manager.cc_issuers = {good: 10, empty: 10, failed: 10}
    for expected in ("first", "second"):
        response = manager._handle_token_refresh_request(SimpleNamespace(payload={"requester": "site"}))
        assert response.get_header(MessageHeaderKey.RETURN_CODE) == subject.F3ReturnCode.OK
        assert response.payload == {
            "site_name": "server",
            "cc_info": [{subject.CC_TOKEN: expected, subject.CC_NAMESPACE: "good", subject.CC_TOKEN_VALIDATED: False}],
        }


@pytest.mark.parametrize("reply", [None, "bad-type", {}, {"site_name": "site"}, {"cc_info": ["token"]}])
def test_collect_tokens_fails_closed_for_missing_or_invalid_peer_response(manager, reply):
    ctx = MagicMock()
    manager._generate_fresh_tokens_for_validation = MagicMock(return_value=["own-token"])
    manager._get_all_cc_enabled_sites = MagicMock(return_value=[("server", "server"), ("site-fqcn", "site")])
    response = None if reply is None else subject.make_reply(subject.F3ReturnCode.OK, "", reply)
    ctx.get_engine.return_value.get_cell.return_value.send_request.return_value = response
    with pytest.raises(RuntimeError, match="site-fqcn"):
        manager._collect_all_site_tokens(ctx)
    request = ctx.get_engine.return_value.get_cell.return_value.send_request.call_args.kwargs
    assert request["target"] == "site-fqcn" and request["topic"] == subject.CC_TOPIC_REQUEST_TOKEN
    assert request["timeout"] == 10.0


def test_collect_tokens_combines_local_and_peer_tokens_and_handles_unavailable_transport(manager):
    ctx = MagicMock()
    manager._generate_fresh_tokens_for_validation = MagicMock(return_value=["own-token"])
    manager._get_all_cc_enabled_sites = MagicMock(return_value=[("server", "server"), ("site-fqcn", "site")])
    cell = ctx.get_engine.return_value.get_cell.return_value
    cell.send_request.return_value = subject.make_reply(
        subject.F3ReturnCode.OK, "", {"site_name": "site", "cc_info": ["peer-token"]}
    )
    assert manager._collect_all_site_tokens(ctx) == {"server": ["own-token"], "site": ["peer-token"]}
    cell.send_request.side_effect = RuntimeError("transport")
    with pytest.raises(RuntimeError, match="Failed to collect"):
        manager._collect_all_site_tokens(ctx)
    ctx.get_engine.return_value.get_cell.return_value = None
    assert manager._collect_all_site_tokens(ctx) == {}


@pytest.mark.parametrize(
    "reply,expected",
    [
        (None, []),
        (subject.make_reply(subject.F3ReturnCode.COMM_ERROR, "failed", {}), []),
        (subject.make_reply(subject.F3ReturnCode.OK, "", {"sites": "bad"}), []),
        (subject.make_reply(subject.F3ReturnCode.OK, "", {"sites": [("server", "server")]}), [("server", "server")]),
    ],
)
def test_client_site_discovery_validates_response_shape(manager, reply, expected):
    ctx = MagicMock()
    cell = ctx.get_engine.return_value.get_cell.return_value
    cell.send_request.return_value = reply
    assert manager._request_sites_from_server(ctx) == expected
    assert cell.send_request.call_args.kwargs["topic"] == subject.CC_TOPIC_GET_SITES
    cell.send_request.side_effect = RuntimeError("offline")
    assert manager._request_sites_from_server(ctx) == []
    ctx.get_engine.return_value.get_cell.return_value = None
    assert manager._request_sites_from_server(ctx) == []


def test_server_site_discovery_filters_sites_and_registers_only_server_handlers(manager):
    engine = MagicMock(spec=ServerEngineSpec)
    engine.get_cell = MagicMock()
    engine.get_clients.return_value = [
        SimpleNamespace(name="site", get_fqcn=lambda: "site-fqcn"),
        SimpleNamespace(name="untrusted"),
    ]
    manager.engine = engine
    assert manager._get_all_sites() == [(subject.FQCN.ROOT_SERVER, "server"), ("site-fqcn", "site")]
    ctx = MagicMock()
    ctx.get_engine.return_value = engine
    assert manager._get_all_cc_enabled_sites(ctx) == manager._get_all_sites()
    manager._register_cc_handlers(ctx)
    calls = engine.get_cell.return_value.register_request_cb.call_args_list
    assert {call.kwargs["topic"] for call in calls} == {subject.CC_TOPIC_REQUEST_TOKEN, subject.CC_TOPIC_GET_SITES}
    response = manager._handle_get_sites_request(SimpleNamespace(payload=None))
    assert response.payload["sites"] == manager._get_all_sites()


def test_client_discovery_rejects_missing_site_list_and_registers_token_handler(manager):
    ctx = MagicMock()
    manager._request_sites_from_server = MagicMock(return_value=[])
    with pytest.raises(RuntimeError, match="Failed to get sites"):
        manager._get_all_cc_enabled_sites(ctx)
    manager._request_sites_from_server.return_value = [("site", "site")]
    assert manager._get_all_cc_enabled_sites(ctx) == [("site", "site")]
    manager._register_cc_handlers(ctx)
    assert (
        ctx.get_engine.return_value.get_cell.return_value.register_request_cb.call_args.kwargs["topic"]
        == subject.CC_TOPIC_REQUEST_TOKEN
    )
    ctx.get_engine.return_value.get_cell.return_value = None
    manager._register_cc_handlers(ctx)


@pytest.mark.parametrize(
    "tokens,error,expected", [({}, "", False), ({"site": []}, "invalid", False), ({"site": []}, "", True)]
)
def test_cross_site_validation_shuts_down_only_on_failed_validation(manager, tokens, error, expected):
    ctx = MagicMock()
    manager._collect_all_site_tokens = MagicMock(return_value=tokens)
    manager._validate_participants_tokens = MagicMock(return_value=error)
    manager._shutdown_system = MagicMock()
    assert manager._perform_cross_site_validation(ctx) is expected
    assert manager._shutdown_system.call_count == (0 if expected else 1)
    manager._collect_all_site_tokens.side_effect = RuntimeError("unexpected")
    assert manager._perform_cross_site_validation(ctx) is False
    assert "unexpected" in manager._shutdown_system.call_args.args[0]


def test_event_policy_blocks_byoc_and_validates_resources_once(manager):
    ctx = MagicMock()
    manager._perform_cross_site_validation = MagicMock(return_value=True)
    manager._shutdown_system = MagicMock()
    manager.handle_event(EventType.BEFORE_CHECK_CLIENT_RESOURCES, ctx)
    manager.handle_event(EventType.BEFORE_CHECK_CLIENT_RESOURCES, ctx)
    manager._perform_cross_site_validation.assert_called_once_with(ctx)
    ctx.get_prop.return_value = {"site": (False, subject.CC_VERIFY_ERROR + ": invalid")}
    manager.handle_event(EventType.AFTER_CHECK_CLIENT_RESOURCES, ctx)
    manager._shutdown_system.assert_called_once()
    ctx.get_prop.return_value = {subject.AppValidationKey.BYOC: True}
    manager.handle_event(EventType.SUBMIT_JOB, ctx)
    assert ctx.set_prop.call_args.kwargs["key"] == FLContextKey.JOB_BLOCK_REASON
    assert ctx.set_prop.call_args.kwargs["value"] == "BYOC job not allowed for CC"


def test_periodic_loop_stops_after_one_validation_without_running_real_thread(manager, monkeypatch):
    current = object()
    manager.cross_validation_thread = current
    stop = MagicMock()
    stop.wait.side_effect = [False, True]
    stop.is_set.return_value = False
    manager.cross_validation_stop_event = stop
    manager._perform_cross_site_validation = MagicMock(return_value=True)
    monkeypatch.setattr(subject, "threading", SimpleNamespace(current_thread=lambda: current))
    manager._cross_site_validation_loop(MagicMock())
    manager._perform_cross_site_validation.assert_called_once()
    assert manager.cross_validation_run_once is True
    assert stop.wait.call_args.kwargs == {"timeout": 600}


def test_start_stop_owns_thread_and_avoids_duplicate_start(manager, monkeypatch):
    thread = MagicMock()
    thread.is_alive.return_value = True
    factory = MagicMock(return_value=thread)
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=factory))
    ctx = MagicMock()
    manager._start_cross_site_validation(ctx)
    thread.start.assert_called_once_with()
    manager._start_cross_site_validation(ctx)
    assert factory.call_count == 1
    manager._stop_cross_site_validation()
    thread.join.assert_called_once_with(timeout=5.0)
    assert manager.cross_validation_stop_event.is_set()
    assert manager.cross_validation_thread is None
