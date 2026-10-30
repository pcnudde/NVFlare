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

from nvflare.apis.fl_constant import FLContextKey, SecureTrainConst
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.private.admin_defs import Message
from nvflare.private.fed.server import cred_keeper, message_send


@pytest.mark.parametrize(
    "job_id,expected_target,expected_channel,optional",
    [
        (None, "root.site-a", CellChannel.CLIENT_MAIN, False),
        ("job-a", "root.site-a.job-a", CellChannel.CLIENT_COMMAND, True),
    ],
)
def test_admin_request_routes_to_parent_or_job_cell_and_maps_reply(job_id, expected_target, expected_channel, optional):
    client = SimpleNamespace(name="site-a", token="token", fqcn="root.site-a")
    request = Message("command", "request")
    reply = Message("command", "response")
    cell = Mock()
    cell.broadcast_multi_requests.return_value = {expected_target: new_cell_message({}, reply)}
    result = message_send.send_requests(
        cell, "control", {"token": request, "unknown": request}, {"token": client}, job_id=job_id, timeout_secs=5
    )
    targets, timeout = cell.broadcast_multi_requests.call_args.args
    assert timeout == 5
    assert cell.broadcast_multi_requests.call_args.kwargs == {"optional": optional}
    routed = targets[expected_target]
    assert routed.target == expected_target
    assert routed.channel == expected_channel
    assert routed.topic == "control"
    assert routed.message.payload is request
    assert len(result) == 1
    assert result[0].client_token == "token"
    assert result[0].client_name == "site-a"
    assert result[0].request is request
    assert result[0].reply is reply


def test_admin_fire_and_forget_uses_no_response_wait():
    client = SimpleNamespace(name="site-a", token="token", fqcn="site-a")
    cell = Mock()
    request = Message("command", "")
    assert (
        message_send.send_requests(
            cell, "command", {"token": request}, {"token": client}, timeout_secs=0, optional=True
        )
        == []
    )
    cell.fire_multi_requests_and_forget.assert_called_once()
    assert cell.fire_multi_requests_and_forget.call_args.kwargs == {"optional": True}
    cell.broadcast_multi_requests.assert_not_called()


@pytest.mark.parametrize("requests,clients", [({}, {}), ({"missing": Message("command", "")}, {})])
def test_unknown_or_empty_admin_targets_do_not_touch_transport(requests, clients):
    cell = Mock()
    assert message_send.send_requests(cell, "command", requests, clients) == []
    assert not cell.mock_calls


def test_admin_requests_require_mapping():
    with pytest.raises(TypeError, match="requests must be a dict"):
        message_send.send_requests(Mock(), "command", [], {})


@pytest.mark.parametrize("config", [None, "wrong", ["wrong"], [{}]])
def test_credential_cache_rejects_missing_or_malformed_server_config(config):
    keeper = cred_keeper.CredKeeper()
    ctx = FLContext()
    if config is not None:
        ctx.set_prop(FLContextKey.SERVER_CONFIG, config)
    assert keeper.get_id_verifier(ctx) is None
    assert keeper.get_id_asserter(ctx) is None
    assert keeper.id_verifier is keeper.id_asserter is None


def test_credential_cache_initializes_each_identity_helper_once(monkeypatch):
    keeper = cred_keeper.CredKeeper()
    ctx = FLContext()
    ctx.set_prop(
        FLContextKey.SERVER_CONFIG,
        [
            {
                SecureTrainConst.SSL_ROOT_CERT: "root.pem",
                SecureTrainConst.SSL_CERT: "server.pem",
                SecureTrainConst.PRIVATE_KEY: "server.key",
            }
        ],
    )
    verifier = Mock()
    asserter = Mock()
    monkeypatch.setattr(cred_keeper, "IdentityVerifier", verifier)
    monkeypatch.setattr(cred_keeper, "IdentityAsserter", asserter)
    assert keeper.get_id_verifier(ctx) is verifier.return_value
    assert keeper.get_id_verifier(FLContext()) is verifier.return_value
    assert keeper.get_id_asserter(ctx) is asserter.return_value
    assert keeper.get_id_asserter(FLContext()) is asserter.return_value
    verifier.assert_called_once_with(root_cert_file="root.pem")
    asserter.assert_called_once_with(private_key_file="server.key", cert_file="server.pem")
