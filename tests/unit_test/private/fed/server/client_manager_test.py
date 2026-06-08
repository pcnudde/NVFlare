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

from unittest.mock import MagicMock, patch

from nvflare.apis.client import Client, ClientPropKey
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegSession, ClientType, InternalFLContextKey
from nvflare.private.fed.server.client_manager import ClientManager


class _FakeDisabledClientStore:
    def __init__(self):
        self.disabled = {}
        self.disable_error = None
        self.enable_error = None

    def get_disabled_client(self, client_name):
        return self.disabled.get(client_name)

    def disable_client(self, client_name, disabled_by=None, reason=None):
        if self.disable_error:
            raise self.disable_error
        row = {"client_name": client_name, "disabled_by": disabled_by, "reason": reason}
        self.disabled[client_name] = row
        return row

    def enable_client(self, client_name):
        if self.enable_error:
            raise self.enable_error
        return self.disabled.pop(client_name, None) is not None


def _make_manager():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    manager.set_state_store(_FakeDisabledClientStore())
    return manager


def _make_request(client_name: str) -> MagicMock:
    shareable = Shareable()
    shareable[IdentityChallengeKey.CERT] = b"fake-cert"
    shareable[IdentityChallengeKey.SIGNATURE] = b"fake-signature"

    request = MagicMock()
    request.payload = shareable
    headers = {
        CellMessageHeaderKeys.CLIENT_NAME: client_name,
        MessageHeaderKey.ORIGIN: f"{client_name}@site",
    }
    request.get_header.side_effect = lambda key: headers.get(key)
    return request


def _make_fl_ctx(secure_mode: bool, client_name: str) -> MagicMock:
    reg = ClientRegSession(client_name)
    fl_ctx = MagicMock()

    def _get_prop(key, default=None):
        if key == FLContextKey.SECURE_MODE:
            return secure_mode
        if key == InternalFLContextKey.CLIENT_REG_SESSION:
            return reg
        return default

    fl_ctx.get_prop.side_effect = _get_prop
    return fl_ctx


def test_authenticated_client_stores_org_extracted_from_cert():
    manager = _make_manager()
    request = _make_request("site-a")
    fl_ctx = _make_fl_ctx(secure_mode=True, client_name="site-a")
    verifier = MagicMock()

    with (
        patch.object(manager, "_get_id_verifier", return_value=verifier),
        patch("nvflare.private.fed.server.client_manager.load_crt_bytes", return_value=object()),
        patch("nvflare.private.fed.server.client_manager.get_org_from_cert", return_value="org_a"),
        patch.object(manager, "_set_client_props"),
    ):
        client = manager.authenticated_client(request, fl_ctx, ClientType.REGULAR)

    assert client is not None
    assert client.get_prop(ClientPropKey.ORG) == "org_a"
    verifier.verify_common_name.assert_called_once()


def test_authenticated_client_sets_empty_org_when_secure_mode_is_disabled():
    manager = _make_manager()
    request = _make_request("site-a")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="site-a")

    with patch.object(manager, "_set_client_props"):
        client = manager.authenticated_client(request, fl_ctx, ClientType.REGULAR)

    assert client is not None
    assert client.get_prop(ClientPropKey.ORG, "") == ""


def test_disable_client_persists_to_state_store_and_removes_active_client():
    manager = _make_manager()
    client = Client("site-a", "token-a")
    manager.clients[client.token] = client
    manager.name_to_clients[client.name] = client

    removed_tokens = manager.disable_client("site-a")

    assert removed_tokens == ["token-a"]
    assert "token-a" not in manager.clients
    assert "site-a" not in manager.name_to_clients
    assert manager.is_client_disabled("site-a")
    assert manager.state_store.get_disabled_client("site-a")["client_name"] == "site-a"


def test_disabled_client_checks_require_state_store():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)

    try:
        manager.is_client_disabled("site-a")
    except AssertionError as e:
        assert "state_store" in str(e)
    else:
        raise AssertionError("expected AssertionError")


def test_disable_client_keeps_active_client_when_store_fails():
    manager = _make_manager()
    client = Client("site-a", "token-a")
    manager.clients[client.token] = client
    manager.name_to_clients[client.name] = client
    manager.state_store.disable_error = RuntimeError("db write failed")

    try:
        manager.disable_client("site-a")
    except RuntimeError as e:
        assert str(e) == "db write failed"
    else:
        raise AssertionError("expected RuntimeError")

    assert not manager.is_client_disabled("site-a")
    assert manager.clients["token-a"] is client
    assert manager.name_to_clients["site-a"] is client


def test_remove_client_unknown_token_returns_none():
    manager = _make_manager()

    assert manager.remove_client("unknown-token") is None


def test_enable_client_persists_to_state_store_and_allows_client():
    manager = _make_manager()
    manager.disable_client("site-a")

    assert manager.enable_client("site-a") is True

    assert not manager.is_client_disabled("site-a")


def test_disabled_client_registration_is_rejected():
    manager = _make_manager()
    manager.disable_client("site-a")
    request = _make_request("site-a")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="site-a")

    client = manager.authenticated_client(request, fl_ctx, ClientType.REGULAR)

    assert client is None
    fl_ctx.set_prop.assert_called_with(FLContextKey.UNAUTHENTICATED, "Client 'site-a' is disabled", sticky=False)


def test_disabled_client_heartbeat_does_not_reactivate():
    manager = _make_manager()
    manager.disable_client("site-a")
    fl_ctx = MagicMock()

    reactivated = manager.heartbeat("token-a", "site-a", "site-a@server", fl_ctx)

    assert reactivated is False
    assert "token-a" not in manager.clients
    fl_ctx.set_prop.assert_called_with(FLContextKey.UNAUTHENTICATED, "Client 'site-a' is disabled", sticky=False)


def test_set_client_props_sets_site_config():
    site_config = {"format_version": 1, "labels": {"region": "us-east"}}
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.CLIENT_SITE_CONFIG, site_config, private=True, sticky=False)

    client = Client(name="site-1", token="token")
    ClientManager._set_client_props(client, "server.site-1", fl_ctx)

    assert client.get_site_config() == site_config
