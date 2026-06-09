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

import time
from unittest.mock import MagicMock, patch

from nvflare.apis.client import Client, ClientPropKey
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegSession, ClientType, InternalFLContextKey
from nvflare.private.fed.server.client_manager import ClientManager
from tests.unit_test.private.fed.server.fake_disabled_client_store import FakeDisabledClientStore


def _make_manager(disabled_cache_ttl=None, disabled_check_fail_open=True):
    manager = ClientManager(
        project_name="project",
        min_num_clients=1,
        max_num_clients=10,
        disabled_cache_ttl=disabled_cache_ttl,
        disabled_check_fail_open=disabled_check_fail_open,
    )
    manager.set_state_store(FakeDisabledClientStore())
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


def test_disabled_check_cache_hit_avoids_second_store_call():
    manager = _make_manager()

    assert manager.is_client_disabled("site-a") is False
    assert manager.is_client_disabled("site-a") is False

    assert manager.state_store.get_calls == ["site-a"]


def test_disabled_check_caches_positive_results():
    manager = _make_manager()
    manager.state_store.disabled["site-a"] = {"client_name": "site-a"}

    assert manager.is_client_disabled("site-a") is True
    assert manager.is_client_disabled("site-a") is True

    assert manager.state_store.get_calls == ["site-a"]


def test_disabled_check_refetches_after_ttl_expiry():
    manager = _make_manager(disabled_cache_ttl=10.0)

    assert manager.is_client_disabled("site-a") is False
    # expire the cached entry
    manager._disabled_cache["site-a"] = (False, time.time() - 11.0)
    # store state changed on another server in the meantime
    manager.state_store.disabled["site-a"] = {"client_name": "site-a"}

    assert manager.is_client_disabled("site-a") is True
    assert manager.state_store.get_calls == ["site-a", "site-a"]


def test_disable_client_updates_cache_immediately():
    manager = _make_manager()

    manager.disable_client("site-a")

    assert manager.is_client_disabled("site-a") is True
    # answered from the cache: no store read happened
    assert manager.state_store.get_calls == []


def test_enable_client_updates_cache_immediately():
    manager = _make_manager()
    manager.disable_client("site-a")

    manager.enable_client("site-a")

    assert manager.is_client_disabled("site-a") is False
    # answered from the cache: no store read happened
    assert manager.state_store.get_calls == []


def test_heartbeat_degrades_open_when_store_read_fails(caplog):
    manager = _make_manager()
    manager.state_store.get_error = RuntimeError("db blipped")
    fl_ctx = MagicMock()

    reactivated = manager.heartbeat("token-a", "site-a", "site-a@server", fl_ctx)

    # no exception, treated as not-disabled: client is re-activated
    assert reactivated is True
    assert manager.clients["token-a"].name == "site-a"
    assert any("db blipped" in r.message or "disabled state" in r.message for r in caplog.records)


def test_disabled_check_falls_back_to_last_cached_value_on_store_error():
    manager = _make_manager(disabled_cache_ttl=10.0)
    manager.state_store.disabled["site-a"] = {"client_name": "site-a"}
    assert manager.is_client_disabled("site-a") is True

    # entry expires, then the store starts failing
    manager._disabled_cache["site-a"] = (True, time.time() - 11.0)
    manager.state_store.get_error = RuntimeError("db down")

    assert manager.is_client_disabled("site-a") is True


def test_fail_closed_treats_uncached_client_as_disabled_on_store_error(caplog):
    manager = _make_manager(disabled_check_fail_open=False)
    manager.state_store.get_error = RuntimeError("db down")

    assert manager.is_client_disabled("site-never-seen") is True
    assert any("fail-closed" in r.message for r in caplog.records)


def test_fail_closed_heartbeat_rejects_uncached_client_on_store_error():
    manager = _make_manager(disabled_check_fail_open=False)
    manager.state_store.get_error = RuntimeError("db down")
    fl_ctx = MagicMock()

    assert manager.heartbeat("token-a", "site-a", "site-a@server", fl_ctx) is False
    assert "token-a" not in manager.clients


def test_fail_closed_still_prefers_cached_value_on_store_error():
    # the with-cached-value fallback is unchanged: an expired not-disabled entry
    # still wins over the fail-closed default
    manager = _make_manager(disabled_cache_ttl=10.0, disabled_check_fail_open=False)
    assert manager.is_client_disabled("site-a") is False  # caches not-disabled

    manager._disabled_cache["site-a"] = (False, time.time() - 11.0)  # expired
    manager.state_store.get_error = RuntimeError("db down")

    assert manager.is_client_disabled("site-a") is False


def test_registration_rechecks_disabled_under_lock():
    """A disable_client that lands between the fast-path check and the locked section
    must still reject the registration (cache recheck under lock, no store I/O)."""
    manager = _make_manager()
    request = _make_request("site-a")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="site-a")

    # simulate the race: the fast-path check saw not-disabled, but the admin disable
    # (store write + cache update under manager.lock) completed before registration
    # acquired the lock
    manager.disable_client("site-a")
    with patch.object(manager, "is_client_disabled", return_value=False):
        client = manager.authenticated_client(request, fl_ctx, ClientType.REGULAR)

    assert client is None
    fl_ctx.set_prop.assert_called_with(FLContextKey.UNAUTHENTICATED, "Client 'site-a' is disabled", sticky=False)


def test_stale_store_read_cannot_clobber_concurrent_disable():
    """Regression for the disable/register cache-poisoning race: a store read that returned
    not-disabled BEFORE an admin disable landed must not overwrite the authoritative cache
    entry (epoch scheme), and the under-lock recheck must keep rejecting the client."""
    manager = _make_manager()
    store = manager.state_store
    orig_get = store.get_disabled_client

    def racy_get(client_name):
        # the stale store read completes (not disabled)...
        result = orig_get(client_name)
        # ...then the admin disable lands (store write + cache True + token sweep, all under
        # manager.lock) before the reader installs its result into the cache
        manager.disable_client(client_name)
        return result

    store.get_disabled_client = racy_get

    # the reader must report the newer authoritative value, not its stale read
    assert manager.is_client_disabled("site-a") is True
    # the authoritative cache entry survived: the under-lock recheck sees disabled
    assert manager._get_cached_disabled("site-a", ignore_ttl=True) is True

    # and a registration attempt right after the race is rejected by the under-lock recheck
    store.get_disabled_client = orig_get
    request = _make_request("site-a")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="site-a")
    with patch.object(manager, "is_client_disabled", return_value=False):  # poisoned fast path
        client = manager.authenticated_client(request, fl_ctx, ClientType.REGULAR)
    assert client is None
    fl_ctx.set_prop.assert_called_with(FLContextKey.UNAUTHENTICATED, "Client 'site-a' is disabled", sticky=False)


def test_stale_store_read_cannot_clobber_concurrent_enable():
    """Mirror race: a store read that returned disabled BEFORE an admin enable landed must not
    overwrite the authoritative not-disabled cache entry."""
    manager = _make_manager()
    manager.disable_client("site-a")
    # expire the cached True so the next check goes to the store
    manager._disabled_cache["site-a"] = (True, time.time() - 1e6)
    store = manager.state_store
    orig_get = store.get_disabled_client

    def racy_get(client_name):
        result = orig_get(client_name)  # stale read: still disabled
        manager.enable_client(client_name)  # admin enable lands before the cache fill
        return result

    store.get_disabled_client = racy_get

    assert manager.is_client_disabled("site-a") is False
    assert manager._get_cached_disabled("site-a", ignore_ttl=True) is False


def test_epoch_does_not_break_normal_cache_fill_after_disable_then_ttl_expiry():
    """An authoritative write followed by TTL expiry must still allow a fresh store read to
    refill the cache (the epoch only blocks fills whose read began before the write)."""
    manager = _make_manager(disabled_cache_ttl=10.0)
    manager.disable_client("site-a")
    # state changed on another server; this server's cache entry expires
    manager.state_store.enable_client("site-a")
    manager._disabled_cache["site-a"] = (True, time.time() - 11.0)

    assert manager.is_client_disabled("site-a") is False
    assert manager._get_cached_disabled("site-a") is False


def test_fail_open_default_is_true_without_env(monkeypatch):
    monkeypatch.delenv("NVFL_DISABLED_CLIENT_FAIL_CLOSED", raising=False)
    manager = ClientManager(project_name="project")
    assert manager.disabled_check_fail_open is True


def test_env_var_makes_default_fail_closed(monkeypatch):
    for value in ("1", "true", "TRUE", "Yes", " yes "):
        monkeypatch.setenv("NVFL_DISABLED_CLIENT_FAIL_CLOSED", value)
        manager = ClientManager(project_name="project")
        assert manager.disabled_check_fail_open is False, f"env value {value!r} should fail closed"


def test_falsy_env_var_keeps_default_fail_open(monkeypatch):
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("NVFL_DISABLED_CLIENT_FAIL_CLOSED", value)
        manager = ClientManager(project_name="project")
        assert manager.disabled_check_fail_open is True, f"env value {value!r} should fail open"


def test_explicit_fail_open_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("NVFL_DISABLED_CLIENT_FAIL_CLOSED", "1")
    manager = ClientManager(project_name="project", disabled_check_fail_open=True)
    assert manager.disabled_check_fail_open is True

    monkeypatch.delenv("NVFL_DISABLED_CLIENT_FAIL_CLOSED")
    manager = ClientManager(project_name="project", disabled_check_fail_open=False)
    assert manager.disabled_check_fail_open is False


def test_set_client_props_sets_site_config():
    site_config = {"format_version": 1, "labels": {"region": "us-east"}}
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.CLIENT_SITE_CONFIG, site_config, private=True, sticky=False)

    client = Client(name="site-1", token="token")
    ClientManager._set_client_props(client, "server.site-1", fl_ctx)

    assert client.get_site_config() == site_config
