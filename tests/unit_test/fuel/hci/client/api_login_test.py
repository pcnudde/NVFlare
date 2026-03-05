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

from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.hci.client.api import AdminAPI, APIStatus, ResultKey
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.proto import InternalCommands


class _FakeIdentityAsserter:
    cert_data = "fake-cert"

    def __init__(self, private_key_file: str, cert_file: str):
        self.private_key_file = private_key_file
        self.cert_file = cert_file

    def sign_common_name(self, nonce: str) -> str:
        return f"sig:{nonce}"


def _make_api(auth_mode="cert") -> AdminAPI:
    api = AdminAPI.__new__(AdminAPI)
    api.auth_mode = auth_mode
    api.user_name = "alice"
    api.client_key = "/tmp/client.key"
    api.client_cert = "/tmp/client.crt"
    api.login_result = None
    api.login_token_value = None
    api.login_token_file = None
    api.login_token_env_var = None
    api.oidc_token_manager = None
    api.server_execute_calls = []
    api.server_sess_active = True

    def _server_execute(command, reply_processor=None, headers=None):
        api.server_execute_calls.append((command, headers))
        api.login_result = "OK"

    def _after_login():
        return {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.DETAILS: "Login success"}

    api.server_execute = _server_execute
    api._after_login = _after_login
    return api


def test_user_login_cert_mode(monkeypatch):
    api = _make_api(auth_mode="cert")
    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", _FakeIdentityAsserter)

    result = api._user_login()

    assert result[ResultKey.STATUS] == APIStatus.SUCCESS
    command, headers = api.server_execute_calls[0]
    assert command == f"{InternalCommands.CERT_LOGIN} alice"
    assert headers["user_name"] == "alice"
    assert headers["cert"] == "fake-cert"
    assert headers["signature"] == "sig:"


def test_user_login_token_mode_with_inline_token():
    api = _make_api(auth_mode="token")
    api.login_token_value = "inline-token"

    result = api._user_login()

    assert result[ResultKey.STATUS] == APIStatus.SUCCESS
    command, headers = api.server_execute_calls[0]
    assert command == InternalCommands.TOKEN_LOGIN
    assert headers == {"authorization": "Bearer inline-token", "auth_mode": "token"}


def test_resolve_login_token_from_file(tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("file-token\n")
    api = _make_api(auth_mode="token")
    api.login_token_file = str(token_file)

    assert api._resolve_login_token() == "file-token"


def test_resolve_login_token_from_env(monkeypatch):
    api = _make_api(auth_mode="token")
    api.login_token_env_var = "NVFLARE_TEST_TOKEN"
    monkeypatch.setenv("NVFLARE_TEST_TOKEN", "env-token")

    assert api._resolve_login_token() == "env-token"


def test_user_login_token_mode_missing_token():
    api = _make_api(auth_mode="token")
    result = api._user_login()

    assert result[ResultKey.STATUS] == APIStatus.ERROR_AUTHENTICATION
    assert "Missing bearer token" in result[ResultKey.DETAILS]
    assert not api.server_execute_calls


def test_token_precedence_inline_over_file_and_env(tmp_path, monkeypatch):
    token_file = tmp_path / "token.txt"
    token_file.write_text("file-token\n")
    monkeypatch.setenv("NVFLARE_TEST_TOKEN", "env-token")

    api = _make_api(auth_mode="token")
    api.login_token_value = "inline-token"
    api.login_token_file = str(token_file)
    api.login_token_env_var = "NVFLARE_TEST_TOKEN"

    assert api._resolve_login_token() == "inline-token"


def test_user_login_oidc_mode_uses_token_login(monkeypatch):
    api = _make_api(auth_mode="oidc")

    class _OIDC:
        def get_access_token(self):
            return "oidc-access-token"

    api.oidc_token_manager = _OIDC()

    result = api._user_login()

    assert result[ResultKey.STATUS] == APIStatus.SUCCESS
    command, headers = api.server_execute_calls[0]
    assert command == InternalCommands.TOKEN_LOGIN
    assert headers == {"authorization": "Bearer oidc-access-token", "auth_mode": "oidc"}


def test_do_command_oidc_auto_login_when_session_inactive():
    class _Reg:
        def __init__(self, entries):
            self._entries = entries

        def get_command_entries(self, cmd):
            return self._entries if cmd == "list_jobs" else []

    class _OIDC:
        invalidated = False

        def invalidate_access_token(self):
            self.invalidated = True

    api = AdminAPI.__new__(AdminAPI)
    api.auth_mode = "oidc"
    api.server_sess_active = False
    api.client_cmd_reg = _Reg([])
    api.server_cmd_reg = _Reg([object()])
    api.oidc_token_manager = _OIDC()

    login_calls = {"count": 0}

    def _login():
        login_calls["count"] += 1
        api.server_sess_active = True
        return {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.DETAILS: "ok"}

    api.login = _login
    api.server_execute = lambda command, cmd_entry=None, props=None: {ResultKey.STATUS: APIStatus.SUCCESS}

    result = api.do_command("list_jobs")

    assert result[ResultKey.STATUS] == APIStatus.SUCCESS
    assert login_calls["count"] == 1


def test_do_command_oidc_retries_after_inactive_result():
    class _Reg:
        def __init__(self, entries):
            self._entries = entries

        def get_command_entries(self, cmd):
            return self._entries if cmd == "list_jobs" else []

    class _OIDC:
        invalidated = False

        def invalidate_access_token(self):
            self.invalidated = True

    api = AdminAPI.__new__(AdminAPI)
    api.auth_mode = "oidc"
    api.server_sess_active = True
    api.client_cmd_reg = _Reg([])
    api.server_cmd_reg = _Reg([object()])
    api.oidc_token_manager = _OIDC()

    login_calls = {"count": 0}
    server_calls = {"count": 0}

    def _login():
        login_calls["count"] += 1
        api.server_sess_active = True
        return {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.DETAILS: "ok"}

    def _server_execute(command, cmd_entry=None, props=None):
        server_calls["count"] += 1
        if server_calls["count"] == 1:
            return {ResultKey.STATUS: APIStatus.ERROR_INACTIVE_SESSION}
        return {ResultKey.STATUS: APIStatus.SUCCESS}

    api.login = _login
    api.server_execute = _server_execute

    result = api.do_command("list_jobs")

    assert result[ResultKey.STATUS] == APIStatus.SUCCESS
    assert login_calls["count"] == 1
    assert server_calls["count"] == 2
    assert api.oidc_token_manager.invalidated


def test_connect_tls_mode_does_not_require_client_cert(monkeypatch, tmp_path):
    ca_cert = tmp_path / "rootCA.pem"
    ca_cert.write_text("dummy-ca")
    created = {}

    class _Cell:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def register_request_cb(self, channel, topic, cb):
            pass

        def start(self):
            pass

    monkeypatch.setattr("nvflare.fuel.hci.client.api.Cell", _Cell)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.NetAgent", lambda cell: None)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.AuxRunner", lambda api: object())
    monkeypatch.setattr("nvflare.fuel.hci.client.api.ObjectStreamer", lambda runner: object())
    monkeypatch.setattr("nvflare.fuel.hci.client.api.flare_decomposers.register", lambda: None)

    admin_config = {
        AdminConfigKey.PROJECT_NAME: "example",
        AdminConfigKey.SERVER_IDENTITY: "server",
        AdminConfigKey.CA_CERT: str(ca_cert),
        AdminConfigKey.CONNECTION_SECURITY: "tls",
        AdminConfigKey.AUTH_MODE: "token",
        AdminConfigKey.HOST: "127.0.0.1",
        AdminConfigKey.PORT: 8003,
    }
    api = AdminAPI(user_name="alice", admin_config=admin_config, cmd_modules=[])

    api.connect()

    assert created["credentials"][DriverParams.CA_CERT.value] == str(ca_cert)
    assert created["credentials"][DriverParams.CONNECTION_SECURITY.value] == "tls"
    assert DriverParams.CLIENT_CERT.value not in created["credentials"]
    assert DriverParams.CLIENT_KEY.value not in created["credentials"]


def test_stream_objects_uses_explicit_server_identity(monkeypatch):
    calls = {}

    class _ObjectStreamer:
        def stream(self, **kwargs):
            calls.update(kwargs)
            return "ok"

    monkeypatch.setattr("nvflare.fuel.hci.client.api.ObjectStreamer", _ObjectStreamer)

    api = AdminAPI.__new__(AdminAPI)
    api.object_streamer = _ObjectStreamer()

    result = api.stream_objects(
        channel="upload",
        topic="folder",
        stream_ctx={"k": "v"},
        targets=["server.admin"],
        producer=object(),
        fl_ctx=object(),
    )

    assert result == "ok"
    assert len(calls["targets"]) == 1
    assert calls["targets"][0].name == "server.admin"
    assert calls["targets"][0].fqcn == "server.admin"
