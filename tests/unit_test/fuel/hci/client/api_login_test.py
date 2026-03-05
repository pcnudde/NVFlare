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

from nvflare.fuel.hci.client.api import APIStatus, AdminAPI, ResultKey
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
    api.server_execute_calls = []

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
    assert headers == {"authorization": "Bearer inline-token"}


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
