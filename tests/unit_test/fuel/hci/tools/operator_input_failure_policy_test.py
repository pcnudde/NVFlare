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

from nvflare.fuel.hci.tools import admin, make_pwd


@pytest.mark.parametrize(
    "passwords,message",
    [
        (["short"], "at least 8 chars"),
        (["long password", "different password"], "Passwords mismatch"),
        (["long password", "long password"], "Password Hash: encoded"),
    ],
)
def test_password_tool_hashes_only_confirmed_minimum_length_identity_bound_secret(
    monkeypatch, capsys, passwords, message
):
    monkeypatch.setattr("builtins.input", Mock(return_value="alice"))
    prompt = Mock(side_effect=passwords)
    monkeypatch.setattr(make_pwd.getpass, "getpass", prompt)
    hash_password = Mock(return_value="encoded")
    monkeypatch.setattr(make_pwd, "hash_password", hash_password)
    make_pwd.main()
    assert message in capsys.readouterr().out
    if passwords == ["long password", "long password"]:
        hash_password.assert_called_once_with("alicelong password")
    else:
        hash_password.assert_not_called()
    assert prompt.call_count == len(passwords)


@pytest.mark.parametrize("case", ["invalid-study", "invalid-config", "missing-admin", "debug-transfer"])
def test_admin_entrypoint_validates_configuration_before_creating_prompt_runtime(tmp_path, monkeypatch, capsys, case):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        admin.sys,
        "argv",
        [
            "admin",
            "--workspace",
            str(tmp_path),
            "--fed_admin",
            "admin.json",
            "--study",
            "bad/name" if case == "invalid-study" else "trial",
        ],
    )
    monkeypatch.setattr(admin, "Workspace", Mock())
    config = {
        admin.AdminConfigKey.WITH_FILE_TRANSFER: True,
        admin.AdminConfigKey.WITH_DEBUG: True,
        admin.AdminConfigKey.UPLOAD_DIR: "uploads",
        admin.AdminConfigKey.DOWNLOAD_DIR: "downloads",
        admin.AdminConfigKey.CLI_HISTORY_SIZE: "invalid",
    }
    conf = SimpleNamespace(get_admin_config=lambda: None if case == "missing-admin" else config, handlers=[])
    monkeypatch.setattr(
        admin,
        "secure_load_admin_config",
        Mock(
            return_value=conf, side_effect=admin.ConfigError("invalid signature") if case == "invalid-config" else None
        ),
    )
    transfer = Mock()
    monkeypatch.setattr(admin, "FileTransferModule", transfer)
    client = Mock()
    monkeypatch.setattr(admin, "AdminClient", client)
    if case == "invalid-study":
        with pytest.raises(SystemExit) as raised:
            admin.main()
        assert raised.value.code == 1
    else:
        admin.main()
    output = capsys.readouterr().out
    if case == "debug-transfer":
        call = client.call_args.kwargs
        assert call["cli_history_size"] == admin.DEFAULT_CLI_HIST_SIZE
        assert call["study"] == "trial" and call["debug"] is True
        assert call["cmd_modules"] == [transfer.return_value]
        assert "Upload Dir: uploads" in output and "Download Dir: downloads" in output
        client.return_value.run.assert_called_once_with()
    else:
        client.assert_not_called()
        assert output
