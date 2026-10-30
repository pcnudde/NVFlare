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

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.client import file_transfer as subject
from nvflare.fuel.hci.client.api_spec import CommandContext
from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.proto import MetaKey, ProtoKey
from nvflare.fuel.hci.reg import CommandEntry


@pytest.fixture
def runtime(tmp_path):
    upload, download = tmp_path / "upload", tmp_path / "download"
    upload.mkdir()
    download.mkdir()
    module = subject.FileTransferModule(str(upload), str(download))
    api = Mock()
    ctx = CommandContext()
    ctx.set_api(api)
    ctx.set_command("file_transfer.pull_folder job")
    ctx.set_command_entry(
        CommandEntry(
            SimpleNamespace(name="file_transfer"), "pull_folder", "description", "usage", Mock(), None, True, None, None
        )
    )
    return module, api, ctx, upload, download


def zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("weights.txt", "model")
    return buffer.getvalue()


@pytest.mark.parametrize("failure", [None, "transfer", "unzip"])
def test_folder_download_defers_unzip_until_all_transfers_and_cleans_failed_transaction(runtime, failure):
    module, api, ctx, _, download = runtime
    commands = []
    metadata = {
        MetaKey.FILES: [("model.zip", "ref-zip"), ("notes.txt", "ref-text")],
        MetaKey.TX_ID: "tx",
        MetaKey.SOURCE_FQCN: "server",
    }

    def server_execute(command, cmd_ctx=None):
        if cmd_ctx is None:
            return {ProtoKey.STATUS: subject.APIStatus.SUCCESS, ProtoKey.META: metadata}
        cmd_ctx.get_requester().send_request(api, None, cmd_ctx)
        return cmd_ctx.get_command_result()

    api.server_execute.side_effect = server_execute

    def download_file(source, ref, destination):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        if failure == "transfer":
            return None
        data = zip_bytes() if ref == "ref-zip" else b"notes"
        if failure == "unzip" and ref == "ref-zip":
            data = b"not a zip"
        path.write_bytes(data)
        return len(data)

    api.download_file.side_effect = download_file

    def do_command(command):
        commands.append(split_to_args(command))
        return module.pull_binary_file(split_to_args(command), ctx)

    api.do_command.side_effect = do_command
    result = module.pull_folder(["pull_folder", "job", "saved"], ctx)
    transaction = download / "job__tx"
    assert not transaction.exists()
    if failure:
        assert result[ProtoKey.STATUS] == subject.APIStatus.ERROR_RUNTIME
        assert not (download / "saved").exists()
    else:
        assert result[ProtoKey.STATUS] == subject.APIStatus.SUCCESS
        assert result[ProtoKey.META][MetaKey.LOCATION] == str(download / "saved")
        assert (download / "saved" / "model" / "weights.txt").read_text() == "model"
        assert (download / "saved" / "notes.txt").read_bytes() == b"notes"
        assert commands[-1][-1] == "end"
        assert not (download / "saved" / "model.zip").exists()


def test_binary_download_validates_arguments_and_preserves_upstream_error(runtime):
    module, api, ctx, _, _ = runtime
    assert module.pull_binary_file(["pull"], ctx)[ProtoKey.STATUS] == subject.APIStatus.ERROR_SYNTAX
    assert module.pull_folder(["pull"], ctx)[ProtoKey.STATUS] == subject.APIStatus.ERROR_SYNTAX
    error = {ProtoKey.STATUS: subject.APIStatus.ERROR_RUNTIME, ProtoKey.DETAILS: "server error"}
    api.server_execute.return_value = error
    assert module.pull_binary_file(["pull", "server", "tx", "ref", "job", "file"], ctx) is error
    assert module.pull_folder(["pull", "job"], ctx) is error


def test_folder_rename_preserves_existing_directory_and_uses_numbered_destination(runtime):
    module, _, _, _, download = runtime
    source, destination = download / "source", download / "saved"
    source.mkdir()
    destination.mkdir()
    (source / "new").write_text("new")
    (destination / "old").write_text("old")
    assert module._rename_folder(str(source), str(destination)) == str(download / "saved__1")
    assert (download / "saved__1" / "new").read_text() == "new"
    assert (destination / "old").read_text() == "old"


def test_folder_rename_exhaustion_is_bounded_and_returns_original_destination(monkeypatch):
    rename = Mock(side_effect=OSError("denied"))
    monkeypatch.setattr(subject, "os", SimpleNamespace(rename=rename))
    assert subject.FileTransferModule._rename_folder("/fake/source", "/fake/destination") == "/fake/destination"
    assert rename.call_count == 1000


def test_file_sender_deletes_archive_after_upload_and_receiver_reports_failure(runtime, tmp_path):
    _, api, ctx, _, _ = runtime
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"data")
    api.upload_file.return_value = "reply"
    assert subject._FileSender(str(archive)).send_request(api, "connection", ctx) == "reply"
    api.upload_file.assert_called_once_with(str(archive), "connection")
    assert not archive.exists()
    api.download_file.return_value = None
    receiver = subject._FileReceiver("server", "reference", "/fake/file")
    assert receiver.send_request(api, None, ctx) is None
    assert ctx.get_command_result()[ProtoKey.STATUS] == subject.APIStatus.ERROR_RUNTIME


def test_transfer_module_rejects_nonexistent_roots_and_describes_local_folders(tmp_path, runtime):
    with pytest.raises(ValueError, match="upload_dir"):
        subject.FileTransferModule(str(tmp_path / "missing"), str(tmp_path))
    with pytest.raises(ValueError, match="download_dir"):
        subject.FileTransferModule(str(tmp_path), str(tmp_path / "missing"))
    module, _, ctx, upload, download = runtime
    info = module.info([], ctx)
    assert str(upload) in info["details"] and str(download) in info["details"]
    assert module.get_spec().name == "file_transfer"


def test_server_command_mapping_exposes_only_supported_transfer_handlers(runtime, capsys):
    module, _, _, _, _ = runtime
    server = subject.CommandSpec(
        "download", "description", "usage", Mock(), client_cmd=subject.ftd.PULL_FOLDER_FQN, scope_name="server"
    )
    generated = module.generate_module_spec(server)
    assert generated.name == "server"
    assert generated.cmd_specs[0].handler_func == module.pull_folder
    assert generated.cmd_specs[0].name == "download"
    assert generated.cmd_specs[0].description == "description"
    server.client_cmd = None
    assert module.generate_module_spec(server) is None
    server.client_cmd = "unknown.handler"
    assert module.generate_module_spec(server) is None
    assert "no cmd handler" in capsys.readouterr().out
