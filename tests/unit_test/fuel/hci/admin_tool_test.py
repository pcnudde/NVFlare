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

import sys
from zipfile import ZipFile

from nvflare.fuel.hci.tools import admin as admin_tool


def test_main_imports_invite_and_exits_without_launch(monkeypatch, capsys, tmp_path):
    imported_workspace = tmp_path / "invite"

    monkeypatch.setattr(
        admin_tool,
        "prepare_workspace",
        lambda workspace, invite_file, fed_admin: str(imported_workspace),
    )
    monkeypatch.setattr(admin_tool, "Workspace", lambda root_dir: (_ for _ in ()).throw(AssertionError(root_dir)))
    monkeypatch.setattr(sys, "argv", ["admin.py", "-i", "invite.zip"])

    admin_tool.main()

    output = capsys.readouterr().out
    assert f"Invite imported to: {imported_workspace}" in output
    assert f"{imported_workspace}/startup/fl_admin.sh" in output


def test_prepare_workspace_makes_fl_admin_executable(tmp_path):
    invite_zip = tmp_path / "invite.zip"
    with ZipFile(invite_zip, "w") as zf:
        zf.writestr("startup/fed_admin.json", "{}")
        zf.writestr("startup/fl_admin.sh", "#!/bin/sh\necho hi\n")
        zf.writestr("local/resources.json", "{}")

    imported_workspace = tmp_path / "invite"
    imported_workspace.mkdir()

    workspace_dir = admin_tool.prepare_workspace(workspace=str(imported_workspace), invite_file=str(invite_zip))

    fl_admin = imported_workspace / "startup" / "fl_admin.sh"
    assert workspace_dir == str(imported_workspace.resolve())
    assert fl_admin.exists()
    assert fl_admin.stat().st_mode & 0o111
