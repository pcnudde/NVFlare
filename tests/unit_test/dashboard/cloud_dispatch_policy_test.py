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

import importlib
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("docker")
module = importlib.import_module("nvflare.dashboard.cli")


@pytest.mark.parametrize("csp", ["aws", "azure"])
def test_cloud_script_generation_uses_installed_version_and_removes_temporary_script(monkeypatch, tmp_path, csp):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "nvflare", SimpleNamespace(__version__="2.7.0"))
    script = "echo {~~NVFLARE~~} {~~START_OPT~~}"
    load = Mock(side_effect=[{"base": "master"}, {"aws_start_dsb_sh": script}, {"azure_start_dsb_sh": script}])
    monkeypatch.setattr(module.utils, "load_yaml", load)
    template = Mock()
    template.get_cloud_script_header.return_value = "header\n"
    factory = Mock(return_value=template)
    monkeypatch.setattr(module.tplt_utils, "Template", factory)
    scripts = []
    process = Mock(side_effect=lambda filename: scripts.append(Path(filename).read_text()))
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=process))
    args = Namespace(cloud=csp, image="image:tag", vpc_id=None, subnet_id=None)
    module.cloud(args)
    assert scripts == ["header\necho nvflare==2.7.0 -i image:tag"]
    assert not list(tmp_path.iterdir())
    process.assert_called_once_with(str(tmp_path / f"{csp}_start_dsb.sh"))
    assert factory.call_args.args[0] == {"base": "master", "aws_start_dsb_sh": script, "azure_start_dsb_sh": script}


def test_cloud_development_version_exits_before_writing_or_invoking_script(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "nvflare", SimpleNamespace(__version__="2.7.0+dev"))
    monkeypatch.setattr(module.utils, "load_yaml", lambda filename: {"aws_start_dsb_sh": "script"})
    write = Mock()
    monkeypatch.setattr(module.utils, "_write", write)
    process = Mock()
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=process))
    with pytest.raises(SystemExit) as caught:
        module.cloud(Namespace(cloud="aws", image="image:tag"))
    assert caught.value.code == 0
    write.assert_not_called()
    process.assert_not_called()
    assert "official NVFlare release" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["stop", "start", "local", "cloud", "unsupported", "empty"])
def test_dashboard_dispatch_prioritizes_actions_and_rejects_unknown_cloud(monkeypatch, mode, capsys):
    calls = {name: Mock() for name in ["stop", "start", "cloud", "_require_image"]}
    for name, callback in calls.items():
        monkeypatch.setattr(module, name, callback)
    args = Namespace(
        stop=mode == "stop",
        start=mode == "start",
        local=mode == "local",
        cloud="aws" if mode == "cloud" else "unknown" if mode == "unsupported" else "",
    )
    module.handle_dashboard(args)
    assert calls["stop"].call_count == int(mode == "stop")
    assert calls["start"].call_count == int(mode in ["start", "local"])
    assert calls["cloud"].call_count == int(mode == "cloud")
    assert calls["_require_image"].call_count == int(mode in ["start", "cloud"])
    if mode in ["unsupported", "empty"]:
        assert ("requested" if mode == "unsupported" else "usage") in capsys.readouterr().out


def test_dashboard_main_parses_flags_and_recognizes_direct_module_invocation(monkeypatch):
    monkeypatch.setattr(module, "_dashboard_parser", None)
    monkeypatch.setattr(module.sys, "argv", ["nvflare.dashboard.cli", "--local", "-i", "image:tag"])
    dispatch = Mock()
    monkeypatch.setattr(module, "handle_dashboard", dispatch)
    module.main()
    assert dispatch.call_args.args[0].local
    assert dispatch.call_args.args[0].image == "image:tag"
    monkeypatch.setattr(module.sys, "argv", ["nvflare.dashboard.cli"])
    assert module.has_no_arguments()
    monkeypatch.setattr(module.sys, "argv", ["nvflare", "--local"])
    assert not module.has_no_arguments()
