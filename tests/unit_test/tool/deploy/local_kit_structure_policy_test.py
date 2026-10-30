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

import json

import pytest

from nvflare.tool import cli_output
from nvflare.tool.deploy import deploy_common as subject


@pytest.fixture(autouse=True)
def json_output(monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "json")


@pytest.mark.parametrize("case", ["missing-kit", "missing-startup", "missing-local", "ambiguous-role"])
def test_kit_structure_requires_exactly_one_role_before_any_runtime_work(tmp_path, capsys, case):
    if case != "missing-kit":
        if case != "missing-startup":
            (tmp_path / "startup").mkdir()
        if case not in ("missing-startup", "missing-local"):
            (tmp_path / "local").mkdir()
    with pytest.raises(SystemExit) as error:
        subject.validate_kit(tmp_path / "absent" if case == "missing-kit" else tmp_path)
    assert error.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_KIT"


@pytest.mark.parametrize("content", [None, "[", "- not-a-mapping"])
def test_json_component_document_reports_missing_parse_and_shape_errors(tmp_path, capsys, content):
    path = tmp_path / "resources.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(SystemExit):
        subject._load_json_file(path, "resources")
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_KIT"


@pytest.mark.parametrize("case", ["mapping", "required-string", "optional-map"])
def test_runtime_configuration_requires_declared_mapping_and_string_shapes(capsys, case):
    with pytest.raises(SystemExit):
        if case == "mapping":
            subject._mapping([], "runtime")
        elif case == "required-string":
            subject._required_str({"image": 3}, "image", "runtime")
        else:
            subject._optional_mapping({"security": "tls"}, "security", "runtime")
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_CONFIG"


def test_port_and_launcher_fallbacks_are_explicit_without_running_commands():
    assert subject._port_from_target(None) is None
    assert subject._port_from_target("server:invalid") is None
    assert not subject._launcher_replacement_discards_config({"id": "k8s_launcher"}, "k8s_launcher", "expected")
    assert subject._launcher_replacement_discards_config(
        {"id": "k8s_launcher", "path": "custom.Launcher"}, "k8s_launcher", "expected"
    )
    assert subject._paths_overlap(subject.Path("/kits"), subject.Path("/kits/site"))
