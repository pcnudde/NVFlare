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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from nvflare.tool import cli_output
from nvflare.tool.deploy import k8s_stage as subject


@pytest.fixture(autouse=True)
def json_output(monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "json")


@pytest.mark.parametrize("values", [{"workspaceConfig": []}, {"workspaceConfig": {"local": [1]}}])
def test_recorded_volume_state_requires_mapping_for_workspace_and_resources(capsys, values):
    with pytest.raises(SystemExit) as error:
        subject._read_k8_stage_state(values)
    assert error.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_KIT"


@pytest.mark.parametrize("arguments", [SimpleNamespace(), SimpleNamespace(kit="first", kit_flag="second")])
def test_kit_selection_requires_exactly_one_prepared_directory(capsys, arguments):
    with pytest.raises(SystemExit):
        subject._resolve_k8_kit(arguments, "stage")
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"


@pytest.mark.parametrize("content", ["[", "- not-a-mapping"])
def test_invalid_helm_values_are_reported_before_any_kubectl_operation(tmp_path, capsys, content):
    chart = tmp_path / subject.HELM_CHART_DIR
    chart.mkdir()
    (chart / "values.yaml").write_text(content)
    with pytest.raises(SystemExit):
        subject._load_k8_stage_values(tmp_path)
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_KIT"


@pytest.mark.parametrize(
    "case", ["no-kit", "no-chart", "empty-folder", "unsafe-relative", "unsafe-absolute", "kubectl"]
)
def test_unstage_and_volume_preflight_refuse_invalid_inputs(tmp_path, capsys, case):
    with pytest.raises(SystemExit):
        if case == "no-kit":
            subject._validate_k8_unstage_inputs(tmp_path / "absent", "namespace", "local", "startup")
        elif case == "no-chart":
            subject._validate_k8_unstage_inputs(tmp_path, "namespace", "local", "startup")
        elif case == "empty-folder":
            subject._build_volume_bundle(tmp_path)
        elif case == "kubectl":
            subject._resolve_kubectl(SimpleNamespace(kubectl="   "))
        else:
            subject._validate_k8_volume_item_path("../outside" if case == "unsafe-relative" else "/outside", tmp_path)
    assert json.loads(capsys.readouterr().out)["error_code"] == ("INVALID_ARGS" if case == "kubectl" else "INVALID_KIT")


def test_stage_metadata_is_written_then_cleared_without_losing_other_chart_values(tmp_path):
    chart = tmp_path / subject.HELM_CHART_DIR
    chart.mkdir()
    path = chart / "values.yaml"
    path.write_text(yaml.safe_dump({"siteName": "site", "replicas": 2}))
    subject._patch_k8_stage_values(
        tmp_path, "namespace", "local", [{"key": "one", "path": "config.json"}], "startup", []
    )
    recorded = subject._read_k8_stage_state(yaml.safe_load(path.read_text()))
    assert recorded["namespace"] == "namespace"
    assert recorded["local_configmap"] == "local" and recorded["startup_secret"] == "startup"
    subject._clear_k8_stage_values(tmp_path, "namespace", "local", "startup")
    values = yaml.safe_load(path.read_text())
    assert values["replicas"] == 2 and values["siteName"] == "site"
    assert subject._read_k8_stage_state(values)["namespace"] is None


def test_unstage_metadata_requires_namespace_and_both_exact_resource_names(tmp_path, capsys):
    chart = tmp_path / subject.HELM_CHART_DIR
    chart.mkdir()
    path = chart / "values.yaml"
    path.write_text("{}")
    args = SimpleNamespace(kit=str(tmp_path))
    with pytest.raises(SystemExit):
        subject._resolve_k8_unstage_inputs(args)
    assert "namespace" in json.loads(capsys.readouterr().out)["message"]
    path.write_text(yaml.safe_dump({"workspaceConfig": {"namespace": "namespace"}}))
    with pytest.raises(SystemExit):
        subject._resolve_k8_unstage_inputs(args)
    assert "ConfigMap" in json.loads(capsys.readouterr().out)["message"]


def test_absent_launcher_and_stage_state_do_not_modify_prepared_files(tmp_path):
    assert subject._find_k8s_launcher(tmp_path) is None
    assert subject._read_k8_launcher_namespace(tmp_path) is None
    local = tmp_path / "local"
    local.mkdir()
    resources = local / subject.RESOURCES_JSON_DEFAULT
    resources.write_text('{"components": null}')
    assert subject._find_k8s_launcher(tmp_path) is None
    subject._clear_k8_stage_values(tmp_path, "namespace", "local", "startup")
    assert resources.read_text() == '{"components": null}'


def test_large_volume_warning_names_resource_and_does_not_invoke_kubernetes(monkeypatch):
    warn = Mock()
    monkeypatch.setattr(subject, "_warn", warn)
    subject._warn_if_large_k8_object("Secret", "startup", subject.K8_STAGE_OBJECT_SIZE_WARN_BYTES + 1)
    assert "Secret" in warn.call_args.args[0] and "startup" in warn.call_args.args[0]
