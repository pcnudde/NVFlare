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

import json
from unittest.mock import Mock

import pytest

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils import wfconf as subject


class RecordingConfigurator(subject.Configurator):
    def start_config(self, ctx):
        self.trace = [("start", ctx.pass_num)]

    def process_config_element(self, ctx, node):
        if node.key == "label":
            self.trace.append((node.element, ctx.pass_num))

    def finalize_config(self, ctx):
        self.trace.append(("finish", ctx.pass_num))


def configured(tmp_path, **kwargs):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({"label": "{priority}", "priority": "workflow", "determinism": {"seed": 4}}))
    return RecordingConfigurator(
        str(tmp_path), kwargs.pop("cmd_vars", {}), kwargs.pop("env_config", {}), str(path), [], [], **kwargs
    )


def test_configuration_precedence_app_paths_variable_processor_and_pass_order(tmp_path):
    processor = Mock()
    config = configured(
        tmp_path,
        cmd_vars={"priority": "command", "APP_INPUT": "input"},
        env_config={"priority": "environment", "APP_OUTPUT": "output"},
        default_vars={"priority": "default", "extra": 9},
        num_passes=2,
        var_processor=processor,
    )
    config.configure()
    assert config.wf_config_data["label"] == "command"
    assert config.all_vars["extra"] == 9 and config.all_vars["determinism"] == {"seed": 4}
    assert config.vars_from_cmd["APP_INPUT"] == str(tmp_path / "input")
    assert config.vars_from_env_config["APP_OUTPUT"] == str(tmp_path / "output")
    assert config.trace == [("start", 0), ("command", 1), ("command", 2), ("finish", 2)]
    processor.process.assert_called_once_with(config.all_vars, app_root=str(tmp_path))
    assert config.config_ctx.config_json is config.wf_config_data


def test_element_filter_leaves_runtime_expressions_untouched(tmp_path):
    config = configured(tmp_path, cmd_vars={"priority": "runtime"}, element_filter=lambda value: False)
    config.configure()
    assert config.wf_config_data["label"] == "{priority}"
    with pytest.raises(ValueError, match="element_filter"):
        subject._EnvUpdater({}, element_filter=42)


def test_component_builder_handles_disabled_values_and_legacy_class_attributes(tmp_path):
    config = configured(tmp_path)
    assert config.build_component(None) is None
    assert config.build_component({"disabled": True, "path": "not.a.Class"}) is None
    with pytest.raises(ConfigError, match="component config must be dict"):
        config.build_component(["invalid"])
    assert config.build_component({"path": "builtins.dict", "args": {"answer": 42}}) == {"answer": 42}
    assert config.build_component({"path": "builtins.dict", "args": {"class_attributes": {"answer": 7}}}) == {
        "answer": 7
    }
    assert config.is_configured_subclass({"path": "builtins.dict"}, dict)


@pytest.mark.parametrize("exception", [ConfigError("invalid component"), RuntimeError("unexpected")])
def test_configuration_errors_preserve_type_and_report_file_context(tmp_path, monkeypatch, exception):
    config = configured(tmp_path)
    monkeypatch.setattr(config, "start_config", Mock(side_effect=exception))
    with pytest.raises(type(exception), match="invalid component|unexpected") as raised:
        config.configure()
    if isinstance(exception, ConfigError):
        assert "workflow.json" in str(raised.value)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_passes": 0}, "num_passes"),
        ({"num_passes": "two"}, "num_passes"),
        ({"default_vars": []}, "default_vars"),
        ({"cmd_vars": [1]}, "cmd_vars"),
        ({"env_config": [1]}, "env_config"),
    ],
)
def test_configuration_rejects_invalid_configuration_inputs(tmp_path, kwargs, message):
    with pytest.raises(AssertionError, match=message):
        configured(tmp_path, **kwargs)


def test_parameterized_variable_definitions_are_copied_and_resolved_locally():
    definition = {"amount": "{count}", "path": "{root}/data"}
    updater = subject._EnvUpdater({"@worker": definition, "root": "/tmp"})
    assert updater.substitute("{@worker:count=3}") == {"amount": 3, "path": "/tmp/data"}
    assert definition == {"amount": "{count}", "path": "{root}/data"}
    with pytest.raises(ConfigError, match="not defined"):
        updater.substitute("{@missing:count=3}")
    updater.vars["@worker"] = "not a dictionary"
    with pytest.raises(ConfigError, match="must be dict"):
        updater.substitute("{@worker:count=3}")
