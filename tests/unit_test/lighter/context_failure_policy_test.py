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

from unittest.mock import Mock

import pytest

from nvflare.lighter import ctx as module
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.entity import Project


@pytest.fixture
def context(tmp_path):
    project = Project("project", "description")
    project.set_server("server", "org", {})
    return module.ProvisionContext(str(tmp_path / "workspace"), project)


def test_template_loader_skips_empty_and_cached_inputs_and_rejects_wrong_type(monkeypatch, context):
    load = Mock(return_value={"section": "hello"})
    monkeypatch.setattr(module, "load_yaml", load)
    context.load_templates(None)
    load.assert_not_called()
    with pytest.raises(ValueError, match="temp_files"):
        context.load_templates(5)
    context.load_templates(["first.yml", "first.yml"])
    context.load_templates("first.yml")
    assert load.call_count == 1
    assert context[CtxKey.TEMP_FILES_LOADED] == ["first.yml"]
    assert context.get_template_section("section") == "hello"
    with pytest.raises(RuntimeError, match="missing section"):
        context.get_template_section("absent")
    context[CtxKey.TEMPLATE] = {}
    with pytest.raises(RuntimeError, match="not available"):
        context.get_template_section("absent")


def test_template_composition_validates_type_and_runs_callback_after_replacement(context):
    context[CtxKey.TEMPLATE] = {"first": "{~~NAME~~}", "second": ":value"}
    callback = Mock(side_effect=lambda text, suffix: text + suffix)
    result = context.build_section_from_template(["first", "second"], {"NAME": "client"}, callback, suffix="!")
    assert result == "client:value!"
    callback.assert_called_once_with("client:value", suffix="!")
    with pytest.raises(ValueError, match="temp_section"):
        context.build_section_from_template(123)


@pytest.mark.parametrize("json_mode", [True, False])
def test_context_logging_keeps_collected_errors_and_uses_cli_output_stream_policy(
    monkeypatch, context, capsys, json_mode
):
    from nvflare.tool import cli_output

    monkeypatch.setattr(cli_output, "_is_json_mode", lambda: json_mode)
    for level in ["info", "debug", "error", "warning"]:
        getattr(context, level)(level + "-message")
    output = capsys.readouterr()
    assert context.get_errors() == ["error-message"]
    assert context.get_warnings() == ["warning-message"]
    assert "info-message" in (output.err if json_mode else output.out)
    assert "debug-message" in (output.err if json_mode else output.out)
    assert ("error-message" in output.err) is (not json_mode)
    assert ("warning-message" in output.err) is (not json_mode)
    detached = context.get_warnings()
    detached.clear()
    assert context.get_warnings() == ["warning-message"]
    logger = Mock()
    context.set_logger(logger)
    for level in ["info", "debug", "error", "warning"]:
        getattr(context, level)("delegated")
        getattr(logger, level).assert_called_once_with("delegated")
    assert context.get_logger() is logger
    with pytest.raises(ValueError, match="invalid provision mode"):
        context.set_provision_mode("unknown")
