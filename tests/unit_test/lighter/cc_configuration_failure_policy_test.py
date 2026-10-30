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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.lighter.cc_provision.cc_constants import CCConfigKey
from nvflare.lighter.cc_provision.impl.cc import CCBuilder
from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName
from nvflare.lighter.entity import Participant


@pytest.mark.parametrize("value, error", [(7, TypeError), ("missing.yml", FileNotFoundError)])
def test_cc_configuration_requires_existing_yaml_path(tmp_path, value, error):
    path = str(tmp_path / value) if isinstance(value, str) else value
    with pytest.raises(error):
        CCBuilder()._load_and_validate_cc_config(path)


def test_cc_configuration_rejects_unknown_environment_and_reports_participant_failure(tmp_path, capsys):
    path = tmp_path / "invalid.yml"
    path.write_text(f"{CCConfigKey.COMPUTE_ENV}: unsupported\n")
    builder = CCBuilder()
    with pytest.raises(ValueError, match="Invalid compute environment"):
        builder._load_and_validate_cc_config(str(path))
    participant = Participant("client", "site", "org", {PropKey.CC_CONFIG: str(path)})
    builder.initialize(SimpleNamespace(name="project", get_all_participants=lambda: [participant]), {})
    assert builder._cc_enabled_sites == [] and builder._cc_builders == {}
    assert "CC is not enabled for site" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Unsupported compute environment"):
        builder._create_builder_for_env({CCConfigKey.COMPUTE_ENV: "unsupported"})
    with pytest.raises(ValueError, match="public-root-only"):
        CCBuilder().initialize(object(), {CtxKey.PUBLIC_ROOT_ONLY: True})


@pytest.mark.parametrize("allowlist", ["package.Class", [""], [7]])
def test_cc_class_allowlist_rejects_invalid_types_before_reading_resources(allowlist):
    participant = Participant(
        "client", "site", "org", {PropKey.CC_CONFIG_DICT: {CCConfigKey.CLASS_ALLOW_LIST: allowlist}}
    )
    context = Mock()
    with pytest.raises(ValueError, match="class_allow_list"):
        CCBuilder()._extend_class_allow_list(participant, context)
    context.get_local_dir.assert_not_called()


@pytest.mark.parametrize("contents", [None, {CCConfigKey.CLASS_ALLOW_LIST: "invalid"}])
def test_cc_class_allowlist_requires_static_resources_and_list_field(tmp_path, contents):
    participant = Participant(
        "client", "site", "org", {PropKey.CC_CONFIG_DICT: {CCConfigKey.CLASS_ALLOW_LIST: ["package.Class"]}}
    )
    if contents is not None:
        (tmp_path / ProvFileName.RESOURCES_JSON_DEFAULT).write_text(json.dumps(contents))
    context = SimpleNamespace(get_local_dir=lambda participant: str(tmp_path))
    with pytest.raises(RuntimeError, match="StaticFileBuilder" if contents is None else "non-list"):
        CCBuilder()._extend_class_allow_list(participant, context)
    assert not list(tmp_path.glob("*.tmp"))
