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

import pytest
from pyhocon import ConfigFactory

from nvflare.tool.job.config import configer as subject
from nvflare.tool.job.config.config_indexer import KeyIndex


def test_custom_config_discovery_includes_nested_user_configuration_and_excludes_reserved_files(tmp_path):
    custom = tmp_path / "app" / "custom" / "nested"
    custom.mkdir(parents=True)
    user_config = custom / "settings.json"
    user_config.write_text(json.dumps({"rate": 0.5, "steps": 3}))
    (custom / f"{subject.NVFLARE_PREFIX}secret.json").write_text('{"excluded": 9}')
    indices = subject.build_config_file_indices(str(tmp_path), ["app"])
    assert set(indices["app"]) == {str(user_config)}
    _, _, keys = indices["app"][str(user_config)]
    assert keys["rate"][0].value == 0.5
    assert keys["steps"][0].value == 3


def test_path_notation_override_updates_nested_array_key_then_deletes_key_and_index():
    config = ConfigFactory.parse_string("components=[{args={rate=1, steps=3}}]")
    indices = {"rate": [KeyIndex("rate", 1)], "steps": [KeyIndex("steps", 3)]}
    subject.handle_key_in_path_notation_or_new_key("settings.conf", "components[0].args.rate", "0.5", config, indices)
    assert config.get("components")[0].get("args.rate") == "0.5"
    assert indices["rate"][0].value == "0.5"
    subject.handle_key_in_path_notation_or_new_key("settings.conf", "components[0].args.steps", None, config, indices)
    assert "steps" not in config.get("components")[0].get("args")
    assert "steps" not in indices
    config.put("obsolete", 1)
    indices["obsolete"] = [KeyIndex("obsolete", 1)]
    subject.handle_key_in_path_notation_or_new_key("settings.conf", "obsolete", None, config, indices)
    assert "obsolete" not in config
    assert "obsolete" not in indices
    with pytest.raises(ValueError, match="invalid key"):
        subject.handle_key_in_path_notation_or_new_key(
            "settings.conf", "missing[0].value", "1", ConfigFactory.parse_string("missing=1"), {}
        )


def test_root_index_follows_nested_parents_without_losing_top_level_owner():
    sentinel = KeyIndex("")
    root = KeyIndex("components", parent_key=sentinel)
    parent = KeyIndex("args", parent_key=root)
    leaf = KeyIndex("rate", 0.5, parent_key=parent)
    assert subject.get_root_index(leaf) is root
    assert subject.get_root_index(KeyIndex("standalone")).key == "standalone"
    assert subject.get_root_index(None) is None
    assert subject.extract_string_with_index("  ") == []
    assert subject.extract_string_with_index("components[2]") == [("components", 2, "")]


def test_cli_config_deletion_markers_preserve_values_containing_equals_and_reject_unknown_apps(tmp_path):
    args = SimpleNamespace(
        job_folder=str(tmp_path),
        config_file=[["worker/settings.conf", "rate=0.5", "obsolete-", "old-=ignored", "url=a=b"]],
        script=None,
    )
    config = subject.get_cli_config(args, ["worker"])
    assert config["worker"][str(tmp_path / "worker" / "config" / "settings.conf")] == {
        "rate": "0.5",
        "obsolete": None,
        "old": None,
        "url": "a=b",
    }
    args.config_file = [["unknown/settings.conf", "rate=1"]]
    with pytest.raises(ValueError, match="app names"):
        subject.get_cli_config(args, ["worker"])
    args.config_file = [["settings.conf", "missing-equals"]]
    with pytest.raises(ValueError, match="key=value"):
        subject.get_cli_config(args, ["worker"])
    assert subject.get_cli_config(SimpleNamespace(script="/source/train.py"), []) == {
        "app": {"config_fed_client.conf": {"app_script": "train.py"}}
    }
