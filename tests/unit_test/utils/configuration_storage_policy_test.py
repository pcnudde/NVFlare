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

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pyhocon import ConfigFactory

from nvflare.tool import job as job_module
from nvflare.utils import cli_utils as subject


@pytest.fixture
def template_locations(tmp_path, monkeypatch):
    explicit, configured, home, package = [tmp_path / name for name in ["explicit", "configured", "home", "package"]]
    for path in [explicit, configured, home / subject.JOB_TEMPLATES, package / "templates"]:
        path.mkdir(parents=True)
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.setattr(job_module, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(subject, "load_hidden_config", Mock(return_value=None))
    return explicit, configured, home, package


def test_template_directory_precedence_is_explicit_then_home_then_config_then_package(template_locations, monkeypatch):
    explicit, configured, home, package = template_locations
    monkeypatch.setenv("NVFLARE_HOME", str(home))
    monkeypatch.setattr(
        subject,
        "load_hidden_config",
        Mock(return_value=ConfigFactory.from_dict({"job_template": {"path": str(configured)}})),
    )
    assert subject.find_job_templates_location(str(explicit)) == str(explicit)
    assert subject.find_job_templates_location() == str(home / subject.JOB_TEMPLATES)
    monkeypatch.delenv("NVFLARE_HOME")
    assert subject.find_job_templates_location() == str(configured)
    monkeypatch.setattr(subject, "load_hidden_config", Mock(return_value=None))
    assert subject.find_job_templates_location() == str(package / "templates")


def test_invalid_explicit_template_does_not_fall_back_and_missing_all_locations_is_explicit(
    template_locations, monkeypatch
):
    _, _, _, package = template_locations
    with pytest.raises(ValueError, match="Invalid job template"):
        subject.find_job_templates_location(str(package / "missing"))
    (package / "templates").rmdir()
    with pytest.raises(ValueError, match="Required job_template"):
        subject.find_job_templates_location()


def test_config_conversion_replaces_original_only_after_new_file_is_written(tmp_path):
    original = tmp_path / "settings.json"
    original.write_text("old")
    config = ConfigFactory.from_dict({"version": 2, "value": 7})
    subject.save_config(config, str(original), keep_origin_format=False)
    converted = tmp_path / "settings.conf"
    assert not original.exists()
    assert ConfigFactory.parse_file(str(converted)).get_int("value") == 7
    subject.save_configs({"job": (config, str(tmp_path / "new.json"))})
    assert ConfigFactory.parse_file(str(tmp_path / "new.json")).get_int("version") == 2


def test_atomic_config_replace_failure_keeps_previous_file_and_removes_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    target.write_text("original")
    local_os = SimpleNamespace(**vars(os))
    local_os.replace = Mock(side_effect=OSError("replace failed"))
    monkeypatch.setattr(subject, "os", local_os)
    with pytest.raises(OSError, match="replace failed"):
        subject.save_config(ConfigFactory.from_dict({"value": 1}), str(target))
    assert target.read_text() == "original"
    assert [path.name for path in tmp_path.iterdir()] == ["settings.json"]
