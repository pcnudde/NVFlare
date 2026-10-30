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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.tool import cli_output
from nvflare.tool.recipe import recipe_cli as subject


@pytest.mark.parametrize("value", ["framework", "framework="])
def test_recipe_filters_require_key_value_and_nonempty_value(value):
    with pytest.raises(ValueError, match="expected key=value|non-empty"):
        subject._parse_recipe_filters([value])


def test_empty_filtered_catalog_explains_exact_filters_in_human_output(monkeypatch, capsys):
    monkeypatch.setattr(cli_output, "_output_format", "human")
    monkeypatch.setattr(subject, "_load_catalog_for_cli", Mock(return_value=[]))
    subject.cmd_recipe_list(SimpleNamespace(framework=None, filters=["algorithm=fedavg"]))
    assert "No recipes matched filters: algorithm=fedavg" in capsys.readouterr().out


@pytest.mark.parametrize("entry", [None, {}, {"summary": {}, "detail": {}}])
def test_malformed_persisted_catalog_entries_are_not_accepted(entry):
    assert not subject._is_valid_catalog_entry(entry)


def test_state_exchange_rendering_differentiates_fixed_and_configurable_protocol():
    assert subject._state_exchange_text({"state_exchange": "full_model"}) == "full_model"
    assert (
        subject._state_exchange_text({"state_exchange": "full_model", "parameters": [{"name": "params_transfer_type"}]})
        == "full_model (configurable with params_transfer_type)"
    )
    assert subject._recipe_cli_name("company.algorithm", "unrecognized") == "algorithm"
    assert subject._module_source_path(None) is None
    assert subject._package_source_path(None) is None
    assert subject._package_source_path("outside_package") is None
