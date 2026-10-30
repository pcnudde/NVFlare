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

from nvflare.apis.filter import Filter, FilterContextKey, FilterSource
from nvflare.apis.fl_constant import FilterKey
from nvflare.private import fed_json_config, json_configer


@pytest.mark.parametrize(
    "passes,filename,error",
    [
        ("two", "config.json", TypeError),
        (0, "config.json", ValueError),
        (1, 123, TypeError),
        (1, "missing.json", FileNotFoundError),
    ],
)
def test_configurator_rejects_invalid_pass_counts_and_config_file_types(monkeypatch, passes, filename, error):
    monkeypatch.setattr(json_configer.ConfigFactory, "has_config", lambda path: False)
    with pytest.raises(error):
        json_configer.JsonConfigurator(filename, [], [], num_passes=passes)


@pytest.mark.parametrize(
    "component,parts,key",
    [
        ({"path": "package.Class#variant"}, ["package.Class", "variant"], "path"),
        ({"class_path": "package.Class#variant"}, ["package.Class", "variant"], "class_path"),
        ({"name": "Class#variant"}, ["Class", "variant"], "name"),
    ],
)
def test_component_references_split_selector_and_normalize_builder_path(component, parts, key):
    assert json_configer.get_component_refs(component) == parts
    assert component[key] == parts[0]


@pytest.mark.parametrize("component", [{}, {"path": None}, {"class_path": 42}, {"name": ""}])
def test_component_references_reject_missing_empty_or_non_string_names(component):
    with pytest.raises(json_configer.ConfigError):
        json_configer.get_component_refs(component)


def test_configure_retains_parameter_templates_for_resolution_but_removes_them_from_scanning(monkeypatch):
    config = object.__new__(json_configer.JsonConfigurator)
    config.config_data = {"@template": {"path": "package.Component"}, "value": "local"}
    config.sys_vars = {"value": "environment"}
    config.num_passes = 2
    config.config_file_names = ["config.json"]
    config.json_scanner = Mock()
    events = []
    config.start_config = lambda ctx: events.append(("start", ctx.pass_num))
    config.finalize_config = lambda ctx: events.append(("finish", ctx.pass_num))
    config.json_scanner.scan.side_effect = lambda scanner: events.append(("scan", config.config_ctx.pass_num))
    resolver = Mock()
    monkeypatch.setattr(json_configer, "resolve_var_refs", resolver)
    config.configure()
    assert events == [("start", 0), ("scan", 1), ("scan", 2), ("finish", 2)]
    assert config.config_data == {"value": "local"}
    assert resolver.call_args.args[1] == {"@template": {"path": "package.Component"}, "value": "environment"}


def test_configure_reports_source_and_reraises_failure(capsys):
    config = object.__new__(json_configer.JsonConfigurator)
    config.config_file_names = ["broken.json"]
    config._do_configure = Mock(side_effect=ValueError("invalid component"))
    with pytest.raises(ValueError, match="invalid component"):
        config.configure()
    assert "broken.json" in capsys.readouterr().out


@pytest.fixture
def fed_config():
    config = object.__new__(fed_json_config.FedJsonConfigurator)
    config.format_version = 2
    config.is_server = True
    config.config_data = {}
    config.components = {}
    config.task_data_filter_chains = []
    config.task_result_filter_chains = []
    return config


@pytest.mark.parametrize("tasks", ["train", [], [1]])
def test_filter_task_names_require_non_empty_string_list(fed_config, tasks):
    with pytest.raises(json_configer.ConfigError):
        fed_config.validate_tasks(tasks)


@pytest.mark.parametrize("filters", ["filter", [], [object()]])
def test_filter_chains_require_non_empty_filter_instances(fed_config, filters):
    chain = fed_json_config.FilterChain("task_data_filters", FilterKey.IN)
    chain.tasks = ["train"]
    chain.filters = filters
    with pytest.raises(json_configer.ConfigError):
        fed_config.validate_filter_chain(chain)


def test_filter_chain_sets_job_source_and_expands_inout_without_duplicate_routes(fed_config):
    item = Mock(spec=Filter)
    chain = fed_json_config.FilterChain("task_data_filters", FilterKey.INOUT)
    chain.tasks = ["train"]
    chain.filters = [item]
    fed_config.validate_filter_chain(chain)
    item.set_prop.assert_any_call(FilterContextKey.CHAIN_TYPE, "task_data_filters")
    item.set_prop.assert_any_call(FilterContextKey.SOURCE, FilterSource.JOB)
    fed_config.task_data_filter_chains = [chain]
    fed_config.finalize_config(None)
    assert fed_config.data_filter_table == {
        "train" + FilterKey.DELIMITER + direction: [item] for direction in (FilterKey.IN, FilterKey.OUT)
    }
    fed_config.task_data_filter_chains.append(chain)
    with pytest.raises(json_configer.ConfigError, match="multiple data filter chains"):
        fed_config.finalize_config(None)


@pytest.mark.parametrize("version", [None, "2", 99])
def test_federated_config_requires_supported_integer_format_version(fed_config, version):
    fed_config.format_version = version
    with pytest.raises(json_configer.ConfigError):
        fed_config.finalize_config(None)


@pytest.mark.parametrize("component_id", [None, 5, "duplicate"])
def test_component_registration_requires_unique_string_id(fed_config, component_id):
    fed_config.components = {"duplicate": object()}
    fed_config.authorize_and_build_component = Mock(return_value=object())
    node = Mock(element={"id": component_id})
    node.path.return_value = "components.#0"
    with pytest.raises(json_configer.ConfigError):
        fed_config.process_config_element(None, node)
    assert list(fed_config.components) == ["duplicate"]
