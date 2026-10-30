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

from nvflare.apis.app_validation import AppValidator
from nvflare.apis.filter import Filter
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FilterKey, SystemConfigs
from nvflare.fuel.utils.wfconf import ConfigError
from nvflare.private.defs import ClientRegMsgKey
from nvflare.private.fed.app import fl_conf


def _node(path, element):
    return SimpleNamespace(element=element, path=lambda: path)


class _PassFilter(Filter):
    def process(self, shareable, fl_ctx):
        return shareable


@pytest.mark.parametrize("config_class", [fl_conf.FLServerStarterConfiger, fl_conf.FLClientStarterConfiger])
def test_starter_constructor_loads_site_config_files_and_command_overrides(monkeypatch, tmp_path, config_class):
    workspace = Mock()
    workspace.get_site_custom_dir.return_value = str(tmp_path)
    workspace.get_config_files_for_startup.return_value = ["startup/config.json", "local/resources.json"]
    init = Mock(return_value=None)
    monkeypatch.setattr(fl_conf.JsonConfigurator, "__init__", init)
    monkeypatch.setattr(fl_conf.sys, "path", list(fl_conf.sys.path))
    args = SimpleNamespace(job_id="job-1")
    config = config_class(workspace, args, ["uid=site-1", "secure_train=true"])
    assert str(tmp_path) in fl_conf.sys.path
    assert config.cmd_vars["uid"] == "site-1"
    assert config.components == {} and config.handlers == []
    assert config.workspace is workspace and config.args is args
    assert init.call_args.kwargs["config_file_name"] == ["startup/config.json", "local/resources.json"]
    expected = config_class is fl_conf.FLServerStarterConfiger
    workspace.get_config_files_for_startup.assert_called_once_with(is_server=expected, for_job=True)
    with pytest.raises(AssertionError, match="cmd_vars must be list"):
        config_class(workspace, args, "uid=site-1")


@pytest.mark.parametrize("config_class", [fl_conf.FLServerStarterConfiger, fl_conf.FLClientStarterConfiger])
def test_component_handlers_are_deduplicated_by_component_type(monkeypatch, config_class):
    config = config_class.__new__(config_class)
    config.handlers = []
    first, duplicate, plain = FLComponent(), FLComponent(), object()
    builder = Mock(side_effect=[first, duplicate, plain])
    monkeypatch.setattr(fl_conf.JsonConfigurator, "build_component", builder)
    assert config.build_component({"path": "first"}) is first
    assert config.build_component({"path": "second"}) is duplicate
    assert config.build_component({"path": "plain"}) is plain
    assert config.handlers == [first]


@pytest.mark.parametrize("config_class", [fl_conf.FLServerStarterConfiger, fl_conf.FLClientStarterConfiger])
@pytest.mark.parametrize(
    "identifier,error", [(None, "missing component id"), (5, "must be str"), ("duplicate", "duplicate component id")]
)
def test_configured_components_require_unique_nonempty_string_ids(config_class, identifier, error):
    config = config_class.__new__(config_class)
    config.components = {"duplicate": object()}
    config.build_component = Mock(return_value=object())
    with pytest.raises(ConfigError, match=error):
        config.process_config_element(None, _node("components.#0", {"id": identifier}))
    assert list(config.components) == ["duplicate"]


@pytest.mark.parametrize("config_class", [fl_conf.FLServerStarterConfiger, fl_conf.FLClientStarterConfiger])
def test_component_and_validator_nodes_populate_only_requested_slots(config_class, caplog):
    config = config_class.__new__(config_class)
    config.components = {}
    config.app_validator = None
    config.snapshot_persistor = None
    component, validator, persistor = object(), object(), object()
    config.build_component = Mock(side_effect=[component, validator, persistor])
    config.process_config_element(None, _node("components.#3", {"id": "usable"}))
    config.process_config_element(None, _node("app_validator", {"path": "validator"}))
    config.process_config_element(None, _node("overseer_agent", {}))
    assert config.components == {"usable": component} and config.app_validator is validator
    assert "obsolete" in caplog.text
    if config_class is fl_conf.FLServerStarterConfiger:
        config.process_config_element(None, _node("snapshot_persistor", {}))
        assert config.snapshot_persistor is persistor
    config.process_config_element(None, _node("ignored", {}))


@pytest.mark.parametrize("config_class", [fl_conf.FLServerStarterConfiger, fl_conf.FLClientStarterConfiger])
def test_invalid_startup_shape_is_reported_with_configuration_file_names(monkeypatch, config_class):
    config = config_class.__new__(config_class)
    config.config_data = {}
    config.server_config_file_names = ["broken-server.json"]
    config.client_config_file_names = ["broken-client.json"]
    monkeypatch.setattr(fl_conf.JsonConfigurator, "start_config", Mock())
    with pytest.raises(ValueError, match="config error.*broken-"):
        config.start_config(None)


@pytest.mark.parametrize("config_class", [fl_conf.FLServerStarterConfiger, fl_conf.FLClientStarterConfiger])
@pytest.mark.parametrize("secure", [False, True])
def test_finalize_builds_deployer_with_site_policy_and_initializes_config_service(monkeypatch, config_class, secure):
    config = config_class.__new__(config_class)
    config.args = SimpleNamespace(workspace="/workspace")
    config.cmd_vars = {"uid": "site-1", "org": "Org", "host": "host", "secure_train": secure}
    config.components, config.handlers = {"one": object()}, [FLComponent()]
    custom_validator = Mock(spec=AppValidator)
    config.app_validator = custom_validator
    config.snapshot_persistor = object()
    config.config_data = {"servers": [{"service": {}}], "client": {ClientRegMsgKey.SITE_CONFIG: {"public": True}}}
    config.server_config_file_names = config.client_config_file_names = ["/kit/startup.json", "/kit/resources.json"]
    deployer, initialize = Mock(), Mock()
    monkeypatch.setattr(fl_conf, "ServerDeployer", lambda: deployer)
    monkeypatch.setattr(fl_conf, "BaseClientDeployer", lambda: deployer)
    monkeypatch.setattr(fl_conf.ConfigService, "initialize", initialize)
    config.finalize_config(None)
    build = deployer.build.call_args.args[0]
    assert build["secure_train"] is secure and build["site_org"] == "Org"
    assert build["server_host"] == "host" and build["server_config"] == [{"service": {}}]
    assert config.site_org == "Org"
    kwargs = initialize.call_args.kwargs
    assert kwargs["section_files"] == {
        SystemConfigs.STARTUP_CONF: "startup.json",
        SystemConfigs.RESOURCES_CONF: "resources.json",
    }
    assert kwargs["config_path"] == ["/workspace"] and kwargs["parsed_args"] is config.args
    assert kwargs["var_dict"] is config.cmd_vars
    if config_class is fl_conf.FLServerStarterConfiger:
        assert build["server_components"] is config.components and build["server_handlers"] is config.handlers
        assert config.deployer is deployer
    else:
        assert build["client_name"] == "site-1" and build["client_components"] is config.components
        assert config.base_deployer is deployer


def test_client_finalize_projects_only_public_site_configuration(monkeypatch):
    config = fl_conf.FLClientStarterConfiger.__new__(fl_conf.FLClientStarterConfiger)
    config.args = SimpleNamespace(workspace="/workspace")
    config.cmd_vars, config.components, config.handlers = {}, {}, []
    config.app_validator = None
    config.config_data = {"client": {}, "custom_public_value": "visible"}
    config.client_config_file_names = ["startup.json", "resources.json"]
    monkeypatch.setattr(fl_conf, "BaseClientDeployer", Mock())
    monkeypatch.setattr(fl_conf.ConfigService, "initialize", Mock())
    config.finalize_config(None)
    assert config.config_data["client"][ClientRegMsgKey.SITE_CONFIG] == {"custom_public_value": "visible"}


@pytest.mark.parametrize("names_only", [False, True])
def test_privacy_configuration_loads_named_scopes_and_default_from_json(tmp_path, names_only):
    path = tmp_path / "privacy.json"
    path.write_text(
        json.dumps(
            {
                "scopes": [{"name": "private", "properties": {"secret": True}}, {"name": "public"}],
                "default_scope": "private",
            }
        )
    )
    workspace = Mock()
    workspace.get_site_privacy_file_path.return_value = str(path)
    manager = fl_conf.create_privacy_manager(workspace, names_only)
    assert manager.is_policy_defined()
    assert manager.get_scope(None).name == "private"
    assert manager.get_scope("public").name == "public"
    assert manager.get_scope("private").props == ({} if names_only else {"secret": True})
    assert manager.get_scope("missing") is None


def test_missing_privacy_file_does_not_define_a_policy(tmp_path):
    workspace = Mock()
    workspace.get_site_privacy_file_path.return_value = str(tmp_path / "missing.json")
    manager = fl_conf.create_privacy_manager(workspace, names_only=False)
    assert manager.is_policy_defined() is False and manager.get_scope(None) is None


@pytest.mark.parametrize("server", [False, True])
@pytest.mark.parametrize("kind", ["task_data_filters", "task_result_filters"])
@pytest.mark.parametrize("explicit", [None, "INOUT"])
def test_privacy_filter_direction_uses_site_role_or_explicit_override(server, kind, explicit):
    config = fl_conf.PrivacyConfiger.__new__(fl_conf.PrivacyConfiger)
    config.scopes, config.components, config.names_only, config.is_server = [], {}, False, server
    config.process_config_element(None, _node("scopes.#0", {}))
    config.process_config_element(None, _node("scopes.#0.name", "private"))
    filter_component = _PassFilter()
    config.build_component = Mock(return_value=filter_component)
    config.process_config_element(None, _node("scopes.#0." + kind + ".#0", {"direction": explicit}))
    filters = getattr(config.current_scope, kind)
    if explicit:
        assert filters == {FilterKey.IN: [filter_component], FilterKey.OUT: [filter_component]}
    else:
        inbound = server == (kind == "task_result_filters")
        assert filters[FilterKey.IN if inbound else FilterKey.OUT] == [filter_component]
        assert filters[FilterKey.OUT if inbound else FilterKey.IN] == []


@pytest.mark.parametrize(
    "identifier,error", [(None, "missing component id"), (42, "must be str"), ("duplicate", "duplicate component id")]
)
def test_privacy_component_registry_rejects_missing_invalid_and_duplicate_ids(identifier, error):
    config = fl_conf.PrivacyConfiger.__new__(fl_conf.PrivacyConfiger)
    config.names_only, config.components = False, {"duplicate": object()}
    config.build_component = Mock(return_value=object())
    with pytest.raises(ConfigError, match=error):
        config.process_config_element(None, _node("components.#0", {"id": identifier}))


def test_privacy_configured_component_is_available_on_final_manager():
    config = fl_conf.PrivacyConfiger.__new__(fl_conf.PrivacyConfiger)
    config.names_only, config.components, config.scopes, config.default_scope_name = False, {}, [], None
    component = object()
    config.build_component = Mock(return_value=component)
    config.process_config_element(None, _node("components.#0", {"id": "custom"}))
    config.finalize_config(None)
    assert config.privacy_manager.components == {"custom": component}
