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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.app.server import server_train


@pytest.fixture
def startup(monkeypatch, tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(workspace=str(tmp_path), set=["secure_train=true", "config_folder=custom"])
    context = FLContext()
    services = Mock(status=server_train.ServerStatus.SHUTDOWN)
    services.engine.new_context.return_value = context
    deployer = Mock(secure_train=True, server_config={"server": "configuration"})
    deployer.deploy.return_value = services
    server_config = {"service": {"target": "host:8002"}}
    config = Mock(
        deployer=deployer,
        cmd_vars={"secure_train": True},
        site_org="Org",
        app_validator=object(),
        config_data={"servers": [server_config]},
    )
    monkeypatch.setattr(server_train, "FLServerStarterConfiger", Mock(return_value=config))
    admin = Mock()
    monkeypatch.setattr(server_train, "create_admin_server", Mock(return_value=admin))
    for name in ("fobs_initialize", "configure_logging", "security_init"):
        monkeypatch.setattr(server_train, name, Mock())
    monkeypatch.setattr(server_train, "create_privacy_manager", Mock(return_value=None))
    monkeypatch.setattr(server_train.PrivacyService, "initialize", Mock())
    monkeypatch.setattr(
        server_train, "time", SimpleNamespace(sleep=Mock(side_effect=AssertionError("unexpected polling")))
    )
    return SimpleNamespace(
        args=args,
        context=context,
        services=services,
        deployer=deployer,
        config=config,
        admin=admin,
        server_config=server_config,
    )


def test_server_bootstrap_preserves_config_and_closes_deployer_admin_and_engine(startup):
    server_train.main(startup.args)
    assert startup.args.server_config == "custom/" + JobConstants.SERVER_JOB_CONFIG
    assert startup.args.config_folder == "custom" and startup.args.job_id is None and startup.args.log_config is None
    startup.config.configure.assert_called_once_with()
    startup.deployer.deploy.assert_called_once_with(startup.args)
    kwargs = server_train.create_admin_server.call_args.kwargs
    assert kwargs == {"server_conf": startup.server_config, "args": startup.args}
    startup.admin.start.assert_called_once_with()
    startup.services.set_admin_server.assert_called_once_with(startup.admin)
    assert startup.context.get_prop(FLContextKey.SERVER_CONFIG) == {"server": "configuration"}
    assert startup.context.get_prop(FLContextKey.SECURE_MODE) is True
    startup.deployer.close.assert_called_once_with()
    startup.admin.stop.assert_called_once_with()
    startup.services.engine.close.assert_called_once_with()
    assert server_train.security_init.call_args.kwargs["site_org"] == "Org"


def test_server_default_config_and_stale_restart_marker_cleanup(startup, tmp_path):
    startup.args.set = []
    for name in ("restart.fl", "shutdown.fl"):
        (tmp_path / name).write_text("stale")
    server_train.main(startup.args)
    assert startup.args.server_config == JobConstants.SERVER_JOB_CONFIG
    assert not (tmp_path / "restart.fl").exists() and not (tmp_path / "shutdown.fl").exists()


def test_deployment_error_closes_deployer_before_propagating(startup):
    startup.deployer.deploy.side_effect = RuntimeError("deployment rejected")
    with pytest.raises(RuntimeError, match="deployment rejected"):
        server_train.main(startup.args)
    startup.deployer.close.assert_called_once_with()
    startup.admin.start.assert_not_called()


def test_server_configuration_error_is_logged_and_propagated(startup, caplog):
    startup.config.configure.side_effect = server_train.ConfigError("configuration rejected")
    with pytest.raises(server_train.ConfigError, match="configuration rejected"):
        server_train.main(startup.args)
    assert "ConfigError" in caplog.text
    startup.deployer.deploy.assert_not_called()


def test_server_arguments_require_startup_configuration(monkeypatch):
    monkeypatch.setattr(
        server_train.sys,
        "argv",
        ["server", "--workspace", "/workspace", "--fed_server", "server.json", "--set", "secure_train=true"],
    )
    args = server_train.parse_arguments()
    assert args.workspace == "/workspace" and args.fed_server == "server.json" and args.set == ["secure_train=true"]
    monkeypatch.setattr(server_train.sys, "argv", ["server", "--workspace", "/workspace"])
    with pytest.raises(SystemExit):
        server_train.parse_arguments()
