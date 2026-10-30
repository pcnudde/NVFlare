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

import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.common.excepts import ConfigError
from nvflare.private.fed.app.server import runner_process as subject


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        set=["secure_train=true", "config_folder=config", "restore_snapshot=saved"],
        workspace=str(tmp_path),
        job_id="job-a",
        app_root="/app",
        root_url="tcp://root:1",
        parent_url="tcp://parent:2",
        token_signature="signature",
        ssid="ssid",
        host="localhost",
        port="1234",
    )
    workspace = Mock()
    workspace.get_app_custom_dir.return_value = "/custom"
    monkeypatch.setattr(subject, "Workspace", Mock(return_value=workspace))
    for name in (
        "download_workspace",
        "refresh_custom_dir_import_path",
        "set_stats_pool_config_for_job",
        "fobs_initialize",
        "security_init_for_job",
        "configure_logging",
        "register_ext_decomposers",
        "set_add_auth_headers_filters",
        "secure_log_traceback",
        "upload_results_on_shutdown",
    ):
        monkeypatch.setattr(subject, name, Mock())
    monkeypatch.setattr(subject.ConfigService, "get_str_var", Mock(return_value="decomposers"))
    logger = Mock()
    monkeypatch.setattr(subject, "get_script_logger", Mock(return_value=logger))
    config = Mock()
    server = Mock()
    config.deployer.create_fl_server.return_value = ({}, server)
    monkeypatch.setattr(subject, "FLServerStarterConfiger", Mock(return_value=config))
    app_runner = Mock()
    monkeypatch.setattr(subject, "ServerAppRunner", Mock(return_value=app_runner))
    monitor = Mock()
    stop = threading.Event()
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=Mock(return_value=monitor), Event=lambda: stop))
    shutdown = Mock()
    monkeypatch.setattr(subject, "shutdown_job_process_runtime", shutdown)
    monkeypatch.setattr(subject, "create_stats_pool_files_for_job", Mock(return_value=""))
    return SimpleNamespace(
        args=args,
        workspace=workspace,
        config=config,
        server=server,
        runner=app_runner,
        monitor=monitor,
        stop=stop,
        shutdown=shutdown,
        logger=logger,
    )


def test_server_runner_sets_auth_snapshot_and_orders_shutdown(runtime):
    env = runtime
    subject.main(env.args)
    assert env.args.server_config == "config/config_fed_server.json"
    assert env.args.config_folder == "config"
    assert env.args.snapshot == "saved"
    assert env.args.log_config is None
    env.config.configure.assert_called_once_with()
    env.server.snapshot_persistor.retrieve_run.assert_called_once_with("job-a")
    env.server.engine.set_cell.assert_called_once_with(env.server.create_job_cell.return_value)
    env.monitor.start.assert_called_once_with()
    env.runner.start_server_app.assert_called_once_with(
        env.workspace,
        env.args,
        "/app",
        "job-a",
        env.server.snapshot_persistor.retrieve_run.return_value,
        env.logger,
        env.args.set,
        event_handlers=env.config.handlers,
    )
    callbacks = env.shutdown.call_args.kwargs
    assert callbacks["stop_command_admission"] == env.server.command_agent.shutdown
    assert callbacks["wait_for_command_callbacks"] == env.server.command_agent.wait_for_callbacks
    assert callbacks["stop_cell"] == env.server.cell.stop
    callbacks["before_streaming_shutdown"]()
    subject.upload_results_on_shutdown.assert_called_once_with(env.args, True, log=env.logger)
    env.config.deployer.close.assert_called_once_with()
    assert env.stop.is_set()


def test_server_runner_closes_deployer_and_signals_monitor_after_app_failure(runtime):
    env = runtime
    env.runner.start_server_app.side_effect = RuntimeError("application failed")
    with pytest.raises(RuntimeError, match="application failed"):
        subject.main(env.args)
    env.shutdown.assert_called_once()
    env.config.deployer.close.assert_called_once_with()
    assert env.stop.is_set()


def test_server_runner_always_closes_deployer_after_shutdown_failure(runtime):
    env = runtime
    env.shutdown.side_effect = OSError("shutdown failed")
    with pytest.raises(OSError, match="shutdown failed"):
        subject.main(env.args)
    env.config.deployer.close.assert_called_once_with()
    assert env.stop.is_set()


def test_server_runner_reports_configuration_errors_before_launch(runtime):
    env = runtime
    env.config.configure.side_effect = ConfigError("bad configuration")
    with pytest.raises(ConfigError, match="bad configuration"):
        subject.main(env.args)
    env.server.create_job_cell.assert_not_called()
    assert "bad configuration" in env.logger.exception.call_args.args[0]
    subject.secure_log_traceback.assert_called_once_with(env.logger)
