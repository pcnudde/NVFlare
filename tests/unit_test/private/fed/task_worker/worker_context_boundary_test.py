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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.statistics_spec import Statistics
from nvflare.private.fed.task_worker import worker as module
from nvflare.private.fed.task_worker.protocol import ContextProperty, TaskAttemptIdentity


@pytest.fixture
def env(monkeypatch):
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    bootstrap = SimpleNamespace(identity=identity, context_properties={}, components=[], executor={"path": "executor"})
    workspace = Mock()
    workspace.get_app_dir.return_value = "/workspace/app"
    workspace.get_root_dir.return_value = "/workspace"
    monkeypatch.setattr(module, "get_job_meta_from_workspace", Mock(return_value={}))
    monkeypatch.setattr(module, "_class_path", lambda config: config["path"])
    return SimpleNamespace(bootstrap=bootstrap, workspace=workspace, identity=identity)


@pytest.mark.parametrize("protected", [True, False])
def test_context_properties_cannot_replace_framework_fields_and_json_values_are_thawed(env, protected):
    name = FLContextKey.TASK_ID if protected else "application_settings"
    env.bootstrap.context_properties = {name: ContextProperty({"values": [1, 2]}, private=False, sticky=False)}
    if protected:
        with pytest.raises(ValueError, match="cannot override"):
            module._new_context(env.bootstrap, env.workspace)
    else:
        runtime, context = module._new_context(env.bootstrap, env.workspace)
        assert context.get_prop(name) == {"values": [1, 2]}
        assert context.get_all_public_props()[name] == {"values": [1, 2]}
        assert context.get_prop(FLContextKey.TASK_ID) == "task"
        assert runtime.new_context().get_prop(name) is None


@pytest.mark.parametrize("invalid", [True, False])
def test_statistics_component_is_type_checked_and_executor_adaptation_preserves_context(env, monkeypatch, invalid):
    env.bootstrap.components = [{"id": "statistics", "path": "generator", "args": {}}]
    generator = object() if invalid else Mock(spec=Statistics)
    monkeypatch.setattr(module, "instantiate_class", Mock(return_value=generator))
    executor = Mock(spec=Executor)
    executor.execute.return_value = Shareable({"result": 7})
    monkeypatch.setattr(module, "_build_executor", Mock(return_value=executor))
    captured = []

    def bind(context):
        captured.append(context)
        return lambda supplied: supplied

    if invalid:
        with pytest.raises(TypeError, match="must implement Statistics"):
            module._execute(env.bootstrap, Shareable(), env.workspace, context_binding=bind)
        assert captured == []
        executor.execute.assert_not_called()
    else:
        result, runtime = module._execute(env.bootstrap, Shareable(), env.workspace, context_binding=bind)
        assert isinstance(result, Shareable) and result["result"] == 7
        assert captured[0].get_engine().get_component("statistics") is generator
        assert executor.execute.call_args.args[2] is captured[0]
        runtime.raise_if_failed()


def test_finalizer_must_preserve_result_type_even_after_successful_execution(env, monkeypatch):
    executor = Mock(spec=Executor)
    executor.execute.return_value = Shareable()
    executor.finalize = Mock(
        side_effect=lambda context: context.set_prop(FLContextKey.TASK_RESULT, "invalid", private=True, sticky=False)
    )
    monkeypatch.setattr(module, "_build_executor", Mock(return_value=executor))
    with pytest.raises(TypeError, match="finalization must leave"):
        module._execute(env.bootstrap, Shareable(), env.workspace)
    executor.finalize.assert_called_once()


@pytest.mark.parametrize("close_error", [False, True])
def test_worker_registers_codecs_before_fake_cell_transport_and_commits_before_session_cleanup(
    env, monkeypatch, close_error, caplog
):
    bootstrap = env.bootstrap
    bootstrap.workspace_root, bootstrap.artifact_root = "/workspace", "/artifacts"
    bootstrap.max_payload_bytes = None
    bootstrap.executor = {"path": module.CLIENT_API_EXECUTOR_PATH, "args": {}}
    bootstrap.components = [{"path": module.DECOMPOSER_REGISTER_PATH, "args": {}}]
    bootstrap.cell_session = {"fake": "transport"}
    store, workspace, transport, runtime = Mock(), Mock(), Mock(), Mock()
    workspace.get_log_config_file_path.return_value = "logging.json"
    store.stage_checkpoint.return_value.to_dict.return_value = {"checkpoint": "validated"}
    runtime.completion_guard = Mock()
    order = Mock()
    order.attach_mock(transport.open, "open")
    order.attach_mock(store.commit_staged_result, "commit")
    order.attach_mock(transport.close, "close")
    if close_error:
        transport.close.side_effect = OSError("already closed")
    monkeypatch.setattr(module, "sys", SimpleNamespace(argv=["worker"], path=list(sys.path), platform=sys.platform))
    monkeypatch.setattr(module, "pop_credential_env", Mock())
    monkeypatch.setattr(module, "read_bootstrap", Mock(return_value=bootstrap))
    monkeypatch.setattr(module, "FileTaskArtifactStore", Mock(return_value=store))
    monkeypatch.setattr(module, "Workspace", Mock(return_value=workspace))
    monkeypatch.setattr(module, "AuditService", SimpleNamespace(get_auditor=lambda: object()))
    monkeypatch.setattr(module, "_new_context", Mock(return_value=(runtime, Mock())))
    monkeypatch.setattr(module, "_authorize_bootstrap", Mock())
    monkeypatch.setattr(module, "configure_logging", Mock())
    monkeypatch.setattr(module, "refresh_custom_dir_import_path", Mock())
    monkeypatch.setattr(module, "fobs_initialize", Mock())
    codecs = Mock()
    monkeypatch.setattr(module, "register_framework_decomposers", codecs)
    component = Mock()
    monkeypatch.setattr(module, "instantiate_class", component)
    monkeypatch.setattr(module, "CellWorkerTransport", Mock(return_value=transport))
    execute = Mock(return_value=(Shareable({"result": 7}), runtime))
    monkeypatch.setattr(module, "_execute", execute)
    bind = Mock()
    result = module.run_worker("bootstrap.json", context_binding=bind)
    assert result is store.commit_staged_result.return_value
    assert [call[0] for call in order.mock_calls] == ["open", "commit", "close"]
    codecs.assert_called_once_with(module.ExchangeFormat.RAW, module.ExchangeFormat.NUMPY)
    component.assert_called_once_with(module.DECOMPOSER_REGISTER_PATH, {})
    assert execute.call_args.kwargs["context_binding"].keywords == {
        "identity": env.identity,
        "store": store,
        "log": transport.log,
    }
    assert store.commit_staged_result.call_args.kwargs["diagnostics"]["checkpoint"] == {"checkpoint": "validated"}
    store.record_failure.assert_not_called()
    assert ("could not close the cell session" in caplog.text) is close_error
