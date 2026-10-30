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

from nvflare.apis.dxo import DataKind
from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.job_config.defs import FilterType
from nvflare.recipe import session_mgr, utils
from nvflare.recipe.prod_env import ProdEnv
from nvflare.recipe.sim_env import SimEnv
from nvflare.recipe.spec import Recipe


def test_session_manager_creates_explicit_sessions_for_status_abort_and_timeout_without_downloading(monkeypatch):
    session = Mock()
    session.get_job_status.return_value = "running"
    session.monitor_job.return_value = MonitorReturnCode.TIMEOUT
    factory = Mock(return_value=session)
    monkeypatch.setattr(session_mgr, "new_secure_session", factory)
    output = Mock()
    monkeypatch.setattr(session_mgr, "_print_output", output)
    manager = session_mgr.SessionManager({"username": "reader", "startup_kit_location": "/kit"})
    assert manager.get_job_status("job") == "running"
    manager.abort_job("job")
    assert manager.get_job_result("job", 2.5) is None
    assert factory.call_count == 3 and session.close.call_count == 3
    assert factory.call_args.kwargs == {"username": "reader", "startup_kit_location": "/kit"}
    session.get_job_status.assert_called_once_with("job")
    session.abort_job.assert_called_once_with("job")
    session.download_job_result.assert_not_called()
    assert any("timed out after 2.5" in str(call) for call in output.call_args_list)


def test_progress_stream_ignores_nonrecords_and_preserves_context_on_valid_log(monkeypatch):
    session = Mock()
    records = [
        [],
        {"fullName": 3},
        {"fullName": "program", "levelname": []},
        {
            "fullName": "program",
            "levelname": "INFO",
            "message": "round complete",
            "fl_ctx": "site",
            "nvflare_progress": True,
        },
    ]
    session.get_job_logs.return_value = {"logs": {"server": "\n".join(json.dumps(row) for row in records)}}
    output = Mock()
    monkeypatch.setattr(session_mgr, "_print_output", output)
    state = {"seen": set()}
    session_mgr._show_job_progress(session, "job", state)
    assert len(state["seen"]) == 1
    assert "site: round complete" in output.call_args.args[0]
    output.reset_mock()
    session_mgr._show_job_progress(session, "job", state)
    output.assert_not_called()
    session.get_job_logs.return_value = {"logs": {"server": ["not-text"]}}
    session_mgr._show_job_progress(session, "job", state)
    assert state["warned"]


def test_recipe_routes_filters_by_direction_and_rejects_nonpath_files_before_job_calls():
    job = SimpleNamespace(name="job", to_server=Mock())
    recipe = Recipe(job)
    recipe._add_to_client_apps = Mock()
    target = object()
    recipe.add_client_input_filter(target, clients=["site"], tasks=["train"])
    assert recipe._add_to_client_apps.call_args.kwargs == {
        "clients": ["site"],
        "filter_type": FilterType.TASK_DATA,
        "tasks": ["train"],
    }
    recipe.add_client_output_filter(target, tasks=["evaluate"])
    assert recipe._add_to_client_apps.call_args.kwargs["filter_type"] == FilterType.TASK_RESULT
    recipe.add_server_input_filter(target, tasks=["train"])
    assert job.to_server.call_args.kwargs["filter_type"] == FilterType.TASK_RESULT
    recipe.add_server_output_filter(target, tasks=["train"])
    assert job.to_server.call_args.kwargs["filter_type"] == FilterType.TASK_DATA
    for method in (recipe.add_client_file, recipe.add_server_file):
        with pytest.raises(TypeError, match="file_path must be a str"):
            method(3)
    assert recipe._get_full_class_name(3) == "int"
    assert recipe._get_full_class_name(SimpleNamespace()) == "types.SimpleNamespace"
    recipe.add_decomposers([])


@pytest.mark.parametrize(
    "framework,job,match",
    [
        (None, SimpleNamespace(_deploy_map={}), "does not have a 'framework'"),
        ("unsupported", SimpleNamespace(_deploy_map={}), "Currently supported"),
        ("pytorch", SimpleNamespace(_deploy_map={}, comp_ids={}), "no persistor_id"),
        ("pytorch", SimpleNamespace(_deploy_map={}), "does not have comp_ids"),
    ],
)
def test_cross_site_evaluation_requires_supported_framework_and_persistor_identity(framework, job, match):
    recipe = SimpleNamespace(_job=job, name="job")
    if framework is not None:
        if framework == "pytorch":
            pytest.importorskip("torch")
            from nvflare.fuel.utils.constants import FrameworkType

            framework = FrameworkType.PYTORCH
        recipe.framework = framework
    with pytest.raises(ValueError, match=match):
        utils.add_cross_site_evaluation(recipe)
    assert not getattr(recipe, "_cse_added", False)


def test_recipe_rejects_unsupported_aggregator_and_empty_site_configuration():
    with pytest.raises(ValueError, match="it declares expected_data_kind"):
        utils.validate_aggregator_data_kind(
            data_kind=DataKind.WEIGHTS,
            recipe_name="recipe",
            aggregator=SimpleNamespace(expected_data_kind=DataKind.METRICS),
        )
    with pytest.raises(ValueError, match="config must not be empty"):
        utils._validate_per_site_config_shape({})


@pytest.mark.parametrize(
    "kind,config,server,client,match",
    [
        ("unknown", {}, True, True, "Invalid tracking type"),
        ("mlflow", {"kw_args": []}, True, True, "kw_args must be a dict"),
        ("tensorboard", {}, False, False, "At least one"),
    ],
)
def test_tracking_configuration_is_rejected_before_job_mutation(kind, config, server, client, match):
    recipe = SimpleNamespace(name="job", _job=SimpleNamespace(name="job"))
    with pytest.raises((TypeError, ValueError), match=match):
        utils.add_experiment_tracking(recipe, kind, config, server_side=server, client_side=client)


def test_prod_environment_routes_job_controls_and_simulation_results_require_completed_workspace(tmp_path, monkeypatch):
    manager = Mock()
    manager.get_job_status.return_value = "running"
    manager.get_job_result.return_value = "/downloaded/job"
    env = ProdEnv(str(tmp_path))
    env._session_manager = manager
    assert env.get_job_status("job") == "running"
    env.abort_job("job")
    manager.abort_job.assert_called_once_with("job")
    assert env.get_job_result("job", 2.5) == "/downloaded/job"
    manager.get_job_result.assert_called_once_with("job", 2.5)
    with pytest.raises(ValueError, match="path does not exist"):
        ProdEnv(str(tmp_path / "missing"))
    sim = SimEnv(num_clients=1, workspace_root=str(tmp_path))
    monkeypatch.setattr(sim, "get_job_status", lambda job: "FINISHED:COMPLETED")
    assert sim.get_job_result("job") is None
    (tmp_path / "job").mkdir()
    assert sim.get_job_result("job") == str(tmp_path / "job")
    sim.workspace_root = None
    with pytest.raises(RuntimeError, match="workspace_root is None"):
        sim.get_job_result("job")
