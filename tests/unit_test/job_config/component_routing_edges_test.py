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

import numpy as np
import pytest

from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.filters.exclude_vars import ExcludeVars
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.api import ClientApp, FedApp, FedJob, JobCtx, ServerApp
from nvflare.job_config.defs import JobTargetType
from nvflare.job_config.fed_job_config import FedJobConfig


def test_app_component_identity_reuse_and_numeric_suffixes_keep_object_references_distinct():
    app = ClientApp()
    first, second, third = FLComponent(), FLComponent(), FLComponent()
    assert app.add_component(first, "layer2") == "layer2"
    assert app.add_component(first, "ignored") == "layer2"
    assert app.add_component(second, "layer2") == "layer3"
    assert app.add_component(third, "layer2") == "layer4"
    assert app.app_config.components == {"layer2": first, "layer3": second, "layer4": third}
    executor = ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py")
    app.add_executor(executor)
    assert app.app_config.executors[0].tasks == ["*"]


def test_job_component_reference_closure_and_custom_context_id_are_registered_once():
    class References(FLComponent):
        def __init__(self, dependency):
            super().__init__()
            self.dependency = dependency

    job = FedJob("refs")
    dependency = FLComponent()
    reference = job.as_id(dependency)
    parent = References(reference)
    job.to_server({"task_check_period": 0.1})
    ctx = JobCtx(parent, "server", "parent")
    assert job.add_component(None, parent, ctx) == "parent"
    components = job._deploy_map["server"].app_config.components
    assert components["parent"] is parent and components[reference] is dependency
    assert not job._components


@pytest.mark.parametrize("target", ["server", "site"])
def test_missing_job_context_target_reports_required_app_instead_of_silently_creating_it(target):
    job = FedJob("missing")
    with pytest.raises(RuntimeError, match="No app found.*missing"):
        job.add_params({"x": 1}, JobCtx(None, target, "component"))
    assert job._deploy_map == {}


@pytest.mark.parametrize("target,expected", [("server", JobTargetType.CLIENT), ("site", JobTargetType.SERVER)])
def test_target_restricted_components_are_rejected_before_component_registration(target, expected):
    job = FedJob("restricted")
    component = SimpleNamespace(get_job_target_type=lambda: expected)
    with pytest.raises(ValueError, match="only be assigned"):
        job.to(component, target)
    assert not job._deploy_map[target].app_config.components


def test_job_add_programmer_parameters_and_config_files_targets_the_existing_client_app(tmp_path):
    job = FedJob("files")
    job.to({"task_check_period": 0.1}, "site")
    ctx = JobCtx(None, "site", "component")
    source = tmp_path / "client.json"
    source.write_text('{"value": 3}')
    job.add_file_source(str(source), "nested", "config", ctx)
    job.add_params({"custom": "value"}, ctx)
    app = job._deploy_map["site"].app_config
    assert app.additional_params["custom"] == "value"
    assert app.file_sources == [(str(source), "nested", "config")]
    controller = FedAvg(num_clients=1, num_rounds=1)
    job.add_controller(controller, JobCtx(controller, "site", "local-controller"))
    assert app.components["local-controller"] is controller
    with pytest.raises(ValueError, match="valid `filter_type`"):
        job.add_filter(ExcludeVars("secret"), "wrong", ["train"], ctx)


def test_job_rejects_cross_side_roles_and_invalid_app_wrappers_before_export(tmp_path):
    job = FedJob("roles")
    with pytest.raises(ValueError, match="for Executors"):
        job.to_server(ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py"))
    with pytest.raises(ValueError, match="for Controllers"):
        job.to_clients(FedAvg(num_clients=1, num_rounds=1))
    with pytest.raises(ValueError, match="not allowed"):
        job.to(ClientApp(), "site")
    with pytest.raises(ValueError, match="client_app.*ClientApp"):
        job._set_all_app(object(), ServerApp())
    with pytest.raises(ValueError, match="server_app.*ServerApp"):
        job._set_all_app(ClientApp(), object())
    with pytest.raises(ValueError, match="FedApp"):
        job._set_site_app(object(), "site")
    with pytest.raises(ValueError, match="ClientAppConfig.*ServerAppConfig"):
        job._set_site_app(FedApp(object()), "site")
    job.to_clients({"task_check_period": 0.1})
    with pytest.raises(ValueError, match="Missing server components"):
        job.export_job(str(tmp_path))


def test_config_export_copies_directory_sources_and_preserves_filter_constructor_options(tmp_path):
    config = FedJobConfig("config", 1)
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested.json").write_text('{"key": 3}')
    output, custom = tmp_path / "config", tmp_path / "custom"
    output.mkdir()
    config._copy_file_sources(str(output), str(custom), [(str(source), "settings", "config")])
    assert (output / "settings" / "nested.json").read_text() == '{"key": 3}'
    filters = config._get_filters([ExcludeVars(["secret"])], str(custom))
    assert filters[0]["path"].endswith("ExcludeVars")
    assert filters[0]["args"]["exclude_vars"] == ["secret"]
    assert config._values_differ(np.array([1]), np.array([1]))

    class Uncomparable:
        def __ne__(self, other):
            raise ValueError("comparison is undefined")

    assert config._values_differ(Uncomparable(), Uncomparable())
