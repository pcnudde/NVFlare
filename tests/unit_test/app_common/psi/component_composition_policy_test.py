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
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.task_handler import TaskHandler
from nvflare.app_common.psi.dh_psi.dh_psi_controller import DhPSIController
from nvflare.app_common.psi.dh_psi.dh_psi_workflow import DhPSIWorkFlow
from nvflare.app_common.psi.psi_executor import PSIExecutor
from nvflare.app_common.psi.psi_spec import PSI
from nvflare.app_common.psi.psi_writer_spec import PSIWriter


class InlinePSI(PSI):
    def load_items(self):
        return ["shared", "local"]


def test_psi_component_loads_writer_and_saves_intersection_under_original_context():
    context, engine = FLContext(), Mock()
    context.set_prop(ReservedKey.ENGINE, engine)
    writer = Mock(spec=PSIWriter)
    engine.get_component.return_value = writer
    psi = InlinePSI("writer")
    assert psi.get_intersection() is None
    psi.initialize(context)
    psi.save(["shared"])
    assert psi.get_intersection() == ["shared"]
    writer.save.assert_called_once_with(intersection=["shared"], overwrite_existing=True, fl_ctx=context)
    engine.get_component.assert_called_once_with("writer")


@pytest.mark.parametrize("component", [None, object(), "valid"])
def test_psi_executor_requires_typed_initialized_handler(component):
    engine, context = Mock(), FLContext()
    context.set_prop(ReservedKey.ENGINE, engine)
    handler = Mock(spec=TaskHandler) if component == "valid" else component
    engine.get_component.return_value = handler
    executor = PSIExecutor("handler")
    assert executor.get_data_kind() == DataKind.PSI
    if component is None:
        with pytest.raises(NotImplementedError):
            executor.get_task_handler(context)
    elif component == "valid":
        assert executor.get_task_handler(context) is handler
        handler.initialize.assert_called_once_with(context)
    else:
        with pytest.raises(TypeError):
            executor.get_task_handler(context)


def test_dh_controller_constructs_a_workflow_bound_to_itself():
    controller, context = DhPSIController(), FLContext()
    workflow = controller.load_psi_workflow(context)
    assert isinstance(workflow, DhPSIWorkFlow)
    assert workflow.controller is controller and workflow.fl_ctx is context


def test_dh_recipe_exports_consistent_task_component_and_writer_ids(tmp_path):
    pytest.importorskip("private_set_intersection")
    from nvflare.app_common.psi.recipes.dh_psi import DhPSIRecipe

    recipe = DhPSIRecipe(
        name="intersection", min_clients=2, local_psi=InlinePSI("writer"), output_path="result/items.txt"
    )
    recipe._job.export_job(str(tmp_path))
    client_config = next((tmp_path / "intersection").rglob("config_fed_client.json"))
    payload = json.loads(client_config.read_text())
    components = {item["id"]: item for item in payload["components"]}
    assert components["dh_psi"]["args"]["local_psi_id"] == "local_psi"
    assert components["local_psi"]["args"]["psi_writer_id"] == "writer"
    assert components["writer"]["args"]["output_path"] == "result/items.txt"
    assert payload["executors"][0]["tasks"] == ["PSI"]
    assert payload["executors"][0]["executor"]["args"]["psi_algo_id"] == "dh_psi"
