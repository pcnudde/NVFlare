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

from nvflare.edge.tools.edge_fed_buff_recipe import (
    DeviceManagerConfig,
    EdgeFedBuffRecipe,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.job_config.api import FedJob


def recipe(**kwargs):
    values = dict(
        job_name="edge-contract",
        model={"class_path": "torch.nn.Identity"},
        model_manager_config=ModelManagerConfig(),
        device_manager_config=DeviceManagerConfig(),
    )
    values.update(kwargs)
    return EdgeFedBuffRecipe(**values)


@pytest.mark.parametrize(
    "kind,options,reason",
    [
        (ModelManagerConfig, {"max_model_version": 1}, "at least 2"),
        (DeviceManagerConfig, {"min_hole_to_fill": 3, "device_selection_size": 2}, "smaller than or equal"),
    ],
)
def test_edge_manager_config_rejects_impossible_round_or_sampling_budget(kind, options, reason):
    with pytest.raises(ValueError, match=reason):
        kind(**options)


@pytest.mark.parametrize("timeout", [0, -1])
def test_recipe_requires_positive_finite_pool_wait_budget_when_supplied(timeout):
    with pytest.raises(ValueError, match="positive number"):
        recipe(device_wait_timeout=timeout)


def test_recipe_rejects_missing_custom_sources_and_update_threshold_larger_than_sample(tmp_path):
    with pytest.raises(ValueError, match="not a valid directory"):
        recipe(custom_source_root=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="never have enough updates"):
        recipe(
            model_manager_config=ModelManagerConfig(num_updates_for_model=3),
            device_manager_config=DeviceManagerConfig(device_selection_size=2),
        )


def test_environment_simulation_override_is_typed_and_configures_only_edge_jobs(monkeypatch):
    item = recipe()
    with pytest.raises(ValueError, match="expect SimulationConfig"):
        item.process_env(SimpleNamespace(get_extra_prop=lambda key: "invalid"))
    config = SimulationConfig(task_processor=None, num_devices=4, num_workers=2, job_timeout=3)
    env = SimpleNamespace(get_extra_prop=lambda key: config)
    real_job = item._job
    configure = Mock()
    monkeypatch.setattr(real_job, "configure_simulation", configure)
    item.process_env(env)
    configure.assert_called_once_with(None, 3, 4, 2)
    item._job = FedJob("ordinary")
    with pytest.raises(TypeError, match="expected recipe job to be an EdgeJob"):
        item.process_env(env)
