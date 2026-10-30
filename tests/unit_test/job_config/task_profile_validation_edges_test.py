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

import pytest

from nvflare.job_config import task_execution as subject


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "5"])
def test_task_timeouts_reject_ambiguous_nonfinite_or_negative_values(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        subject._timeout("timeout", value)


@pytest.mark.parametrize("value", [True, -1, 0.5, "1"])
def test_task_gpu_resource_counts_require_nonnegative_integers(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        subject.validate_task_resources(
            {"launcher_spec": {"site": {"slurm": {"gpus_per_node": value}}}}, "site", allow_gpu=True
        )


def test_nested_plain_task_constructor_dictionaries_are_retained_but_components_are_rejected():
    plain = {"args": {"options": {"nested": {"weights": [1, 2]}}}}
    assert subject.task_constructor_args(plain) == plain["args"]
    with pytest.raises(ValueError, match="component args must be a dict"):
        subject.task_constructor_args({"args": []})
    with pytest.raises(ValueError, match="nested components"):
        subject.task_constructor_args(
            {"args": {"options": {"path": "nvflare.app_common.filters.exclude_vars.ExcludeVars"}}}
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"executors": []}, "requires client executors"),
        ({"components": {}}, "must be a list"),
        ({"components": [{"id": ""}]}, "nonempty string IDs"),
        ({"components": [{"id": "same"}, {"id": "same"}]}, "duplicate component ID"),
        ({"executors": [{}]}, "component specification"),
        ({"executors": [{"executor": {"path": subject.TASK_EXECUTOR_PATH}}]}, "framework-managed"),
    ],
)
def test_task_profile_rejects_incomplete_component_graph_before_worker_materialization(changes, match):
    config = {
        "execution_lifetime": "task",
        "components": [],
        "executors": [
            {
                "tasks": ["train"],
                "executor": {"path": subject.CLIENT_API_EXECUTOR_PATH, "args": {"task_script_path": "train.py"}},
            }
        ],
    }
    config.update(changes)
    with pytest.raises(ValueError, match=match):
        subject.prepare_task_execution(config)
    assert config["execution_lifetime"] == "task"
    with pytest.raises(TypeError, match="client config must be a dict"):
        subject.prepare_task_execution([])
