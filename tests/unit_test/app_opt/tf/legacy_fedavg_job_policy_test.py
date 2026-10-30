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

tf = pytest.importorskip("tensorflow")

from nvflare.app_common.workflows.fedavg import FedAvg  # noqa: E402
from nvflare.app_opt.tf.job_config.fed_avg import FedAvgJob  # noqa: E402


def test_tensorflow_legacy_job_routes_round_count_persistence_and_loss_selection_to_server():
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Flatten()])
    job = FedAvgJob(
        model,
        n_clients=3,
        num_rounds=4,
        name="tiny",
        min_clients=2,
        mandatory_clients=["site-1"],
        key_metric="loss",
        key_metric_mode="min",
    )
    app = job._deploy_map["server"].app_config
    controllers = [workflow.controller for workflow in app.workflows]
    assert len(controllers) == 1 and isinstance(controllers[0], FedAvg)
    assert controllers[0].num_clients == 3 and controllers[0].num_rounds == 4
    assert controllers[0]._persistor_id == job.comp_ids["persistor_id"]
    assert job.comp_ids["persistor_id"] in app.components
    assert job.job.min_clients == 2
    assert job.job.mandatory_clients == ["site-1"]
    with pytest.raises(ValueError, match="Expected initial model"):
        FedAvgJob(object(), n_clients=1, num_rounds=1)
