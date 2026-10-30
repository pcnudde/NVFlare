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

import pytest

from nvflare.app_common.response_processors.global_weights_initializer import WeightMethod
from nvflare.app_common.workflows.initialize_global_weights import InitializeGlobalWeights


@pytest.mark.parametrize("clients,expected", [(None, None), ("site", ["site"]), (["one", "two"], ["one", "two"])])
def test_global_weights_initializer_routes_only_to_configured_clients(clients, expected):
    controller = InitializeGlobalWeights(
        weights_client_name=clients,
        task_name="weights",
        min_responses_required=2,
        wait_time_after_min_received=3,
        task_timeout=7,
        weights_prop_name="initial",
    )
    assert controller.clients == expected
    assert controller.task_name == "weights" and controller.timeout == 7
    assert controller.min_responses_required == 2 and controller.wait_time_after_min_received == 3
    assert controller.processor.weights_prop_name == "initial"
    assert controller.processor.weight_method == WeightMethod.FIRST


def test_specific_weight_provider_uses_matching_processor_selection_policy():
    controller = InitializeGlobalWeights(weight_method=WeightMethod.CLIENT, weights_client_name="chosen")
    assert controller.clients == ["chosen"]
    assert controller.processor.client_name == "chosen" and controller.processor.weight_method == WeightMethod.CLIENT
