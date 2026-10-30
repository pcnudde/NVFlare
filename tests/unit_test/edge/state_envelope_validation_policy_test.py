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

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.shareable import Shareable
from nvflare.edge.mud import BaseState, PropKey, StateUpdateReport


@pytest.mark.parametrize(
    "model_version, model, selection_version, selection",
    [
        (1, None, 0, {}),
        (0, DXO(DataKind.WEIGHTS, {}), 0, {}),
        (0, None, 1, {}),
        (0, None, 0, {"device": 1}),
    ],
)
def test_versions_must_agree_with_model_and_selection_presence(model_version, model, selection_version, selection):
    with pytest.raises(ValueError, match="invalid .*version"):
        BaseState(model_version, model, selection_version, selection)


@pytest.mark.parametrize(
    "headers, values, message",
    [
        ({PropKey.MODEL_VERSION: "one"}, {}, "model_version must be int"),
        ({PropKey.MODEL_VERSION: 1}, {PropKey.MODEL: {}}, "model must be DXO"),
        (
            {PropKey.MODEL_VERSION: 0, PropKey.DEVICE_SELECTION_VERSION: "one"},
            {},
            "device_selection_version must be int",
        ),
        (
            {PropKey.MODEL_VERSION: 0, PropKey.DEVICE_SELECTION_VERSION: 1},
            {PropKey.DEVICE_SELECTION: []},
            "device_selection must be dict",
        ),
    ],
)
def test_wire_state_rejects_wrong_header_or_payload_types(headers, values, message):
    data = Shareable(values)
    for key, value in headers.items():
        data.set_header(key, value)
    with pytest.raises(ValueError, match=message):
        BaseState.from_shareable(data)


@pytest.mark.parametrize("key", [PropKey.MODEL_UPDATES, PropKey.DEVICES])
def test_update_report_rejects_mistyped_member_collections(key):
    data = Shareable({key: ["invalid"]})
    data.set_header(PropKey.MODEL_VERSION, 0)
    data.set_header(PropKey.DEVICE_SELECTION_VERSION, 0)
    with pytest.raises(ValueError, match=f"prop {key} must be dict"):
        StateUpdateReport.from_shareable(data)


def test_empty_base_state_roundtrip_and_device_selection_do_not_enable_training():
    original = BaseState(0, None, 0, {})
    restored = BaseState.from_shareable(original.to_shareable())
    assert restored.model is None
    assert restored.device_selection == {}
    assert restored.is_device_selected("device", 0) == (False, 0)
    assert restored.get_converted_model("torch") is None
    model = {"weights": [1]}
    restored.set_converted_model(model, "torch")
    assert restored.get_converted_model("torch") is model
