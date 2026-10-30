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

from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator


def test_weight_diff_without_global_base_panics_and_cannot_create_new_model(monkeypatch):
    generator, context = FullModelShareableGenerator(), FLContext()
    panic = Mock()
    monkeypatch.setattr(generator, "system_panic", panic)
    assert generator.shareable_to_learnable(DXO(DataKind.WEIGHT_DIFF, {"w": 2}).to_shareable(), context) is None
    assert "No global base model" in panic.call_args.kwargs["reason"]
    assert context.get_prop(AppConstants.GLOBAL_MODEL) is None


def test_empty_full_weights_preserve_existing_parameters_and_replace_metadata():
    generator, context = FullModelShareableGenerator(), FLContext()
    model = make_model_learnable({"w": 1}, {"source": "old"})
    context.set_prop(AppConstants.GLOBAL_MODEL, model)
    result = generator.shareable_to_learnable(DXO(DataKind.WEIGHTS, {}, {"source": "new"}).to_shareable(), context)
    assert result is model
    assert result[ModelLearnableKey.WEIGHTS] == {"w": 1}
    assert result[ModelLearnableKey.META] == {"source": "new"}


def test_full_weights_create_a_model_when_no_global_model_exists():
    result = FullModelShareableGenerator().shareable_to_learnable(
        DXO(DataKind.WEIGHTS, {"w": 3}).to_shareable(), FLContext()
    )
    assert result[ModelLearnableKey.WEIGHTS] == {"w": 3}


@pytest.mark.parametrize(
    "value,error", [(object(), TypeError), (DXO(DataKind.METRICS, {"m": 1}).to_shareable(), ValueError)]
)
def test_unrecognized_model_payload_is_rejected_without_mutating_global_model(value, error):
    context = FLContext()
    model = make_model_learnable({"w": 1}, {})
    context.set_prop(AppConstants.GLOBAL_MODEL, model)
    with pytest.raises(error):
        FullModelShareableGenerator().shareable_to_learnable(value, context)
    assert model[ModelLearnableKey.WEIGHTS] == {"w": 1}
