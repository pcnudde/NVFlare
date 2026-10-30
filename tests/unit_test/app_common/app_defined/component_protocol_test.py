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

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_defined.aggregator import AppDefinedAggregator
from nvflare.app_common.app_defined.component_base import ComponentBase
from nvflare.app_common.app_defined.model_persistor import AppDefinedModelPersistor
from nvflare.app_common.app_defined.shareable_generator import AppDefinedShareableGenerator
from nvflare.app_common.app_event_type import AppEventType


class CustomAggregator(AppDefinedAggregator):
    def reset(self):
        self.updates = []

    def processing_training_result(self, client_name, trained_weights, trained_meta):
        if not trained_meta["valid"]:
            return False
        self.updates.append((client_name, trained_weights))
        return True

    def aggregate_training_result(self):
        return [sum(row[i] for _, row in self.updates) / len(self.updates) for i in range(2)], {
            "clients": [name for name, _ in self.updates]
        }


class CustomGenerator(AppDefinedShareableGenerator):
    def model_to_trainable(self, model_obj):
        return list(model_obj), {"offset": 1}

    def update_model(self, model_obj, training_result, meta):
        return [base + value + meta["offset"] for base, value in zip(model_obj, training_result)]


class MemoryPersistor(AppDefinedModelPersistor):
    def read_model(self):
        return self.model

    def write_model(self, model_obj):
        self.model = model_obj


def test_custom_aggregation_tracks_round_base_and_client_identity_before_emitting_app_defined_dxo():
    ctx = FLContext()
    ctx.set_prop(AppConstants.CURRENT_ROUND, 3)
    ctx.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable([1, 2], {}))
    aggregator = CustomAggregator()
    aggregator.handle_event(AppEventType.ROUND_STARTED, ctx)
    assert aggregator.current_round == 3 and aggregator.base_model_obj == [1, 2]
    for name, values in (("one", [2, 4]), ("two", [4, 6])):
        peer = FLContext()
        peer.set_prop(ReservedKey.IDENTITY_NAME, name)
        ctx.set_peer_context(peer)
        assert aggregator.accept(DXO(DataKind.APP_DEFINED, values, {"valid": True}).to_shareable(), ctx)
    assert not aggregator.accept(DXO(DataKind.APP_DEFINED, [99, 99], {"valid": False}).to_shareable(), ctx)
    result = from_shareable(aggregator.aggregate(ctx))
    assert result.data_kind == DataKind.APP_DEFINED
    assert result.data == [3, 5] and result.meta == {"clients": ["one", "two"]}
    aggregator.handle_event(AppEventType.ROUND_STARTED, ctx)
    assert aggregator.updates == []


def test_custom_generator_carries_round_and_training_metadata_into_model_update():
    generator = CustomGenerator()
    ctx = FLContext()
    ctx.set_prop(AppConstants.CURRENT_ROUND, 4)
    base = make_model_learnable([1, 2], {"unused": 3})
    request = generator.learnable_to_shareable(base, ctx)
    dxo = from_shareable(request)
    assert dxo.data_kind == DataKind.APP_DEFINED and dxo.data == [1, 2]
    assert dxo.meta == {"offset": 1}
    assert generator.current_round == 4 and generator.fl_ctx is ctx
    ctx.set_prop(AppConstants.GLOBAL_MODEL, base)
    result = generator.shareable_to_learnable(DXO(DataKind.APP_DEFINED, [3, 5], {"offset": 1}).to_shareable(), ctx)
    assert result[ModelLearnableKey.WEIGHTS] == [5, 8]
    assert result[ModelLearnableKey.META] == {}
    assert base[ModelLearnableKey.WEIGHTS] == [1, 2]


def test_generator_requires_a_typed_base_model_before_applying_updates():
    generator = CustomGenerator()
    generator.system_panic = Mock()
    ctx = FLContext()
    update = DXO(DataKind.APP_DEFINED, [1, 2], {"offset": 1}).to_shareable()
    assert generator.shareable_to_learnable(update, ctx) is None
    generator.system_panic.assert_called_once()
    ctx.set_prop(AppConstants.GLOBAL_MODEL, {"weights": [1, 2]})
    with pytest.raises(ValueError, match="ModelLearnable"):
        generator.shareable_to_learnable(update, ctx)


def test_custom_persistor_saves_only_model_object_and_restores_standard_learnable_envelope():
    persistor = MemoryPersistor()
    persistor.model = [1, 2]
    ctx = FLContext()
    restored = persistor.load_model(ctx)
    assert restored == make_model_learnable([1, 2], {})
    persistor.save_model(make_model_learnable([3, 4], {"metadata": "not-model"}), ctx)
    assert persistor.model == [3, 4]
    assert persistor.fl_ctx is ctx


@pytest.mark.parametrize("level", ["debug", "info", "error", "warning", "exception", "critical"])
def test_component_logging_wrappers_bind_current_context(monkeypatch, level):
    component = ComponentBase()
    ctx = FLContext()
    component.fl_ctx = ctx
    log = Mock()
    monkeypatch.setattr(component, "log_" + level, log)
    getattr(component, level)("training status")
    log.assert_called_once_with(ctx, "training status")
