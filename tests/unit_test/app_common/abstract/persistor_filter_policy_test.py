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

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.abstract.persistor_filter import PersistorFilter


class MemoryPersistor(ModelPersistor):
    def __init__(self, filter_id=None):
        super().__init__(filter_id)
        self.model = make_model_learnable({"w": [1]}, {})
        self.saved = None

    def load_model(self, fl_ctx):
        return self.model

    def save_model(self, model, fl_ctx):
        self.saved = model

    def get_model(self, model_file, fl_ctx):
        assert model_file == "best"
        return self.model


def test_persistor_invokes_filter_hooks_in_order_and_saves_transformed_model():
    persistor = MemoryPersistor("filter")
    model = make_model_learnable({"w": [9]}, {})
    filter_component = Mock(spec=PersistorFilter)
    filter_component.process_post_load.return_value = model
    filter_component.process_pre_save.return_value = model
    filter_component.process_post_get.return_value = model
    engine = Mock()
    engine.get_component.return_value = filter_component
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    assert persistor.load(ctx) is model
    assert persistor.get("best", ctx) is model
    persistor.save(persistor.model, ctx)
    assert persistor.saved is model
    filter_component.process_post_load.assert_called_once_with(learnable=persistor.model, fl_ctx=ctx)
    filter_component.process_pre_save.assert_called_once_with(learnable=persistor.model, fl_ctx=ctx)
    filter_component.process_post_save.assert_called_once_with(learnable=model, fl_ctx=ctx)
    filter_component.process_post_get.assert_called_once_with(learnable=persistor.model, fl_ctx=ctx)
    passthrough = PersistorFilter()
    for hook in [
        passthrough.process_pre_save,
        passthrough.process_post_save,
        passthrough.process_post_load,
        passthrough.process_post_get,
    ]:
        assert hook(model, ctx) is model


@pytest.mark.parametrize("operation", ["load", "save", "get"])
def test_persistor_rejects_invalid_filter_before_persisting(operation):
    persistor = MemoryPersistor("filter")
    engine = Mock()
    engine.get_component.return_value = object()
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    args = [ctx] if operation == "load" else [persistor.model if operation == "save" else "best", ctx]
    with pytest.raises(ValueError, match="PersistorFilter"):
        getattr(persistor, operation)(*args)
    assert persistor.saved is None
