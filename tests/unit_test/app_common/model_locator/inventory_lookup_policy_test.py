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

from nvflare.apis.dxo import DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.model_locator.list_model_locator import ListModelLocator


def test_named_inventory_resolves_persistor_location_and_preserves_model_metadata():
    persistor = Mock(spec=ModelPersistor)
    persistor.get_model_from_location = Mock(return_value=make_model_learnable({"w": [1, 2]}, {"epoch": 3}))
    engine = Mock()
    engine.get_component.return_value = persistor
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    locator = ListModelLocator("models", {"best": "checkpoints/best"})
    locator.handle_event(EventType.START_RUN, ctx)
    assert locator.get_model_names(ctx) == ["best"]
    result = locator.locate_model("best", ctx)
    assert result.data_kind == DataKind.WEIGHTS
    assert result.data == {"w": [1, 2]}
    assert result.meta == {"epoch": 3}
    persistor.get_model_from_location.assert_called_once_with("checkpoints/best", ctx)
    with pytest.raises(ValueError, match="does not contain"):
        locator.locate_model("missing", ctx)


@pytest.mark.parametrize("component", [None, object()])
def test_model_locator_requires_model_persistor_component(component):
    engine = Mock()
    engine.get_component.return_value = component
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    with pytest.raises(ValueError, match="persistor_id component"):
        ListModelLocator("models", {}).handle_event(EventType.START_RUN, ctx)
