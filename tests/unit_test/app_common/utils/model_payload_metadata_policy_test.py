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

from types import SimpleNamespace

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.abstract.fl_model import FLModel, MetaKey, ParamsType
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


def test_model_conversion_retains_validation_job_and_site_headers_and_initial_metrics():
    model = FLModel(params={"w": [1]}, metrics={"accuracy": 0.8}, meta={MetaKey.VALIDATE_TYPE: "before_train"})
    model.params_type = None
    message = FLModelUtils.to_shareable(model)
    message.set_header(MetaKey.JOB_ID, "job")
    message.set_header(MetaKey.SITE_NAME, "site")
    restored = FLModelUtils.from_shareable(message)
    assert restored.params_type == ParamsType.FULL and restored.params == model.params
    assert restored.metrics == {"accuracy": 0.8}
    assert restored.meta[MetaKey.JOB_ID] == "job" and restored.meta[MetaKey.SITE_NAME] == "site"
    assert message.get_header(AppConstants.VALIDATE_TYPE) == "before_train"


@pytest.mark.parametrize(
    "failure", ["empty", "invalid-params-type", "wrong-dxo-kind", "wrong-dxo-data", "wrong-base", "wrong-update"]
)
def test_model_conversion_rejects_invalid_envelope_and_update_contracts(failure):
    with pytest.raises((ValueError, RuntimeError)):
        if failure == "empty":
            FLModelUtils.to_shareable(FLModel())
        elif failure == "invalid-params-type":
            model = FLModel(params={})
            model.params_type = SimpleNamespace(value="invalid")
            FLModelUtils.to_shareable(model)
        elif failure == "wrong-dxo-kind":
            FLModelUtils.from_dxo(DXO(DataKind.WEIGHTS, {}))
        elif failure == "wrong-dxo-data":
            message = DXO(DataKind.FL_MODEL, {})
            message.data = []
            FLModelUtils.from_dxo(message)
        elif failure == "wrong-base":
            FLModelUtils.update_model(FLModel(params_type=ParamsType.DIFF, params={}), FLModel(params={}))
        else:
            update = FLModel(params={})
            update.params_type = None
            FLModelUtils.update_model(FLModel(params={}), update)


def test_model_learnable_conversion_and_metadata_helpers_preserve_parameters():
    model = FLModelUtils.from_model_learnable(make_model_learnable({"w": [1]}, {"source": "saved"}))
    assert model.params == {"w": [1]} and model.params_type == ParamsType.FULL
    assert FLModelUtils.get_meta_prop(model, "missing", 4) == 4
    FLModelUtils.set_meta_prop(model, MetaKey.CONFIGS, {"epochs": 3})
    assert FLModelUtils.get_configs(model) == {"epochs": 3}
    model.meta = None
    assert FLModelUtils.get_meta_prop(model, "missing", 4) == 4
