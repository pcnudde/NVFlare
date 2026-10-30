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
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.np.np_model_persistor import NPModelPersistor


def test_numpy_persistor_prefers_saved_array_over_configured_initial_values(tmp_path):
    ctx = FLContext()
    ctx.set_prop(ReservedKey.RUN_NUM, "job")
    ctx.set_prop(
        ReservedKey.ENGINE,
        Mock(
            get_workspace=Mock(
                return_value=SimpleNamespace(
                    get_run_dir=lambda job: str(tmp_path), get_result_root=lambda job: str(tmp_path)
                )
            )
        ),
    )
    persistor = NPModelPersistor(model=[[1, 2]])
    first = persistor.load_model(ctx)
    np.testing.assert_array_equal(first["weights"][NPConstants.NUMPY_KEY], [[1, 2]])
    persistor.save_model(make_model_learnable({NPConstants.NUMPY_KEY: np.array([[3, 4]])}, {}), ctx)
    np.testing.assert_array_equal(persistor.load_model(ctx)["weights"][NPConstants.NUMPY_KEY], [[3, 4]])
    assert (tmp_path / "models" / "server.npy").exists()


def test_numpy_persistor_rejects_missing_job_identity_before_loading():
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, Mock())
    with pytest.raises(RuntimeError, match="job_id is missing"):
        NPModelPersistor().load_model(ctx)
