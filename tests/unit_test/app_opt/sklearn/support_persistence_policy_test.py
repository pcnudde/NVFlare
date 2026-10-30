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

import joblib
import numpy as np
import pytest

pytest.importorskip("sklearn")

from nvflare.apis.dxo import DXO, DataKind  # noqa: E402
from nvflare.apis.event_type import EventType  # noqa: E402
from nvflare.apis.fl_constant import FLContextKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.app_common.abstract.model import make_model_learnable  # noqa: E402
from nvflare.app_common.app_constant import AppConstants  # noqa: E402
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor  # noqa: E402
from nvflare.app_opt.sklearn.svm_assembler import SVMAssembler  # noqa: E402


@pytest.fixture(autouse=True)
def preserve_numpy_rng():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def context(tmp_path, round_number=0):
    ctx = FLContext()
    ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path), private=True, sticky=True)
    ctx.set_prop(AppConstants.CURRENT_ROUND, round_number, private=True, sticky=True)
    ctx.set_prop(AppConstants.NUM_ROUNDS, 2, private=True, sticky=True)
    return ctx


def test_svm_assembler_selects_exact_pooled_support_rows_and_reuses_the_round_zero_fit(tmp_path):
    assembler = SVMAssembler(kernel="linear")
    first = {"support_x": np.array([[-2.0], [-1.0]]), "support_y": np.array([0, 0])}
    second = {"support_x": np.array([[1.0], [2.0]]), "support_y": np.array([1, 1])}
    assert assembler.get_model_params(DXO(DataKind.WEIGHTS, first)) is not first
    assembler.collection.update({"a": first, "b": second})
    ctx = context(tmp_path)
    selected = assembler.assemble({}, ctx)
    assert selected.data_kind == DataKind.WEIGHTS
    np.testing.assert_array_equal(selected.data["support_x"], [[-1], [1]])
    np.testing.assert_array_equal(selected.data["support_y"], [0, 1])
    assembler.collection.clear()
    ctx.set_prop(AppConstants.CURRENT_ROUND, 1, private=True, sticky=True)
    repeated = assembler.assemble({}, ctx)
    np.testing.assert_array_equal(repeated.data["support_x"], [[-1], [1]])


def test_joblib_persistor_initial_config_last_round_checkpoint_and_reload_precedence(tmp_path):
    ctx = context(tmp_path / "server")
    persistor = JoblibModelParamPersistor(initial_params={"n_clusters": 3})
    persistor.handle_event(EventType.START_RUN, ctx)
    assert persistor.load_model(ctx)["weights"] == {"n_clusters": 3}
    final = make_model_learnable({"centers": np.array([[2.0, 4.0]])}, {})
    persistor.save_model(final, ctx)
    assert not (tmp_path / "server" / persistor.save_name).exists()
    ctx.set_prop(AppConstants.CURRENT_ROUND, 1, private=True, sticky=True)
    persistor.save_model(final, ctx)
    np.testing.assert_array_equal(persistor.load_model(ctx)["weights"]["centers"], [[2, 4]])
    explicit = tmp_path / "explicit.joblib"
    joblib.dump({"source": "explicit"}, explicit)
    persistor.model_path = str(explicit)
    assert persistor.load_model(ctx)["weights"] == {"source": "explicit"}
    persistor.model_path = None
    persistor.initial_params = {"model_path": str(explicit), "n_clusters": 5}
    assert persistor.load_model(ctx)["weights"] == {"source": "explicit"}


@pytest.mark.parametrize(
    "params,match",
    [
        ({"model_path": "relative.joblib"}, "absolute path"),
        ({"model_path": "/missing/unit-model.joblib"}, "Model file not found"),
        ({}, "No model parameters"),
    ],
)
def test_joblib_persistor_rejects_unavailable_initial_source(tmp_path, params, match):
    ctx = context(tmp_path)
    item = JoblibModelParamPersistor(initial_params=params)
    item.handle_event(EventType.START_RUN, ctx)
    with pytest.raises(ValueError, match=match):
        item.load_model(ctx)
