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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.higgs_data_loader import HIGGSDataLoader
from nvflare.app_opt.xgboost.tree_based import executor as module
from nvflare.app_opt.xgboost.tree_based.executor import FedXGBTreeExecutor


def _matrix():
    return xgb.DMatrix(np.array([[0], [1], [2], [3]], dtype=np.float32), label=[0, 0, 1, 1], nthread=1)


@pytest.mark.parametrize(
    "mode,lr_mode,expected", [("bagging", "uniform", 0.1), ("bagging", "scaled", 0.075), ("cyclic", "scaled", 0.3)]
)
def test_initialization_resolves_loader_paths_rank_and_effective_learning_rate(
    monkeypatch, tmp_path, mode, lr_mode, expected
):
    loader = MagicMock(spec=XGBDataLoader)
    train, val = _matrix(), _matrix()
    loader.load_data.return_value = (train, val)
    engine = MagicMock()
    engine.get_workspace.return_value.get_app_dir.return_value = str(tmp_path)
    engine.get_component.return_value = loader
    context = FLContext()
    context.get_engine = lambda: engine
    context.get_identity_name = lambda: "site-2"
    writer = MagicMock()
    monkeypatch.setattr(
        module, "optional_import", lambda module: (SimpleNamespace(SummaryWriter=lambda path: writer), True)
    )
    subject = FedXGBTreeExecutor(
        mode, 0.25, "loader", num_client_bagging=3, learning_rate=0.3, lr_mode=lr_mode, nthread=1
    )
    subject.handle_event(EventType.START_RUN, context)
    assert subject.rank == 1
    assert subject.client_id == "site-2"
    assert subject.local_model_path == str(tmp_path / "model.json")
    assert subject.global_model_path == str(tmp_path / "model_global.json")
    assert subject.writer is writer
    loader.initialize.assert_called_once_with(client_id="site-2", rank=1)
    assert subject.train_data is train and subject.val_data is val
    assert subject.lr == pytest.approx(expected)
    params = subject._get_xgb_train_params()
    assert params["eta"] == pytest.approx(expected)
    assert params["nthread"] == 1
    subject.handle_event(EventType.END_RUN, context)
    assert not hasattr(subject, "bst") and not hasattr(subject, "train_data")


def test_initialization_rejects_training_mode_and_reports_loader_failure(monkeypatch, tmp_path):
    engine = MagicMock()
    engine.get_workspace.return_value.get_app_dir.return_value = str(tmp_path)
    context = FLContext()
    context.get_engine = lambda: engine
    context.get_identity_name = lambda: "site-1"
    monkeypatch.setattr(module, "optional_import", lambda module: (None, False))
    invalid = FedXGBTreeExecutor("invalid", 1, "loader")
    invalid.system_panic = MagicMock()
    invalid.initialize(context)
    assert "Only support" in invalid.system_panic.call_args.args[0]
    engine.get_component.assert_not_called()
    loader = MagicMock(spec=XGBDataLoader)
    loader.load_data.side_effect = ValueError("bad split")
    engine.get_component.return_value = loader
    subject = FedXGBTreeExecutor("cyclic", 1, "loader")
    subject.system_panic = MagicMock()
    subject.initialize(context)
    assert "load_data failed" in subject.system_panic.call_args.args[0]
    with pytest.raises(ValueError, match="Only support"):
        FedXGBTreeExecutor("cyclic", 1, "loader", lr_mode="unknown")


@pytest.mark.parametrize("mode", ["cyclic", "bagging"])
def test_second_training_round_consumes_server_model_and_masks_hessian_metadata(tmp_path, mode):
    subject = FedXGBTreeExecutor(
        mode, 1, "loader", learning_rate=0.3, nthread=1, max_depth=1, num_client_bagging=1, eval_metric="logloss"
    )
    subject.lr = 0.3
    subject.train_data = subject.val_data = _matrix()
    subject.client_id = "site-1"
    subject.writer = MagicMock()
    context = FLContext()
    first = subject.train(DXO(DataKind.WEIGHTS, {}).to_shareable(), context, Signal())
    data = from_shareable(first).data["model_data"]
    update = {"model_data": [data]} if mode == "bagging" else {"model_data": data}
    second = from_shareable(subject.execute("train", DXO(DataKind.WEIGHTS, update).to_shareable(), context, Signal()))
    local = json.loads(second.data["model_data"])
    trees = local["learner"]["gradient_booster"]["model"]["trees"]
    assert len(trees) == (1 if mode == "bagging" else 2)
    assert all(all(value == 1.0 for value in tree["sum_hessian"]) for tree in trees)
    assert "logloss" in second.meta[AppConstants.PROGRESS_METRICS]
    assert subject.writer.flush.call_count == 2
    assert subject.writer.add_scalar.call_count == 1
    subject.handle_event(EventType.END_RUN, context)


def test_executor_maps_unknown_malformed_and_abort_tasks_to_return_codes():
    from nvflare.apis.fl_constant import ReturnCode

    context = FLContext()
    subject = FedXGBTreeExecutor("cyclic", 1, "loader", nthread=1)
    assert subject.execute("unknown", Shareable(), context, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN
    assert subject.execute("train", Shareable(), context, Signal()).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    abort = Signal()
    abort.trigger(True)
    subject.finalize = MagicMock()
    assert subject.execute("train", Shareable(), context, abort).get_return_code() == ReturnCode.TASK_ABORTED
    subject.finalize.assert_called_once_with(context)


@pytest.mark.parametrize("missing", [None, "site-1", "valid"])
def test_higgs_loader_reads_exact_csv_ranges_and_rejects_missing_splits(tmp_path, missing):
    data = tmp_path / "higgs.csv"
    data.write_text("0,1,2\n1,3,4\n0,5,6\n1,7,8\n")
    indices = {"site-1": {"start": 0, "end": 2}, "valid": {"start": 2, "end": 4}}
    if missing:
        indices.pop(missing)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"data_path": str(data), "data_index": indices}))
    loader = HIGGSDataLoader(str(split))
    loader.initialize(client_id="site-1", rank=0)
    if missing:
        with pytest.raises(ValueError, match="split"):
            loader.load_data()
    else:
        train, val = loader.load_data()
        assert train.num_row() == val.num_row() == 2
        assert train.num_col() == val.num_col() == 2
        np.testing.assert_array_equal(train.get_label(), [0, 1])
        np.testing.assert_array_equal(val.get_label(), [0, 1])
        np.testing.assert_array_equal(train.get_data().toarray(), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(val.get_data().toarray(), [[5, 6], [7, 8]])
