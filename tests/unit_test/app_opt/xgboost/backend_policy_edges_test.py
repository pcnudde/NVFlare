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
from unittest.mock import Mock

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from nvflare.apis.event_type import EventType  # noqa: E402
from nvflare.apis.fl_constant import FLContextKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.app_common.abstract.model import make_model_learnable  # noqa: E402
from nvflare.app_common.app_constant import AppConstants  # noqa: E402
from nvflare.app_common.tracking.log_writer import LogWriter  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2 import controller as subject  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader import CSVDataLoader  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.sec import processor_data_converter as codec  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.sec.dam import DamDecoder, DamEncoder  # noqa: E402
from nvflare.app_opt.xgboost.metrics_cb import MetricsCallback  # noqa: E402
from nvflare.app_opt.xgboost.tree_based.model_persistor import XGBModelPersistor  # noqa: E402


@pytest.mark.parametrize("as_bytes", [False, True])
def test_tree_model_persistor_roundtrip_is_loadable_and_only_writes_final_round(tmp_path, as_bytes):
    features, labels = np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0.0, 0.0, 1.0, 1.0])
    data = xgb.DMatrix(features, label=labels)
    booster = xgb.train(
        {"objective": "binary:logistic", "nthread": 1, "max_depth": 1, "min_child_weight": 0}, data, num_boost_round=1
    )
    raw = booster.save_raw(raw_format="json")
    persistor = XGBModelPersistor(load_as_dict=not as_bytes)
    ctx = FLContext()
    for name, value in [
        (FLContextKey.APP_ROOT, str(tmp_path / "server")),
        (AppConstants.NUM_ROUNDS, 2),
        (AppConstants.CURRENT_ROUND, 0),
    ]:
        ctx.set_prop(name, value, private=True, sticky=True)
    persistor.handle_event(EventType.START_RUN, ctx)
    assert persistor.load_model(ctx)["weights"] is None
    model = raw if as_bytes else json.loads(raw)
    persistor.save_model(make_model_learnable(model, {}), ctx)
    assert not (tmp_path / "server" / persistor.save_name).exists()
    ctx.set_prop(AppConstants.CURRENT_ROUND, 1, private=True, sticky=True)
    persistor.save_model(make_model_learnable(model, {}), ctx)
    reloaded = persistor.load_model(ctx)["weights"]
    assert isinstance(reloaded, bytearray if as_bytes else dict)
    restored = xgb.Booster()
    restored.load_model(bytearray(json.dumps(reloaded), "utf-8") if isinstance(reloaded, dict) else reloaded)
    np.testing.assert_array_equal(restored.predict(data), booster.predict(data))


def test_xgb_csv_loader_uses_site_specific_row_split_and_label_column(tmp_path):
    folder = tmp_path / "site-2"
    folder.mkdir()
    (folder / "train.csv").write_text("0,1,2\n1,3,4\n")
    (folder / "valid.csv").write_text("1,5,6\n")
    loader = CSVDataLoader(str(tmp_path))
    loader.initialize("site-2", 1, xgb.core.DataSplitMode.ROW)
    train, valid = loader.load_data()
    assert train.num_row() == 2 and train.num_col() == 2
    np.testing.assert_array_equal(train.get_label(), [0, 1])
    np.testing.assert_array_equal(valid.get_label(), [1])


class RecordingWriter(LogWriter):
    def __init__(self):
        super().__init__()
        self.records = []

    def get_writer_name(self):
        return "recording"

    def write(self, **record):
        self.records.append(record)


def test_metric_callback_keeps_latest_tuple_mean_and_scalar_per_dataset():
    writer = RecordingWriter()
    callback = MetricsCallback(writer)
    assert callback.after_iteration(None, 4, {}) is False
    assert callback.after_iteration(None, 4, {"train": {"loss": [3, 2], "auc": [(0.5, 0.1), (0.7, 0.2)]}}) is False
    assert len(writer.records) == 1
    assert writer.records[0]["tag"] == "train_metrics"
    assert writer.records[0]["global_step"] == 4
    assert writer.records[0]["value"] == {"loss": 2, "auc": 0.7}
    with pytest.raises(RuntimeError, match="writer is not valid"):
        MetricsCallback(object())


@pytest.mark.parametrize("mode", ["active", "done", "inactive-client", "stalled-job"])
def test_federated_controller_deadlines_distinguish_silent_client_from_global_stall(monkeypatch, mode):
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: 100.0))
    controller = subject.XGBController(
        "adaptor", 1, 0, False, {"objective": "reg:squarederror"}, max_client_op_interval=50, progress_timeout=10
    )
    controller.system_panic, controller.log_info = Mock(), Mock()
    status = subject.ClientStatus()
    status.xgb_done = mode == "done"
    status.last_op_time = 0 if mode == "inactive-client" else 80 if mode == "stalled-job" else 99
    controller.client_statuses = {"site": status}
    assert controller._check_job_status(FLContext()) is (mode != "active")
    if mode in ("inactive-client", "stalled-job"):
        controller.system_panic.assert_called_once()
        assert ("activity" if mode == "inactive-client" else "no progress") in controller.system_panic.call_args.args[0]
    else:
        controller.system_panic.assert_not_called()


def test_histogram_codec_roundtrip_checks_dataset_identity_and_invalid_buffers():
    converter = codec.ProcessorDataConverter()
    ctx = FLContext()
    assert converter.decode_histograms(b"not-dam", ctx) is None
    assert converter.decode_gh_pairs(b"not-dam", ctx) is None
    assert converter.decode_aggregation_context(b"not-dam", ctx) is None
    encoder = DamEncoder(codec.DATA_SET_HISTOGRAMS)
    encoder.add_float_array([1.25, -2.5, 4.0])
    assert converter.decode_histograms(encoder.finish(), ctx) == [1.25, -2.5, 4.0]
    decoder = DamDecoder(converter.encode_histograms_result([3.5, 2.0], ctx))
    assert decoder.get_data_set_id() == codec.DATA_SET_HISTOGRAMS_RESULT
    assert decoder.decode_float_array() == [3.5, 2.0]
    wrong = DamEncoder(codec.DATA_SET_AGGREGATION_RESULT)
    wrong.add_float_array([1.0, 2.0])
    with pytest.raises(RuntimeError, match="Invalid DataSet"):
        converter.decode_histograms(wrong.finish(), ctx)
    with pytest.raises(RuntimeError, match="not for GH Pairs"):
        converter.decode_gh_pairs(wrong.finish(), ctx)
    assert converter.slot_to_bin([0, 2, 5], -1) == (0, -1)
    with pytest.raises(RuntimeError, match="out of range"):
        converter.slot_to_bin([0, 2, 5], 5)
