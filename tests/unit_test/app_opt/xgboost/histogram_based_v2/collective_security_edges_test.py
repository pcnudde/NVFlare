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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("xgboost")

from nvflare.apis.fl_constant import FLContextKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.shareable import Shareable  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.sec import client_handler as subject  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.sec.dam import DamEncoder  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.sec.processor_data_converter import (  # noqa: E402
    DATA_SET_GH_PAIRS,
    DATA_SET_HISTOGRAMS,
)


@pytest.fixture
def handler(monkeypatch):
    if subject.tenseal_imported:
        monkeypatch.setattr(subject, "decomposers", SimpleNamespace(register=Mock()))
    return subject.ClientSecurityHandler()


def test_clear_or_non_dam_collectives_are_not_misclassified_or_encrypted(handler):
    ctx = FLContext()
    ctx.set_prop(Constant.PARAM_KEY_ROOT, 0)
    ctx.set_prop(Constant.PARAM_KEY_RANK, 0)
    ctx.set_prop(Constant.PARAM_KEY_SEND_BUF, b"not-dam")
    handler._process_before_broadcast(ctx)
    handler._process_before_all_gather_v(ctx)
    handler._process_before_all_gather_v_vertical(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_SEND_BUF) == b"not-dam"
    assert ctx.get_prop(Constant.PARAM_KEY_HEADERS) is None
    ctx.set_prop(Constant.PARAM_KEY_REPLY, Shareable())
    handler._process_after_broadcast(ctx)
    handler._process_after_all_gather_v(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_RCV_BUF) is None
    assert handler._do_aggregation([], ctx) is None


@pytest.mark.parametrize("horizontal", [False, True])
def test_secure_collective_dispatch_uses_dataset_identity_before_and_reply_marker_after(
    handler, monkeypatch, horizontal
):
    ctx = FLContext()
    encoder = DamEncoder(DATA_SET_HISTOGRAMS if horizontal else DATA_SET_GH_PAIRS)
    encoder.add_float_array([1.0, 2.0])
    ctx.set_prop(Constant.PARAM_KEY_SEND_BUF, encoder.finish())
    before_h, before_v, after_h, after_v = Mock(), Mock(), Mock(), Mock()
    for name, method in [
        ("_process_before_all_gather_v_horizontal", before_h),
        ("_process_before_all_gather_v_vertical", before_v),
        ("_process_after_all_gather_v_horizontal", after_h),
        ("_process_after_all_gather_v_vertical", after_v),
    ]:
        monkeypatch.setattr(handler, name, method)
    handler._process_before_all_gather_v(ctx)
    reply = Shareable()
    reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
    reply.set_header(Constant.HEADER_KEY_HORIZONTAL, horizontal)
    ctx.set_prop(Constant.PARAM_KEY_REPLY, reply)
    handler._process_after_all_gather_v(ctx)
    for method in (before_h, after_h) if horizontal else (before_v, after_v):
        method.assert_called_once_with(ctx)
    for method in (before_v, after_v) if horizontal else (before_h, after_h):
        method.assert_not_called()


def test_secure_job_rejects_unsupported_xgboost_before_crypto_setup(handler, monkeypatch):
    monkeypatch.setattr(handler, "_check_xgboost_version", lambda disabled: False)
    ctx = FLContext()
    ctx.set_prop(
        FLContextKey.TASK_DATA, {Constant.CONF_KEY_SECURE_TRAINING: True, Constant.CONF_KEY_DATA_SPLIT_MODE: 0}
    )
    handler.handle_event(Constant.EVENT_XGB_JOB_CONFIGURED, ctx)
    assert "doesn't support secure training" in ctx.get_prop(Constant.PARAM_KEY_CONFIG_ERROR)
    assert handler.tenseal_context is None and handler.encryptor is None
