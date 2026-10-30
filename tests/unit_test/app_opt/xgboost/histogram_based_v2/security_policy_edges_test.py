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

import pytest

xgb = pytest.importorskip("xgboost")

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.histogram_based_v2 import fed_eval_executor as eval_module
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.sec import client_handler as module
from nvflare.app_opt.xgboost.histogram_based_v2.sec.partial_he.util import combine
from nvflare.app_opt.xgboost.histogram_based_v2.sec.sec_handler import SecurityHandler

pytestmark = pytest.mark.usefixtures("isolated_fobs")


@pytest.mark.parametrize(
    "version,disabled,expected",
    [("1.7.0", False, False), ("3.3.0", False, True), ("invalid", False, False), ("1.7.0", True, True)],
)
def test_secure_training_version_policy_handles_disabled_old_supported_and_invalid_versions(
    monkeypatch, version, disabled, expected
):
    monkeypatch.setattr(module, "xgboost", SimpleNamespace(__version__=version, core=xgb.core))
    assert module.ClientSecurityHandler()._check_xgboost_version(disabled) is expected


def _configuration(split):
    context = FLContext()
    context.set_prop(
        FLContextKey.TASK_DATA,
        {
            Constant.CONF_KEY_DATA_SPLIT_MODE: split,
            Constant.CONF_KEY_SECURE_TRAINING: True,
            Constant.CONF_KEY_DISABLE_VERSION_CHECK: True,
        },
    )
    return context


def test_secure_column_configuration_constructs_key_bound_pool_helpers(monkeypatch):
    subject = module.ClientSecurityHandler(key_length=2048, num_workers=3)
    public, private = object(), object()
    keys = MagicMock(return_value=(public, private))
    encryptor, decrypter, adder = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(module, "ipcl_imported", True)
    monkeypatch.setattr(module, "generate_keys", keys)
    monkeypatch.setattr(module, "Encryptor", encryptor)
    monkeypatch.setattr(module, "Decrypter", decrypter)
    monkeypatch.setattr(module, "Adder", adder)
    subject.handle_event(Constant.EVENT_XGB_JOB_CONFIGURED, _configuration(xgb.core.DataSplitMode.COL))
    keys.assert_called_once_with(2048)
    encryptor.assert_called_once_with(public, 3)
    decrypter.assert_called_once_with(private, 3)
    adder.assert_called_once_with(3)
    assert subject.public_key is public and subject.private_key is private
    assert subject.encryptor is encryptor.return_value
    assert subject.decrypter is decrypter.return_value
    assert subject.adder is adder.return_value


@pytest.mark.parametrize("mode", ["missing_backend", "load_failure", "success"])
def test_secure_horizontal_configuration_reports_context_failure_or_clears_on_end_run(monkeypatch, mode):
    if mode != "missing_backend":
        pytest.importorskip("tenseal")
    subject = module.ClientSecurityHandler(tenseal_context_file="private.tenseal")
    monkeypatch.setattr(module, "tenseal_error", "backend unavailable")
    monkeypatch.setattr(module, "tenseal_imported", mode != "missing_backend")
    context = object()
    load = MagicMock(
        side_effect=ValueError("missing context") if mode == "load_failure" else None, return_value=context
    )
    if mode != "missing_backend":
        monkeypatch.setattr(module, "load_tenseal_context_from_workspace", load)
    fl_ctx = _configuration(xgb.core.DataSplitMode.ROW)
    subject.handle_event(Constant.EVENT_XGB_JOB_CONFIGURED, fl_ctx)
    if mode == "success":
        load.assert_called_once_with("private.tenseal", fl_ctx)
        assert subject.tenseal_context is context
        subject.handle_event(EventType.END_RUN, fl_ctx)
        assert subject.tenseal_context is None
    else:
        assert subject.tenseal_context is None
        assert (
            "backend unavailable" in fl_ctx.get_prop(Constant.PARAM_KEY_CONFIG_ERROR)
            if mode == "missing_backend"
            else "missing context" in fl_ctx.get_prop(Constant.PARAM_KEY_CONFIG_ERROR)
        )
        assert load.call_count == int(mode == "load_failure")


def test_non_label_broadcast_preserves_key_and_ciphertext_but_substitutes_original_sized_dummy(monkeypatch):
    subject = module.ClientSecurityHandler()
    public = object()
    ciphertext = object()
    decode = MagicMock(return_value=(public, [ciphertext]))
    monkeypatch.setattr(module, "decode_encrypted_data", decode)
    monkeypatch.setattr(module, "os", SimpleNamespace(urandom=lambda length: b"x" * length))
    context = FLContext()
    reply = Shareable()
    reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
    reply.set_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE, 17)
    context.set_prop(Constant.PARAM_KEY_REPLY, reply)
    context.set_prop(Constant.PARAM_KEY_RCV_BUF, "encrypted-data", sticky=False)
    subject._process_after_broadcast(context)
    decode.assert_called_once_with("encrypted-data")
    assert subject.public_key is public
    assert subject.encrypted_ghs == [ciphertext]
    assert context.get_prop(Constant.PARAM_KEY_RCV_BUF) == b"x" * 17


def test_decrypted_feature_results_preserve_feature_and_group_identity():
    subject = module.ClientSecurityHandler()
    data = [(4, 7, json.dumps([combine(1, 2)])), (5, 9, json.dumps([combine(3, 4)]))]
    subject.decrypter = SimpleNamespace(decrypt=MagicMock(return_value=[[combine(1, 2)], [combine(3, 4)]]))
    result = subject._decrypt_aggr_result(json.dumps(data), FLContext())
    assert result == [(4, 7, [combine(1, 2)]), (5, 9, [combine(3, 4)])]
    subject.decrypter.decrypt.assert_called_once_with([[combine(1, 2)], [combine(3, 4)]])
    assert subject._decrypt_aggr_result(None, FLContext()) is None


@pytest.mark.parametrize(
    "event,method",
    [
        (Constant.EVENT_BEFORE_BROADCAST, "_process_before_broadcast"),
        (Constant.EVENT_AFTER_BROADCAST, "_process_after_broadcast"),
        (Constant.EVENT_BEFORE_ALL_GATHER_V, "_process_before_all_gather_v"),
        (Constant.EVENT_AFTER_ALL_GATHER_V, "_process_after_all_gather_v"),
    ],
)
def test_security_event_dispatch_records_context_event_before_processing(event, method):
    subject = SecurityHandler()
    handler = MagicMock()
    setattr(subject, method, handler)
    context = FLContext()
    subject.handle_event(event, context)
    handler.assert_called_once_with(context)
    assert context.get_prop(Constant.PARAM_KEY_EVENT) == event


def test_security_abort_marks_fatal_state_and_emits_abort_event_without_logging_recursion():
    subject = SecurityHandler()
    context = FLContext()
    for key, value in [
        (Constant.PARAM_KEY_RANK, 2),
        (Constant.PARAM_KEY_ROOT, 1),
        (Constant.PARAM_KEY_SEQ, 3),
        (Constant.PARAM_KEY_EVENT, "before"),
    ]:
        context.set_prop(key, value)
    subject.fire_event = MagicMock()
    subject._abort("bad ciphertext", context)
    assert context.get_prop(FLContextKey.FATAL_SYSTEM_ERROR) == "bad ciphertext"
    subject.fire_event.assert_called_once_with(Constant.EVENT_XGB_ABORTED, context)
    for method, target in [(subject.info, "log_info"), (subject.debug, "log_debug"), (subject.error, "log_error")]:
        logger = MagicMock()
        setattr(subject, target, logger)
        method(context, "message")
        args = logger.call_args
        assert args.args[0] is context
        assert "rank=2" in args.args[1] and "seq=3" in args.args[1] and "root=1" in args.args[1]
        assert args.kwargs == {"fire_event": False}


def test_federated_evaluation_executor_registers_security_and_configures_runner_adaptor():
    subject = eval_module.FedXGBEvalExecutor(
        "loader", "/train/workspace", per_msg_timeout=3, tx_timeout=9, in_process=False
    )
    engine = MagicMock()
    engine.get_component.return_value = MagicMock(spec=XGBDataLoader)
    context = FLContext()
    context.get_engine = lambda: engine
    adaptor = subject.get_adaptor(context)
    handler = engine.add_component.call_args.args[1]
    assert isinstance(handler, module.ClientSecurityHandler)
    assert adaptor.in_process is False
    assert adaptor.per_msg_timeout == 3 and adaptor.tx_timeout == 9
    assert adaptor.app_runner.data_loader_id == "loader"
    assert adaptor.app_runner.train_workspace_path == "/train/workspace"
    assert adaptor.app_runner.fl_ctx is context
