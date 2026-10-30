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

from unittest.mock import MagicMock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.xgb_adaptor import XGBClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.executor import XGBExecutor


@pytest.fixture
def executor():
    ctx = FLContext()
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site")
    ctx.set_prop(ReservedKey.ENGINE, MagicMock())
    adaptor = MagicMock(spec=XGBClientAdaptor)
    ctx.get_engine().get_component.return_value = adaptor
    executor = XGBExecutor("adaptor")
    executor.system_panic = MagicMock()
    executor.handle_event(EventType.START_RUN, ctx)
    return executor, adaptor, ctx


def test_configure_and_start_send_validated_task_to_adaptor_and_stop_on_end_run(executor):
    executor, adaptor, ctx = executor
    adaptor.set_abort_signal.assert_called_once_with(executor.abort_signal)
    adaptor.initialize.assert_called_once_with(ctx)
    request = Shareable({Constant.CONF_KEY_CLIENT_RANKS: {"site": 0}, Constant.CONF_KEY_NUM_ROUNDS: 3})
    reply = executor.execute(executor.configure_task_name, request, ctx, Signal())
    assert reply.get_return_code() == ReturnCode.OK
    adaptor.configure.assert_called_once_with(request, ctx)
    assert executor.execute(executor.start_task_name, Shareable(), ctx, Signal()).get_return_code() == ReturnCode.OK
    adaptor.monitor_target.assert_called_once_with(ctx, executor._notify_client_done)
    executor.handle_event(EventType.END_RUN, ctx)
    assert executor.abort_signal.triggered


@pytest.mark.parametrize(
    "config_request",
    [
        Shareable(),
        Shareable({Constant.CONF_KEY_CLIENT_RANKS: [0]}),
        Shareable({Constant.CONF_KEY_CLIENT_RANKS: {"other": 0}}),
        Shareable({Constant.CONF_KEY_CLIENT_RANKS: {"site": 0}}),
    ],
)
def test_configure_rejects_missing_or_malformed_rank_and_round_fields_before_adaptor(executor, config_request):
    executor, adaptor, ctx = executor
    assert (
        executor.execute(executor.configure_task_name, config_request, ctx, Signal()).get_return_code()
        == ReturnCode.BAD_TASK_DATA
    )
    adaptor.configure.assert_not_called()


def test_security_configuration_failure_and_adaptor_start_exception_have_distinct_codes(executor):
    executor, adaptor, ctx = executor
    ctx.set_prop(Constant.PARAM_KEY_CONFIG_ERROR, "bad certificate")
    config = Shareable({Constant.CONF_KEY_CLIENT_RANKS: {"site": 0}, Constant.CONF_KEY_NUM_ROUNDS: 3})
    reply = executor.execute(executor.configure_task_name, config, ctx, Signal())
    assert reply.get_return_code() == ReturnCode.SERVICE_UNAVAILABLE
    assert reply.get_header(ReservedKey.EXCEPTIONS) == "bad certificate"
    adaptor.start.side_effect = RuntimeError("launch")
    assert (
        executor.execute(executor.start_task_name, Shareable(), ctx, Signal()).get_return_code()
        == ReturnCode.EXECUTION_EXCEPTION
    )
    assert executor.execute("unknown", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNSUPPORTED


@pytest.mark.parametrize("rc", [0, 2])
def test_done_notification_sends_exit_code_to_server_and_panics_for_failure(executor, rc):
    executor, _, ctx = executor
    ctx.set_prop(FLContextKey.FATAL_SYSTEM_ERROR, "failure")
    executor._notify_client_done(rc, ctx)
    kwargs = ctx.get_engine().send_aux_request.call_args.kwargs
    assert kwargs["topic"] == Constant.TOPIC_CLIENT_DONE
    assert kwargs["request"][Constant.MSG_KEY_EXIT_CODE] == rc
    assert kwargs["timeout"] == 0 and kwargs["optional"]
    assert executor.system_panic.call_count == (0 if rc == 0 else 1)


@pytest.mark.parametrize("bad", [None, object()])
def test_start_rejects_missing_or_wrong_adaptor(executor, bad):
    executor, _, ctx = executor
    ctx.get_engine().get_component.return_value = bad
    executor.handle_event(EventType.START_RUN, ctx)
    executor.system_panic.assert_called_once()
