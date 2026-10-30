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

pytest.importorskip("torch")

from nvflare.apis.dxo import DXO, DataKind  # noqa: E402
from nvflare.apis.fl_constant import FLContextKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.app_common.app_constant import AppConstants  # noqa: E402
from nvflare.app_common.app_event_type import AppEventType  # noqa: E402
from nvflare.app_opt.tensor_stream.receiver import TensorReceiver  # noqa: E402
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer  # noqa: E402


def context():
    ctx = FLContext()
    ctx.set_peer_context(SimpleNamespace(get_identity_name=lambda: "site"))
    return ctx


def test_round_done_releases_only_observed_round_tensors_and_all_its_counters():
    streamer = TensorServerStreamer()
    streamer.receiver = SimpleNamespace(tensors={"seen": {"w": 1}, "other-round": {"w": 2}})
    streamer.seen_tasks[2] = {"seen", "absent"}
    streamer.seen_tasks[3] = {"other-round"}
    for mapping in (
        streamer.num_task_data_sent,
        streamer.num_task_skipped,
        streamer.data_cleaned,
        streamer.start_sending_time,
    ):
        mapping[2] = 1
        mapping[3] = 2
    ctx = context()
    ctx.set_prop(AppConstants.CURRENT_ROUND, 2)
    streamer.handle_event(AppEventType.ROUND_DONE, ctx)
    assert streamer.receiver.tensors == {"other-round": {"w": 2}}
    for mapping in (
        streamer.seen_tasks,
        streamer.num_task_data_sent,
        streamer.num_task_skipped,
        streamer.data_cleaned,
        streamer.start_sending_time,
    ):
        assert 2 not in mapping and 3 in mapping
    streamer.handle_event(AppEventType.ROUND_DONE, ctx)
    assert streamer.receiver.tensors == {"other-round": {"w": 2}}


@pytest.mark.parametrize(
    "case,reason",
    [("missing-id", "No task_id"), ("empty-data", "empty data"), ("unsupported-format", "Unsupported tensor format")],
)
def test_receiver_rejects_unaddressed_empty_or_unsupported_tensor_payloads(case, reason):
    ctx = context()
    receiver = TensorReceiver(Mock(), FLContextKey.TASK_RESULT, format="unsupported")
    if case != "missing-id":
        ctx.set_prop(FLContextKey.TASK_ID, "task")
        receiver.tensors["task"] = {} if case == "empty-data" else {"w": [1]}
        ctx.set_prop(FLContextKey.TASK_RESULT, DXO(DataKind.WEIGHTS, {}).to_shareable())
    with pytest.raises((ValueError, RuntimeError), match=reason):
        receiver.set_ctx_with_tensors(ctx)
    if case != "missing-id":
        assert ctx.get_prop(FLContextKey.TASK_RESULT)["DXO"]["data"] == {}


def test_stream_completion_without_task_identity_does_not_create_tensor_event():
    receiver = TensorReceiver(Mock(), FLContextKey.TASK_RESULT)
    with pytest.raises(ValueError, match="No task_id"):
        receiver._save_tensors_cb(True, context())
    assert not receiver.tensor_events and not receiver.tensors and not receiver.error_events
