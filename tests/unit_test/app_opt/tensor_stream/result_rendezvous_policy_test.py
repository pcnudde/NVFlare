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

from nvflare.apis.event_type import EventType  # noqa: E402
from nvflare.apis.fl_constant import FLContextKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.shareable import Shareable  # noqa: E402
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer  # noqa: E402
from nvflare.app_opt.tensor_stream.types import TensorCustomKeys  # noqa: E402
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef  # noqa: E402


@pytest.fixture
def context():
    ctx = FLContext()
    ctx.set_prop(FLContextKey.TASK_NAME, "train")
    ctx.set_prop(FLContextKey.TASK_ID, "task")
    ctx.set_peer_context(SimpleNamespace(get_identity_name=lambda: "site"))
    return ctx


@pytest.mark.parametrize("ordinary", [True, False])
def test_payload_results_skip_tensor_rendezvous_for_header_or_nested_download_reference(context, ordinary):
    result = Shareable()
    if ordinary:
        result.set_header(TensorCustomKeys.TASK_RESULT_STREAMING_SKIPPED, True)
    else:
        result["nested"] = [{"weights": LazyDownloadRef("source", "ref", "tensor")}]
    context.set_prop(FLContextKey.TASK_RESULT, result)
    streamer = TensorServerStreamer()
    streamer.receiver = Mock()
    streamer.log_info = Mock()
    streamer.handle_event(EventType.BEFORE_TASK_RESULT_FILTER, context)
    streamer.receiver.wait_for_tensors.assert_not_called()
    streamer.receiver.set_ctx_with_tensors.assert_not_called()
    assert context.get_prop(FLContextKey.TASK_RESULT) is result
    streamer.log_info.assert_called_once()


def test_result_rendezvous_error_panics_without_inserting_incomplete_tensors(context):
    context.set_prop(FLContextKey.TASK_RESULT, Shareable())
    streamer = TensorServerStreamer()
    streamer.receiver = Mock()
    streamer.receiver.wait_for_tensors.side_effect = TimeoutError("missing tensor chunk")
    streamer.system_panic = Mock()
    streamer.handle_event(EventType.BEFORE_TASK_RESULT_FILTER, context)
    streamer.receiver.wait_for_tensors.assert_called_once_with("task", "site")
    streamer.receiver.set_ctx_with_tensors.assert_not_called()
    streamer.system_panic.assert_called_once_with("missing tensor chunk", context)
