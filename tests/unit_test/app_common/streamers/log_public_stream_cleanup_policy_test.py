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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import StreamableEngine
from nvflare.app_common.streamers import log_streamer as subject


@pytest.mark.parametrize("fails", [False, True])
def test_public_log_stream_closes_producer_after_engine_success_or_failure(tmp_path, monkeypatch, fails):
    context, engine, producer = FLContext(), Mock(spec=StreamableEngine), Mock()
    context.set_prop(ReservedKey.ENGINE, engine)
    engine.stream_objects.side_effect = OSError("transport failed") if fails else None
    engine.stream_objects.return_value = {"site": "complete"}
    create = Mock(return_value=producer)
    monkeypatch.setattr(subject, "_LogTailProducer", create)
    args = dict(
        channel="logs",
        topic="tail",
        stream_ctx=None,
        targets=["site"],
        file_name=str(tmp_path / "job.log"),
        fl_ctx=context,
    )
    if fails:
        with pytest.raises(OSError, match="transport failed"):
            subject.LogStreamer.stream_log(**args)
    else:
        assert subject.LogStreamer.stream_log(**args) == {"site": "complete"}
    producer.close.assert_called_once_with()
    call = engine.stream_objects.call_args.kwargs
    assert call["producer"] is producer and call["fl_ctx"] is context
    assert call["stream_ctx"][subject.KEY_FILE_NAME] == "job.log"
    assert call["targets"] == ["site"] and call["channel"] == "logs"
    assert create.call_args.args[1:4] == (65536, 5.0, 0.5)
    assert not create.call_args.args[4].is_set()


def test_public_log_stream_requires_streamable_engine_before_creating_producer(tmp_path, monkeypatch):
    create = Mock()
    monkeypatch.setattr(subject, "_LogTailProducer", create)
    with pytest.raises(RuntimeError, match="StreamableEngine"):
        subject.LogStreamer.stream_log("logs", "tail", {}, [], str(tmp_path / "job.log"), FLContext())
    create.assert_not_called()


def test_closed_log_consumer_rejects_late_chunk_without_dispatching_callback():
    callback = Mock()
    consumer = subject._LogChunkConsumer({}, callback, 0, {})
    consumer._done.set()
    keep, result = consumer.consume(Shareable({"text": "late"}), {}, FLContext())
    assert not keep and result.get_return_code() == ReturnCode.TASK_ABORTED
    callback.assert_not_called()
