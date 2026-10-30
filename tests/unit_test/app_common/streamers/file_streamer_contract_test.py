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

import io
from pathlib import Path
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.streaming import StreamableEngine
from nvflare.app_common.streamers import file_streamer as subject


@pytest.mark.parametrize("data", [b"", b"short", b"12345678"])
def test_file_chunks_reassemble_exact_bytes_and_finish_at_eof(tmp_path, data):
    ctx = {subject.KEY_FILE_NAME: "model.bin", subject.KEY_FILE_SIZE: len(data)}
    producer = subject._ChunkProducer(io.BytesIO(data), 4, 7)
    consumer = subject._ChunkConsumerFactory(str(tmp_path)).get_consumer(ctx, FLContext())
    try:
        for _ in range(len(data) // 4 + 2):
            item, timeout = producer.produce(ctx, FLContext())
            assert timeout == 7
            keep_going, reply = consumer.consume(item, ctx, FLContext())
            assert reply.get_return_code() == ReturnCode.OK
            if not keep_going:
                break
        else:
            pytest.fail("file stream did not terminate within its chunk bound")
        assert consumer.received_size == len(data)
    finally:
        consumer.finalize(ctx, FLContext())
    assert Path(subject.FileStreamer.get_file_location(ctx)).read_bytes() == data
    assert subject.FileStreamer.get_file_name(ctx) == "model.bin"
    assert subject.FileStreamer.get_file_size(ctx) == len(data)
    assert consumer.file.closed


@pytest.mark.parametrize("declared_chunk,declared_file", [(5, 2), (2, 5)])
def test_file_consumer_rejects_chunk_and_total_size_mismatches(tmp_path, declared_chunk, declared_file):
    ctx = {subject.KEY_FILE_NAME: "model", subject.KEY_FILE_SIZE: declared_file}
    consumer = subject._ChunkConsumer(ctx, str(tmp_path))
    try:
        with pytest.raises(ValueError, match="does not match"):
            consumer.consume(
                Shareable({subject.KEY_DATA: b"ab", subject.KEY_DATA_SIZE: declared_chunk, subject.KEY_EOF: True}),
                ctx,
                FLContext(),
            )
    finally:
        consumer.finalize(ctx, FLContext())
    assert consumer.file.closed


def test_file_streamer_registers_consumer_options_and_closes_source_after_send(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"payload")
    engine = Mock(spec=StreamableEngine)
    ctx = Mock(spec=FLContext)
    ctx.get_engine.return_value = engine
    consumed, finished = Mock(), Mock()
    subject.FileStreamer.register_stream_processing(
        ctx, "models", "latest", str(tmp_path), finished, consumed, role="receiver"
    )
    registered = engine.register_stream_processing.call_args.kwargs
    assert registered["factory"].dest_dir == str(tmp_path)
    assert registered["stream_done_cb"] is finished and registered["consumed_cb"] is consumed
    assert registered["role"] == "receiver"

    def stream(**kwargs):
        item, timeout = kwargs["producer"].produce(kwargs["stream_ctx"], ctx)
        assert item[subject.KEY_DATA] == b"payload"
        assert timeout == 5
        assert kwargs["optional"] and kwargs["secure"]
        assert kwargs["targets"] == ["site"]
        assert kwargs["stream_ctx"][subject.KEY_FILE_NAME] == "source.bin"
        assert kwargs["stream_ctx"][subject.KEY_FILE_SIZE] == 7
        return True

    engine.stream_objects.side_effect = stream
    assert subject.FileStreamer.stream_file(
        "models", "latest", {}, ["site"], str(path), ctx, optional=True, secure=True
    )
    assert engine.stream_objects.call_args.kwargs["producer"].file.closed


def test_file_streamer_rejects_invalid_files_destinations_engines_and_chunk_settings(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"payload")
    ctx = Mock(spec=FLContext)
    ctx.get_engine.return_value = object()
    with pytest.raises(ValueError, match="valid dir"):
        subject.FileStreamer.register_stream_processing(ctx, "c", "t", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="StreamableEngine"):
        subject.FileStreamer.register_stream_processing(ctx, "c", "t")
    with pytest.raises(RuntimeError, match="StreamableEngine"):
        subject.FileStreamer.stream_file("c", "t", None, ["s"], str(path), ctx)
    with pytest.raises(ValueError, match="valid file"):
        subject.FileStreamer.stream_file("c", "t", None, ["s"], str(tmp_path / "missing"), ctx)
    for kwargs in ({"chunk_size": -1}, {"chunk_timeout": -1}):
        with pytest.raises(ValueError):
            subject.FileStreamer.stream_file("c", "t", None, ["s"], str(path), ctx, **kwargs)
