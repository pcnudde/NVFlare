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

from nvflare.fuel.hci import chunk as subject


@pytest.mark.parametrize(
    "arguments,fragment", [((0, 0, 0), "invalid chunk marker"), ((subject.MARKER_END, 1, 0), "num1 must be 0")]
)
def test_chunk_header_rejects_invalid_marker_and_terminal_sequence(arguments, fragment):
    with pytest.raises(ValueError, match=fragment):
        subject.Header(*arguments)


def test_incomplete_header_and_out_of_order_chunk_never_dispatch_payload():
    with pytest.raises(ValueError, match="Prefix too short"):
        subject.Header.from_bytes(b"short")
    callback = Mock()
    receiver = subject.Receiver(callback)
    with pytest.raises(RuntimeError, match="received seq 2.*expected seq 1"):
        receiver.receive(subject.Header(subject.MARKER_DATA, 2, 1).to_bytes() + b"x")
    callback.assert_not_called()


def test_empty_send_is_valid_frame_and_closed_endpoints_reject_reuse():
    chunks = []
    sender = subject.Sender(chunks.append)
    sender.send(None)
    sender.close()
    receiver = subject.Receiver(Mock())
    assert receiver.receive(b"".join(chunks))
    assert receiver.done and sender.closed
    with pytest.raises(RuntimeError, match="receiver is already done"):
        receiver.receive(b"more")
    with pytest.raises(RuntimeError, match="sender is already closed"):
        sender.send(b"more")
    with pytest.raises(RuntimeError, match="sender is already closed"):
        sender.close()


def test_bad_terminal_checksum_cannot_be_reported_as_success():
    receiver = subject.Receiver(None)
    bad_checksum = receiver.checksum.result() + 1
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        receiver.receive(subject.Header(subject.MARKER_END, 0, bad_checksum).to_bytes())


@pytest.mark.parametrize(
    "data,cursor,error",
    [("text", 0, "only chunk bytes"), (b"data", -1, "out of data range"), (b"data", 4, "out of data range")],
)
def test_chunk_parser_rejects_invalid_buffers_or_cursor_without_processing(data, cursor, error):
    callback = Mock()
    with pytest.raises(ValueError, match=error):
        subject.chunk_it(subject.ChunkState(), data, cursor, callback)
    callback.assert_not_called()


def test_empty_partial_input_preserves_parser_state_without_dispatch():
    state, callback = subject.ChunkState(), Mock()
    assert subject.chunk_it(state, b"", 0, callback) is state
    callback.assert_not_called()
