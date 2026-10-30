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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci import binary_proto as subject


@pytest.mark.parametrize("chunks,expected", [([b"a", b"bc"], b"abc"), ([b"a", b""], None)])
def test_fragmented_header_reassembly_honors_peer_close_without_polling_forever(chunks, expected):
    receiver = Mock()
    receiver.recv.side_effect = chunks
    handler = subject.ExchangeHandler(receiver, Mock())
    assert handler._must_recv(3) == expected
    assert receiver.recv.call_count == 2


@pytest.mark.parametrize("failure", ["body", "footer", "marker", "checksum"])
def test_binary_parser_reports_incomplete_or_corrupted_frames_before_acknowledging(failure):
    receiver, processor = Mock(), Mock()
    if failure == "body":
        chunks, reason = [b""], "failed to receive"
    elif failure == "footer":
        chunks, reason = [b"abc", b""], "cannot get footer"
    else:
        chunks = [b"abc", subject.FOOTER_STRUCT.pack(1 if failure == "marker" else 0, 0)]
        reason = "footer marker" if failure == "marker" else "checksum mismatch"
    receiver.recv.side_effect = chunks
    handler = subject.ExchangeHandler(receiver, processor)
    with pytest.raises(RuntimeError, match=reason):
        handler._parse_binary(3)
    if failure != "body":
        processor.process.assert_called_once_with(b"abc", subject.CT_BINARY)


@pytest.mark.parametrize("data", [b"", bytes([subject.BINARY_MARKER])])
def test_exchange_requires_type_marker_and_complete_header(data):
    receiver = Mock()
    stream = io.BytesIO(data)
    receiver.recv.side_effect = stream.read
    handler = subject.ExchangeHandler(receiver, Mock())
    with pytest.raises(RuntimeError, match="marker|header"):
        handler.receive_and_parse()


def test_text_exchange_stops_at_protocol_terminator_and_discards_later_bytes():
    source = iter([b"f", b"irst" + subject.LINE_END.encode() + b"second" + subject.ALL_END.encode(), b"not consumed"])
    receiver = Mock(recv=Mock(side_effect=lambda size: next(source)))
    content_type, text, path = subject.receive_all(receiver)
    assert content_type == subject.CT_TEXT and text == "firstsecond" and path is None
    assert receiver.recv.call_count == 2


def test_message_processor_writes_binary_chunks_to_owned_tempfile_and_closes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "tempfile", SimpleNamespace(gettempdir=lambda: str(tmp_path)))
    processor = subject.MsgDataProcessor()
    try:
        assert processor.process(b"first", subject.CT_BINARY) is False
        assert processor.process(b"second", subject.CT_BINARY) is False
        processor.finalize()
        assert processor.file.closed
        assert list(tmp_path.iterdir())[0].read_bytes() == b"firstsecond"
    finally:
        if processor.file:
            processor.file.close()


def test_mismatched_generator_length_closes_source_and_never_emits_footer():
    sender, generator = Mock(), Mock()
    generator.data_size.return_value = 3
    generator.generate.side_effect = [b"x", b""]
    with pytest.raises(RuntimeError, match="generated body size 1 != expected body size 3"):
        subject.send_binary_data(sender, generator, "meta")
    generator.close.assert_called_once_with()
    assert sender.sendall.call_count == 4


def test_source_stat_failure_closes_open_file_before_propagating(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"data")
    with path.open("rb") as handle:
        monkeypatch.setattr(subject, "open", Mock(return_value=handle), raising=False)
        monkeypatch.setattr(subject.os, "fstat", Mock(side_effect=OSError("stat failed")))
        with pytest.raises(OSError, match="stat failed"):
            subject.GenerateDataFromFile(str(path))
        assert handle.closed
