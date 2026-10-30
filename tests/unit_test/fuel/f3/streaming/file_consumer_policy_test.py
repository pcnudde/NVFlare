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

from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.streaming import file_downloader as subject


@pytest.mark.parametrize("failure", [False, True])
def test_file_consumer_counts_chunks_and_closes_file_on_terminal_outcome(tmp_path, failure):
    consumer = subject._ChunkConsumer(str(tmp_path))
    try:
        assert consumer.consume("reference", {}, b"abc") == {subject._StateKey.RECEIVED_BYTES: 3}
        assert consumer.consume("reference", {}, b"defg") == {subject._StateKey.RECEIVED_BYTES: 7}
        if failure:
            consumer.download_failed("reference", "source unavailable")
        else:
            consumer.download_completed("reference")
        assert consumer.file.closed
        assert consumer.error == ("source unavailable" if failure else None)
        with open(consumer.file_path, "rb") as stream:
            assert stream.read() == b"abcdefg"
    finally:
        consumer.file.close()


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_download_file_rejects_invalid_destination_before_starting_transfer(tmp_path, monkeypatch, path_kind):
    destination = tmp_path / "destination"
    if path_kind == "file":
        destination.write_text("not a directory")
    transfer = Mock()
    monkeypatch.setattr(subject, "download_object", transfer)
    with pytest.raises(ValueError, match="does not exist" if path_kind == "missing" else "not a valid dir"):
        subject.download_file("site-a", "ref", 1, Mock(), location=str(destination))
    transfer.assert_not_called()


def test_download_file_passes_transfer_policy_and_returns_completed_path(tmp_path, monkeypatch):
    def transfer(**kwargs):
        consumer = kwargs["consumer"]
        try:
            consumer.consume(kwargs["ref_id"], {}, b"file-content")
            consumer.download_completed(kwargs["ref_id"])
        finally:
            consumer.file.close()

    download = Mock(side_effect=transfer)
    monkeypatch.setattr(subject, "download_object", download)
    cell = Mock()
    error, path = subject.download_file(
        "site-a", "reference", 3, cell, location=str(tmp_path), secure=True, optional=True
    )
    assert error is None
    with open(path, "rb") as stream:
        assert stream.read() == b"file-content"
    kwargs = download.call_args.kwargs
    assert (kwargs["from_fqcn"], kwargs["ref_id"], kwargs["per_request_timeout"], kwargs["cell"]) == (
        "site-a",
        "reference",
        3,
        cell,
    )
    assert kwargs["secure"] is kwargs["optional"] is True
