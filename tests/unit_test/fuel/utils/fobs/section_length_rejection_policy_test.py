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

import io

import pytest

from nvflare.fuel.utils.fobs import lobs as subject


@pytest.mark.parametrize(
    "payload, expect_datum, message",
    [
        (b"x", False, "header bytes"),
        (subject._Header(subject.MARKER_MAIN, 0, 0).to_bytes(), False, "invalid size"),
        (subject._Header(subject.MARKER_DATUM_TEXT, 0, 1).to_bytes(), False, "expect main"),
        (subject._Header(subject.MARKER_MAIN, 0, 1).to_bytes(), True, "expect datum"),
        (subject._Header(subject.MARKER_MAIN, 0, 3).to_bytes(), False, "cannot get 3 data bytes"),
        (subject._Header(subject.MARKER_MAIN, 0, 3).to_bytes() + b"xx", False, "expect 3 bytes but got 2"),
        (
            subject._Header(subject.MARKER_DATUM_BLOB, 0, 15).to_bytes() + b"x" * 15,
            True,
            "not enough data for datum ID",
        ),
        (subject._Header(subject.MARKER_DATUM_BLOB, 0, 16).to_bytes(), True, "cannot get 16 for datum ID"),
        (subject._Header(subject.MARKER_DATUM_BLOB, 0, 16).to_bytes() + b"x", True, "expect 16 bytes for datum ID"),
    ],
)
def test_truncated_or_misordered_section_fails_before_deserialization(payload, expect_datum, message):
    with pytest.raises(RuntimeError, match=message):
        subject._get_one_section(io.BytesIO(payload), expect_datum)


def test_short_header_rejected_and_empty_stream_has_no_main_body():
    with pytest.raises(ValueError, match="Header too short"):
        subject._Header.from_bytes(b"short")
    with pytest.raises(RuntimeError, match="missing main body"):
        subject.load_from_stream(io.BytesIO())
