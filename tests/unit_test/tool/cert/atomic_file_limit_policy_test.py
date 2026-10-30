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

from nvflare.tool.cert import file_utils as subject


@pytest.mark.parametrize("operation", ["read", "write"])
def test_descriptor_wrapper_failure_closes_owned_descriptor_and_preserves_error(tmp_path, monkeypatch, operation):
    path = tmp_path / "file"
    path.write_bytes(b"original")
    real_close, closed = subject.os.close, []

    def close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(subject.os, "close", close)
    monkeypatch.setattr(subject.os, "fdopen", Mock(side_effect=OSError("wrapper failed")))
    with pytest.raises(OSError, match="wrapper failed"):
        if operation == "read":
            subject.read_file_nofollow(str(path), 100)
        else:
            subject.write_file_nofollow(str(path), b"secret")
    assert len(closed) == 1
    if operation == "write":
        assert not path.exists()
    else:
        assert path.read_bytes() == b"original"


@pytest.mark.parametrize("custom", [False, True])
def test_bounded_file_reader_rejects_oversize_with_requested_exception_type(tmp_path, custom):
    path = tmp_path / "request.json"
    path.write_bytes(b"12345")
    factory = (lambda filename: RuntimeError("oversize: " + filename)) if custom else None
    with pytest.raises(RuntimeError if custom else ValueError, match="oversize|maximum size"):
        subject.read_file_nofollow(str(path), 4, too_large_error_factory=factory)
    assert subject.read_file_nofollow(str(path), 5) == b"12345"


def test_project_name_length_limit_provides_specific_recovery_hint():
    reason, hint = subject.safe_project_name_error("p" * 65, field_label="Deployment")
    assert reason == "Deployment must be 64 characters or fewer."
    assert "64 characters" in hint
