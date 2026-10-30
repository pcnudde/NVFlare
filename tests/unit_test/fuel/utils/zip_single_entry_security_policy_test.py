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
import stat
from zipfile import ZipFile, ZipInfo

import pytest

from nvflare.fuel.utils.zip_utils import unzip_single_file_from_bytes


def archive(name, content=b"data", symlink=False):
    data = io.BytesIO()
    entry = ZipInfo(name)
    if symlink:
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(data, "w") as zipped:
        zipped.writestr(entry, content)
    return data.getvalue()


@pytest.mark.parametrize("member", ["../escaped", "nested/../../escaped", "/absolute", "\\absolute"])
def test_requested_unsafe_member_is_rejected_before_output_creation(tmp_path, member):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="unsafe"):
        unzip_single_file_from_bytes(archive("safe"), str(output), member)
    assert not output.exists()
    assert not (tmp_path / "escaped").exists()


def test_single_plain_member_extracts_only_requested_file(tmp_path):
    output = tmp_path / "output"
    unzip_single_file_from_bytes(archive("chosen", b"expected"), str(output), "chosen")
    assert (output / "chosen").read_bytes() == b"expected"
    assert [p.name for p in output.iterdir()] == ["chosen"]


def test_single_symlink_entry_is_rejected_without_materializing_target(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="unsafe symlink"):
        unzip_single_file_from_bytes(archive("link", b"../target", symlink=True), str(output), "link")
    assert not (output / "link").exists()
    assert list(output.iterdir()) == []
