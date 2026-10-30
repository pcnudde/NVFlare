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

from nvflare.job_config.file_source import FileSource


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"app_folder_type": "invalid"}, "app_folder_type"),
        ({"dest_dir": 42}, "dest_dir must be str"),
        ({"dest_dir": "/absolute"}, "must not be absolute"),
    ],
)
def test_source_validation_rejects_invalid_destination_before_export(tmp_path, kwargs, message):
    source = tmp_path / "data.txt"
    source.write_text("data")
    if kwargs.get("dest_dir") == "/absolute":
        kwargs = {**kwargs, "dest_dir": str(tmp_path / "absolute")}
    with pytest.raises(ValueError, match=message):
        FileSource(str(source), **kwargs)
    with pytest.raises(ValueError, match="does not exist"):
        FileSource(str(tmp_path / "missing"))


def test_export_registers_exact_source_destination_and_context(tmp_path):
    source = tmp_path / "data.txt"
    source.write_text("data")
    item = FileSource(str(source), "assets", "config")
    job, ctx = Mock(), object()
    item.add_to_fed_job(job, ctx)
    job.check_kwargs.assert_called_once_with(args_to_check={}, args_expected={})
    job.add_file_source.assert_called_once_with(
        src_path=str(source), dest_dir="assets", ctx=ctx, app_folder_type="config"
    )
