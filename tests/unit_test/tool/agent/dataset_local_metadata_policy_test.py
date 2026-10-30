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

from pathlib import Path

import pytest

from nvflare.tool.agent import dataset_inspect as subject


def reads():
    return {"files_read": 0, "bytes_read": 0}


def test_unreadable_csv_has_no_invented_schema_or_read_accounting(tmp_path):
    missing = tmp_path / "absent.csv"
    audit = reads()
    assert subject._read_bounded(missing, 100, audit) == (None, False)
    schema = subject._tabular_schema([missing], 100, audit)
    assert schema["features"] is None and schema["row_count"] is None
    assert audit == reads()
    assert subject._tabular_schema([tmp_path / "image.png"], 100, audit) == {}
    assert subject._data_extension(Path("volume.nii.gz")) == ".nii.gz"


def test_real_image_depth_inspection_reports_metadata_without_reading_pixel_values_into_output(tmp_path):
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.png"
    image.new("RGB", (2, 3), color=(1, 2, 3)).save(path)
    audit = reads()
    assert subject._sample_pixel_depth([path], audit) == "uint8"
    assert audit == {"files_read": 1, "bytes_read": path.stat().st_size}


def test_invalid_parquet_shard_marks_row_count_approximate_without_fabricating_schema(tmp_path):
    pytest.importorskip("pyarrow.parquet")
    corrupt = tmp_path / "broken.parquet"
    corrupt.write_bytes(b"not parquet")
    audit = reads()
    schema = subject._parquet_schema([corrupt], audit)
    assert schema["schema_available"] is False and schema["row_count_approximate"] is True
    assert schema["features"] is None and audit == reads()


def test_real_parquet_schema_respects_column_budget_and_records_valid_footer_only(tmp_path, monkeypatch):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    valid, corrupt = tmp_path / "valid.parquet", tmp_path / "bad.parquet"
    parquet.write_table(arrow.table({"age": [1, 2], "label": ["a", "b"]}), valid)
    corrupt.write_bytes(b"bad")
    monkeypatch.setattr(subject, "MAX_SCHEMA_COLUMNS", 1)
    audit = reads()
    schema = subject._parquet_schema([corrupt, valid], audit)
    assert schema["features"] == ["age"] and schema["dtypes"] == ["numeric"]
    assert schema["column_count"] == 1 and schema["columns_truncated"]
    assert schema["row_count"] == 2 and schema["row_count_approximate"]
    assert audit["files_read"] == 1 and audit["bytes_read"] == valid.stat().st_size
