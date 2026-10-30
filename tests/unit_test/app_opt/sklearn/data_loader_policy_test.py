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

import numpy as np
import pandas as pd
import pytest

from nvflare.app_opt.sklearn import data_loader as module


def test_csv_loading_splits_label_and_features_and_range_without_header(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("0,1,2\n1,3,4\n0,5,6\n")
    features, labels, count = module.load_data(str(path))
    np.testing.assert_array_equal(features, [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(labels, [0, 1, 0])
    assert count == 3
    features, labels, count = module.load_data_for_range(str(path), 1, 3)
    np.testing.assert_array_equal(features, [[3, 4], [5, 6]])
    np.testing.assert_array_equal(labels, [1, 0])
    assert count == 2
    assert module.get_pandas_reader("sample.xlsx") is pd.read_excel
    with pytest.raises(ValueError, match="no pandas reader"):
        module.get_pandas_reader("sample.unknown")


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("range_support", [True, False])
def test_reader_capabilities_control_range_and_header_arguments(monkeypatch, header, range_support):
    calls = []
    frame = pd.DataFrame([[0, 1], [1, 2], [0, 3]])

    def read(path, **kwargs):
        calls.append((path, kwargs))
        return frame.iloc[1:3] if "skiprows" in kwargs else frame

    read.header = True
    if range_support:
        read.skiprows = True
    monkeypatch.setattr(module, "get_pandas_reader", lambda path: read)
    _, labels, count = module.load_data_for_range("file.csv", 1, 3, require_header=header)
    assert count == 2
    np.testing.assert_array_equal(labels, [1, 0])
    expected = {"skiprows": 1, "nrows": 2} if range_support else {}
    if not header:
        expected["header"] = None
    assert calls == [("file.csv", expected)]
    module.load_data("file.csv", require_header=header)
    assert calls[-1] == ("file.csv", {} if header else {"header": None})
