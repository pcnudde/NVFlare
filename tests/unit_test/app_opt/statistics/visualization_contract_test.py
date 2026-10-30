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

from unittest.mock import MagicMock

import pandas as pd
import pytest

from nvflare.app_opt.statistics.visualization import statistics_visualization as subject


def _histograms():
    return {"age": {"histogram": {"a": {"train": [(0, 1, 2), (1, 2, 6)]}, "b": {"test": [(0, 1, 3), (1, 2, 1)]}}}}


@pytest.mark.parametrize(
    "display_format,expected",
    [
        ("sample_count", {"a-train": [2, 6], "b-test": [3, 1]}),
        ("percent", {"a-train": [0.25, 0.75], "b-test": [0.75, 0.25]}),
    ],
)
def test_histogram_dataframe_preserves_bin_edges_and_normalizes_each_dataset(monkeypatch, display_format, expected):
    view = subject.Visualization()
    monkeypatch.setattr(view, "import_modules", lambda: (MagicMock(), pd))
    frames = view.get_histogram_dataframes(_histograms(), display_format=display_format)
    assert list(frames) == ["age"]
    assert frames["age"].index.tolist() == [0, 1]
    assert frames["age"].to_dict("list") == expected
    assert view.sum_counts_in_histogram([(0, 1, 2), (1, 2, 6)]) == 8


def test_show_stats_filters_feature_names_and_displays_real_dataframe(monkeypatch):
    view = subject.Visualization()
    display = MagicMock()
    monkeypatch.setattr(view, "import_modules", lambda: (display, pd))
    values = {"mean": {"site": {"train": 1.5}}, "count": {"site": {"train": 3}}}
    view.show_stats({"age": values, "ignored": values}, white_list_features=["age"])
    frame = display.call_args.args[0]
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc["site-train", "mean"] == 1.5 and frame.loc["site-train", "count"] == 3
    display.assert_called_once()
    assert subject.convert_data(values) == {"mean": {"site-train": 1.5}, "count": {"site-train": 3}}


@pytest.mark.parametrize(
    "mode,count,subplots", [("both", 2, True), ("main", 1, False), ("subplot", 1, True), ("unknown", 0, False)]
)
def test_plot_policy_dispatches_expected_main_and_subplot_requests(mode, count, subplots, capsys):
    frame = MagicMock(spec=pd.DataFrame)
    view = subject.Visualization()
    view.show_dataframe_plots({"age": frame}, plot_type=mode)
    assert frame.plot.line.call_count == count
    if count:
        assert frame.plot.line.call_args.kwargs.get("subplots", False) is subplots
        assert frame.plot.line.call_args.kwargs["title"] == "age"
    else:
        assert "not supported plot type" in capsys.readouterr().out


def test_show_histograms_passes_transformed_data_and_selected_plot_type(monkeypatch):
    view = subject.Visualization()
    monkeypatch.setattr(view, "import_modules", lambda: (MagicMock(), pd))
    plot = MagicMock()
    monkeypatch.setattr(view, "show_dataframe_plots", plot)
    view.show_histograms(_histograms(), plot_type="main")
    assert plot.call_args.args[1] == "main"
    assert plot.call_args.args[0]["age"].to_dict("list")["a-train"] == [2, 6]
    assert view._get_target_features(["a", "b"], None) == ["a", "b"]
    assert view._get_target_features(["a", "b"], ["b"]) == ["b"]
