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

import json
import sys

import pytest

from nvflare.edge.simulation.config import ConfigParser, load_class
from nvflare.edge.simulation.devices.num import NumProcessor


@pytest.fixture(autouse=True)
def isolate_import_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


def write_config(tmp_path, **changes):
    config = {"job_name": "edge-job", "processor": {"path": "nvflare.edge.simulation.devices.num.NumProcessor"}}
    config.update(changes)
    filename = tmp_path / "device.json"
    filename.write_text(json.dumps(config))
    return str(filename)


def test_defaults_and_explicit_options_are_loaded_without_modifying_processor_arguments(tmp_path):
    parser = ConfigParser(write_config(tmp_path))
    assert parser.get_job_name() == "edge-job"
    assert (parser.get_num_devices(), parser.get_num_workers(), parser.get_endpoint(), parser.get_job_timeout) == (
        100,
        10,
        None,
        60.0,
    )
    assert isinstance(parser.get_processor(), NumProcessor)
    parser = ConfigParser(
        write_config(
            tmp_path,
            endpoint="http://edge:8080",
            num_devices=7,
            num_workers=2,
            get_job_timeout=3,
            processor={
                "path": "nvflare.edge.simulation.devices.num.NumProcessor",
                "python_path": "/custom",
                "args": {"min_train_time": "{low}", "max_train_time": 4},
            },
        )
    )
    processor = parser.get_processor({"low": "0.1"})
    assert processor.min_train_time == "0.1" and processor.max_train_time == 4
    assert parser.processor_args["min_train_time"] == "{low}"
    assert (parser.get_num_devices(), parser.get_num_workers(), parser.get_endpoint(), parser.get_job_timeout) == (
        7,
        2,
        "http://edge:8080",
        3,
    )
    assert "/custom" in sys.path


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"processor": None}, "processor is not defined"),
        ({"processor": {}}, "path for processor"),
        ({"processor": {"path": "dict"}}, "not a subclass"),
        ({"job_name": None}, "job_name"),
        ({"endpoint": 3}, "endpoint"),
        ({"num_devices": -1}, "num_devices"),
        ({"num_workers": 1.5}, "num_workers"),
        ({"get_job_timeout": 0}, "get_job_timeout"),
    ],
)
def test_invalid_configuration_rejected_before_processor_execution(tmp_path, changes, error):
    with pytest.raises((ValueError, TypeError), match=error):
        ConfigParser(write_config(tmp_path, **changes))


def test_nested_variable_substitution_handles_adjacent_and_unknown_tokens():
    parser = ConfigParser.__new__(ConfigParser)
    original = {"nested": ["{a}{b}/{a}", "{missing}", 3, None], "flag": False}
    result = parser._variable_substitution(original, {"a": "long-value", "b": "x"})
    assert result == {"nested": ["long-valuex/long-value", "{missing}", 3, None], "flag": False}
    assert original["nested"][0] == "{a}{b}/{a}"


def test_load_class_supports_builtin_and_qualified_classes_and_reports_bad_names():
    assert load_class("dict") is dict
    assert load_class("json.JSONDecoder") is json.JSONDecoder
    for name in ("missing_builtin", "json.DoesNotExist", "missing_module.Class"):
        with pytest.raises(TypeError, match="Can't load class"):
            load_class(name)
