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

from argparse import Namespace

import pytest

from nvflare.fuel.utils.config_service import ConfigService, search_file


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch):
    for name, value in {
        "_sections": {},
        "_config_path": [],
        "_cmd_args": None,
        "_var_dict": None,
        "_var_values": {},
    }.items():
        monkeypatch.setattr(ConfigService, name, value)


def test_configuration_initialization_loads_named_sections_and_preserves_first_search_path(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "settings.json").write_text('{"value": 4}')
    (second / "settings.json").write_text('{"value": 9}')
    ConfigService.initialize({"app": "settings.json"}, [str(first), str(second)], Namespace(argument=3), {"extra": 7})
    assert ConfigService.get_section("app") == {"value": 4}
    assert ConfigService._cmd_args == {"argument": 3} and ConfigService._var_dict == {"extra": 7}
    assert ConfigService.find_file("settings.json") == str(first / "settings.json")
    ConfigService.initialize({}, [str(first)])
    assert ConfigService._config_path == [str(first), str(second)]
    assert search_file("absent", str(first)) is None
    ConfigService.reset()
    assert ConfigService.get_section("app") is None and ConfigService._config_path == []


@pytest.mark.parametrize(
    "kwargs,exception,message",
    [
        ({"section_files": []}, TypeError, "section_files"),
        ({"config_path": "invalid"}, TypeError, "config_dirs"),
        ({"config_path": []}, ValueError, "empty"),
        ({"config_path": [1]}, ValueError, "contain str"),
        ({"var_dict": [1]}, ValueError, "var_dict"),
        ({"parsed_args": "invalid"}, ValueError, "parsed_args"),
    ],
)
def test_invalid_configuration_arguments_are_rejected(tmp_path, kwargs, exception, message):
    values = {"section_files": {}, "config_path": [str(tmp_path)], **kwargs}
    with pytest.raises(exception, match=message):
        ConfigService.initialize(**values)


def test_nonexistent_or_file_configuration_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        ConfigService.initialize({}, [str(tmp_path / "missing")])
    file = tmp_path / "not-a-directory"
    file.write_text("value")
    with pytest.raises(ValueError, match="not a valid directory"):
        ConfigService.initialize({}, [str(file)])
