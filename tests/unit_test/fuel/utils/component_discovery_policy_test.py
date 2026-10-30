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

import importlib
import sys
from unittest.mock import Mock

import pytest

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils import class_utils as subject


@pytest.fixture
def scanner_package(tmp_path, monkeypatch):
    root = tmp_path / "round95_scanpkg"
    (root / "components" / "libs").mkdir(parents=True)
    (root / "other").mkdir()
    for folder in [root, root / "components", root / "components" / "libs", root / "other"]:
        (folder / "__init__.py").write_text("")
    public = "from nvflare.apis.fl_component import FLComponent\nclass Public(FLComponent): pass\nclass _Private(FLComponent): pass\nclass Plain: pass\n"
    (root / "components" / "first.py").write_text(public)
    (root / "components" / "second.py").write_text(public)
    (root / "components" / "libs" / "optional.py").write_text(
        "from nvflare.apis.fl_component import FLComponent\nclass Optional(FLComponent): pass\n"
    )
    (root / "components" / "broken.py").write_text("raise RuntimeError('unavailable native extension')\n")
    (root / "components" / "deprecated.py").write_text("raise AssertionError('deprecated module imported')\n")
    (root / "other" / "unselected.py").write_text("raise AssertionError('unselected module imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(subject, "DEPRECATED_PACKAGES", ["round95_scanpkg.components.deprecated"])
    monkeypatch.setattr(subject, "create_classes_table_static", Mock(return_value={}))
    importlib.invalidate_caches()
    try:
        yield "round95_scanpkg"
    finally:
        for key in list(sys.modules):
            if key == "round95_scanpkg" or key.startswith("round95_scanpkg."):
                sys.modules.pop(key)
        importlib.invalidate_caches()


@pytest.mark.parametrize("exclude_libs", [True, False])
def test_scanner_registers_only_public_local_components_and_contains_import_errors(
    scanner_package, exclude_libs, caplog
):
    scanner = subject.ModuleScanner([scanner_package], ["components"], exclude_libs=exclude_libs)
    table = scanner.create_classes_table()
    assert set(table["Public"]) == {f"{scanner_package}.components.first", f"{scanner_package}.components.second"}
    assert "_Private" not in table and "Plain" not in table and "FLComponent" not in table
    assert ("Optional" in table) is not exclude_libs
    assert "unavailable native extension" in caplog.text
    scanner._class_table = table
    with pytest.raises(ConfigError, match="Multiple modules"):
        scanner.get_module_name("Public")
    with pytest.raises(ConfigError, match="Cannot find class"):
        scanner.get_module_name("Unknown")
    if not exclude_libs:
        assert scanner.get_module_name("Optional") == f"{scanner_package}.components.libs.optional"


@pytest.mark.parametrize(
    "config",
    [
        {"path": None, "class_path": "builtins.dict"},
        {"path": "", "name": "dict"},
        {"class_path": None, "name": "dict"},
        {"class_path": ""},
        {"class_path": "dict"},
        {"name": None},
        {"name": ""},
        {},
    ],
)
def test_invalid_higher_priority_component_key_cannot_fall_through(config):
    with pytest.raises(ConfigError):
        subject.get_class_path_from_config(config, resolve_name=lambda _: "builtins")


def test_component_name_resolution_requires_explicit_known_module_and_constructor_errors_are_contextual():
    with pytest.raises(ConfigError, match="resolve_name required"):
        subject.get_class_path_from_config({"name": "dict"})
    with pytest.raises(ConfigError, match="Cannot find component"):
        subject.get_class_path_from_config({"name": "dict"}, resolve_name=lambda _: None)
    assert subject.get_class_path_from_config({"name": "dict"}, resolve_name=lambda _: "builtins") == "builtins.dict"
    assert subject.instantiate_class("builtins.dict", {"answer": 42}) == {"answer": 42}
    with pytest.raises(ValueError, match="builtins.int has parameters error"):
        subject.instantiate_class("builtins.int", {"unsupported": 1})
