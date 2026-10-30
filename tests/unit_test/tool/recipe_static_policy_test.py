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

import ast
import json
from enum import Enum
from unittest.mock import Mock

import pytest

from nvflare.tool.recipe import recipe_cli as subject


@pytest.fixture
def static_modules(monkeypatch):
    modules = {
        "pkg.defaults": ast.parse('class Constants:\n    DEFAULT: str = "full"\n    UNKNOWN = factory()\n'),
        "pkg.base": ast.parse("class Base: pass\n"),
        "pkg.recipe": ast.parse(
            "from pkg.defaults import Constants\nimport pkg.base as roots\nclass Example(roots.Base):\n    def __init__(self, required: int, /, mode=Constants.DEFAULT, *metrics: str, note=None, **kwargs): pass\n"
        ),
        "pkg.export": ast.parse("from pkg.recipe import Example\n"),
        "pkg.a": ast.parse("from pkg.b import Example\n"),
        "pkg.b": ast.parse("from pkg.a import Example\n"),
    }
    monkeypatch.setattr(subject, "_static_module_tree", Mock(side_effect=modules.get))
    return modules


def test_static_catalog_resolves_reexported_class_bases_and_constant_defaults_without_importing_modules(static_modules):
    module, tree, node = subject._static_class_node("pkg.export", "Example")
    assert module == "pkg.recipe"
    assert subject._static_base_references(module, tree, node) == [("pkg.base", "Base")]
    parameters = subject._static_recipe_parameters("pkg.export", "Example")
    assert [p["name"] for p in parameters] == ["required", "mode", "metrics", "note", "kwargs"]
    assert parameters[0]["kind"] == "positional_only"
    assert parameters[0]["required"]
    assert parameters[1]["default"] == "full"
    assert parameters[2]["kind"] == "var_positional"
    assert parameters[-1]["kind"] == "var_keyword"
    assert subject._static_member_value("pkg.defaults", "Constants", "DEFAULT") == "full"
    assert subject._static_member_value("pkg.defaults", "Constants", "UNKNOWN") is None
    assert subject._static_member_value("pkg.defaults", "Constants", "MISSING") is None
    assert subject._static_member_value("pkg.defaults", "Absent", "DEFAULT") is None


def test_cyclic_reexports_and_absent_sources_are_not_executed_or_reported_as_valid_classes(static_modules):
    assert subject._static_class_node("pkg.a", "Example")[2] is None
    assert subject._static_class_node("missing", "Example")[2] is None
    assert subject._static_recipe_parameters("missing", "Example") == []
    assert (
        subject._ast_default_value(
            ast.parse("Constants.UNKNOWN", mode="eval").body, {"Constants": ("pkg.defaults", "Constants")}
        )
        == "Constants.UNKNOWN"
    )


@pytest.mark.parametrize(
    "value,normalized,preserved",
    [
        ({"B", "A"}, ["a", "b"], ["A", "B"]),
        (7, ["7"], ["7"]),
        ("  ", [], []),
        (("Raw-Values", ""), ["raw_values"], ["Raw-Values"]),
    ],
)
def test_recipe_metadata_normalization_preserves_display_values_and_normalizes_filters(value, normalized, preserved):
    assert subject._as_string_list(value) == normalized
    assert subject._as_preserved_string_list(value) == preserved


@pytest.mark.parametrize(
    "algorithm,expected",
    [
        ("xgboost_vertical", ["vertical"]),
        ("xgboost_horizontal", ["horizontal"]),
        ("fedstats", ["cross_site_statistics"]),
        ("psi", ["sample_intersection"]),
    ],
)
def test_catalog_heterogeneity_describes_algorithm_data_partitioning(algorithm, expected):
    assert subject._heterogeneity_support({"algorithm": algorithm}, {}) == expected


def test_catalog_values_are_json_safe_for_enums_sets_nested_containers_and_unknown_defaults():
    class Mode(Enum):
        FULL = "full"

    class Opaque:
        def __repr__(self):
            return "opaque-default"

    result = subject._json_safe_value({"mode": Mode.FULL, "values": {2, 1}, "nested": (Opaque(),)})
    assert json.loads(json.dumps(result)) == {"mode": "full", "values": [1, 2], "nested": ["opaque-default"]}
