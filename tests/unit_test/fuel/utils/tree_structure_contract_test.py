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

import pytest

from nvflare.fuel.utils.tree_utils import build_forest, build_path, forest_to_dict


def test_forest_reconstructs_parent_links_multiple_roots_and_nested_display():
    objects = [
        {"name": "leaf", "fqn": "root.child.leaf"},
        {"name": "root", "fqn": "root"},
        {"name": "child", "fqn": "root.child"},
        {"name": "other", "fqn": "other"},
    ]
    forest = build_forest(objects, lambda obj: obj["name"], lambda obj: obj["fqn"])
    assert forest.nodes["leaf"].parent is forest.nodes["child"]
    assert forest.nodes["child"].parent is forest.nodes["root"]
    assert forest.nodes["root"].children == [forest.nodes["child"]]
    assert forest_to_dict(forest, lambda obj: obj["name"]) == {
        "roots": ["root", "other"],
        "trees": [{"root": [{"child": ["leaf"]}]}, "other"],
    }


@pytest.mark.parametrize(
    "objects,error",
    [
        ([{"name": "one", "fqn": "a"}, {"name": "two", "fqn": "a"}], "same FQN"),
        ([{"name": "one", "fqn": "a"}, {"name": "one", "fqn": "b"}], "same name"),
        ([{"name": "one", "fqn": "missing.child"}], "missing node definition"),
    ],
)
def test_invalid_forest_identity_or_missing_parent_is_rejected(objects, error):
    with pytest.raises(ValueError, match=error):
        build_forest(objects, lambda obj: obj["name"], lambda obj: obj["fqn"])


def test_path_builder_propagates_accessors_and_reports_cycles_without_recursion_overflow():
    parents = {"leaf": "root", "root": None}
    assert build_path("leaf", lambda obj, suffix: obj + suffix, lambda obj, suffix: parents[obj], suffix="!") == (
        "",
        ["root!", "leaf!"],
    )
    error, path = build_path("self", lambda obj: obj, lambda obj: obj)
    assert "circular parent ref self" == error and path == ["self"]
    for name_func, parent_func in ((None, lambda obj: None), (lambda obj: obj, None)):
        with pytest.raises(ValueError, match="not callable"):
            build_path("leaf", name_func, parent_func)
