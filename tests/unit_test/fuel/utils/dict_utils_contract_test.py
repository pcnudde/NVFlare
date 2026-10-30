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

from nvflare.fuel.utils.dict_utils import (
    augment,
    extract_first_level_primitive,
    merge_dict,
    update_components,
    update_configs_with_envs,
)


def test_environment_expansion_visits_nested_mapping_and_list_mapping_entries():
    config = {"path": "/{site}", "nested": {"port": "{port}", "count": 2}, "items": [{"site": "{site}"}, 3]}
    assert update_configs_with_envs(config, {"site": "alpha", "port": 9}) is config
    assert config == {"path": "/alpha", "nested": {"port": "9", "count": 2}, "items": [{"site": "alpha"}, 3]}
    with pytest.raises(KeyError, match="missing"):
        update_configs_with_envs({"path": "{missing}"}, {})


def test_merge_and_primitive_extraction_do_not_retain_nested_objects_as_primitives():
    left, right = {"a": 1, "b": 2}, {"b": 3, "c": 4}
    assert merge_dict(left, right) == {"a": 1, "b": 3, "c": 4}
    assert left == {"a": 1, "b": 2} and right == {"b": 3, "c": 4}
    assert extract_first_level_primitive({"i": 1, "f": 0.5, "s": "x", "b": True, "none": None, "d": {}, "l": []}) == {
        "i": 1,
        "f": 0.5,
        "s": "x",
        "b": True,
    }


def test_augmentation_recurses_into_mappings_and_aligned_lists_and_appends_components():
    target = {"nested": {"kept": 1}, "items": [{"a": 1}], "components": [{"id": "first"}], "n": 1}
    source = {"nested": {"added": 2}, "items": [{"b": 2}], "components": [{"id": "second"}], "new": 3, "n": 9}
    assert augment(target, source) == ""
    assert target == {
        "nested": {"kept": 1, "added": 2},
        "items": [{"a": 1, "b": 2}],
        "components": [{"id": "first"}, {"id": "second"}],
        "new": 3,
        "n": 1,
    }
    assert augment(target, {"n": 9}, from_override_to=True) == ""
    assert target["n"] == 9
    assert augment({"custom": [1]}, {"custom": [2, 3]}, append_list=["custom"]) == ""


@pytest.mark.parametrize(
    "target,source,error",
    [
        ({"x": 1}, {"x": {}}, "dict in from_dict"),
        ({"x": 1}, {"x": []}, "list in from_dict"),
        ({"x": []}, {"x": [{}]}, "list length conflict"),
        ({"x": [{}]}, {"x": [1]}, "from_dict: must be dict"),
        ({"x": [1]}, {"x": [{}]}, "to_dict: must be dict"),
        ({"x": 1}, {"x": "1"}, "type conflict"),
        ({"x": {"a": 1}}, {"x": {"a": {}}}, "type conflict"),
        ({"x": [{"a": 1}]}, {"x": [{"a": {}}]}, "type conflict"),
    ],
)
def test_conflicting_configuration_shapes_produce_actionable_errors(target, source, error):
    assert error in augment(target, source)


def test_augmentation_validates_dictionary_and_append_policy_inputs():
    assert "append_list must be" in augment({}, {}, append_list=3)
    with pytest.raises(TypeError, match="to_dict"):
        augment([], {})
    with pytest.raises(TypeError, match="from_dict"):
        augment({}, [])


def test_component_updates_replace_duplicate_ids_and_preserve_unaffected_order():
    target = {"components": [{"id": "keep", "n": 1}, {"id": "replace", "n": 2}]}
    source = {"components": [{"id": "replace", "n": 3}, {"id": "new", "n": 4}]}
    assert update_components(target, source) == ""
    assert target["components"] == [{"id": "keep", "n": 1}, {"id": "replace", "n": 3}, {"id": "new", "n": 4}]
    assert update_components(target, {}) == ""
    empty = {}
    assert update_components(empty, source) == "" and empty["components"] is source["components"]


@pytest.mark.parametrize("bad_side", ["source", "target"])
def test_component_validation_identifies_missing_ids_and_wrong_collection_types(bad_side):
    target, source = {"components": [{"id": "a"}]}, {"components": [{"id": "b"}]}
    (source if bad_side == "source" else target)["components"] = [{}]
    assert f"error in {'from_dict' if bad_side == 'source' else 'target_dict'}" in update_components(target, source)
    (source if bad_side == "source" else target)["components"] = "not-list"
    with pytest.raises(TypeError, match="list"):
        update_components(target, source)
