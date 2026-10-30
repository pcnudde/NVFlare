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

import pytest
from pyhocon import ConfigFactory, ConfigTree

from nvflare.tool.job.config import config_indexer as subject


def test_nested_list_index_keeps_physical_index_and_component_hints():
    indices = subject.build_list_reverse_order_index([[1, None], [], None], "items", None, None, None)
    assert sorted(indices) == ["items[0][0]", "items[1]"]
    assert indices["items[0][0]"][0].value == 1
    assert indices["items[0][0]"][0].parent_key.key == "items[0]"
    assert indices["items[1]"][0].value == []
    names = subject.build_list_reverse_order_index(["Dense"], "name", None, None, None)
    assert names["name[0]"][0].component_name == "Dense"
    paths = subject.build_list_reverse_order_index(["package.Layer"], "path", None, None, None)
    assert paths["path[0]"][0].component_name == "Layer"


def test_config_tree_list_and_excluded_keys_are_indexed_independently():
    config = ConfigFactory.from_dict({"drop": 1, "by_value": "omit", "keep": 3})
    indices = subject.build_dict_reverse_order_index(config, excluded_keys=["drop", "omit"])
    assert set(indices) == {"keep"} and indices["keep"][0].value == 3
    mixed = subject.build_list_reverse_order_index(
        [ConfigFactory.from_dict({"name": "Block", "units": 2})], "layers", [], None, {}
    )
    assert mixed["units"][0].value == 2
    assert mixed["name"][0].component_name == "Block"


@pytest.mark.parametrize("container", ["list", "dict"])
def test_unhandled_configuration_value_reports_its_type(container):
    with pytest.raises(RuntimeError, match="Unhandled data type.*object"):
        if container == "list":
            subject.build_list_reverse_order_index([object()], "layers", [], None, {})
        else:
            config = ConfigTree()
            config.put("unsupported", object())
            subject.build_dict_reverse_order_index(config)


def test_missing_config_is_reported_with_input_filename(tmp_path):
    with pytest.raises(RuntimeError, match="failed|filed") as error:
        subject.load_pyhocon_conf("absent.json", str(tmp_path))
    assert "absent.json" in str(error.value)


def test_index_without_parent_retains_its_own_component_name():
    index = subject.KeyIndex("root", component_name="Block")
    assert subject.update_index_comp_name(index) is index
    assert index.component_name == "Block"
