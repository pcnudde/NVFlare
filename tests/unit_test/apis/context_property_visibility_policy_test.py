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

from nvflare.apis.fl_context import FLContext, FLContextManager, to_string


def test_private_custom_properties_remain_local_and_public_keys_are_explicit():
    manager = FLContextManager()
    first = manager.new_context()
    assert first.get_custom_prop("missing", "fallback") == "fallback"
    first.set_custom_prop("answer", 42)
    first.set_custom_prop("second", 7)
    first.set_prop("public", 1, private=False, sticky=False)
    assert first.get_custom_prop("answer") == 42
    assert first.get_custom_prop("second") == 7
    assert first.get_custom_prop("missing", "fallback") == "fallback"
    assert first.public_key_exists("public")
    assert not first.public_key_exists("absent")
    assert first.get_prop_detail("public") == {"value": 1, "private": False, "sticky": False}
    assert first.get_prop_detail("absent") is None
    assert "public" in first.get_prop_keys()
    assert "answer" not in first.get_all_public_props()
    assert manager.new_context().get_custom_prop("answer") is None
    assert to_string(0) == "public:non-sticky"


def test_invalid_property_keys_and_missing_engine_cannot_mutate_context():
    ctx = FLContext()
    with pytest.raises(ValueError, match="prop key must be str"):
        ctx.set_prop(1, "value")
    ctx.remove_prop(1)
    assert ctx.get_prop_keys() == []
    with pytest.raises(RuntimeError, match="missing engine"):
        ctx.get_workspace()


def test_conflicting_sticky_visibility_does_not_overwrite_shared_value():
    manager = FLContextManager(public_stickers={"shared": "safe"})
    ctx = manager.new_context()
    ctx.remove_prop("shared")
    assert ctx.set_prop("shared", "private", private=True, sticky=True) is False
    assert ctx.get_prop("shared") == "safe"
    assert ctx.public_key_exists("shared")
