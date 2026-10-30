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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FilterKey
from nvflare.private import privacy_manager as module


@pytest.mark.parametrize("method,value", [("set_name", 7), ("set_props", [])])
def test_scope_configuration_rejects_wrong_types_without_replacing_defaults(method, value):
    scope = module.Scope()
    with pytest.raises(TypeError):
        getattr(scope, method)(value)
    assert scope.name == "" and scope.props == {}


@pytest.mark.parametrize("method", ["add_task_data_filter", "add_task_result_filter"])
@pytest.mark.parametrize("direction", ["invalid", FilterKey.IN])
def test_filter_policy_rejects_unsupported_direction_or_nonfilter_before_installation(method, direction):
    scope = module.Scope()
    with pytest.raises(TypeError):
        getattr(scope, method)(object(), direction)
    assert all(not chain for chain in scope.task_data_filters.values())
    assert all(not chain for chain in scope.task_result_filters.values())


def test_duplicate_or_missing_default_scope_is_rejected_and_service_policy_is_restored(monkeypatch):
    scope = module.Scope()
    scope.set_name("allowed")
    with pytest.raises(ValueError, match="duplicate scopes"):
        module.PrivacyManager([scope, scope], None, {})
    with pytest.raises(ValueError, match="does not exist"):
        module.PrivacyManager([scope], "missing", {})
    monkeypatch.setattr(module.PrivacyService, "manager", None)
    assert module.PrivacyService.get_scope("allowed") is None
    with pytest.raises(TypeError, match="PrivacyManager"):
        module.PrivacyService.initialize(Mock())
    manager = module.PrivacyManager([scope], "allowed", {})
    module.PrivacyService.initialize(manager)
    assert module.PrivacyService.get_manager() is manager
    assert module.PrivacyService.is_scope_allowed("allowed")
    assert not module.PrivacyService.is_scope_allowed("missing")
