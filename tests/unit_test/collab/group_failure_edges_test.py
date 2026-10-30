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

from nvflare.apis.signal import Signal
from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.call_utils import check_call_args, get_collab_object_name
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.exceptions import CollabCallError, RunAborted
from nvflare.collab.api.group import Group
from nvflare.collab.api.group_call_context import GroupCallContext, ResultWaiter
from nvflare.collab.api.proxy import Proxy
from nvflare.collab.api.publish_interface import MethodInterface, MethodParameter


@pytest.fixture(autouse=True)
def preserve_context():
    previous = get_call_context()
    try:
        yield
    finally:
        set_call_context(previous)


@pytest.mark.parametrize("abort,raise_on_abort", [(False, True), (True, True), (True, False)])
def test_synchronous_group_dispatch_failure_releases_only_acquired_slots_and_propagates_abort(abort, raise_on_abort):
    app = ClientApp(object())
    proxies = [Proxy(app, name, name, Mock(), {"work": []}) for name in ("first", "second")]
    group = Group(app, Signal(), proxies)
    waiter = ResultWaiter([proxy.name for proxy in proxies])
    options = CallOption(parallel=2)
    items = []
    for proxy in proxies:
        context = app.new_context("client", proxy.name, set_call_ctx=False)
        gcc = GroupCallContext(app, proxy.name, options, "work", None, {}, context, waiter)
        gcc.set_completion_cb(group._call_completed, gcc=gcc, proxy=proxy)
        items.append((proxy, gcc, (), {}))
    failure = RunAborted("run ended") if abort else ValueError("cannot serialize")
    proxies[0].backend.call_target_in_group.side_effect = failure

    def second(gcc, *args, **kwargs):
        gcc.set_result(7)
        gcc.call_completed()

    proxies[1].backend.call_target_in_group.side_effect = second
    if abort and raise_on_abort:
        with pytest.raises(RunAborted) as caught:
            group._dispatch_work_items(items, "work", waiter, 2)
        assert caught.value is failure
    else:
        group._dispatch_work_items(items, "work", waiter, 2, raise_on_abort=raise_on_abort)
    assert waiter.standing_call_count == 0
    assert waiter.results.num_whole_items_received == 2
    assert waiter.results.failures["first"].cause is failure
    if abort:
        assert waiter.results.failures["second"].cause is failure
        proxies[1].backend.call_target_in_group.assert_not_called()
    else:
        assert len(waiter.results.freeze()) == 1
        assert list(waiter.results.freeze()) == [("second", 7)]
    assert group.size == 2 and group.members == proxies


def test_proxy_child_resolution_requires_unambiguous_qualified_target_before_dispatch():
    app = ClientApp(object())
    backend = Mock()
    parent = Proxy(app, "site", "site", backend, {})
    first = Proxy(app, "site.first", "", backend, {"work": ["value"]})
    second = Proxy(app, "site.second", "", backend, {"other": []})
    parent.add_child("first", first)
    parent.add_child("second", second)
    selected, method = parent._find_interface("work")
    assert selected is first and list(method) == ["value"]
    second.target_interface = first.target_interface
    with pytest.raises(RuntimeError, match="multiple collab objects"):
        parent._find_interface("work")
    with pytest.raises(CollabCallError, match="does not have collab target"):
        parent.call_func(CallOption(target="absent"), "work", (), {})
    group = Group(app, Signal(), [parent], CallOption(target="first"))
    assert group._get_work_proxy(parent, "work") is first
    group._call_opt.target = "absent"
    with pytest.raises(RuntimeError, match="does not have collab target"):
        group._get_work_proxy(parent, "work")
    backend.call_target.assert_not_called()
    with pytest.raises(ValueError, match="no proxies"):
        Group(app, Signal(), [])


@pytest.mark.parametrize(
    "name,kind,required,match",
    [
        (3, "KEYWORD_ONLY", True, "name must be str"),
        ("value", "VAR_POSITIONAL", True, "unsupported parameter kind"),
        ("value", "KEYWORD_ONLY", "yes", "must be bool"),
    ],
)
def test_wire_method_parameters_reject_nonserializable_or_flexible_argument_contracts(name, kind, required, match):
    with pytest.raises((TypeError, ValueError), match=match):
        MethodParameter(name, kind, required)


def test_method_validation_rejects_missing_unknown_and_unnormalized_arguments_without_mutating_input():
    method = MethodInterface([{"name": "value", "kind": "POSITIONAL_OR_KEYWORD", "required": True}])
    with pytest.raises(TypeError, match="unexpected keyword"):
        method.bind((), {"unknown": 1})
    with pytest.raises(TypeError, match="missing required"):
        method.bind((), {})
    for args, kwargs, message in [
        ([1], {}, "normalized"),
        ([], {"unknown": 1}, "unexpected keyword"),
        ([], {}, "missing required"),
    ]:
        with pytest.raises(TypeError, match=message):
            check_call_args("work", method, args, kwargs)
    with pytest.raises(RuntimeError, match="only supports"):
        check_call_args("legacy", ["value"], [1, 2], {})
    with pytest.raises(RuntimeError, match="not supported"):
        check_call_args("legacy", ["value"], [], {"unknown": 1})
    check_call_args("legacy", ["value"], [], {"value": 1})
    assert get_collab_object_name("site") == "_app_"
    assert get_collab_object_name("site.helper") == "helper"
