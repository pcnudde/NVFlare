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

from nvflare.fuel.sec import authz as subject


def context(right="read", name="alice", org="site", role="member", submitter=None):
    return subject.AuthzContext(right, subject.Person(name, org, role), submitter)


@pytest.mark.parametrize(
    "expression, permitted",
    [
        ("org:site", True),
        ("org:other", False),
        ("name:alice", True),
        ("name:bob", False),
        ("org:submitter", True),
        ("name:submitter", True),
        ("all", True),
        ("none", False),
        (["org:site", "not name:alice"], False),
        (["name:bob", "org:site"], True),
        (["not name:bob"], True),
    ],
)
def test_policy_conditions_enforce_block_priority_and_site_or_submitter_identity(expression, permitted):
    authorizer = subject.Authorizer(" Site ")
    assert authorizer.load_policy({"format_version": "1.0", "permissions": {"member": {"read": expression}}}) == ""
    ctx = context(submitter=subject.Person("alice", "site", "submitter"))
    result, error = authorizer.authorize(ctx)
    assert result is permitted
    assert bool(error) is not permitted


def test_explicit_right_overrides_category_and_wildcard_applies_to_unknown_right():
    authorizer = subject.Authorizer("site", right_categories={"read": "data", "write": "data"})
    assert (
        authorizer.load_policy(
            {
                "format_version": "1.0",
                "permissions": {
                    "member": {"data": "all", "write": "none"},
                    "observer": ["name:alice"],
                },
            }
        )
        == ""
    )
    assert authorizer.authorize(context("read")) == (True, "")
    assert authorizer.authorize(context("write"))[0] is False
    assert authorizer.authorize(context("custom", role="observer")) == (True, "")
    assert authorizer.authorize(context(role="unknown"))[0] is False
    policy = authorizer.get_policy()
    assert policy.get_roles() == ["member", "observer"]
    assert set(policy.get_rights()) == {"*", "data", "read", "write"}
    assert authorizer.last_load_time > 0


@pytest.mark.parametrize(
    "permissions, message",
    [
        ({"member": 2}, "bad right config"),
        ({2: "all"}, "bad role name"),
        ({"member": {2: "all"}}, "bad right name"),
        ({"member": {"read": []}}, "no conditions"),
        ({"member": {"read": ["all", "invalid"]}}, "bad condition"),
        ({"member": {"read": "group:site"}}, "invalid type"),
        ({"member": {"read": 1}}, "bad condition expression type"),
    ],
)
def test_invalid_policy_does_not_replace_last_successful_policy(permissions, message):
    authorizer = subject.Authorizer("site")
    assert authorizer.load_policy({"format_version": "1.0", "permissions": {"member": "all"}}) == ""
    previous = authorizer.get_policy()
    error = authorizer.load_policy({"format_version": "1.0", "permissions": permissions})
    assert message in error
    assert authorizer.get_policy() is previous
    assert authorizer.authorize(context()) == (True, "")


@pytest.mark.parametrize(
    "configuration, message",
    [
        (None, "must be a dict"),
        ({}, "empty"),
        ({"format_version": "2.0"}, "format_version"),
        ({"format_version": "1.0"}, "missing permissions"),
        ({"format_version": "1.0", "permissions": ["all"]}, "invalid permissions"),
    ],
)
def test_malformed_policy_is_rejected(configuration, message):
    policy, error = subject.parse_policy_config(configuration, {})
    assert policy is None and message in error


def test_authorization_context_attributes_and_singleton_lifecycle_are_isolated(monkeypatch):
    monkeypatch.setattr(subject.AuthorizationService, "the_authorizer", None)
    assert subject.AuthorizationService.authorize(context()) == (True, "")
    first, second = subject.Authorizer("site"), subject.Authorizer("other")
    assert subject.AuthorizationService.initialize(first) == (first, "")
    assert subject.AuthorizationService.initialize(second) == (first, "")
    assert subject.AuthorizationService.get_authorizer() is first
    assert first.authorize(None) == (True, "")
    assert first.authorize("invalid")[0] is False
    assert first.authorize(context(role="super")) == (True, "")
    assert first.authorize(context())[1] == "policy not defined"
    ctx = context()
    ctx.set_attr("resource", "job")
    assert ctx.get_attr("resource") == "job" and ctx.get_attr("missing", 4) == 4
    assert str(subject.Person(" Alice ", "ORG", " Member ")) == "alice:org:member"
    assert str(subject.Person("", "", "")) == "None:None:None"
    with pytest.raises(ValueError, match="user needs"):
        subject.AuthzContext("read", object())
    with pytest.raises(ValueError, match="submitter needs"):
        subject.AuthzContext("read", subject.Person("a", "b", "c"), object())
    with pytest.raises(ValueError, match="authorizer must"):
        subject.AuthorizationService.initialize(object())
