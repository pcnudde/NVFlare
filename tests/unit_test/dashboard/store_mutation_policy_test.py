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

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.dashboard.application import store as module


class Record(SimpleNamespace):
    def asdict(self):
        return vars(self).copy()


@pytest.fixture
def database(monkeypatch):
    records = {
        "Project": Record(id=1, short_name="old", title="old", frozen=False, root_key="private"),
        "Client": Record(id=2, creator_id=3, approval_state=0, download_count=0),
        "User": Record(
            id=3,
            email="owner@example.com",
            role=Record(name=""),
            organization=Record(name=""),
            approval_state=0,
            role_id=9,
            organization_id=8,
        ),
        "Organization": Record(id=4),
        "Role": Record(id=5),
    }
    models = {}
    for name, record in records.items():
        model = Mock(return_value=record)
        model.query.first.return_value = record
        model.query.get.return_value = record
        model.query.filter_by.return_value.first.return_value = record
        model.query.all.return_value = [record]
        model.query.count.return_value = 2
        monkeypatch.setattr(module, name, model)
        models[name] = model
    session = Mock()
    monkeypatch.setattr(module, "db", SimpleNamespace(session=session))
    monkeypatch.setattr(module, "generate_password_hash", lambda password: "hash:" + password)
    return SimpleNamespace(records=records, models=models, session=session)


def test_project_mutation_truncates_name_serializes_props_and_ignores_private_fields(monkeypatch, database):
    build = Mock()
    monkeypatch.setattr(module.Store, "build_project", build)
    result = module.Store.set_project(
        {
            "id": 99,
            "short_name": "0123456789abcdefghijkl",
            "title": "new",
            "project_props": {"rounds": 7},
            "server_props": {"port": 8002},
            "root_key": "attacker",
            "frozen": True,
        }
    )
    project = database.records["Project"]
    assert project.short_name == "0123456789abcdef"
    assert project.id == 1 and project.root_key == "private"
    assert project.title == "new" and project.frozen
    assert json.loads(project.project_props) == {"rounds": 7}
    assert json.loads(project.server_props) == {"port": 8002}
    build.assert_called_once_with(project)
    database.session.commit.assert_called_once_with()
    assert result["status"] == "ok"
    assert result["project"]["num_clients"] == result["project"]["num_orgs"] == result["project"]["num_users"] == 2
    database.session.reset_mock()
    assert module.Store.set_project({"title": "forbidden"}) == {"status": "Project is frozen"}
    database.session.commit.assert_not_called()
    assert project.title == "new"


def test_project_creation_and_certificate_build_persist_expected_public_and_private_material(monkeypatch, database):
    created = module.Store.create_project()
    assert created["project"]["id"] == 1
    cert = Mock(return_value=SimpleNamespace(ser_cert="certificate", ser_pri_key="private-key"))
    monkeypatch.setattr(module, "make_root_cert", cert)
    result = module.Store.build_project(database.records["Project"])
    assert cert.call_args.args[0].name == "old"
    assert result["project"]["root_cert"] == "certificate"
    assert result["project"]["root_key"] == "private-key"
    assert database.session.commit.call_count == 2


@pytest.mark.parametrize("actor", ["admin", "creator"])
@pytest.mark.parametrize("failure", [True, False])
def test_client_patch_serializes_optional_data_enforces_actor_fields_and_handles_commit_failure(
    monkeypatch, database, actor, failure
):
    client = database.records["Client"]
    create = Mock(return_value=Record(id=77))
    monkeypatch.setattr(module, "get_or_create", create)
    database.session.commit.side_effect = ValueError("db unavailable") if failure else None
    request = {
        "organization": "org",
        "capacity": {"gpu": 2},
        "props": {"region": "west"},
        "name": "client",
        "description": "updated",
        "approval_state": 200,
        "creator_id": 999,
    }
    patch = module.Store.patch_client_by_project_admin if actor == "admin" else module.Store.patch_client_by_creator
    result = patch(2, request)
    assert json.loads(client.capacity) == {"gpu": 2}
    assert json.loads(client.props) == {"region": "west"}
    assert client.name == "client" and client.description == "updated" and client.creator_id == 3
    assert client.approval_state == (200 if actor == "admin" else 0)
    assert create.call_count == int(actor == "admin")
    if actor == "admin":
        assert client.organization_id == 77
    assert (result is None) is failure
    if not failure:
        assert result["status"] == "ok"


def test_client_creation_contains_database_failure_and_retains_serialized_properties(monkeypatch, database):
    monkeypatch.setattr(module, "get_or_create", Mock(return_value=Record(id=77)))
    database.session.commit.side_effect = ValueError("duplicate client")
    result = module.Store.create_client(
        {"name": "client", "organization": "org", "props": {"secure": True}}, "owner@example.com"
    )
    assert result is None
    assert database.models["Client"].call_args.kwargs == {"name": "client", "description": "", "creator_id": 3}
    assert database.records["Client"].organization_id == 77
    assert json.loads(database.records["Client"].props) == {"secure": True}


@pytest.mark.parametrize("role", [42, {"name": "member"}])
def test_user_creation_rejects_nonstring_roles_before_database_writes(database, role):
    assert module.Store.create_user({"email": "new@example.com", "role": role}) is None
    database.session.add.assert_not_called()


@pytest.mark.parametrize("empty_identity", [True, False])
def test_creator_cannot_change_established_role_organization_or_approval(monkeypatch, database, empty_identity):
    user = database.records["User"]
    user.role.name = "" if empty_identity else "member"
    user.organization.name = "" if empty_identity else "original"
    create = Mock(return_value=Record(id=77))
    monkeypatch.setattr(module, "get_or_create", create)
    result = module.Store.patch_user_by_creator(
        3,
        {
            "role": "lead",
            "organization": "new-org",
            "password": "new-password",
            "approval_state": 200,
            "email": "attacker@example.com",
            "name": "new",
        },
    )
    assert result["status"] == "ok"
    assert user.role_id == (77 if empty_identity else 9)
    assert user.organization_id == (77 if empty_identity else 8)
    assert user.approval_state == 0 and user.email == "owner@example.com" and user.name == "new"
    assert user.password_hash == "hash:new-password"
    assert create.call_count == (2 if empty_identity else 0)


@pytest.mark.parametrize("role", ["project_admin", 7])
def test_creator_role_escalation_rejected_without_commit(database, role):
    assert module.Store.patch_user_by_creator(3, {"role": role}) == {"status": "error"}
    database.session.commit.assert_not_called()


def test_admin_user_patch_applies_identity_bindings_password_and_only_allowed_fields(monkeypatch, database):
    create = Mock(return_value=Record(id=77))
    monkeypatch.setattr(module, "get_or_create", create)
    result = module.Store.patch_user_by_project_admin(
        3,
        {
            "role": "lead",
            "organization": "org",
            "password": "secret",
            "approval_state": 200,
            "description": "updated",
            "email": "forbidden",
        },
    )
    user = database.records["User"]
    assert result["status"] == "ok"
    assert user.role_id == user.organization_id == 77
    assert user.password_hash == "hash:secret" and user.approval_state == 200 and user.description == "updated"
    assert user.email == "owner@example.com"
    assert create.call_count == 2


def test_user_and_client_absence_and_rejection_are_explicit(database, monkeypatch):
    database.models["User"].query.get.return_value = None
    database.models["Client"].query.get.return_value = None
    assert module.Store._get_email_by_id(99) is None
    assert module.Store.get_creator_id_by_client_id(99) is None
    database.models["Organization"].query.filter_by.return_value.first.return_value = None
    assert module.Store.get_users("missing") == {"status": "ok", "user_list": []}
    user = database.records["User"]
    user.approval_state = -1
    user.password_hash = "hash:secret"
    monkeypatch.setattr(module, "check_password_hash", lambda stored, incoming: stored == "hash:" + incoming)
    assert module.Store.verify_user("owner@example.com", "secret") is None
    assert module.Store.get_orgs()["client_list"] == [{"id": 4}]


def test_get_or_create_commits_only_new_records_and_download_counter_is_persisted(database):
    session = Mock()
    existing = Record(id=7)
    session.query.return_value.filter_by.return_value.first.return_value = existing
    model = Mock(return_value=Record(id=8))
    assert module.get_or_create(session, model, name="org") is existing
    model.assert_not_called()
    session.commit.assert_not_called()
    session.query.return_value.filter_by.return_value.first.return_value = None
    assert module.get_or_create(session, model, name="org").id == 8
    model.assert_called_once_with(name="org")
    session.add.assert_called_once_with(model.return_value)
    session.commit.assert_called_once_with()
    module.inc_dl(database.models["Client"], 2)
    assert database.records["Client"].download_count == 1
    database.session.add.assert_called_once_with(database.records["Client"])
