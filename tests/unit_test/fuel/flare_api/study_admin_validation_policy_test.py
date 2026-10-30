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

from unittest.mock import Mock

import pytest

from nvflare.fuel.flare_api import flare_api as subject
from nvflare.fuel.hci.proto import ProtoKey


@pytest.fixture
def session():
    value = object.__new__(subject.Session)
    value._do_command = Mock(
        return_value={ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.DICT, ProtoKey.DATA: {"study": "trial"}}]}
    )
    return value


@pytest.mark.parametrize(
    "validate, value",
    [
        ("_validate_study_name", None),
        ("_validate_study_name", ""),
        ("_validate_study_sites", "site"),
        ("_validate_study_sites", []),
        ("_validate_study_sites", [None]),
        ("_validate_study_user", None),
        ("_validate_study_user", ""),
        ("_validate_study_site_orgs", "org:site"),
        ("_validate_study_site_orgs", []),
        ("_validate_study_site_orgs", [None]),
    ],
)
def test_study_validators_reject_wrong_types_and_empty_identifiers(session, validate, value):
    with pytest.raises(subject.InvalidArgumentError):
        getattr(session, validate)(value)
    session._do_command.assert_not_called()


@pytest.mark.parametrize(
    "method, command, arguments",
    [
        ("remove_study", "remove_study trial", ("trial",)),
        ("list_studies", "list_studies", ()),
        ("add_study_user", "add_study_user trial alice@example.com", ("trial", "alice@example.com")),
        ("remove_study_user", "remove_study_user trial alice@example.com", ("trial", "alice@example.com")),
    ],
)
def test_study_admin_operations_build_expected_command_and_decode_dictionary_reply(session, method, command, arguments):
    assert getattr(session, method)(*arguments) == {"study": "trial"}
    session._do_command.assert_called_once_with(command)


def test_application_command_refuses_unserializable_data_without_contacting_server(session):
    with pytest.raises(ValueError, match="cannot be JSON serialized"):
        session.do_app_command("job", "inspect", {"bad": object()})
    session._do_command.assert_not_called()


@pytest.mark.parametrize(
    "job_id,target,targets,error",
    [
        (None, "server", None, TypeError),
        ("", "server", None, ValueError),
        ("job", "invalid", None, ValueError),
        ("job", "client", [None], SyntaxError),
    ],
)
def test_stats_collection_rejects_malformed_job_and_target_parameters(session, job_id, target, targets, error):
    with pytest.raises(error):
        session._collect_info("show_stats", job_id, target, targets)
    session._do_command.assert_not_called()


def test_client_stats_command_keeps_target_names_as_separate_arguments(session):
    assert session._collect_info("show_stats", "job", "client", ["site-a", "site-b"]) == {"study": "trial"}
    session._do_command.assert_called_once_with("show_stats job client site-a site-b", enforce_meta=False)


@pytest.mark.parametrize("tail", ["invalid", -1])
def test_log_tail_rejects_invalid_line_limits_without_modifying_original_payload(tail):
    with pytest.raises(ValueError, match="tail_lines"):
        subject.Session._filter_job_log_text("first\nlast\n", tail, None)
    original = {"logs": {"site": "first\nlast\n", "missing": None}}
    assert subject.Session._filter_job_logs_payload(original, 1, None) == {"logs": {"site": "last\n", "missing": None}}
    assert original["logs"]["site"] == "first\nlast\n"


def test_available_uploads_are_only_direct_child_directories(session, tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "file.zip").write_bytes(b"archive")
    session.upload_dir = str(tmp_path)
    assert session.get_available_apps_to_upload() == ["app"]


@pytest.mark.parametrize("meta,expected", [(None, None), ({"status": "RUNNING"}, "RUNNING"), ({}, None)])
def test_job_status_preserves_missing_metadata_and_server_status(session, meta, expected):
    session.get_job_meta = Mock(return_value=meta)
    assert session.get_job_status("job") == expected
    session.get_job_meta.assert_called_once_with("job")


def test_public_command_passes_explicit_properties_to_admin_api(session):
    session.api = Mock()
    assert session.do_command("inspect", props={"x": 1}) is session.api.do_command.return_value
    session.api.do_command.assert_called_once_with("inspect", {"x": 1})
