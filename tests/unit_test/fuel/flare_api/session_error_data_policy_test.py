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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.fuel.flare_api import api_spec
from nvflare.fuel.flare_api import flare_api as subject
from nvflare.fuel.hci.client.api import APIStatus, ResultKey
from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ReplyKeyword
from tests.timing_utils import isolate_time


@pytest.fixture
def session():
    result = object.__new__(subject.Session)
    result.api = Mock()
    result.api.closed = False
    result.api.session_expired_reason = None
    return result


@pytest.mark.parametrize(
    "meta_status,error",
    [
        (MetaStatusValue.INVALID_JOB_DEFINITION, api_spec.InvalidJobDefinition),
        (MetaStatusValue.NOT_AUTHORIZED, api_spec.AuthorizationError),
        (MetaStatusValue.NOT_AUTHENTICATED, api_spec.AuthenticationError),
        (MetaStatusValue.SYNTAX_ERROR, api_spec.InternalError),
        (MetaStatusValue.INVALID_JOB_ID, api_spec.JobNotFound),
        (MetaStatusValue.JOB_RUNNING, api_spec.JobNotDone),
        (MetaStatusValue.JOB_NOT_RUNNING, api_spec.JobNotRunning),
        (MetaStatusValue.CLIENTS_RUNNING, api_spec.ClientsStillRunning),
        (MetaStatusValue.NO_CLIENTS, api_spec.NoClientsAvailable),
        (MetaStatusValue.INTERNAL_ERROR, api_spec.InternalError),
        (MetaStatusValue.INVALID_TARGET, api_spec.InvalidTarget),
        (MetaStatusValue.NO_REPLY, api_spec.NoReply),
        ("unknown-status", api_spec.InternalError),
    ],
)
def test_command_metadata_maps_server_failures_to_public_exception_types(session, meta_status, error):
    session.api.do_command.return_value = {
        ResultKey.STATUS: APIStatus.SUCCESS,
        ResultKey.META: {MetaKey.STATUS: meta_status, MetaKey.INFO: "detail"},
    }
    with pytest.raises(error):
        session._do_command("custom.action")
    session.api.do_command.assert_called_once_with("custom.action", props=None)


@pytest.mark.parametrize(
    "reply,error",
    [
        (None, api_spec.InternalError),
        ({ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: [1]}, api_spec.InternalError),
        ({}, api_spec.InternalError),
        ({ResultKey.STATUS: APIStatus.ERROR_AUTHORIZATION}, api_spec.AuthorizationError),
        ({ResultKey.STATUS: APIStatus.ERROR_INACTIVE_SESSION}, api_spec.SessionClosed),
        ({ResultKey.STATUS: APIStatus.ERROR_SYNTAX, ResultKey.DETAILS: "syntax"}, api_spec.InternalError),
        ({ResultKey.STATUS: APIStatus.ERROR_PROTOCOL}, api_spec.InternalError),
        ({ResultKey.STATUS: APIStatus.ERROR_SERVER_CONNECTION}, api_spec.NoConnection),
        ({ResultKey.STATUS: APIStatus.ERROR_RUNTIME, ResultKey.DETAILS: "runtime"}, RuntimeError),
    ],
)
def test_transport_status_and_malformed_result_map_to_public_errors(session, reply, error):
    session.api.do_command.return_value = reply
    with pytest.raises(error):
        session._do_command("custom.action")


def test_system_info_translation_preserves_server_clients_jobs_and_display_contracts(session):
    session._do_command = Mock(
        return_value={
            ResultKey.META: {
                MetaKey.SERVER_STATUS: "running",
                MetaKey.SERVER_START_TIME: None,
                MetaKey.CLIENTS: [{MetaKey.CLIENT_NAME: "one", MetaKey.CLIENT_LAST_CONNECT_TIME: None}],
                MetaKey.JOBS: [{MetaKey.JOB_ID: "job", MetaKey.APP_NAME: "app"}],
            }
        }
    )
    info = session._do_get_system_info("status")
    assert info.server_info.status == "running"
    assert info.client_info[0].name == "one"
    assert info.job_info[0].job_id == "job"
    assert info.job_info[0].app_name == "app"
    displayed = str(info)
    assert "start_time: unknown" in displayed
    assert "one (last_connect_time: unknown)" in displayed
    assert "app_name: app" in displayed
    session._do_command.assert_called_once_with("status server")


@pytest.mark.parametrize("names,expected", [(None, []), ("one", ["one"]), (["one", "two"], ["one", "two"])])
def test_client_environment_query_normalizes_selection_and_requires_server_payload(session, names, expected):
    environments = [{"client_name": "one", "workspace": "/workspace"}]
    session._do_command = Mock(return_value={ResultKey.META: {MetaKey.CLIENTS: environments}})
    assert session.get_client_env(names) is environments
    assert split_to_args(session._do_command.call_args.args[0]) == [AdminCommandNames.REPORT_ENV, *expected]
    session._do_command.return_value = {ResultKey.META: {}}
    with pytest.raises(RuntimeError, match="missing"):
        session.get_client_env(names)
    with pytest.raises(ValueError, match="str or list"):
        session.get_client_env(7)


def test_no_clients_target_detail_uses_no_clients_exception_and_closed_session_never_sends(session):
    session.api.do_command.return_value = {
        ResultKey.STATUS: APIStatus.SUCCESS,
        ResultKey.META: {MetaKey.STATUS: MetaStatusValue.INVALID_TARGET, MetaKey.INFO: ReplyKeyword.NO_CLIENTS},
    }
    with pytest.raises(api_spec.NoClientsAvailable):
        session._do_command("custom.action")
    session.api.do_command.reset_mock()
    session.api.closed = True
    with pytest.raises(api_spec.SessionClosed):
        session._do_command("custom.action")
    session.api.do_command.assert_not_called()


def system_info(names, time=1):
    return SimpleNamespace(client_info=[SimpleNamespace(name=name, last_connect_time=time) for name in names])


@pytest.mark.parametrize("timeout", [False, True])
def test_client_shutdown_wait_tracks_selected_sites_and_uses_finite_clock_budget(session, monkeypatch, timeout):
    session.get_system_info = Mock(
        side_effect=[system_info(["one", "other"]), system_info(["one"] if timeout else ["other"])]
    )
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(side_effect=[0, 1, 2, 11] if timeout else [0, 1, 2, 3])
    clock.sleep = Mock()
    if timeout:
        with pytest.raises(TimeoutError, match="one"):
            session._wait_for_clients_shutdown(["one"], 10)
    else:
        session._wait_for_clients_shutdown(["one"], 10)
    assert clock.sleep.call_count == 1


def test_client_restart_wait_tolerates_transient_errors_and_requires_new_connection_timestamp(session, monkeypatch):
    session.get_system_info = Mock(
        side_effect=[api_spec.NoConnection("offline"), RuntimeError("query"), system_info(["one"], 2)]
    )
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(side_effect=range(32))
    clock.sleep = Mock()
    session._wait_for_clients_restart({"one": 1, "initially-absent": None}, 30)
    assert session.get_system_info.call_count == 3
    assert clock.sleep.call_count == 2
    assert clock.time.call_count < 32
    session.get_system_info = Mock(return_value=system_info(["one"], 3))
    assert session._client_last_connect_times(["one", "absent"]) == {"one": 3, "absent": None}


def test_default_monitor_callback_updates_counter_and_switches_to_compact_progress(capsys):
    count = {"count": 0}
    for _ in range(4):
        assert subject.basic_cb_with_print(None, "job", {"status": "RUNNING"}, cb_run_counter=count)
    assert subject.basic_cb_with_print(None, "job", {"status": "FINISHED"}, cb_run_counter=count)
    assert count["count"] == 5
    output = capsys.readouterr().out
    assert output.count("RUNNING") == 3
    assert "." in output and "FINISHED" in output


@pytest.mark.parametrize(
    "operation,command_name",
    [("add_study_site", AdminCommandNames.ADD_STUDY_SITE), ("remove_study_site", AdminCommandNames.REMOVE_STUDY_SITE)],
)
@pytest.mark.parametrize("by_org", [False, True])
def test_study_membership_mutations_quote_study_names_and_select_one_routing_mode(
    session, operation, command_name, by_org
):
    from nvflare.fuel.hci.proto import ProtoKey

    session._do_command = Mock(
        return_value={ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.DICT, ProtoKey.DATA: {"updated": True}}]}
    )
    kwargs = {"site_orgs": ["org:site-1,site-2"]} if by_org else {"sites": ["site-1", "site-2"]}
    assert getattr(session, operation)("study name", **kwargs) == {"updated": True}
    expected = (
        [command_name, "study name", "--site-org", "org:site-1,site-2"]
        if by_org
        else [command_name, "study name", "--sites", "site-1,site-2"]
    )
    assert split_to_args(session._do_command.call_args.args[0]) == expected
    with pytest.raises(api_spec.InvalidArgumentError, match="mutually exclusive"):
        getattr(session, operation)("study", sites=["one"], site_orgs=["org:one"])


def test_downloaded_job_artifacts_move_to_requested_destination_and_component_lookup_uses_metadata(session, tmp_path):
    job_id = "01234567-89ab-cdef-0123-456789abcdef"
    downloaded = tmp_path / "staging" / "job"
    downloaded.mkdir(parents=True)
    (downloaded / "weights").write_bytes(b"model")
    session._do_command = Mock(return_value={ResultKey.META: {MetaKey.LOCATION: str(downloaded)}})
    location = session.download_job_result(job_id, str(tmp_path / "results"))
    assert location == str(tmp_path / "results" / "job")
    assert not downloaded.exists()
    assert (tmp_path / "results" / "job" / "weights").read_bytes() == b"model"
    session._do_command.return_value = {
        ResultKey.META: {MetaKey.JOB_COMPONENTS: ["ERRORLOG_site-1"], MetaKey.LOCATION: "/fake/components"}
    }
    assert session.list_job_components(job_id) == ["ERRORLOG_site-1"]
    assert session.download_job_components(job_id) == "/fake/components"
