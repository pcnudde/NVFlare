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

from nvflare.fuel.f3.message import Message
from nvflare.fuel.hci.client import api as subject
from nvflare.fuel.hci.client.api_spec import CommandContext, CommandInfo, ReplyProcessor
from nvflare.fuel.hci.proto import Buffer, MetaKey, MetaStatusValue, ProtoKey, ReplyKeyword, make_meta, validate_proto
from nvflare.fuel.hci.reg import CommandEntry, ConfirmMethod
from nvflare.fuel.hci.table import Table


@pytest.fixture
def api():
    api = object.__new__(subject.AdminAPI)
    api.debug, api.fire_event = Mock(), Mock()
    api.logger = Mock()
    api.in_logout = False
    api.user_name = "admin"
    api._debug = False
    api.token, api.cmd_timeout = "token", 5
    api.cell = Mock()
    api.server_sess_active = True
    api.client_cmd_reg, api.server_cmd_reg = Mock(), Mock()
    api.client_cmd_reg.get_command_entries.return_value = []
    api.server_cmd_reg.get_command_entries.return_value = []
    return api


def entry(confirm=None):
    return CommandEntry("scope", "command", "description", "usage", Mock(), None, True, confirm, None)


def context(api, processor=None):
    ctx = CommandContext()
    ctx.set_api(api)
    ctx.set_command("scope.command argument")
    ctx.set_command_args(["scope.command", "argument"])
    ctx.set_reply_processor(processor)
    return ctx


def test_reply_processor_dispatches_structured_types_and_finishes_with_reply_metadata(api):
    processor = Mock(spec=ReplyProcessor)
    ctx = context(api, processor)
    reply = {
        ProtoKey.DATA: [
            {ProtoKey.TYPE: ProtoKey.STRING, ProtoKey.DATA: "message"},
            {ProtoKey.TYPE: ProtoKey.SUCCESS, ProtoKey.DATA: "done"},
            {ProtoKey.TYPE: ProtoKey.TABLE, ProtoKey.ROWS: [["header"], ["row"]]},
            {ProtoKey.TYPE: ProtoKey.DICT, ProtoKey.DATA: {"value": 1}},
            {ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: "token"},
        ],
        ProtoKey.META: make_meta(MetaStatusValue.OK),
    }
    subject._ServerReplyJsonProcessor(ctx).process_server_reply(reply)
    processor.reply_start.assert_called_once_with(ctx, reply)
    processor.process_string.assert_called_once_with(ctx, "message")
    processor.process_success.assert_called_once_with(ctx, "done")
    assert processor.process_table.call_args.args[1].rows == [["header"], ["row"]]
    processor.process_dict.assert_called_once_with(ctx, {"value": 1})
    processor.process_token.assert_called_once_with(ctx, "token")
    processor.reply_done.assert_called_once_with(ctx)
    assert ctx.get_command_result() is reply
    assert ctx.get_meta() == make_meta(MetaStatusValue.OK)


@pytest.mark.parametrize(
    "kind,method",
    [(ProtoKey.ERROR, "process_error"), (ProtoKey.SHUTDOWN, "process_shutdown"), ("unknown", "protocol_error")],
)
def test_terminal_or_invalid_reply_item_prevents_processing_following_items(api, kind, method):
    processor = Mock(spec=ReplyProcessor)
    ctx = context(api, processor)
    reply = {
        ProtoKey.DATA: [
            {ProtoKey.TYPE: kind, ProtoKey.DATA: "terminal"},
            {ProtoKey.TYPE: ProtoKey.STRING, ProtoKey.DATA: "ignored"},
        ]
    }
    subject._ServerReplyJsonProcessor(ctx).process_server_reply(reply)
    getattr(processor, method).assert_called_once()
    processor.process_string.assert_not_called()
    processor.reply_done.assert_called_once_with(ctx)


def test_absent_reply_reports_protocol_failure_and_default_shutdown_updates_api_state(api):
    processor = Mock(spec=ReplyProcessor)
    ctx = context(api, processor)
    subject._ServerReplyJsonProcessor(ctx).process_server_reply(None)
    processor.protocol_error.assert_called_once_with(ctx, "Protocol Error")
    ctx.set_reply_processor(None)
    subject._ServerReplyJsonProcessor(ctx).process_server_reply(
        {ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.SHUTDOWN, ProtoKey.DATA: "maintenance"}]}
    )
    assert api.shutdown_received
    assert api.shutdown_msg == "maintenance"


def test_command_list_reply_maps_scoped_command_visibility_and_short_rows_stop_registration(api):
    ctx = context(api)
    table = Table(["scope", "name", "description", "usage", "confirm"])
    table.add_row(["scope", "one", "description", "usage", "none"])
    table.add_row(["scope", "two", "description", "usage", "auth", "client.cmd", "no"])
    subject._CmdListReplyProcessor().process_table(ctx, table)
    assert api.server_cmd_received
    calls = api.server_cmd_reg.add_command.call_args_list
    assert calls[0].kwargs["visible"]
    assert calls[1].kwargs["visible"] is False
    assert calls[1].kwargs["client_cmd"] == "client.cmd"
    malformed = Table(["header"])
    malformed.add_row(["short"])
    api.server_cmd_received = False
    subject._CmdListReplyProcessor().process_table(ctx, malformed)
    assert not api.server_cmd_received
    assert api.server_cmd_reg.add_command.call_count == 2


@pytest.mark.parametrize(
    "confirm,expected",
    [
        (ConfirmMethod.AUTH, CommandInfo.CONFIRM_AUTH),
        (ConfirmMethod.YESNO, CommandInfo.CONFIRM_YN),
        (None, CommandInfo.OK),
    ],
)
def test_command_confirmation_respects_registered_policy(api, confirm, expected):
    api.client_cmd_reg.get_command_entries.return_value = [entry(confirm)]
    assert api.check_command("command argument") == expected
    api.client_cmd_reg.get_command_entries.return_value = [entry(), entry()]
    assert api.check_command("command") == CommandInfo.AMBIGUOUS
    assert api.do_command("command")[subject.ResultKey.STATUS] == subject.APIStatus.ERROR_SYNTAX
    api.client_cmd_reg.get_command_entries.return_value = []
    assert api.check_command("unknown") == CommandInfo.UNKNOWN
    assert api.do_command("unknown")[subject.ResultKey.STATUS] == subject.APIStatus.ERROR_SYNTAX


def test_client_command_uses_context_result_or_direct_return_and_reports_missing_response(api):
    command = entry()
    api.client_cmd_reg.get_command_entries.return_value = [command]
    command.handler.side_effect = lambda args, ctx: ctx.set_command_result({"reply": args[1]})
    assert api.do_command("command value", props={"study": "default"}) == {"reply": "value"}
    command.handler.side_effect = None
    command.handler.return_value = {"direct": True}
    assert api.do_command("command") == {"direct": True}
    command.handler.return_value = None
    assert api.do_command("command")[subject.ResultKey.STATUS] == subject.APIStatus.ERROR_RUNTIME
    api.client_cmd_reg.get_command_entries.return_value = []
    api.server_cmd_reg.get_command_entries.return_value = [entry()]
    api.server_sess_active = False
    assert api.do_command("command")[subject.ResultKey.STATUS] == subject.APIStatus.ERROR_INACTIVE_SESSION


def test_server_execute_encodes_token_headers_and_metadata_over_fake_cell_and_returns_decoded_reply(api):
    buffer = Buffer()
    buffer.append_success("done", meta=make_meta(MetaStatusValue.OK))
    api.cell.send_request.return_value = Message(payload=buffer.encode())
    result = api.server_execute("scope.command argument", props={"study": "default"}, headers={"auth": "proof"})
    assert result[subject.ResultKey.STATUS] == subject.APIStatus.SUCCESS
    request = api.cell.send_request.call_args.kwargs["request"]
    assert request.headers == {"auth": "proof"}
    sent = validate_proto(request.payload)
    assert sent[ProtoKey.META][MetaKey.CMD_TIMEOUT] == 5
    assert sent[ProtoKey.META][MetaKey.CMD_PROPS] == {"study": "default"}
    assert sent[ProtoKey.DATA] == [
        {ProtoKey.TYPE: ProtoKey.COMMAND, ProtoKey.DATA: "scope.command argument"},
        {ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: "token"},
    ]


@pytest.mark.parametrize(
    "message,status",
    [
        (ReplyKeyword.SESSION_INACTIVE, subject.APIStatus.ERROR_INACTIVE_SESSION),
        (ReplyKeyword.WRONG_SERVER, subject.APIStatus.ERROR_SERVER_CONNECTION),
        (ReplyKeyword.COMM_FAILURE, subject.APIStatus.ERROR_SERVER_CONNECTION),
        (ReplyKeyword.INVALID_CLIENT, subject.APIStatus.ERROR_INVALID_CLIENT),
        (ReplyKeyword.UNKNOWN_SITE, subject.APIStatus.ERROR_INVALID_CLIENT),
        (ReplyKeyword.NOT_AUTHORIZED, subject.APIStatus.ERROR_AUTHORIZATION),
    ],
)
def test_legacy_reply_status_detection_classifies_server_error_phrases(api, message, status):
    assert (
        api._determine_api_status({ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.ERROR, ProtoKey.DATA: message}]}) == status
    )
    assert api._determine_api_status({}) == subject.APIStatus.ERROR_RUNTIME


def test_before_command_event_failure_prevents_transport_and_returns_error_payload(api):
    api.fire_event.side_effect = RuntimeError("event handler")
    result = api.server_execute("scope.command")
    assert any("before_execute_cmd" in item.get(ProtoKey.DATA, "") for item in result[ProtoKey.DATA])
    api.cell.send_request.assert_not_called()


def test_requester_failure_is_converted_to_protocol_reply_without_using_cell(api):
    processor = Mock(spec=ReplyProcessor)
    ctx = context(api, processor)
    requester = Mock()
    requester.send_request.side_effect = RuntimeError("requester")
    ctx.set_requester(requester)
    ctx.set_json_processor(subject._ServerReplyJsonProcessor(ctx))
    api._send_to_cell(ctx)
    processor.process_error.assert_called_once()
    assert "failed to send request" in processor.process_error.call_args.args[1]
    api.cell.send_request.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_upload_command_passes_encoded_command_to_stream_boundary_and_returns_end_result(api, monkeypatch, failure):
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.shareable import Shareable
    from nvflare.fuel.hci.conn import Connection

    api.new_context = Mock(return_value=FLContext())
    response = Shareable()
    response.set_header(subject.HeaderKey.END_RESULT, "encoded result")
    stream = Mock(
        return_value=(
            subject.ReturnCode.EXECUTION_EXCEPTION if failure else subject.ReturnCode.OK,
            {subject.FQCN.ROOT_SERVER: response},
        )
    )
    monkeypatch.setattr(subject.FileStreamer, "stream_file", stream)
    connection = Connection()
    connection.append_command("upload")
    assert api.upload_file("/fake/archive.zip", connection) == (None if failure else "encoded result")
    kwargs = stream.call_args.kwargs
    assert kwargs["file_name"] == "/fake/archive.zip"
    assert kwargs["targets"] == [subject.FQCN.ROOT_SERVER]
    assert validate_proto(kwargs["stream_ctx"]["conn_data"])[ProtoKey.DATA][0][ProtoKey.DATA] == "upload"


@pytest.mark.parametrize("error", [None, "transfer failed"])
def test_download_moves_completed_file_or_removes_partial_file(api, monkeypatch, tmp_path, error):
    from nvflare.apis.signal import Signal

    temporary = tmp_path / "temporary"
    temporary.write_bytes(b"payload")
    destination = tmp_path / "download" / "result.bin"
    api.file_download_progress_timeout = 2
    api.session_abort_signal = Signal()
    api._print_hci = Mock()
    download = Mock(return_value=(error, str(temporary)))
    monkeypatch.setattr(subject.downloader, "download_file", download)
    result = api.download_file("server", "reference", str(destination))
    assert not temporary.exists()
    if error:
        assert result is None
        assert not destination.exists()
        api._print_hci.assert_called_once()
    else:
        assert result == 7
        assert destination.read_bytes() == b"payload"
    assert download.call_args.kwargs["from_fqcn"] == "server"
    assert download.call_args.kwargs["per_request_timeout"] == 2
