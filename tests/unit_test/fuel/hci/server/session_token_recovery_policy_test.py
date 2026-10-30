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

import json
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import ProtoKey, validate_proto
from nvflare.fuel.hci.server import sess as subject


@pytest.fixture
def manager(monkeypatch):
    thread = Mock()
    monkeypatch.setattr(subject.threading, "Thread", Mock(return_value=thread))
    manager = subject.SessionManager(Mock(), monitor_interval=0)
    thread.start.assert_called_once_with()
    assert manager.monitor_interval == 5
    return manager


@pytest.mark.parametrize("token", [None, "missing-separator", "a:b:c"])
def test_invalid_session_token_envelope_is_rejected_before_signature_verification(token):
    with pytest.raises(ValueError, match="token"):
        subject.Session.decode_token(token, Mock())


def test_signed_token_with_nonstring_certificate_study_is_rejected(monkeypatch):
    verifier = Mock()
    verifier.verify.return_value = True
    monkeypatch.setattr(subject, "TokenVerifier", Mock(return_value=verifier))
    payload = json.dumps({"cs": ["study", 3]})
    token = subject.str_to_b64str(payload) + ":signature"
    with pytest.raises(ValueError, match="certificate studies"):
        subject.Session.decode_token(token, Mock())
    verifier.verify.assert_called_once_with("", payload, "signature")


@pytest.mark.parametrize("outcome", ["invalid", "expired", "valid"])
def test_recreated_session_is_registered_only_after_token_and_expiry_validation(manager, monkeypatch, outcome):
    session = subject.Session("session", "user", "org", "lead", "old")
    if outcome == "expired":
        session.cert_exp = 1
    decode = Mock(return_value=None if outcome == "invalid" else session)
    monkeypatch.setattr(subject.Session, "decode_token", decode)
    if outcome != "valid":
        with pytest.raises(ValueError, match="invalid session|expired"):
            manager.recreate_session("token", "admin.new", Mock())
        assert not manager.sessions
    else:
        assert manager.recreate_session("token", "admin.new", Mock()) is session
        assert manager.sessions == {"session": session}
        assert session.origin_fqcn == "admin.new"


@pytest.mark.parametrize("failure", ["none", "decode-error", "expired"])
def test_invalid_or_expired_token_cannot_recover_stored_session(manager, monkeypatch, failure):
    session = subject.Session("session", "user", "org", "lead", "admin")
    if failure == "expired":
        session.cert_exp = 1
    manager.sessions["session"] = session
    decode = (
        Mock(side_effect=ValueError("invalid token"))
        if failure == "decode-error"
        else Mock(return_value=None if failure == "none" else session)
    )
    monkeypatch.setattr(subject.Session, "decode_token", decode)
    assert manager.get_session("token", Mock()) is None
    if failure == "expired":
        assert "session" not in manager.sessions
        manager.cell.fire_and_forget.assert_called_once()


def test_session_listing_is_sorted_and_missing_token_reports_protocol_error(manager):
    manager.create_session("zoe", "org", "lead", "admin.zoe")
    manager.create_session("alice", "org", "lead", "admin.alice")
    connection = Connection()
    manager.handle_list_sessions(connection, [])
    reply = validate_proto(connection.close())
    rows = reply[ProtoKey.DATA][0][ProtoKey.ROWS]
    assert [row[0] for row in rows[1:]] == ["alice", "zoe"]
    connection = Connection()
    manager.handle_check_session(connection, [])
    reply = validate_proto(connection.close())
    assert reply[ProtoKey.DATA][0] == {ProtoKey.TYPE: ProtoKey.ERROR, ProtoKey.DATA: "invalid_session"}
