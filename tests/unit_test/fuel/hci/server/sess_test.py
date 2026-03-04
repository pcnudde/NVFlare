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

from nvflare.fuel.hci.server.sess import Session


class _DummyIdentityAsserter:
    def sign(self, data, return_str=True):
        return "signature"


def _new_session():
    return Session(sess_id="sess-1", user_name="alice", org="org_a", role="lead", origin_fqcn="site.server")


def test_session_idle_timeout_check():
    session = _new_session()
    session.last_active_time = 100
    assert not session.is_idle_timed_out(idle_timeout=20, now=119)
    assert session.is_idle_timed_out(idle_timeout=20, now=121)


def test_session_hard_ttl_check():
    session = _new_session()
    session.start_time = 100
    assert not session.is_session_ttl_expired(session_ttl=20, now=119)
    assert session.is_session_ttl_expired(session_ttl=20, now=121)


def test_session_token_expiry_and_refresh_window():
    session = _new_session()
    session.token_expiry_time = 200
    assert not session.is_token_expired(now=199)
    assert session.is_token_expired(now=200)
    assert not session.should_refresh(refresh_window=30, now=160)
    assert session.should_refresh(refresh_window=30, now=175)
    assert not session.should_refresh(refresh_window=30, now=201)


def test_session_combined_expiry_logic():
    session = _new_session()
    session.start_time = 100
    session.last_active_time = 190
    session.token_expiry_time = 300
    assert not session.is_expired(idle_timeout=20, session_ttl=150, now=195)
    assert session.is_expired(idle_timeout=20, session_ttl=150, now=211)
    assert session.is_expired(idle_timeout=20, session_ttl=90, now=195)
    assert session.is_expired(idle_timeout=20, session_ttl=150, now=301)


def test_session_token_roundtrip_preserves_token_expiry():
    session = _new_session()
    session.token_expiry_time = 1234.5
    token = session.make_token(_DummyIdentityAsserter())
    decoded = Session.decode_token(token=token)
    assert decoded is not None
    assert decoded.token_expiry_time == 1234.5
