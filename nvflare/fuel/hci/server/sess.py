# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import threading
import time
import uuid
from typing import List, Optional

from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.base64_utils import b64str_to_str, str_to_b64str
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ReplyKeyword
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.utils.time_utils import time_to_string
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, TokenVerifier

LIST_SESSIONS_CMD_NAME = InternalCommands.LIST_SESSIONS
CHECK_SESSION_CMD_NAME = InternalCommands.CHECK_SESSION


class Session(object):
    def __init__(
        self,
        sess_id,
        user_name,
        org,
        role,
        origin_fqcn,
        token_expiry_time: Optional[float] = None,
        auth_source: str = "cert",
    ):
        """Object keeping track of an admin client session with token and time data."""
        self.sess_id = sess_id
        self.user_name = user_name
        self.user_org = org
        self.user_role = role
        self.origin_fqcn = origin_fqcn
        self.start_time = time.time()
        self.last_active_time = time.time()
        self.token_expiry_time = token_expiry_time
        self.auth_source = auth_source or "cert"

    def mark_active(self):
        self.last_active_time = time.time()

    def is_idle_timed_out(self, idle_timeout: float, now: Optional[float] = None) -> bool:
        if idle_timeout <= 0:
            return False
        ts = now if now is not None else time.time()
        return (ts - self.last_active_time) > idle_timeout

    def is_session_ttl_expired(self, session_ttl: float, now: Optional[float] = None) -> bool:
        if session_ttl <= 0:
            return False
        ts = now if now is not None else time.time()
        return (ts - self.start_time) > session_ttl

    def is_token_expired(self, now: Optional[float] = None) -> bool:
        if self.token_expiry_time is None:
            return False
        ts = now if now is not None else time.time()
        return ts >= self.token_expiry_time

    def should_refresh(self, refresh_window: float, now: Optional[float] = None) -> bool:
        if self.token_expiry_time is None or refresh_window <= 0:
            return False
        ts = now if now is not None else time.time()
        seconds_to_expire = self.token_expiry_time - ts
        return 0 < seconds_to_expire <= refresh_window

    def is_expired(self, idle_timeout: float, session_ttl: float = 0, now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        return (
            self.is_idle_timed_out(idle_timeout=idle_timeout, now=ts)
            or self.is_session_ttl_expired(session_ttl=session_ttl, now=ts)
            or self.is_token_expired(now=ts)
        )

    def make_token(self, id_asserter: IdentityAsserter):
        user = {
            "n": self.user_name,
            "r": self.user_role,
            "o": self.user_org,
            "s": self.sess_id,
        }
        if self.token_expiry_time is not None:
            user["e"] = self.token_expiry_time
        if self.auth_source:
            user["a"] = self.auth_source
        ds = json.dumps(user)
        bds = str_to_b64str(ds)
        signature = id_asserter.sign(ds, return_str=True)

        # both bds and signature are b64 str
        return f"{bds}:{signature}"

    @staticmethod
    def decode_token(token: str, id_asserter: IdentityAsserter = None):
        if not isinstance(token, str):
            raise ValueError(f"token must be str but got {type(token)}")

        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid token {token}: expects 2 parts but got {len(parts)}")

        bds = parts[0]
        signature = parts[1]
        ds = b64str_to_str(bds)
        if id_asserter:
            token_verifier = TokenVerifier(id_asserter.cert)
            is_valid = token_verifier.verify("", ds, signature)
            if not is_valid:
                return None

        user = json.loads(ds)
        return Session(
            user_name=user.get("n"),
            role=user.get("r"),
            org=user.get("o"),
            sess_id=user.get("s"),
            origin_fqcn="",
            token_expiry_time=user.get("e"),
            auth_source=user.get("a", "cert"),
        )


class SessionManager(CommandModule):
    def __init__(self, cell, idle_timeout=1800, monitor_interval=5, session_ttl=0):
        """Session manager.

        Args:
            idle_timeout: session idle timeout
            monitor_interval: interval for obtaining updates when monitoring
            session_ttl: maximum lifetime of session from creation time, 0 disables limit
        """
        if monitor_interval <= 0:
            monitor_interval = 5

        self.cell = cell
        self.sess_update_lock = threading.Lock()
        self.sessions = {}  # token => Session
        self.idle_timeout = idle_timeout
        self.monitor_interval = monitor_interval
        self.session_ttl = session_ttl
        self.id_asserter_getter = None
        self.asked_to_stop = False
        self.monitor = threading.Thread(target=self.monitor_sessions)
        self.monitor.daemon = True
        self.monitor.start()

    def set_id_asserter_getter(self, getter):
        self.id_asserter_getter = getter

    def _decode_session_token(self, token: str):
        id_asserter = None
        if self.id_asserter_getter:
            id_asserter = self.id_asserter_getter()
        return Session.decode_token(token, id_asserter)

    def monitor_sessions(self):
        """Runs loop in a thread to end sessions that time out."""
        while True:
            # print('checking for dead sessions ...')
            if self.asked_to_stop:
                break

            dead_sess = None
            with self.sess_update_lock:
                sess_list = list(self.sessions.values())
            for sess in sess_list:
                if sess.is_expired(idle_timeout=self.idle_timeout, session_ttl=self.session_ttl):
                    dead_sess = sess
                    break

            if dead_sess:
                # print('ending dead session {}'.format(dead_sess.token))
                self.end_session_by_id(dead_sess.sess_id, "Your session is closed due to inactivity.")
            else:
                # print('no dead sessions found')
                pass

            time.sleep(self.monitor_interval)

    def shutdown(self):
        self.asked_to_stop = True

    def create_session(
        self,
        user_name,
        user_org,
        user_role,
        origin_fqcn,
        token_expiry_time: Optional[float] = None,
        auth_source: str = "cert",
    ):
        """Creates new session with a new session token.

        Args:
            user_name: username for session
            user_org: org of the user
            user_role: user's role
            origin_fqcn: request origin FQCN
            token_expiry_time: optional absolute epoch time for upstream-token expiry
            id_asserter: used to sign session token

        Returns: Session

        """
        sess_id = str(uuid.uuid4())
        sess = Session(
            sess_id=sess_id,
            user_name=user_name,
            org=user_org,
            role=user_role,
            origin_fqcn=origin_fqcn,
            token_expiry_time=token_expiry_time,
            auth_source=auth_source,
        )
        with self.sess_update_lock:
            self.sessions[sess_id] = sess
        return sess

    def recreate_session(self, token: str, origin_fqcn, id_asserter: IdentityAsserter):
        sess = Session.decode_token(token, id_asserter)
        if not isinstance(sess, Session):
            raise ValueError("invalid session token")
        if sess.is_expired(idle_timeout=self.idle_timeout, session_ttl=self.session_ttl):
            raise ValueError("session token expired")
        sess.origin_fqcn = origin_fqcn
        with self.sess_update_lock:
            self.sessions[sess.sess_id] = sess
        return sess

    def get_session(self, token: str):
        try:
            sess = self._decode_session_token(token)
        except:
            return None

        with self.sess_update_lock:
            stored = self.sessions.get(sess.sess_id)
            if stored and stored.is_expired(idle_timeout=self.idle_timeout, session_ttl=self.session_ttl):
                self.sessions.pop(sess.sess_id, None)
                return None
            return stored

    def get_sessions(self):
        result = []
        with self.sess_update_lock:
            for _, s in self.sessions.items():
                result.append(s)
        return result

    def end_session_by_token(self, token, reason=None):
        try:
            sess = self._decode_session_token(token)
        except:
            return
        self.end_session_by_id(sess.sess_id, reason)

    def end_session_by_id(self, sess_id: str, reason=None):
        with self.sess_update_lock:
            sess = self.sessions.pop(sess_id, None)
            if sess and reason:
                self.cell.fire_and_forget(
                    channel=CellChannel.HCI,
                    topic="SESSION_EXPIRED",
                    targets=sess.origin_fqcn,
                    message=CellMessage(payload=reason),
                    optional=True,
                )

    def get_spec(self):
        return CommandModuleSpec(
            name="sess",
            cmd_specs=[
                CommandSpec(
                    name=LIST_SESSIONS_CMD_NAME,
                    description="list user sessions",
                    usage=LIST_SESSIONS_CMD_NAME,
                    handler_func=self.handle_list_sessions,
                    visible=False,
                    enabled=True,
                ),
                CommandSpec(
                    name=CHECK_SESSION_CMD_NAME,
                    description="check if session is active",
                    usage=CHECK_SESSION_CMD_NAME,
                    handler_func=self.handle_check_session,
                    visible=False,
                ),
            ],
        )

    def handle_list_sessions(self, conn: Connection, args: List[str]):
        """Lists sessions and the details in a table.

        Registered in the FedAdminServer with ``cmd_reg.register_module(sess_mgr)``.
        """
        with self.sess_update_lock:
            sess_list = list(self.sessions.values())
        sess_list.sort(key=lambda x: x.user_name, reverse=False)
        table = conn.append_table(["User", "Org", "Role", "Session ID", "Start", "Last Active", "Idle"])
        for s in sess_list:
            table.add_row(
                [
                    s.user_name,
                    s.user_org,
                    s.user_role,
                    s.sess_id,
                    time_to_string(s.start_time),
                    time_to_string(s.last_active_time),
                    f"{(time.time() - s.last_active_time)}",
                ]
            )

    def handle_check_session(self, conn: Connection, args: List[str]):
        token = conn.get_token()
        if not token:
            conn.append_error("invalid_session")
            return

        sess = self.get_session(token)
        if sess:
            conn.append_string("OK")
        else:
            conn.append_error(ReplyKeyword.SESSION_INACTIVE)
            conn.append_string(
                "admin client session timed out after {} seconds of inactivity - logging out".format(self.idle_timeout)
            )
