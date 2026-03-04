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
import traceback
from typing import Any, Callable, List, Mapping, Optional

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ReplyKeyword
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.security import IdentityKey, get_identity_info
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.token_auth import ClaimMapper, TokenValidator
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.lighter.utils import cert_to_dict, load_crt_bytes
from nvflare.security.logging import secure_format_exception

from .reg import CommandFilter
from .sess import Session, SessionManager


class LoginModule(CommandModule, CommandFilter):
    def __init__(
        self,
        sess_mgr: SessionManager,
        token_validator: Optional[TokenValidator] = None,
        claim_mapper: Optional[ClaimMapper] = None,
        token_jwks: Optional[Mapping[str, Any]] = None,
        jwks_fetcher: Optional[Callable[[], Mapping[str, Any]]] = None,
    ):
        """Login module.

        CommandModule containing the login commands to handle login and logout of admin clients, as well as the
        CommandFilter pre_command to check that a client is logged in with a valid session.

        Args:
            sess_mgr: SessionManager
            token_validator: validator for token login mode
            claim_mapper: claim mapper for token login mode
            token_jwks: static JWKS for token verification
            jwks_fetcher: function returning current JWKS for token verification
        """
        if not isinstance(sess_mgr, SessionManager):
            raise TypeError("sess_mgr must be SessionManager but got {}.".format(type(sess_mgr)))

        token_enabled = token_validator is not None or claim_mapper is not None
        if token_enabled and not (token_validator and claim_mapper):
            raise ValueError("token login requires both token_validator and claim_mapper")
        if token_enabled and not (token_jwks is not None or jwks_fetcher is not None):
            raise ValueError("token login requires token_jwks or jwks_fetcher")

        self.session_mgr = sess_mgr
        self.token_validator = token_validator
        self.claim_mapper = claim_mapper
        self.token_jwks = token_jwks
        self.jwks_fetcher = jwks_fetcher
        self.logger = get_obj_logger(self)

    def get_spec(self):
        return CommandModuleSpec(
            name="login",
            cmd_specs=[
                CommandSpec(
                    name=InternalCommands.CERT_LOGIN,
                    description="login to server with SSL cert",
                    usage="login userName",
                    handler_func=self.handle_cert_login,
                    visible=False,
                ),
                CommandSpec(
                    name=InternalCommands.TOKEN_LOGIN,
                    description="login to server with bearer token",
                    usage=InternalCommands.TOKEN_LOGIN,
                    handler_func=self.handle_token_login,
                    visible=False,
                ),
                CommandSpec(
                    name=InternalCommands.LOGOUT,
                    description="logout from server",
                    usage="logout",
                    handler_func=self.handle_logout,
                    visible=False,
                ),
            ],
        )

    def handle_cert_login(self, conn: Connection, args: List[str]):
        if len(args) != 2:
            conn.append_string("REJECT")
            return

        user_name = args[1]
        headers = conn.get_prop(ConnProps.CMD_HEADERS)
        cert_data = headers.get("cert")
        signature = headers.get("signature")

        self.logger.debug(f"got cert login headers: {headers=}")
        hci = conn.get_prop(ConnProps.HCI_SERVER)
        identity_verifier = hci.get_id_verifier()
        id_asserter = hci.get_id_asserter()

        cert = load_crt_bytes(cert_data)
        try:
            ok = identity_verifier.verify_common_name(
                asserter_cert=cert,
                asserted_cn=user_name,
                signature=signature,
                nonce="",
            )
            self.logger.debug(f"verify common name: {ok=}")
        except Exception as ex:
            self.logger.error(f"identity_verifier.verify_common_name got exception: {ex}")
            traceback.print_exc()
            ok = False

        if not ok:
            conn.append_string("REJECT")
            return

        cert_dict = cert_to_dict(cert)
        self.logger.debug(f"got cert dict: {cert_dict}")
        identity = get_identity_info(cert_dict)

        request = conn.get_prop(ConnProps.REQUEST)
        assert isinstance(request, CellMessage)
        origin = request.get_header(MessageHeaderKey.ORIGIN)

        session = self.session_mgr.create_session(
            user_name=user_name,
            user_org=identity.get(IdentityKey.ORG, ""),
            user_role=identity.get(IdentityKey.ROLE, ""),
            origin_fqcn=origin,
        )
        token = session.make_token(id_asserter)
        self.logger.info(f"Created user session for {user_name}")
        conn.append_string("OK")
        conn.append_token(token)

    def handle_token_login(self, conn: Connection, args: List[str]):
        if len(args) != 1 or not self._token_login_enabled():
            conn.append_string("REJECT")
            return

        headers = conn.get_prop(ConnProps.CMD_HEADERS)
        token = self._extract_token_from_headers(headers)
        if not token:
            conn.append_string("REJECT")
            return

        hci = conn.get_prop(ConnProps.HCI_SERVER)
        id_asserter = hci.get_id_asserter()
        request = conn.get_prop(ConnProps.REQUEST)
        origin = request.get_header(MessageHeaderKey.ORIGIN) if isinstance(request, CellMessage) else ""

        try:
            claims = self.token_validator.validate(token=token, jwks=self._get_jwks())
            mapped_identity = self.claim_mapper.map(claims)
            token_expiry_time = claims.get("exp")
            if token_expiry_time is not None and not isinstance(token_expiry_time, (int, float)):
                raise ValueError("invalid token exp claim")

            session = self.session_mgr.create_session(
                user_name=mapped_identity.user_name,
                user_org=mapped_identity.user_org,
                user_role=mapped_identity.user_role,
                origin_fqcn=origin,
                token_expiry_time=float(token_expiry_time) if token_expiry_time is not None else None,
            )
            session_token = session.make_token(id_asserter)
            self.logger.info(f"Created token-auth user session for {mapped_identity.user_name}")
            conn.append_string("OK")
            conn.append_token(session_token)
        except Exception as ex:
            self.logger.error(f"token login failed: {secure_format_exception(ex)}")
            conn.append_string("REJECT")

    def _token_login_enabled(self) -> bool:
        return self.token_validator is not None and self.claim_mapper is not None

    def _get_jwks(self) -> Mapping[str, Any]:
        jwks = self.jwks_fetcher() if self.jwks_fetcher else self.token_jwks
        if not isinstance(jwks, Mapping):
            raise ValueError("token login jwks must be a mapping")
        return jwks

    @staticmethod
    def _extract_token_from_headers(headers: Any) -> Optional[str]:
        if not isinstance(headers, dict):
            return None

        token = headers.get("token")
        if isinstance(token, str) and token.strip():
            return token.strip()

        auth_header = headers.get("authorization")
        if not isinstance(auth_header, str):
            return None
        auth_header = auth_header.strip()
        if auth_header.lower().startswith("bearer "):
            bearer_token = auth_header[7:].strip()
            return bearer_token if bearer_token else None
        return None

    def handle_logout(self, conn: Connection, args: List[str]):
        if self.session_mgr:
            token = conn.get_prop(ConnProps.TOKEN)
            if token:
                self.session_mgr.end_session_by_token(token)
        conn.append_string("OK")

    def pre_command(self, conn: Connection, args: List[str]):
        if args[0] in [InternalCommands.CERT_LOGIN, InternalCommands.TOKEN_LOGIN, InternalCommands.CHECK_SESSION]:
            # skip login and check session commands
            return True

        # validate token
        token = conn.get_token()
        if token is None:
            conn.append_error("not authenticated - no token")
            return False

        sess = self.session_mgr.get_session(token)
        if not sess:
            # try to recreate the session
            request = conn.get_prop(ConnProps.REQUEST)
            assert isinstance(request, CellMessage)
            origin = request.get_header(MessageHeaderKey.ORIGIN)

            hci = conn.get_prop(ConnProps.HCI_SERVER)
            id_asserter = hci.get_id_asserter()

            try:
                sess = self.session_mgr.recreate_session(token, origin, id_asserter)
                self.logger.info(f"recreated admin session for {sess.user_name}")
            except Exception as ex:
                self.logger.error(f"cannot recreate admin session: {secure_format_exception(ex)}")
                conn.append_error(ReplyKeyword.SESSION_INACTIVE)
                conn.append_string(
                    "user not authenticated or session timed out after {} seconds of inactivity - logged out".format(
                        self.session_mgr.idle_timeout
                    )
                )
                return False

        assert isinstance(sess, Session)
        sess.mark_active()
        conn.set_prop(ConnProps.SESSION, sess)
        conn.set_prop(ConnProps.USER_NAME, sess.user_name)
        conn.set_prop(ConnProps.USER_ORG, sess.user_org)
        conn.set_prop(ConnProps.USER_ROLE, sess.user_role)
        conn.set_prop(ConnProps.TOKEN, token)
        return True

    def close(self):
        self.session_mgr.shutdown()
