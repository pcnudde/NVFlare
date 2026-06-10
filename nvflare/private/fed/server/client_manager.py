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

import os
import threading
import time
import uuid
from typing import Optional, Tuple

from nvflare.apis.client import Client, ClientPropKey
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.state_store import StateStore
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.fuel.utils.admin_name_utils import is_valid_admin_client_name
from nvflare.fuel.utils.argument_utils import str2bool
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegSession, ClientType, InternalFLContextKey
from nvflare.private.fed.server.cred_keeper import CredKeeper
from nvflare.private.fed.utils.identity_utils import get_org_from_cert, load_crt_bytes
from nvflare.security.logging import secure_format_exception


class _DisabledClientCache:
    """Read-through TTL cache of disabled-client lookups: client_name => (is_disabled, fetched_at).

    Both positive and negative results are cached so heartbeats don't hit the state store DB on
    every beat. Guarded by its own lock so cache reads/writes never wait behind ClientManager.lock
    (the HOT paths' store I/O is never performed while holding either lock).

    Stale-fill protection (epoch scheme): a per-client monotonic counter is bumped (under the
    cache lock) whenever an authoritative value is installed (set_authoritative, called by
    disable_client/enable_client). Readers record the epoch BEFORE their store read (lookup) and
    only install the read result if the epoch is unchanged afterward (fill) — so a stale store
    read that raced with an admin disable/enable can never clobber the authoritative cache entry.
    """

    def __init__(self, ttl: float):
        self.ttl = ttl
        self._entries = dict()  # client_name => (is_disabled, fetched_at)
        self._epoch = dict()  # client_name => int; see epoch scheme above
        self._lock = threading.Lock()

    def get(self, client_name: str, ignore_ttl: bool = False) -> Optional[bool]:
        """Return the cached disabled state of the client, or None if not cached (or expired)."""
        with self._lock:
            entry = self._entries.get(client_name)
        return self._unexpired_value(entry, ignore_ttl)

    def lookup(self, client_name: str) -> Tuple[Optional[bool], int]:
        """Return (cached value or None, current epoch) in a single lock acquisition.

        The returned epoch must be recorded before a store read so the result can be installed
        safely via fill().
        """
        with self._lock:
            entry = self._entries.get(client_name)
            epoch = self._epoch.get(client_name, 0)
        return self._unexpired_value(entry, ignore_ttl=False), epoch

    def _unexpired_value(self, entry, ignore_ttl: bool) -> Optional[bool]:
        if entry is None:
            return None
        is_disabled, fetched_at = entry
        if not ignore_ttl and time.time() - fetched_at >= self.ttl:
            return None
        return is_disabled

    def set_authoritative(self, client_name: str, is_disabled: bool):
        """Install an authoritative (disable_client/enable_client) value and bump the epoch.

        The epoch bump invalidates any in-flight store read that began before this write, so a
        stale read result can never overwrite this entry (see fill).
        """
        with self._lock:
            self._epoch[client_name] = self._epoch.get(client_name, 0) + 1
            self._entries[client_name] = (is_disabled, time.time())

    def fill(self, client_name: str, is_disabled: bool, read_epoch: int) -> bool:
        """Install a store-read result, unless an authoritative write landed since the read began.

        Args:
            client_name: the client the read was for
            is_disabled: the store-read result
            read_epoch: the client's epoch recorded BEFORE the store read began

        Returns:
            The value now effective for the client: the read result if it was installed, or the
            newer authoritative cached value if the epoch advanced during the read.
        """
        with self._lock:
            if self._epoch.get(client_name, 0) != read_epoch:
                # set_authoritative updated the cache while our store read was in flight; our
                # result is stale. Keep the authoritative entry and report it.
                entry = self._entries.get(client_name)
                if entry is not None:
                    return entry[0]
                return is_disabled
            self._entries[client_name] = (is_disabled, time.time())
            return is_disabled


class ClientManager:

    DISABLED_CACHE_TTL = 10.0  # seconds; cross-server HA convergence happens via this TTL

    # Env var consulted when disabled_check_fail_open is not passed explicitly: a truthy value
    # ("1", "true", "t", "yes", "y", case-insensitive) makes the disabled-client check fail CLOSED.
    DISABLED_FAIL_CLOSED_ENV = "NVFL_DISABLED_CLIENT_FAIL_CLOSED"

    def __init__(
        self,
        project_name=None,
        min_num_clients=2,
        max_num_clients=10,
        disabled_cache_ttl=None,
        disabled_check_fail_open: Optional[bool] = None,
    ):
        """Manages client adding and removing.

        Args:
            project_name: project name
            min_num_clients: minimum number of clients allowed.
            max_num_clients: maximum number of clients allowed.
            disabled_cache_ttl: TTL (seconds) of the disabled-client read-through cache.
            disabled_check_fail_open: behavior of the disabled-client check when the state store
                read fails and no cached value exists for the client. True (fail open):
                treat the client as not-disabled, favoring availability — heartbeats keep working
                through a DB blip, but a disabled client that was never cached on this server could
                slip through while the DB is down. False (fail closed): treat the client as
                DISABLED, favoring security — no disabled client is ever admitted, at the cost of
                rejecting never-before-seen clients during a DB outage. With either setting, a
                cached value (even expired) is preferred over the failure default, so only clients
                never cached on this server are affected. The default None consults the
                NVFL_DISABLED_CLIENT_FAIL_CLOSED environment variable: when set to a truthy value
                ("1", "true", "t", "yes", "y", case-insensitive) the check fails closed; otherwise
                (unset, falsy, or unrecognized) it fails open. An explicit True/False always wins
                over the environment variable.
        """
        self.project_name = project_name
        # TODO:: remove min num clients
        self.min_num_clients = min_num_clients
        self.max_num_clients = max_num_clients
        self.clients = dict()  # token => Client
        self.name_to_clients = dict()  # name => Client
        self.state_store: Optional[StateStore] = None
        self.cred_keeper = CredKeeper()
        self.lock = threading.Lock()
        self.num_relays = 0

        self.disabled_cache_ttl = self.DISABLED_CACHE_TTL if disabled_cache_ttl is None else disabled_cache_ttl
        if disabled_check_fail_open is None:
            # str2bool returns None for unset/unrecognized values, so those default to fail open.
            fail_closed = str2bool(os.environ.get(self.DISABLED_FAIL_CLOSED_ENV, "").strip())
            disabled_check_fail_open = not fail_closed
        self.disabled_check_fail_open = disabled_check_fail_open
        self._disabled_cache = _DisabledClientCache(self.disabled_cache_ttl)

        self.logger = get_obj_logger(self)

    def set_state_store(self, state_store: StateStore):
        assert state_store is not None, "state_store must be configured before disabled-client checks are used"
        self.state_store = state_store

    def _require_state_store(self) -> StateStore:
        assert self.state_store is not None, "state_store must be configured before disabled-client checks are used"
        return self.state_store

    def _reject_disabled(self, fl_ctx: FLContext, client_name: str, action: str) -> None:
        """Mark the FLContext unauthenticated and log the rejection of a disabled client."""
        fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, f"Client '{client_name}' is disabled", sticky=False)
        self.logger.warning(f"Reject disabled client {action}: {client_name}")

    def is_client_disabled(self, client_name: str) -> bool:
        """Check whether the client is disabled, via a read-through TTL cache over the state store.

        Must NOT be called while holding self.lock: a cache miss performs store I/O, and a slow
        store must not serialize the client control plane.

        On store read errors, falls back to the last cached value (ignoring TTL) if one exists.
        With no cached value the result depends on disabled_check_fail_open: True (default)
        degrades open and treats the client as not-disabled, so heartbeats don't fail because
        the DB blipped; False fails closed and treats the client as DISABLED, so an outage can
        never admit a disabled client that was never cached on this server. See the constructor
        docstring for the tradeoff. A missing state store is a configuration error and still
        raises.

        Stale-fill protection: the client's epoch is recorded before the store read, and the
        result is installed only if no authoritative disable/enable bumped the epoch in the
        meantime (see _DisabledClientCache.fill), so a stale read can never poison the cache.
        """
        store = self._require_state_store()
        cached, read_epoch = self._disabled_cache.lookup(client_name)
        if cached is not None:
            return cached
        try:
            is_disabled = store.get_disabled_client(client_name) is not None
        except Exception as ex:
            stale = self._disabled_cache.get(client_name, ignore_ttl=True)
            if stale is None:
                result = not self.disabled_check_fail_open  # fail open: not disabled; fail closed: disabled
                reason = (
                    "no cached value, fail-open" if self.disabled_check_fail_open else "no cached value, fail-closed"
                )
            else:
                result = bool(stale)
                reason = "last cached value"
            self.logger.error(
                f"failed to read disabled state of client {client_name} from state store: "
                f"{secure_format_exception(ex)}; treating as {'disabled' if result else 'not disabled'} ({reason})"
            )
            return result
        return self._disabled_cache.fill(client_name, is_disabled, read_epoch)

    def disable_client(self, client_name: str) -> list:
        # The whole admin action — store write, cache update (with epoch bump), token sweep — is
        # serialized under self.lock so store order and cache order can never cross with a
        # concurrent enable_client/disable_client. Store write first (errors propagate to the
        # admin caller and leave the cache untouched); the cache update under self.lock means
        # registration/heartbeat rechecking the cache under self.lock always see a disable that
        # committed before they acquired the lock, and the epoch bump prevents an in-flight stale
        # store read from clobbering it. Admin ops are rare, so store I/O under self.lock is
        # acceptable here (the HOT paths' store reads stay outside self.lock).
        with self.lock:
            self._require_state_store().disable_client(client_name)
            self._disabled_cache.set_authoritative(client_name, True)
            removed_clients = []
            for token, client in list(self.clients.items()):
                if client.name == client_name:
                    removed_clients.append((token, client))
                    self.clients.pop(token, None)
            self.name_to_clients.pop(client_name, None)
        removed_tokens = [token for token, _client in removed_clients]
        self.logger.info(f"Client {client_name} disabled. Removed active tokens: {removed_tokens}")
        return removed_tokens

    def enable_client(self, client_name: str) -> bool:
        # Serialized under self.lock for the same reason as disable_client: store write and cache
        # update (with epoch bump) must not cross with a concurrent disable/enable, and an
        # in-flight stale store read must not clobber the result. Store write errors propagate to
        # the admin caller and leave the cache untouched.
        with self.lock:
            was_disabled = self._require_state_store().enable_client(client_name)
            self._disabled_cache.set_authoritative(client_name, False)
        self.logger.info(f"Client {client_name} enabled. Was disabled: {was_disabled}")
        return was_disabled

    def set_clients(self, clients: dict):
        self.clients = clients
        self.name_to_clients = {}
        for c in clients.values():
            self.name_to_clients[c.name] = c

    def authenticate(self, request, fl_ctx: FLContext) -> Optional[Client]:
        client_type = request.get_header(CellMessageHeaderKeys.CLIENT_TYPE)
        client = self.login_client(request, fl_ctx, client_type)
        if not client:
            return None

        # client_ip = context.peer().split(":")[1]
        client_ip = request.get_header(CellMessageHeaderKeys.CLIENT_IP)

        # new client join
        with self.lock:
            if client_type == ClientType.REGULAR:
                self.name_to_clients[client.name] = client
                self.clients.update({client.token: client})
                client_kind = "client"
            else:
                # do not update self.clients for non-regular clients
                client_kind = client_type

            self.logger.info(
                "Client: New {} {} joined. Sent token: {}.  Total clients: {}".format(
                    client_kind, client.name + "@" + client_ip, client.token, len(self.clients)
                )
            )
        return client

    def remove_client(self, token):
        """Remove a registered client's active token entry.

        Args:
            token: client token

        Returns:
            The removed Client object, if the token was active
        """
        with self.lock:
            client = self.clients.pop(token, None)
            if client:
                self.name_to_clients.pop(client.name, None)
                self.logger.info(
                    "Client Name:{} \tToken: {} left.  Total clients: {}".format(client.name, token, len(self.clients))
                )
            else:
                self.logger.warning("remove_client: unknown token %s", token)
            return client

    def login_client(self, client_login, fl_ctx: FLContext, client_type):
        proj_name = client_login.get_header(CellMessageHeaderKeys.PROJECT_NAME)
        if not self.is_valid_task(proj_name):
            fl_ctx.set_prop(
                FLContextKey.UNAUTHENTICATED, "Requested task does not match the current server task", sticky=False
            )
            self.logger.error(f"login_client failed: {proj_name}")
            return None
        return self.authenticated_client(client_login, fl_ctx, client_type)

    def has_relays(self):
        return self.num_relays > 0

    def validate_client(self, request, fl_ctx: FLContext, allow_new=False):
        """Validate the client state message.

        Args:
            request: A request from client.
            fl_ctx: FLContext
            allow_new: whether to allow new client. Note that its task should still match server's.

        Returns:
             client id if it's a valid client
        """
        # token = client_state.token
        token = request.get_header(CellMessageHeaderKeys.TOKEN)
        if not token:
            fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, "Could not read client uid from the payload", sticky=False)
            client = None
        elif not self.is_valid_task(request.get_header(CellMessageHeaderKeys.PROJECT_NAME)):
            fl_ctx.set_prop(
                FLContextKey.UNAUTHENTICATED, "Requested task does not match the current server task", sticky=False
            )
            client = None
        elif not (allow_new or self.is_from_authorized_client(token)):
            fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, "Unknown client identity", sticky=False)
            client = None
        else:
            client = self.clients.get(token)
        return client

    def _get_id_verifier(self, fl_ctx: FLContext):
        return self.cred_keeper.get_id_verifier(fl_ctx)

    def authenticated_client(self, request, fl_ctx: FLContext, client_type) -> Optional[Client]:
        """Use SSL certificate for authenticate the client.

        Args:
            request: client login request Message
            fl_ctx: FL_Context
            client_type: type of the client

        Returns:
            Client object.
        """
        client_name = request.get_header(CellMessageHeaderKeys.CLIENT_NAME)
        if self.is_client_disabled(client_name):
            self._reject_disabled(fl_ctx, client_name, "registration")
            return None

        shareable = request.payload
        if not isinstance(shareable, Shareable):
            self.logger.error(f"payload must be Shareable but got {type(shareable)}")
            return None

        secure_mode = fl_ctx.get_prop(FLContextKey.SECURE_MODE, False)
        client_org = ""
        asserter_cert_data = shareable.get(IdentityChallengeKey.CERT)
        if secure_mode:
            # verify client identity
            if not asserter_cert_data:
                self.logger.error("missing client cert in register request")
                return None

            signature = shareable.get(IdentityChallengeKey.SIGNATURE)
            if not signature:
                self.logger.error("missing signature in register request")
                return None

            asserter_cert = load_crt_bytes(asserter_cert_data)
            id_verifier = self._get_id_verifier(fl_ctx)
            reg = fl_ctx.get_prop(InternalFLContextKey.CLIENT_REG_SESSION)
            if not reg:
                self.logger.error(f"missing {InternalFLContextKey.CLIENT_REG_SESSION} in FLContext!")
                return None

            if not isinstance(reg, ClientRegSession):
                self.logger.error(f"reg should be ClientRegSession but got {type(reg)}")
                return None

            try:
                id_verifier.verify_common_name(
                    asserted_cn=client_name,
                    asserter_cert=asserter_cert,
                    signature=signature,
                    nonce=reg.nonce,
                )
            except Exception as ex:
                self.logger.error(f"failed to verify client identity: {secure_format_exception(ex)}")
                return None

            self.logger.debug(f"identity verified for client '{client_name}'")
            client_org = get_org_from_cert(asserter_cert)
        elif asserter_cert_data:
            try:
                asserter_cert = load_crt_bytes(asserter_cert_data)
                client_org = get_org_from_cert(asserter_cert)
            except Exception:
                pass

        with self.lock:
            # Recheck under lock (cache only, no store I/O) so disable_client cannot race with
            # registration after the fast-path check above. Invariant: disable_client updates the
            # cache (with an epoch bump) while holding self.lock, so any disable that completed
            # before we acquired the lock is visible here — and the epoch scheme guarantees no
            # in-flight stale store read can have overwritten it with not-disabled. A disable that
            # lands after we release the lock removes our token in its own sweep. Either way a
            # disabled client never stays registered.
            if self._disabled_cache.get(client_name, ignore_ttl=True):
                self._reject_disabled(fl_ctx, client_name, "registration")
                return None

            clients_to_be_removed = [token for token, client in self.clients.items() if client.name == client_name]
            for item in clients_to_be_removed:
                client = self.clients.pop(item, None)
                if client:
                    self.name_to_clients.pop(client.name, None)
                self.logger.info(f"Client: {client_name} already registered. Re-login the client with a new token.")

        client = Client(client_name, str(uuid.uuid4()))
        client.set_prop(ClientPropKey.ORG, client_org)
        client_fqcn = request.get_header(MessageHeaderKey.ORIGIN)
        self._set_client_props(client, client_fqcn, fl_ctx)
        self.logger.debug(f"authenticated client {client_name}: {client_fqcn=}")

        if client_type == ClientType.REGULAR and len(self.clients) >= self.max_num_clients:
            # only impose the limit to REGULAR clients
            fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, "Maximum number of clients reached", sticky=False)
            self.logger.info(f"Maximum number of clients reached. Reject client: {client_name} login.")
            return None

        if client_type == ClientType.RELAY:
            self.num_relays += 1

        return client

    def is_from_authorized_client(self, token):
        """Check if a client is authorized.

        Args:
            token: client token

        Returns:
            True if it is a recognised client
        """
        return token in self.clients

    def is_valid_task(self, task):
        """Check whether the requested task matches the server's project_name.

        Returns:
            True if task name is the same as server's project name.
        """
        # TODO: change the name of this method
        return task == self.project_name

    def heartbeat(self, token, client_name, client_fqcn, fl_ctx: FLContext):
        """Update the heartbeat of the client.

        Args:
            token: client token
            client_name: client name
            client_fqcn: FQCN of the client
            fl_ctx: FLContext

        Returns:
            If a new client needs to be created.
        """
        # Check (with possible store I/O) outside self.lock so a slow store can't serialize
        # the client control plane.
        if self.is_client_disabled(client_name):
            self._reject_disabled(fl_ctx, client_name, "heartbeat")
            return False

        with self.lock:
            # Recheck under lock (cache only, no store I/O) so disable_client cannot race with
            # re-activation below; see the registration recheck in authenticated_client for the
            # full epoch invariant. A later disable removes the token itself.
            if self._disabled_cache.get(client_name, ignore_ttl=True):
                self._reject_disabled(fl_ctx, client_name, "heartbeat")
                return False

            client = self.clients.get(token)
            if client:
                client.last_connect_time = time.time()
                self.logger.debug(f"Receive heartbeat from Client:{token}")
                return False
            else:
                for _token, _client in self.clients.items():
                    if _client.name == client_name:
                        fl_ctx.set_prop(
                            FLContextKey.COMMUNICATION_ERROR,
                            "Client ID already registered as a client: {}".format(client_name),
                            sticky=False,
                        )
                        self.logger.info(
                            f"Failed to re-activate the client:{client_name} with token: {token}. "
                            f"Client already exist with token: {_token}."
                        )
                        return False

                client = Client(client_name, token)
                self._set_client_props(client, client_fqcn, fl_ctx)
                self.clients.update({token: client})
                self.name_to_clients[client.name] = client
                self.logger.info(f"Re-activate the client: {client_name} at {client_fqcn} with token: {token}")
                return True

    @staticmethod
    def _set_client_props(client: Client, fqcn: str, fl_ctx: FLContext):
        client.set_fqcn(fqcn)
        client.last_connect_time = time.time()
        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx:
            client.set_fqsn(peer_ctx.get_prop(ReservedKey.FQSN, "?"))
            client.set_is_leaf(peer_ctx.get_prop(ReservedKey.IS_LEAF, "?"))
        site_config = fl_ctx.get_prop(FLContextKey.CLIENT_SITE_CONFIG)
        if site_config is not None:
            client.set_site_config(site_config)

    def get_clients(self):
        """Get the list of registered clients.

        Returns:
            A dict of {client_token: client}
        """
        return self.clients

    def get_min_clients(self):
        return self.min_num_clients

    def get_max_clients(self):
        return self.max_num_clients

    def get_all_clients_from_inputs(self, inputs):
        clients = []
        invalid_inputs = []
        for item in inputs:
            client = self.clients.get(item)
            # if item in self.get_all_clients():
            if client:
                clients.append(client)
            else:
                client = self.get_client_from_name(item)
                if client:
                    clients.append(client)
                else:
                    invalid_inputs.append(item)
        return clients, invalid_inputs

    def get_client_from_name(self, client_name):
        result = self.name_to_clients.get(client_name)
        if not result:
            # Check whether this is a valid admin client.
            # Note that since admin clients are not kept in name_to_clients, we assume that the admin client
            # is valid and dynamically create the Client object as the result.
            if is_valid_admin_client_name(client_name):
                result = Client(client_name, None)
                result.set_fqcn(client_name)
            else:
                self.logger.debug(
                    f"no client for {client_name}: I have {self.name_to_clients.keys()} {self.clients.keys()}"
                )
        return result
