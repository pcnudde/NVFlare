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

"""Pure state machine for server-side job completion."""

from dataclasses import dataclass, replace
from enum import Enum

from nvflare.apis.job_def import RunStatus
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.fuel.common.exit_codes import ProcessExitCode


class Phase(str, Enum):
    WAITING_FOR_SERVER = "waiting_for_server"
    WAITING_FOR_CLIENTS = "waiting_for_clients"
    ARCHIVING = "archiving"
    CLEANING = "cleaning"
    PUBLISHING = "publishing"
    DONE = "done"


class Action(str, Enum):
    PARTICIPANTS_SELECTED = "ParticipantsSelected"
    RECORD_CLIENT_OUTCOME = "RecordClientOutcome"
    SERVER_EXITED = "ServerExited"
    TERMINAL_OVERRIDE = "TerminalOverride"
    CLIENT_WAIT_EXPIRED = "ClientWaitExpired"
    ARCHIVE_COMMITTED = "ArchiveCommitted"
    ARCHIVE_ABANDONED = "ArchiveAbandoned"
    CLEANUP_SETTLED = "CleanupSettled"
    STATUS_PUBLISHED = "StatusPublished"


class ClientOutcome(str, Enum):
    NO_OVERRIDE = "no_override"
    EXECUTION_FAILURE = "execution_failure"
    ABNORMAL = "abnormal"
    ABORTED = "aborted"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class Event:
    action: Action
    client: str | None = None
    clients: frozenset[str] | None = None
    status: RunStatus | None = None
    outcome: ClientOutcome | None = None


@dataclass(frozen=True)
class State:
    phase: Phase
    pending_clients: frozenset[str]
    status: RunStatus | None = None
    archive_committed: bool = False

    @classmethod
    def initial(cls, clients=()) -> "State":
        return cls(Phase.WAITING_FOR_SERVER, frozenset(clients))


class InvalidTransition(ValueError):
    pass


_TERMINAL_STATUSES = {
    RunStatus.FINISHED_COMPLETED,
    RunStatus.FINISHED_ABORTED,
    RunStatus.FINISHED_EXECUTION_EXCEPTION,
    RunStatus.FINISHED_ABNORMAL,
}
_NON_SUCCESS_STATUSES = _TERMINAL_STATUSES - {RunStatus.FINISHED_COMPLETED}

_WAITING = (Phase.WAITING_FOR_SERVER, Phase.WAITING_FOR_CLIENTS)


def transition(state: State, event: Event) -> State:
    """Apply one executable protocol rule."""
    phase = state.phase
    match event.action:
        case Action.PARTICIPANTS_SELECTED if phase == Phase.WAITING_FOR_SERVER and event.clients is not None:
            return replace(state, pending_clients=state.pending_clients.intersection(event.clients))
        case Action.RECORD_CLIENT_OUTCOME if (
            phase in _WAITING and event.client in state.pending_clients and event.outcome in ClientOutcome
        ):
            if event.outcome == ClientOutcome.NO_OVERRIDE:
                pending = state.pending_clients - {event.client}
                next_phase = Phase.ARCHIVING if phase == Phase.WAITING_FOR_CLIENTS and not pending else phase
                return replace(state, phase=next_phase, pending_clients=pending)
            next_phase = Phase.ARCHIVING if phase == Phase.WAITING_FOR_CLIENTS else phase
            return replace(
                state,
                phase=next_phase,
                pending_clients=frozenset(),
                status=merge_terminal_status(state.status, _OUTCOME_STATUSES[event.outcome]),
            )
        case Action.SERVER_EXITED if phase == Phase.WAITING_FOR_SERVER and event.status in _TERMINAL_STATUSES:
            next_phase = Phase.WAITING_FOR_CLIENTS if state.pending_clients else Phase.ARCHIVING
            return replace(state, phase=next_phase, status=merge_terminal_status(state.status, event.status))
        case Action.TERMINAL_OVERRIDE if phase in _WAITING and event.status in _NON_SUCCESS_STATUSES:
            next_phase = Phase.ARCHIVING if phase == Phase.WAITING_FOR_CLIENTS else phase
            return replace(
                state,
                phase=next_phase,
                pending_clients=frozenset(),
                status=merge_terminal_status(state.status, event.status),
            )
        case Action.CLIENT_WAIT_EXPIRED if phase == Phase.WAITING_FOR_CLIENTS:
            return replace(
                state,
                phase=Phase.ARCHIVING,
                pending_clients=frozenset(),
                status=_fail_candidate_success(state.status),
            )
        case Action.ARCHIVE_COMMITTED if phase == Phase.ARCHIVING:
            return replace(state, phase=Phase.CLEANING, archive_committed=True)
        case Action.ARCHIVE_ABANDONED if phase == Phase.ARCHIVING:
            return replace(state, phase=Phase.PUBLISHING, status=_fail_candidate_success(state.status))
        case Action.CLEANUP_SETTLED if phase == Phase.CLEANING:
            return replace(state, phase=Phase.PUBLISHING)
        case Action.STATUS_PUBLISHED if phase == Phase.PUBLISHING and (
            state.status != RunStatus.FINISHED_COMPLETED or state.archive_committed
        ):
            return replace(state, phase=Phase.DONE)
    raise InvalidTransition(f"cannot {event.action.value} while {phase.value}")


def _fail_candidate_success(status: RunStatus | None) -> RunStatus | None:
    return RunStatus.FINISHED_EXECUTION_EXCEPTION if status == RunStatus.FINISHED_COMPLETED else status


_OUTCOME_STATUSES = {
    ClientOutcome.EXECUTION_FAILURE: RunStatus.FINISHED_EXECUTION_EXCEPTION,
    ClientOutcome.ABNORMAL: RunStatus.FINISHED_ABNORMAL,
    ClientOutcome.ABORTED: RunStatus.FINISHED_ABORTED,
    ClientOutcome.UNSAFE: RunStatus.FINISHED_ABORTED,
}


_STATUS_PRECEDENCE = {
    None: 0,
    RunStatus.FINISHED_COMPLETED: 1,
    RunStatus.FINISHED_ABORTED: 2,
    RunStatus.FINISHED_EXECUTION_EXCEPTION: 3,
    RunStatus.FINISHED_ABNORMAL: 4,
}


def merge_terminal_status(existing: RunStatus | None, candidate: RunStatus) -> RunStatus:
    """Keep the strongest observed terminal result, independent of event order."""
    return candidate if _STATUS_PRECEDENCE[candidate] > _STATUS_PRECEDENCE[existing] else existing


@dataclass(frozen=True)
class ReturnCodeDisposition:
    client_outcome: ClientOutcome
    server_status: RunStatus


_RETURN_CODE_DISPOSITIONS = {
    ProcessExitCode.CONFIG_ERROR: ReturnCodeDisposition(
        ClientOutcome.EXECUTION_FAILURE, RunStatus.FINISHED_EXECUTION_EXCEPTION
    ),
    ProcessExitCode.EXCEPTION: ReturnCodeDisposition(
        ClientOutcome.EXECUTION_FAILURE, RunStatus.FINISHED_EXECUTION_EXCEPTION
    ),
    ProcessExitCode.INFRASTRUCTURE_ERROR: ReturnCodeDisposition(ClientOutcome.ABNORMAL, RunStatus.FINISHED_ABNORMAL),
    JobReturnCode.ABORTED: ReturnCodeDisposition(ClientOutcome.ABORTED, RunStatus.FINISHED_ABORTED),
    ProcessExitCode.UNSAFE_COMPONENT: ReturnCodeDisposition(
        ClientOutcome.UNSAFE, RunStatus.FINISHED_EXECUTION_EXCEPTION
    ),
    JobReturnCode.EXECUTION_ERROR: ReturnCodeDisposition(
        ClientOutcome.NO_OVERRIDE, RunStatus.FINISHED_EXECUTION_EXCEPTION
    ),
}


def classify_client_outcome(return_code: int) -> ClientOutcome:
    disposition = _RETURN_CODE_DISPOSITIONS.get(return_code)
    return disposition.client_outcome if disposition else ClientOutcome.NO_OVERRIDE


def status_for_process_failure(return_code: int) -> RunStatus:
    disposition = _RETURN_CODE_DISPOSITIONS.get(return_code)
    return disposition.server_status if disposition else RunStatus.FINISHED_EXECUTION_EXCEPTION


def merge_process_return_code(
    existing: int | None,
    candidate: int,
    *,
    existing_is_authoritative: bool,
    candidate_is_authoritative: bool,
) -> int | None:
    """Merge an external failure report or secondary process-exit code."""
    if existing_is_authoritative and not candidate_is_authoritative:
        return existing
    if existing is None:
        return candidate
    if not candidate_is_authoritative:
        return existing
    if existing == ProcessExitCode.INFRASTRUCTURE_ERROR or candidate == JobReturnCode.ABORTED:
        return existing
    return candidate
