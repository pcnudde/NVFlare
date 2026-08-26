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

"""Check the production completion machine against its bounded TLA+ graph."""

import tempfile
from collections import deque
from itertools import combinations
from pathlib import Path

from tlc import canonical, parse_graph, parse_set, parse_value, run_cli, verify_model

from nvflare.apis.job_def import RunStatus
from nvflare.private.fed.server.job_completion import (
    Action,
    ClientOutcome,
    Event,
    InvalidTransition,
    Phase,
    State,
    transition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "formal_models" / "job_completion"
MODEL_FIELDS = {
    "phase": parse_value,
    "pendingClients": parse_set,
    "status": parse_value,
    "archiveCommitted": parse_value,
}
EXPECTED_MUTATIONS = {
    "JobCompletionUnsafePublication.cfg": "Invariant CompletedPublicationHasArchive is violated.",
    "JobCompletionUnsafePrecedence.cfg": "Temporal properties were violated.",
    "JobCompletionUnsafeRearchive.cfg": "Invariant CommittedArchiveIsNotRewritten is violated.",
    "JobCompletionUnsafeRetryForever.cfg": "Temporal properties were violated.",
}


def _snapshot(state: State) -> dict:
    statuses = {
        None: "none",
        RunStatus.FINISHED_COMPLETED: "completed",
        RunStatus.FINISHED_ABORTED: "aborted",
        RunStatus.FINISHED_EXECUTION_EXCEPTION: "failed",
        RunStatus.FINISHED_ABNORMAL: "abnormal",
    }
    return {
        "phase": state.phase.value,
        "pendingClients": sorted(state.pending_clients),
        "status": statuses[state.status],
        "archiveCommitted": state.archive_committed,
    }


def python_state_graph(clients=("client_1", "client_2")) -> tuple[set[State], set[tuple[State, Action, State]]]:
    participant_sets = [
        frozenset(selected) for size in range(len(clients) + 1) for selected in combinations(clients, size)
    ]
    events = [Event(Action.PARTICIPANTS_SELECTED, clients=selected) for selected in participant_sets]
    events += [
        Event(Action.RECORD_CLIENT_OUTCOME, client=client, outcome=outcome)
        for client in clients
        for outcome in ClientOutcome
    ]
    events += [
        *(Event(Action.SERVER_EXITED, status=s) for s in _terminal_statuses()),
        *(Event(Action.TERMINAL_OVERRIDE, status=s) for s in _terminal_statuses() if s != RunStatus.FINISHED_COMPLETED),
        *(
            Event(action)
            for action in Action
            if action
            not in {
                Action.RECORD_CLIENT_OUTCOME,
                Action.PARTICIPANTS_SELECTED,
                Action.SERVER_EXITED,
                Action.TERMINAL_OVERRIDE,
            }
        ),
    ]
    initial = State.initial(clients)
    pending = deque([initial])
    seen = {initial}
    edges = set()
    while pending:
        state = pending.popleft()
        for event in events:
            try:
                next_state = transition(state, event)
            except InvalidTransition:
                continue
            edges.add((state, event.action, next_state))
            if next_state not in seen:
                seen.add(next_state)
                pending.append(next_state)
    return seen, edges


def python_graph(clients=("client_1", "client_2")) -> tuple[set[str], set[tuple[str, str, str]]]:
    states, edges = python_state_graph(clients)
    return (
        {canonical(_snapshot(state)) for state in states},
        {
            (canonical(_snapshot(source)), action.value, canonical(_snapshot(target)))
            for source, action, target in edges
        },
    )


def mermaid_phase_diagram() -> str:
    """Generate the readable phase/action projection of the production graph."""
    _, edges = python_state_graph()
    phase_order = {phase: index for index, phase in enumerate(Phase)}
    action_order = {action: index for index, action in enumerate(Action)}
    phase_edges = {(source.phase, action, target.phase) for source, action, target in edges}
    lines = ["stateDiagram-v2", "    direction LR", "    [*] --> waiting_for_server"]
    for source, action, target in sorted(
        phase_edges,
        key=lambda edge: (phase_order[edge[0]], phase_order[edge[2]], action_order[edge[1]]),
    ):
        lines.append(f"    {source.value} --> {target.value}: {action.value}")
    lines.append("    done --> [*]")
    return "\n".join(lines)


def _terminal_statuses():
    return (
        RunStatus.FINISHED_COMPLETED,
        RunStatus.FINISHED_ABORTED,
        RunStatus.FINISHED_EXECUTION_EXCEPTION,
        RunStatus.FINISHED_ABNORMAL,
    )


def check(java: str, jar: Path) -> None:
    documented_diagram = f"```mermaid\n{mermaid_phase_diagram()}\n```"
    readme = (MODEL_DIR / "README.md").read_text(encoding="utf-8")
    if documented_diagram not in readme:
        raise RuntimeError("README phase diagram does not match the production transition graph")

    with tempfile.TemporaryDirectory(prefix="nvflare_completion_graph_") as temp_dir:
        dot = Path(temp_dir) / "graph.dot"
        verify_model(
            java,
            jar,
            MODEL_DIR,
            "JobCompletion",
            "JobCompletion.cfg",
            "JobCompletionLiveness.cfg",
            EXPECTED_MUTATIONS,
            "-dump",
            "dot,actionlabels",
            str(dot),
        )
        tla_states, tla_edges = parse_graph(dot, MODEL_FIELDS)

    py_states, py_edges = python_graph()
    if py_states != tla_states or py_edges != tla_edges:
        raise RuntimeError(
            "Python/TLA+ graph mismatch: "
            f"states={len(py_states)}/{len(tla_states)}, edges={len(py_edges)}/{len(tla_edges)}, "
            f"python-only-state={next(iter(py_states - tla_states), None)}, "
            f"tla-only-state={next(iter(tla_states - py_states), None)}, "
            f"python-only-edge={next(iter(py_edges - tla_edges), None)}, "
            f"tla-only-edge={next(iter(tla_edges - py_edges), None)}"
        )

    print(f"PASS safe model and exact graph: {len(py_states)} states, {len(py_edges)} transitions")
    print("PASS README phase diagram matches the production graph")
    print("PASS bounded retries satisfy TerminalProgress")
    for config, expected in EXPECTED_MUTATIONS.items():
        print(f"PASS {config} fails with {expected}")


if __name__ == "__main__":
    run_cli(check)
