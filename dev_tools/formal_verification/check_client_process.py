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

"""Check the production client-process machine against its TLA+ model."""

import tempfile
from collections import deque
from pathlib import Path

from tlc import canonical, parse_graph, parse_value, run_cli, verify_model

from nvflare.private.fed.client.client_process import (
    Action,
    Event,
    InvalidTransition,
    Phase,
    State,
    StopIntent,
    transition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "formal_models" / "client_process"
MODEL_FIELDS = {"phase": parse_value, "handleAttached": parse_value, "stopIntent": parse_value}
EXPECTED_MUTATIONS = {
    "ClientProcessUnsafeAttach.cfg": "Invariant AcceptedStopPreserved is violated.",
    "ClientProcessUnsafePrecedence.cfg": "Invariant AcceptedStopPreserved is violated.",
    "ClientProcessUnsafeStopped.cfg": "Invariant RemovalAfterExit is violated.",
    "ClientProcessUnsafeCompletion.cfg": "Invariant CompletionIsClean is violated.",
    "ClientProcessUnsafeRetryForever.cfg": "Temporal properties were violated.",
}


def _events() -> list[Event]:
    events = [
        Event(Action.REQUEST_STOP, stop_intent=StopIntent.HEARTBEAT_CLEANUP),
        Event(Action.REQUEST_STOP, stop_intent=StopIntent.USER_ABORT),
    ]
    events.extend(Event(action) for action in Action if action != Action.REQUEST_STOP)
    return events


def python_state_graph() -> tuple[set[State], set[tuple[State, Action, State]]]:
    pending = deque([State()])
    seen = {State()}
    edges = set()
    while pending:
        state = pending.popleft()
        for event in _events():
            try:
                next_state = transition(state, event)
            except InvalidTransition:
                continue
            edges.add((state, event.action, next_state))
            if next_state not in seen:
                seen.add(next_state)
                pending.append(next_state)
    return seen, edges


def _snapshot(state: State) -> dict:
    return {
        "phase": state.phase.value,
        "handleAttached": state.handle_attached,
        "stopIntent": state.stop_intent.value,
    }


def python_graph() -> tuple[set[str], set[tuple[str, str, str]]]:
    states, edges = python_state_graph()
    return (
        {canonical(_snapshot(state)) for state in states},
        {
            (canonical(_snapshot(source)), action.value, canonical(_snapshot(target)))
            for source, action, target in edges
        },
    )


def mermaid_phase_diagram() -> str:
    _, edges = python_state_graph()
    phase_order = {phase: index for index, phase in enumerate(Phase)}
    action_order = {action: index for index, action in enumerate(Action)}
    phase_edges = {(source.phase, action, target.phase) for source, action, target in edges}
    lines = ["stateDiagram-v2", "    direction LR", "    [*] --> launching"]
    for source, action, target in sorted(
        phase_edges,
        key=lambda edge: (phase_order[edge[0]], phase_order[edge[2]], action_order[edge[1]]),
    ):
        lines.append(f"    {source.value} --> {target.value}: {action.value}")
    lines.extend(("    done --> [*]", "    launch_failed --> [*]"))
    return "\n".join(lines)


def check(java: str, jar: Path) -> None:
    documented_diagram = f"```mermaid\n{mermaid_phase_diagram()}\n```"
    if documented_diagram not in (MODEL_DIR / "README.md").read_text(encoding="utf-8"):
        raise RuntimeError("README phase diagram does not match the production transition graph")

    with tempfile.TemporaryDirectory(prefix="nvflare_client_graph_") as temp_dir:
        dot = Path(temp_dir) / "graph.dot"
        verify_model(
            java,
            jar,
            MODEL_DIR,
            "ClientProcess",
            "ClientProcess.cfg",
            "ClientProcessLiveness.cfg",
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
    print("PASS bounded termination retries satisfy process-exit and cleanup liveness")
    for config, expected in EXPECTED_MUTATIONS.items():
        print(f"PASS {config} fails with {expected}")


if __name__ == "__main__":
    run_cli(check)
