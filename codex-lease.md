# Transfer Lease Design Spike

## Context

PR [NVIDIA/NVFlare#4736](https://github.com/NVIDIA/NVFlare/pull/4736) addresses large streamed task and result transfers by adding progress-aware waiting. The core problem is that several independent fixed-duration timeout knobs currently guard different parts of the same transfer lifecycle:

- `peer_read_timeout`
- `heartbeat_timeout`
- `download_complete_timeout`
- `*_min_download_timeout`
- `*_streaming_per_request_timeout`

The goal of the transfer lease approach is to avoid requiring users to tune these values manually in lockstep. A large transfer should be allowed to run as long as real monotonic transfer progress continues, and should fail clearly when progress stops.

## Current PR Approach

The current PR adds `_STREAM_PROGRESS_` events and a direction-neutral progress tracker. Progress events report monotonic counters such as `sequence`, `bytes_done`, and `items_done`. Waiters such as `TaskExchanger` and `FlareAgent` inspect those records and continue waiting while progress is recent.

High-level behavior:

```text
if peer ACKed:
    success
elif stream progress is recent:
    keep waiting
elif stream progress is idle too long:
    fail or retry
```

This is a pragmatic fix. It preserves most existing structure and adds progress awareness at the points that currently time out too early.

## Transfer Lease Alternative

A transfer lease makes the streamed transfer lifecycle a first-class object. Instead of each waiter interpreting raw progress events, the streaming subsystem owns the lease and exposes a simple lifecycle question:

```text
is this transfer still alive, complete, failed, aborted, or expired?
```

The transfer subsystem renews leases when real transfer progress occurs. Waiters ask the lease manager whether it is still valid to wait. Existing timeout knobs remain as fallback or transport-level details, not normal user-facing transfer policy.

## Comparison

| Area | Current progress-aware PR | Transfer lease design |
| --- | --- | --- |
| Authority | Each waiter interprets stream progress | Transfer subsystem owns lease state |
| Policy location | `TaskExchanger`, `FlareAgent`, `Cell`, progress trackers | `TransferLeaseManager` plus small wait adapters |
| Signal | Raw progress counters | Lease lifecycle updates: created, renewed, completed, failed |
| Cleanup | Mostly still `DownloadService` timeout/tombstones | Lease state can drive cleanup and subprocess lifetime |
| Config | Adds generic idle timeout but still aligns old knobs | One lease policy; old knobs become fallback/transport details |
| Scope | Lower-risk incremental change | Bigger change, cleaner long-term model |

The recommended implementation path is to spike leases as a mirror over the existing pipe event path, then promote lease ownership into `DownloadService` once the behavior is proven.

## Design Goal

Normal users should not set any timeout values for large streamed transfers.

Lease-enabled behavior:

```text
create a transfer lease when streamed refs are created
renew the lease when real monotonic progress occurs
complete the lease when all expected refs/receivers complete
fail or expire the lease when required progress stops
abort the lease on job cancel, peer gone, pipe close, or source shutdown
```

The transfer policy belongs to the transfer subsystem. Callers should not need to know which combination of timeouts keeps the source object, subprocess, pipe wait, and receiver alive.

## Core Model

Add a new module:

```text
nvflare/fuel/f3/streaming/transfer_lease.py
```

Core objects:

```python
class TransferLeaseState:
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    EXPIRED = "expired"


@dataclass
class TransferLeasePolicy:
    idle_timeout: float
    start_timeout: float
    completion_ack_grace: float
    max_peer_silence: float | None = None


@dataclass
class TransferLeasePair:
    ref_id: str
    receiver_id: str | None
    state: str
    sequence: int
    bytes_done: int
    items_done: int | None
    started_time: float | None
    last_renewed_time: float | None
    failure_reason: str | None = None


@dataclass
class TransferLease:
    lease_id: str
    direction: str
    owner_fqcn: str
    job_id: str
    task_id: str
    tx_id: str | None
    state: str
    created_time: float
    last_renewed_time: float
    policy: TransferLeasePolicy
    pairs: dict[tuple[str, str | None], TransferLeasePair]
    failure_reason: str | None = None
```

`direction` should initially support:

- `task_payload_download`
- `result_upload`

The same lease model can later apply to raw byte streams if those paths are wired.

## TransferLeaseManager API

The manager should own all lifecycle decisions:

```python
class TransferLeaseManager:
    def create_lease(
        self,
        *,
        lease_id: str | None,
        direction: str,
        owner_fqcn: str,
        job_id: str,
        task_id: str,
        tx_id: str | None,
        policy: TransferLeasePolicy,
    ) -> TransferLease:
        ...

    def register_pair(
        self,
        *,
        lease_id: str,
        ref_id: str,
        receiver_id: str | None,
    ) -> None:
        ...

    def renew(
        self,
        *,
        lease_id: str,
        ref_id: str,
        receiver_id: str | None,
        sequence: int,
        bytes_done: int,
        items_done: int | None,
        timestamp: float | None = None,
    ) -> LeaseUpdateResult:
        ...

    def complete_pair(
        self,
        *,
        lease_id: str,
        ref_id: str,
        receiver_id: str | None,
        timestamp: float | None = None,
    ) -> None:
        ...

    def fail_pair(
        self,
        *,
        lease_id: str,
        ref_id: str,
        receiver_id: str | None,
        reason: str,
        timestamp: float | None = None,
    ) -> None:
        ...

    def abort_lease(self, *, lease_id: str, reason: str) -> None:
        ...

    def snapshot(self, lease_id: str) -> TransferLeaseSnapshot | None:
        ...

    def decide(self, scope: TransferLeaseWaitScope) -> TransferLeaseDecision:
        ...

    def expire_idle_leases(self, now: float | None = None) -> list[TransferLease]:
        ...
```

Renewal rules:

- `sequence` must increase.
- `bytes_done` must not regress.
- `items_done` must not regress when present.
- At least one monotonic counter must advance to renew an active lease.
- Completion is always a terminal lifecycle event, but it should not hide an incomplete sibling ref or receiver.
- One progressing pair must not mask a stalled sibling pair.

## Lease Ticket

FOBS download references should carry a lease ticket while preserving legacy ref behavior.

Example serialized ref:

```json
{
  "fqcn": "server",
  "ref_id": "R...",
  "lease": {
    "lease_id": "L...",
    "direction": "task_payload_download",
    "owner_fqcn": "server",
    "tx_id": "T...",
    "job_id": "job-id",
    "task_id": "task-id"
  }
}
```

Refs without `lease` continue using the existing timeout behavior. This keeps the spike backward compatible and lets the implementation roll out gradually.

## Pipe Event

Add an internal pipe topic, for example:

```text
_TRANSFER_LEASE_
```

The pipe event is not the policy. It is only a transport for lease lifecycle updates between processes.

Event schema:

```json
{
  "event": "renewed",
  "lease_id": "L...",
  "direction": "task_payload_download",
  "job_id": "job-id",
  "task_id": "task-id",
  "tx_id": "T...",
  "ref_id": "R...",
  "receiver_id": "site-1",
  "sequence": 12,
  "bytes_done": 123456789,
  "items_done": 430,
  "state": "active",
  "timestamp": 1790000000.0
}
```

Supported event names:

- `created`
- `pair_registered`
- `renewed`
- `pair_completed`
- `pair_failed`
- `lease_completed`
- `lease_failed`
- `lease_aborted`
- `lease_expired`

For the spike, this can reuse the same side-channel mechanics as `_STREAM_PROGRESS_`. The final design should keep all liveness decisions inside `TransferLeaseManager`.

## Forward Task Payload Flow

Current failure:

```text
CJ sends task to subprocess
subprocess receives task but Cell/FOBS decode materializes large refs before ACK
CJ peer_read_timeout fires
CJ resends while subprocess is still downloading/materializing
```

Lease-enabled flow:

```text
server/aggr DownloadService creates lease and refs
CJ receives lazy refs and forwards lease tickets to subprocess
subprocess starts materializing refs
download_object renews lease as chunks/items arrive
subprocess sends _TRANSFER_LEASE_ updates to CJ
CJ suppresses resend while task-scoped leases remain alive
subprocess ACKs when task is accepted/materialized according to existing call path
```

`TaskExchanger` should not interpret bytes or item counts directly. It should own a small `TransferLeaseWaiter` keyed by `(job_id, task_id, msg_id)` and call:

```python
lease_waiter.should_continue_task_send_waiting(
    job_id=job_id,
    task_id=task_id,
    send_start_time=send_start_time,
)
```

Decision rules:

```text
if peer ACKed:
    success
elif no lease has arrived and send_start_time is within startup budget:
    keep waiting
elif all task_payload_download leases for the task are alive:
    keep waiting
elif all task_payload_download leases completed recently:
    keep waiting for short ACK grace
else:
    fail or allow existing resend policy
```

The startup budget should use the lease start timeout, not `peer_read_timeout`. This covers the gap before the first lease update crosses the subprocess/CJ pipe.

## Reverse Result Upload Flow

Reverse result upload is the strongest long-term reason for leases.

Current behavior:

```text
subprocess serializes result refs
CJ ACKs lightweight result
server/aggr pulls tensors from subprocess DownloadService
subprocess waits up to download_complete_timeout
```

This is fixed-duration and can fail even while the server is still making progress.

Lease-enabled flow:

```text
subprocess serializes large result
DownloadService creates result_upload lease
CJ ACKs lightweight refs
server/aggr pulls result chunks from subprocess DownloadService
DownloadService renews/completes lease pairs locally
FlareAgent waits on local lease manager
subprocess exits only after lease complete, failed, aborted, or expired
```

Expected pairs are `(ref_id, receiver_id)`. In swarm, `receiver_id` should be the aggregation client FQCN. A progressing receiver cannot hide a stalled receiver.

Decision rules:

```text
if no DownloadService transaction was created:
    proceed immediately
elif no lease tracking was installed:
    fall back to download_complete_timeout
elif any expected pair failed/aborted/expired:
    fail clearly
elif all expected pairs completed:
    wait short completion grace for callback ordering, then succeed
elif any started pair has not renewed within idle_timeout:
    expire/fail
elif any unstarted pair has not started within start_timeout:
    expire/fail
else:
    keep waiting
```

## Relationship To DownloadService

`DownloadService` already has lease-shaped concepts:

- transaction id
- ref id
- expected receiver count
- last active time
- timeout monitor
- completion status
- finished-ref tombstones

The final implementation should make `DownloadService` the canonical source of lease lifecycle for FOBS object transfers.

Initial integration points:

- `DownloadService.new_transaction()` creates or attaches a lease.
- `DownloadService.add_object()` registers lease pairs when receiver identities are known.
- `_handle_download()` renews a pair when bytes/items are served.
- EOF completes a pair.
- producer error fails a pair.
- transaction timeout expires the lease.
- transaction deletion aborts the lease.
- finished-ref tombstones preserve enough terminal lease state for delayed EOF retries.

For task-payload downloads, the source-side `DownloadService` owns served-byte progress, but the waiting CJ is not necessarily the source. The subprocess must still forward lease lifecycle events to CJ until there is a shared/distributed lease query mechanism.

For result uploads, the subprocess owns both source-side `DownloadService` progress and the wait decision, so no pipe round trip is needed.

## Configuration

Add advanced config:

```json
{
  "transfer_lease_idle_timeout": 600,
  "transfer_lease_start_timeout": 600,
  "transfer_lease_max_peer_silence": 900
}
```

Defaults:

```text
transfer_lease_idle_timeout = 600
transfer_lease_start_timeout = transfer_lease_idle_timeout
transfer_lease_max_peer_silence = max(900, 1.5 * transfer_lease_idle_timeout)
```

Normal users should not set these.

Existing knobs remain:

- `peer_read_timeout`: transport ACK/polling timeout, not large-transfer budget.
- `download_complete_timeout`: fallback only when no lease exists.
- `*_min_download_timeout`: fallback for legacy refs without leases.
- `*_streaming_per_request_timeout`: per-request transport timeout for object download requests.
- `heartbeat_timeout`: unchanged in phase 1.

Phase 2 may include lease renewals in peer liveness, bounded by `transfer_lease_max_peer_silence`.

## Spike Scope

The spike should validate the model without refactoring all streaming ownership at once.

Implement:

1. Add `TransferLeaseManager` and focused unit tests.
2. Add `_TRANSFER_LEASE_` pipe topic.
3. Add a receiver-side lease renewal callback from `download_object()`.
4. Add `TaskExchanger` lease waiter for forward task payload only.
5. Preserve old timeout behavior for refs without lease tickets.

Do not implement in the spike:

- heartbeat integration
- full source-side cleanup changes
- full reverse result upload replacement
- durable distributed lease storage
- raw byte stream integration outside FOBS object downloads

## Spike Tests

Required tests:

1. Delayed ACK plus lease renewals avoids resend.
2. No lease and no renewal fails after startup/idle budget.
3. Stale sequence is ignored.
4. Regressed bytes/items are ignored.
5. Sibling ref stall fails despite another ref renewing.
6. Terminal completion gives a short ACK grace.
7. Abort marks active leases terminal.
8. Legacy refs without lease tickets preserve existing behavior.
9. Bad lease events are ignored without crashing the pipe handler.
10. Lease manager prunes terminal/expired leases.

## Full Implementation Plan

### Phase 1: Forward Spike

- Implement local `TransferLeaseManager`.
- Carry lease tickets in FOBS refs for task-payload downloads.
- Forward lease renewals from subprocess to CJ.
- Use lease waiter in `TaskExchanger` to suppress duplicate resend.
- Keep all old timeout behavior available as fallback.

### Phase 2: DownloadService Ownership

- Make `DownloadService` create and own leases for FOBS object transactions.
- Register expected `(ref_id, receiver_id)` pairs when receiver identities are known.
- Emit lease lifecycle events from `_handle_download()`.
- Align transaction timeout/deletion/completion with lease state.

### Phase 3: Reverse Result Upload

- Replace `download_complete_timeout` wait with local result-upload leases when available.
- Keep fixed timeout fallback for non-lease transactions.
- Wire swarm receiver IDs so expected result-upload pairs are receiver-scoped.

### Phase 4: Configuration And Docs

- Rewrite timeout warnings around lease policy.
- Document that normal users do not tune transfer timeouts.
- Keep old timeout knobs documented as advanced fallback/transport controls.

### Phase 5: Optional Liveness Integration

- Include lease renewals in peer liveness calculations.
- Enforce `transfer_lease_max_peer_silence`.
- Keep heartbeat semantics unchanged unless this phase is explicitly enabled.

## Implementation Notes

- Keep lease updates monotonic and idempotent.
- Do not let a progressing sibling ref hide a stalled ref.
- Do not let one receiver hide another receiver for the same ref.
- Bound retained lease state and prune terminal records.
- Mark leases terminal on job kill, task abort, peer gone, pipe close, source shutdown, or transaction deletion.
- Make the spike fail closed: if lease state is malformed or unavailable, fall back to existing timeout behavior rather than waiting forever.
- Avoid exposing lease internals to normal users. The public operational story should be: large streamed transfers keep going while they make progress.
