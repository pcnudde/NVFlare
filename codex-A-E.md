# A+E Design: Lazy Forward Receive And Data-Plane Idle Authority

## Summary

This design is an alternative to PR [NVIDIA/NVFlare#4736](https://github.com/NVIDIA/NVFlare/pull/4736). Instead of adding a parallel progress-signaling subsystem around existing control-plane waits, it changes the layering so control-plane ACKs no longer depend on large payload materialization, and data-plane idle timeout becomes the authority for streamed object transfer.

The two parts are:

- **A. Decouple forward task ACK from materialization**: receive CJ -> subprocess task messages in PASS_THROUGH mode, ACK after receiving the lightweight task/ref envelope, then materialize lazy refs in `FlareAgent.get_task()` before returning data to user code.
- **E. Make DownloadService/data-plane idle timeout authoritative**: for reverse result uploads, wait for the `DownloadService` transaction to finish or time out by inactivity instead of waiting for a fixed `download_complete_timeout` wall-clock budget.

The target user-facing behavior is:

```text
normal users set no transfer timeouts
small control messages ACK quickly
large streamed objects live while the data plane is active
stalled data-plane transfers fail via DownloadService inactivity timeout
dead peer/job abort still terminates waits promptly
```

## Current Failure

Forward task delivery currently couples control-plane ACK to payload materialization:

```text
CJ sends task to subprocess over CellPipe
subprocess Adapter.call() decodes FOBS payload before CellPipe._receive_message()
ViaDownloaderDecomposer.process_datum() downloads tensors inline when PASS_THROUGH is false
CellPipe._receive_message() returns ACK only after decode/materialization finishes
CJ peer_read_timeout fires if the download takes too long
CJ resends the task while the first materialization is still active
```

This makes `peer_read_timeout` depend on model size, network throughput, receiver fanout, and server-side chunk transfer behavior. That is the root of the timeout tuning problem for the documented 5GB x 16-client case.

The data plane is already closer to the desired model:

- `DownloadService` transaction timeout is inactivity-based: it expires after no download activity for `timeout`, not after total wall-clock duration.
- `download_object()` does per-request timeout, retry, and exponential backoff.
- `ViaDownloaderDecomposer` already supports PASS_THROUGH: it can preserve refs as `LazyDownloadRef` placeholders instead of materializing tensors at an intermediate hop.

The defect is that control-plane waits cannot see or defer to that data-plane lifecycle.

## Goals

- Make `peer_read_timeout` independent of streamed task payload size.
- Keep user training code receiving normal materialized `DXO`/`FLModel` data, not `LazyDownloadRef`.
- Let `DownloadService` transaction inactivity decide streamed object failure.
- Remove the need for users to coordinate `peer_read_timeout`, `download_complete_timeout`, `*_min_download_timeout`, and `*_streaming_per_request_timeout`.
- Keep old fixed timeout behavior as fallback for non-CellPipe, non-PASS_THROUGH, and legacy paths.
- Keep implementation smaller and more structural than a new cross-pipe progress subsystem.

## Non-Goals

- Do not add a new progress event protocol in this spike.
- Do not integrate transfer activity into heartbeat/liveness yet.
- Do not build a durable distributed transfer lease service in this design.
- Do not expose `LazyDownloadRef` to user application code.
- Do not remove true job/task deadlines such as controller-level task timeout. Those remain job policy, not transfer tuning.

## Design A: Lazy Forward Receive

### Core Idea

Use receiver-side PASS_THROUGH on the subprocess task `CellPipe` channel.

Today, PASS_THROUGH is used in two relevant places:

- CJ receiver-side PASS_THROUGH on `CellChannel.SERVER_COMMAND`, so the CJ can receive server task refs as `LazyDownloadRef` instead of materializing tensors.
- subprocess reverse `pass_through_on_send=True`, so CJ ACKs lightweight result refs and the server/aggregator downloads result tensors directly from the subprocess.

This design adds the missing forward subprocess receive step:

```text
CJ -> subprocess task message
subprocess Adapter.call() decodes with PASS_THROUGH=True
ViaDownloader creates LazyDownloadRef placeholders instead of downloading
CellPipe._receive_message() queues the lightweight message and returns ACK quickly
FlareAgent.get_task() resolves LazyDownloadRef to real tensors before user code receives task data
```

The ACK path now depends on receiving and decoding a small control envelope, not on downloading 5GB of tensor data.

### Why Receiver-Side PASS_THROUGH

Prefer receiver-side enablement over sender stamping:

- The subprocess is the endpoint that needs fast ACK behavior.
- The CJ `TaskExchanger` does not need to know whether the subprocess wants lazy forward receive.
- Metrics and other pipes can remain eager.
- Existing `Cell.decode_pass_through_channels` already supports receiver-side PASS_THROUGH.

### Proposed Code Changes

#### `nvflare/fuel/utils/pipe/cell_pipe.py`

Add a receive-side flag:

```python
class CellPipe(Pipe):
    def __init__(...):
        ...
        self.pass_through_on_send = False
        self.pass_through_on_receive = False
```

Update `open()`:

```python
def open(self, name: str):
    with self.pipe_lock:
        if self.closed:
            raise BrokenPipeError("pipe already closed")
        self.ci.start()
        self.set_cell_cb(name)
        if self.pass_through_on_receive:
            self.cell.decode_pass_through_channels.add(self.channel)
```

Update `close()` to clean up:

```python
def close(self):
    with self.pipe_lock:
        if self.pass_through_on_receive and self.channel:
            self.cell.decode_pass_through_channels.discard(self.channel)
        ...
```

This makes `Adapter.call()` decode incoming task messages with `FOBSContextKey.PASS_THROUGH=True` before `_receive_message()` is called.

#### `nvflare/client/config.py`

Add a task exchange config key for controlled rollout:

```python
class ConfigKey:
    LAZY_FORWARD_RECEIVE = "lazy_forward_receive"
```

Add:

```python
def get_lazy_forward_receive(self) -> bool:
    return bool(
        self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(
            ConfigKey.LAZY_FORWARD_RECEIVE,
            True,
        )
    )
```

Default should be `True` for CellPipe external process jobs after the spike passes. During the spike, it can default to `False` behind a targeted test flag if we want lower rollout risk.

#### `nvflare/client/ex_process/api.py`

When creating the subprocess task pipe:

```python
if isinstance(pipe, CellPipe):
    pipe.pass_through_on_send = True       # existing reverse path
    pipe.pass_through_on_receive = client_config.get_lazy_forward_receive()
```

Only set this on the task pipe, not the metric pipe.

#### `nvflare/client/flare_agent.py`

Add lazy-ref detection and resolution helpers. Reuse the swarm pattern, but put it in the client agent because this is now part of the normal Client API task receive path.

```python
def _has_lazy_refs(self, obj) -> bool:
    from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef

    if isinstance(obj, LazyDownloadRef):
        return True
    if isinstance(obj, dict):
        return any(self._has_lazy_refs(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(self._has_lazy_refs(v) for v in obj)
    return False


def _resolve_lazy_refs(self, shareable: Shareable) -> Shareable:
    import nvflare.fuel.utils.fobs as fobs

    if not isinstance(self.pipe, CellPipe):
        return shareable
    cell = self.pipe.cell
    encoded = fobs.dumps(shareable)
    decode_ctx = cell.get_fobs_context(props={fobs.FOBSContextKey.PASS_THROUGH: False})
    return fobs.loads(encoded, fobs_ctx=decode_ctx)
```

Call this from `get_task()` after dequeuing the message and before `shareable_to_task_data()`:

```python
shareable = req.data
task_id = shareable.get_header(FLContextKey.TASK_ID)
task_name = shareable.get_header(FLContextKey.TASK_NAME)
tc = _TaskContext(task_id=task_id, task_name=task_name, msg_id=req.msg_id)
self.current_task = tc

try:
    if self._has_lazy_refs(shareable):
        shareable = self._resolve_lazy_refs(shareable)
    task_data = self.shareable_to_task_data(shareable)
except Exception:
    self._submit_task_receive_failure(tc, ...)
    self.current_task = None
    raise

return Task(task_name=tc.task_name, task_id=tc.task_id, data=task_data)
```

Important ordering change: set `current_task` before materialization. If materialization fails after ACK, the subprocess must be able to submit a failure result back to CJ instead of leaving CJ waiting for `task_wait_time`.

Add a failure helper:

```python
def _submit_task_receive_failure(self, current_task: _TaskContext, reason: str) -> None:
    result = Shareable()
    result.set_return_code(RC.EXECUTION_EXCEPTION)
    result.set_header(ReservedHeaderKey.ERROR, reason)
    self._do_submit_result(current_task, None, RC.EXECUTION_EXCEPTION)
```

The exact return code can be refined:

- malformed task payload: `BAD_TASK_DATA`
- source download/materialization failure: `EXECUTION_EXCEPTION`
- abort while resolving: `TASK_ABORTED`

The key requirement is that the failure is actively reported after the ACK path has succeeded.

### Materialization Semantics

User code should not see `LazyDownloadRef`.

The `get_task()` path should return the same app-friendly object types as today:

- default `FlareAgent`: `DXO`
- `FlareAgentWithFLModel`: `FLModel`
- framework-specific parameter containers after converter logic

Materialization moves from `Adapter.call()` to `FlareAgent.get_task()`, but it must still occur before the task is returned to the application.

### Abort And Stop Behavior

Today, eager decode in `Adapter.call()` does not receive a task abort signal either, so the first spike does not need to solve all cancellation semantics. Still, the final implementation should make lazy materialization interruptible:

- Check `self.asked_to_stop` before resolving.
- Check pipe handler status after resolution.
- If possible, pass an abort signal into FOBS context in a follow-up so `download_object()` can stop quickly.

### Expected Forward Flow

```text
CJ TaskExchanger.send_to_peer()
  CellPipe.send()
    cell.send_request(timeout=peer_read_timeout)

subprocess Adapter.call()
  decode_payload(PASS_THROUGH=True)
  ViaDownloader.process_datum() stores _LazyBatchInfo
  ViaDownloader.recompose() creates LazyDownloadRef
  CellPipe._receive_message() queues message
  returns ACK

CJ peer_read_timeout is satisfied quickly
CJ waits for task result

subprocess app calls get_task()
  FlareAgent detects LazyDownloadRef
  FOBS round-trip with PASS_THROUGH=False
  ViaDownloader downloads tensors through data plane
  returns materialized DXO/FLModel to app
```

## Design E: Data-Plane Idle Timeout Authority

### Core Idea

For streamed object transactions, stop using a fixed wall-clock `download_complete_timeout` as the subprocess lifetime budget.

Instead:

```text
if result has no streamed tensors:
    proceed immediately
if result creates a DownloadService transaction:
    wait until the transaction_done callback fires
    the callback fires on FINISHED, TIMEOUT, or DELETED
    TIMEOUT is DownloadService inactivity timeout, not total transfer duration
if no transaction metadata/callback is available:
    fall back to legacy download_complete_timeout
```

This makes the data plane the single authority for streamed result transfer completion and stall detection.

### Current Reverse Behavior

`FlareAgent._do_submit_result()` currently:

```text
registers DOWNLOAD_COMPLETE_CB
sends result refs to CJ
if download transaction was created:
    wait up to download_complete_timeout for callback
    if callback does not fire, log warning and proceed
```

That can still let the subprocess exit while the server is actively downloading if the fixed wall-clock budget is too short.

### Proposed Reverse Behavior

Change to:

```text
register DOWNLOAD_COMPLETE_CB
register transaction-created callback or thread-local transaction info
send result refs to CJ
if no transaction was created:
    proceed immediately
if transaction was created:
    wait until DownloadService says FINISHED, TIMEOUT, or DELETED
    return success only for FINISHED
    return failure for TIMEOUT or DELETED
```

The wait loop is bounded by liveness/abort, not by transfer duration:

```python
while not download_done.is_set():
    if self.asked_to_stop:
        delete/cancel transaction if known
        return False
    if self.pipe_handler and self.pipe_handler.asked_to_stop:
        delete/cancel transaction if known
        return False
    if getattr(self.pipe, "closed", False):
        delete/cancel transaction if known
        return False
    download_done.wait(timeout=0.5)
```

### Transaction Idle Timeout

`DownloadService` already treats transaction timeout as inactivity:

```text
if now - tx.last_active_time > tx.timeout:
    transaction_done(TIMEOUT)
```

The design should make `tx.timeout` the one large-transfer idle policy.

Recommended config shape:

```json
{
  "streaming_idle_timeout": 600
}
```

Implementation detail:

- Use `streaming_idle_timeout` as the default for `*_min_download_timeout`.
- For reverse PASS_THROUGH, stamp the result message so `_create_downloader()` uses `streaming_idle_timeout` as the transaction idle timeout.
- Keep `download_complete_timeout` only for legacy fallback paths where no `DownloadService` transaction metadata/callback exists.

Short term, this can reuse the existing `MSG_ROOT_TTL` plumbing by setting `reply._dl_ttl = streaming_idle_timeout`. Long term, rename or add an explicit FOBS context key such as `DOWNLOAD_TX_IDLE_TIMEOUT` because `TTL` sounds like wall-clock lifetime even though `DownloadService` uses it as inactivity timeout.

### Capturing Transaction IDs

The existing `was_download_initiated()` bool is not enough for robust E behavior because the wait loop should be able to cancel/delete transactions on abort.

Add transaction info capture in `via_downloader.py`:

```python
@dataclass(frozen=True)
class DownloadTransactionInfo:
    tx_id: str
    ref_ids: tuple[str, ...]
    created_time: float


def get_download_transactions() -> tuple[DownloadTransactionInfo, ...]:
    ...


def clear_download_transactions() -> None:
    ...
```

In `_finalize_download_tx()`:

```python
if downloadable_objs:
    downloader = self._create_downloader(fobs_ctx)
    info = DownloadTransactionInfo(
        tx_id=downloader.tx_id,
        ref_ids=tuple(ref_id for ref_id, _ in downloadable_objs),
        created_time=time.time(),
    )
    _append_download_transaction(info)
    for ref_id, obj in downloadable_objs:
        downloader.add_object(obj, ref_id=ref_id)
    _tls.download_initiated = True
```

If `ObjectDownloader` does not expose `tx_id`, add it there. It already creates a `DownloadService` transaction internally.

### FlareAgent Reverse Wait

Update `_do_submit_result()`:

```python
clear_download_initiated()
clear_download_transactions()
sent = self.pipe_handler.send_to_peer(reply, self.submit_result_timeout)
transactions = get_download_transactions()

if transactions:
    result_ok = self._wait_for_download_transactions(download_done, download_status, transactions)
elif was_download_initiated():
    result_ok = self._wait_for_download_complete_fixed(...)
else:
    result_ok = True
```

`_wait_for_download_transactions()`:

```python
def _wait_for_download_transactions(self, download_done, download_status, transactions):
    while not download_done.is_set():
        reason = self._get_download_wait_abandon_reason()
        if reason:
            self._delete_download_transactions(transactions)
            self.logger.warning(...)
            return False
        download_done.wait(timeout=0.5)

    return download_status[0] == TransactionDoneStatus.FINISHED
```

Delete/cancel helper:

```python
def _delete_download_transactions(self, transactions):
    for tx in transactions:
        DownloadService.delete_transaction(tx.tx_id)
```

Avoid calling delete after the callback has already reported terminal status.

### Expected Reverse Flow

```text
subprocess _do_submit_result()
  sets DOWNLOAD_COMPLETE_CB
  serializes result
  ViaDownloader creates DownloadService transaction with idle timeout
  CJ ACKs lightweight refs
  subprocess waits for transaction_done

server/aggr pulls refs
  DownloadService mark_active on each request
  transaction stays alive while chunks are requested
  FINISHED fires when expected receivers complete
  TIMEOUT fires when no request arrives for streaming_idle_timeout

subprocess exits only after FINISHED/TIMEOUT/DELETED or abort
```

## Combined Behavior

With A+E:

```text
forward peer_read_timeout:
    only protects small CJ -> subprocess control-message ACK

forward large task payload:
    materialized after ACK in get_task()
    governed by download_object per-request timeout/retry and source DownloadService inactivity timeout

reverse submit_result_timeout:
    only protects small subprocess -> CJ control-message ACK

reverse large result payload:
    governed by subprocess DownloadService transaction inactivity timeout
    subprocess stays alive until transaction terminal state
```

This removes the main reason operators currently tune several unrelated wall-clock values.

## Configuration Plan

Add one generic data-plane idle timeout:

```json
{
  "streaming_idle_timeout": 600
}
```

Derived defaults:

```text
np_min_download_timeout = streaming_idle_timeout
tensor_min_download_timeout = streaming_idle_timeout
reverse DownloadService transaction timeout = streaming_idle_timeout
```

Keep existing keys for compatibility:

- `peer_read_timeout`: advanced control-message ACK timeout.
- `submit_result_timeout`: advanced control-message ACK timeout.
- `download_complete_timeout`: fallback for legacy reverse paths without transaction metadata.
- `*_streaming_per_request_timeout`: per-request object download RPC timeout.
- `*_min_download_timeout`: advanced override of data-plane idle timeout.

Startup warnings should change from "set all these timeouts consistently" to "legacy timeout fallback active" only when the system cannot use A+E paths.

## Failure Semantics

### Forward Materialization Failure

If lazy ref resolution fails after the CJ already received the task ACK:

1. The subprocess logs the materialization failure.
2. The subprocess submits a task failure result to CJ.
3. `current_task` is cleared.
4. `get_task()` raises to the local application or returns `None` depending on existing Client API expectations.

Prefer raising during the spike because current eager decode failures are already exceptional. The important new behavior is that CJ gets a terminal task result instead of waiting until `task_wait_time`.

### Reverse Transaction Timeout

If `DownloadService` reports `TIMEOUT`, treat result submission as failed:

```text
transaction status FINISHED -> success
transaction status TIMEOUT -> failure
transaction status DELETED -> failure unless explicitly caused by normal shutdown after success
```

Do not log "server may still be downloading" and exit. With E, timeout means the data plane observed no activity for the configured idle timeout.

### Dead Peer Or Abort

Heartbeat/liveness still guards peer death. If the subprocess is asked to stop while waiting for reverse transaction completion, delete known transactions and return failure.

## Spike Implementation Plan

### Step 1: Forward PASS_THROUGH Receive Flag

- Add `CellPipe.pass_through_on_receive`.
- Enable it on the subprocess task pipe in `client/ex_process/api.py`.
- Clean up `decode_pass_through_channels` on close.

### Step 2: Resolve Lazy Refs In `get_task()`

- Add lazy-ref detection/resolution helpers to `FlareAgent`.
- Resolve lazy refs before `shareable_to_task_data()`.
- Set `current_task` before materialization so failures can be reported.
- Add failure reporting for materialization exceptions.

### Step 3: Reverse Wait Uses Transaction Terminal State

- Add download transaction info capture in `via_downloader.py`.
- Add transaction-aware wait loop in `FlareAgent._do_submit_result()`.
- Keep fixed `download_complete_timeout` fallback only when transaction metadata is missing.

### Step 4: Use One Idle Timeout

- Add or reuse `streaming_idle_timeout`.
- Feed it into reverse `DownloadService` transaction timeout.
- Use it as the default min download timeout for NumPy/Tensor decomposers.

### Step 5: Remove Or Downgrade Timeout Zoo Warnings

- Keep warnings only for explicit advanced overrides that disable A+E behavior.
- Stop telling normal users to set several timeouts manually.

## Test Plan

### Unit Tests For A

1. **Forward receive ACK does not materialize**
   - Configure subprocess `CellPipe.pass_through_on_receive=True`.
   - Send a task containing a downloadable NumPy/Tensor ref.
   - Assert `Adapter.call()` / `CellPipe.send()` ACKs without calling `_download_from_remote_cell()`.
   - Assert queued task contains `LazyDownloadRef`.

2. **`get_task()` materializes before returning**
   - Queue a task containing `LazyDownloadRef`.
   - Mock or simulate `download_object()` success.
   - Assert returned `Task.data` contains real arrays/tensors, not `LazyDownloadRef`.

3. **Materialization failure reports task failure**
   - Make lazy ref resolution raise.
   - Assert subprocess sends a failure reply to CJ.
   - Assert `current_task` is cleared.

4. **No lazy refs follows old path**
   - Task with small payload returns normally.
   - No FOBS round-trip resolver is invoked.

5. **Metric pipe is unaffected**
   - `pass_through_on_receive` is not enabled on metric pipe.

### Unit Tests For E

1. **Reverse result waits past old fixed timeout**
   - Set legacy `download_complete_timeout` very small.
   - Create a DownloadService transaction.
   - Delay completion callback beyond the old fixed value.
   - Assert subprocess waits until callback and succeeds on `FINISHED`.

2. **No tensors skip wait**
   - Submit metrics-only result.
   - Assert no transaction wait occurs.

3. **Transaction TIMEOUT fails**
   - Trigger callback with `TransactionDoneStatus.TIMEOUT`.
   - Assert `_do_submit_result()` returns false.

4. **Abort deletes transactions**
   - Capture tx id.
   - Set `asked_to_stop=True` while waiting.
   - Assert `DownloadService.delete_transaction(tx_id)` is called.

5. **Fallback path preserved**
   - Simulate `was_download_initiated=True` with no transaction info.
   - Assert old fixed wait fallback is used and logs a compatibility warning.

### Integration/Simulation Tests

1. **5GB x N delayed forward ACK equivalent**
   - Simulate many clients with slow task payload materialization.
   - Assert CJ does not resend because ACK arrives before materialization.

2. **Forward data-plane idle failure**
   - Source `DownloadService` receives no activity or stalls.
   - Assert materialization fails and CJ receives task failure result.

3. **Reverse large result slow active download**
   - Server/aggregator pulls chunks slowly but continuously.
   - Assert subprocess stays alive beyond old `download_complete_timeout`.
   - Assert completion succeeds.

4. **Reverse no-start timeout**
   - Result transaction is created but receiver never pulls.
   - Assert `DownloadService` inactivity timeout fires and subprocess reports failure.

## Risks And Mitigations

### Lazy Refs Leak To User Code

Risk: user code receives `LazyDownloadRef` and fails in unexpected ways.

Mitigation: resolve in `FlareAgent.get_task()` before `shareable_to_task_data()` returns. Add tests that recursively check no lazy refs remain.

### Failure After ACK Needs Explicit Result

Risk: once ACK is decoupled, materialization failures happen after CJ believes task delivery succeeded.

Mitigation: set `current_task` before materialization and submit an explicit failure result on lazy resolution errors.

### Heartbeat Starvation

Risk: heartbeats share cell resources with large transfer activity.

Mitigation: not part of the first spike. Existing heartbeat still guards peer death. If tests show starvation, address with heartbeat priority or separate channel later.

### FOBS Round-Trip Cost

Risk: resolving lazy refs via `fobs.dumps()` / `fobs.loads()` adds an extra serialization pass over the task envelope.

Mitigation: the heavy tensors are refs, not inline bytes, so this should be cheap relative to materialization. Optimize with a direct recursive resolver only if profiling shows it matters.

### Transaction Metadata Race

Risk: reverse wait may miss transaction creation if metadata capture is not thread-local.

Mitigation: follow the existing thread-local pattern used by `was_download_initiated()`.

## Relationship To Transfer Leases

A+E should be implemented before a full transfer lease subsystem.

If A+E works, leases may be reduced to a thin formalization of `DownloadService` transaction state rather than a separate cross-pipe progress protocol. The forward path no longer needs CJ to watch payload progress because CJ no longer waits for payload materialization during task send. The reverse path can use local `DownloadService` terminal state directly.

A lease-like abstraction may still be useful later for:

- uniform observability;
- cross-process transaction queries;
- heartbeat/liveness integration;
- raw byte stream support outside FOBS object downloads.

But it should not be required to fix the timeout tuning problem.

## Open Questions

- Should `lazy_forward_receive` default to true immediately for all CellPipe Client API jobs, or start as an opt-in spike flag?
- What exact return code should represent forward materialization failure after ACK: `BAD_TASK_DATA`, `EXECUTION_EXCEPTION`, or a more specific new code?
- Should `get_task()` raise after auto-submitting failure, or return `None` to application code?
- Should `MSG_ROOT_TTL` be renamed/replaced for DownloadService inactivity timeout to avoid misleading TTL semantics?
- Can `download_object()` receive an abort signal during `get_task()` materialization without larger Client API changes?
- Does controller-level `task.timeout` need clarification as a true job deadline rather than a transfer timeout?
