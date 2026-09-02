# Tensor Disk Offload

## Objective

Reduce aggregation peak memory for large PyTorch model updates by materializing
incoming streamed tensor payloads on disk and resolving them lazily during
aggregation.

## Scope

- Applies to streamed **PyTorch tensor** payloads handled by `TensorDecomposer`.
- Controlled by `enable_tensor_disk_offload` in a supported receiving workflow/controller config:
  FedAvg (including FedProx), FedOpt, SCAFFOLD, or Swarm.
- Default is `False` (legacy in-memory behavior).
- If model updates are converted to NumPy before transport, tensor disk offload is not engaged.

Tensor streaming, ref pass-through, and tensor disk offload are separate
behaviors. The sender's `TensorDownloadable` remains memory-backed while it
serves a transport ref. An intermediate process may preserve that ref without
downloading it. Disk offload applies when the receiving aggregation workflow
terminates the ref and downloads its tensor chunks. It does not spool the
trainer's model or training result to disk at the source, and it does not reduce
the memory required to load or train the model in the trainer.

## How To Enable

FedAvg:

- `nvflare/recipe/fedavg.py` -> `FedAvgRecipe(..., enable_tensor_disk_offload=True)`
- `nvflare/app_opt/pt/recipes/fedavg.py` -> PT recipe forwards the same flag
- `nvflare/app_common/workflows/fedavg.py` -> `FedAvg(..., enable_tensor_disk_offload=True)`
- PyTorch FedProx uses the same FedAvg recipe and server aggregation path

Other server-controlled workflows:

- `nvflare/app_opt/pt/recipes/fedopt.py` -> `FedOptRecipe(..., enable_tensor_disk_offload=True)`
- `nvflare/app_opt/pt/recipes/scaffold.py` -> `ScaffoldRecipe(..., enable_tensor_disk_offload=True)`
- both require `server_expected_format=ExchangeFormat.PYTORCH`; custom configurations can enable
  the same setting directly on `ScatterAndGather` or `Scaffold`

Swarm/CCWF:

- `nvflare/app_opt/pt/recipes/swarm.py` -> use
  `SwarmLearningRecipe(aggregation_format=ExchangeFormat.PYTORCH, enable_tensor_disk_offload=True)`
- `nvflare/app_common/ccwf/ccwf_job.py` -> use
  `SwarmClientConfig(..., enable_tensor_disk_offload=True)` for custom Job API configurations
- `nvflare/app_common/ccwf/swarm_client_ctl.py` owns an offload root on each
  eligible client because the aggregation role moves between clients, but
  enables disk materialization only for terminal aggregation downloads

If no active Cell is available, the offload context is not enabled and the runtime falls back to in-memory download.

## Data Flow

```
TensorDownloadable chunks
        |
        v
TensorDecomposer.download()
  - enable_tensor_disk_offload=False -> deserialize in memory
  - enable_tensor_disk_offload=True  -> write safetensors temp files
        |
        v
LazyTensorDict
        |
        v
ViaDownloaderDecomposer.recompose()
        |
        v
Lazy refs in payload tree
        |
        +--> aggregator consumes lazy refs (materialize on demand)
```

Server-controlled FedAvg/FedProx, FedOpt, and SCAFFOLD workflows install their
disk-offload setting on the receiving server Cell. Swarm does not enable disk
offload globally on its client Cells because the same Cells also receive
learner tasks and final-result broadcasts. Instead, the selected aggregation
controller puts the setting and its job-scoped root in the terminal result
download's FOBS decode context. The destination therefore does not depend on
mutable Cell-wide state, and non-aggregation deliveries remain ordinary
in-memory tensors.

## Runtime Behavior

### FedAvg

In `nvflare/app_common/workflows/fedavg.py`:

- custom aggregators receive `result.params` as-is
- with `enable_tensor_disk_offload=True`, lazy refs are passed through directly
- built-in weighted aggregation uses `LazyWeightedAggregationHelper`: contributions are only
  recorded, and the aggregate is computed one tensor at a time and written to disk (see
  "Lazy Aggregation and Safetensors Persistence" below)
- without an active Cell the in-memory `WeightedAggregationHelper` materializes per tensor inside
  `add()` and relies on lazy-ref object lifetime / GC for temp-resource cleanup

### Filter Compatibility

Tensor disk offload produces per-tensor lazy values with a `materialize()` method. These values are
different from the transport-level `LazyDownloadRef` objects resolved at the Client API site-filter
boundary. Conventional content filters, including model quantizers and dequantizers, expect concrete
NumPy arrays or PyTorch tensors and are not compatible with tensor disk-offload lazy values.

Do not combine `enable_tensor_disk_offload=True` with conventional content filters on the same
receiving path. Disable tensor disk offload for that path, move the transformation to an explicit
send/receive endpoint, or use a consumer such as the built-in weighted aggregator that explicitly
materializes each tensor. Supporting filters while retaining bounded-memory disk offload requires a
separate lazy-ref-aware or streaming/per-tensor filter contract.

### Swarm/CCWF

`ClientAPIExecutor` preserves the Cell/FOBS large-payload references between an
external trainer and the CCWF controller. When a training result is sent to a
remote aggregation client, `SwarmClientController` requests PASS_THROUGH on that
message. The aggregation controller therefore receives refs instead of
materializing the tensors inside the Cell receive callback, and explicitly
resolves them with its job-scoped disk root. A result returned by a local
external trainer also crosses a Cell boundary as a ref and is resolved by the
local aggregation controller with the same disk root. In both cases tensor disk
offload yields lazy tensor refs, and the built-in
`InTimeAccumulateWeightedAggregator` materializes one tensor at a time.

The Swarm controller owns the decision to preserve or resolve transport refs.
On a non-aggregation client it keeps refs when the local learn executor is
`ClientAPIExecutor(execution_mode="external_process")`, making the external
trainer their single consumer. If that external-process site is also the selected
aggregator, the controller resolves the task once for its aggregation base model
and passes the same in-memory payload to the trainer. This avoids two consumers
racing a one-receiver download transaction. The controller also resolves tasks
into memory for `in_process`, `attach`, and non-`ClientAPIExecutor` learners.
This conservative fallback supports jobs where sites use different learner
execution modes. Disk-backed aggregation refs remain local to the aggregation
client and are never passed to a learner.

The external trainer process's Cell is not configured as a disk-offload
receiver. It is the terminal consumer of an incoming learn task, and the Client
API materializes ordinary in-memory tensors for model loading. For an outgoing
training result, `TensorDownloadable` still holds the tensors in trainer memory
while serving a transport ref. `ClientAPIExecutor` and the client job preserve
and route that ref; the selected aggregation controller decides whether its
terminal download is materialized on disk. Source-side tensor spooling is not
part of this feature.

If an in-process learner is also the local aggregation client, its own result is
already an in-memory object with no transport ref to preserve. That one local
contribution remains in memory; remote contributions still use terminal
aggregation-CJ disk offload.

`SwarmLearningRecipe` defaults to NumPy exchange for compatibility. Disk offload
therefore requires `aggregation_format=ExchangeFormat.PYTORCH`; streamed
NumPy arrays are not handled by `TensorDecomposer`.

## Lazy Aggregation and Safetensors Persistence

Inbound offload alone still leaves three model-sized copies on the server: the accumulator built
by `WeightedAggregationHelper`, the global model held in memory, and the deep copy of the task data
taken at broadcast time. With `enable_tensor_disk_offload=True`, FedAvg's built-in aggregation
therefore works on lazy refs end to end (`nvflare/app_opt/pt/lazy_aggregation.py`):

- `LazyWeightedAggregationHelper.add()` records each contribution's refs, weight and key
  statistics; no tensor is loaded during result callbacks.
- `get_result()` walks the model keys in sorted order. For each key it materializes one client
  tensor at a time, accumulates into a single accumulator (float32 for bf16/fp16 inputs, cast back
  on write) and streams the tensor into a new safetensors file below the offload root
  (`<root>/nvflare_aggregate_*/model.safetensors`). The header is derived from the contributions'
  metadata before any tensor is produced.
- The result is a dict of `_LazyRef` into that file, owned by a `_TempDirRef` exactly like inbound
  chunk files. The file is removed when the last ref is released, normally when the next round
  replaces the global model. Aggregate files are never rewritten, so refs held by other components
  stay valid.
- `ParamsType.DIFF` contributions are applied to the base model by the helper and keys without
  contributions are copied from it, one tensor at a time. FedAvg treats the result as a full model.
- Custom aggregators are unaffected and still receive refs.

Once an aggregate exists the global model is a dict of refs. Outbound task data is serialized by
`TensorDecomposer` through the FOBS type alias `_LazyRef -> torch.Tensor`
(`fobs.register_type_alias`), so `TensorDownloadable` materializes one tensor per item while
producing chunks, keeps normal chunk batching, and disables its cross-receiver chunk cache for
ref-backed payloads. A deep copy of a ref is a ref. The wire format and the clients are unchanged.

### PTSafetensorsModelPersistor

`PTFileModelPersistor` keeps a state dict in memory and loads the initial checkpoint with
`torch.load`. To drop that copy, and to start from a checkpoint that does not fit in server memory,
use `PTSafetensorsModelPersistor` (`nvflare/app_opt/pt/safetensors_model_persistor.py`):

- `load_model()` returns refs into a `.safetensors` file, a Hugging Face
  `model.safetensors.index.json`, or a directory containing either. Nothing is loaded.
- `save_model()` writes `FL_global_model.safetensors` in the app log dir. When the weights are refs
  into one complete file, which is what lazy aggregation produces, the file is hard-linked (copied
  when the offload root is on another file system); otherwise tensors are streamed one at a time.
  Writes go to a `.tmp` file and are published with `os.replace`.
- `GLOBAL_BEST_MODEL_AVAILABLE` saves the current global model to `best_FL_global_model.safetensors`
  the same way.
- Only tensors are persisted, not `FLModel.meta`. The model inventory used by cross-site evaluation
  is not implemented. As with `PTFileModelPersistor`, the source checkpoint is loaded on every start.

`PTFileModelPersistor` accepts lazy aggregation output as well: it materializes refs while updating
its in-memory state dict, so existing `enable_tensor_disk_offload` jobs keep working with one
model-sized copy at save time. `FLModelUtils.update_model` likewise materializes a lazy base value
when applying an in-memory DIFF.

### Configuration

```python
from nvflare.app_opt.pt import PTSafetensorsModelPersistor
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat

recipe = FedAvgRecipe(
    name="qwen_fedavg",
    min_clients=2,
    num_rounds=3,
    train_script="client.py",
    model_persistor=PTSafetensorsModelPersistor("/models/Qwen2.5-72B-Instruct"),
    server_expected_format=ExchangeFormat.PYTORCH,
    enable_tensor_disk_offload=True,
)
```

The persistor requires `ExchangeFormat.PYTORCH`; the PyTorch recipe rejects other formats.

### Memory and Disk

```text
aggregation  O(largest tensor)               one input tensor + one accumulator
outbound     O(receivers * largest tensor)   one serialized item per receiver, plus bounded prefetch
inbound      O(clients * response)           unchanged
persistence  O(largest tensor)               streamed, or a zero-copy hard link
```

Disk below the offload root holds the current round's client contributions, the previous aggregate
until the new one replaces it, and the new aggregate. The offload root follows `TMPDIR`; on a tmpfs
`/tmp` the aggregate would live in RAM, so point `TMPDIR` at a disk, ideally on the same file system
as the workspace so saves are links rather than copies. Unit tests establish the code path; the 72B
memory target still needs a production measurement.

## Custom Aggregator Contract

When a custom aggregator is used, payload params may contain lazy refs (duck-typed object with `materialize()`).

Custom aggregators are responsible for:

1. materializing refs when tensor math is required
2. releasing lazy-ref object references after use so temp resources can be reclaimed

## Temp File Lifecycle

- Each workflow creates a job-scoped offload root (`nvflare_tensor_offload_<job>_*`),
  with safetensors download directories beneath it.
- Temp dir selection follows Python `tempfile` behavior (`TMPDIR` / OS default, typically `/tmp`).
- In containerized deployments, `/tmp` may be tmpfs (RAM-backed); set `TMPDIR` to a disk-backed mount to realize memory offload benefits.
- `LazyTensorDict` owns a shared `_TempDirRef`; each lazy ref keeps this reference alive.
- Lazy download directories are reclaimed when their refs are released, with GC as
  a fallback.
- FedAvg-style workflows restore the prior FOBS context and remove their
  job-scoped root when the workflow exits.
- Swarm keeps its job-scoped root and tensor-forwarding route through workflow
  finalization. Explicit terminal downloads inherit the run abort signal. At job
  `END_RUN`, Swarm cancels active disk consumers under its root, gives the
  controller-owned learning and aggregation threads a bounded drain window, then
  removes the root and its remaining contents. A receiver that abandons an active
  download also sends a capability-negotiated cancellation to the producer. The
  producer records that receiver as failed across every ref in the transaction and
  can release an accepted external-process result source without waiting for its
  normal transfer timeout. Older producers keep the existing bounded timeout path.
- These workflow-local cleanup paths require the server/client job process to reach
  normal teardown. Reclaiming scratch after an abrupt server/client job process
  death requires launcher-owned storage whose cleanup capability survives process
  and container boundaries; that lifecycle is outside this workflow-level feature.

## Failure Behavior

- Download failures trigger `DiskTensorConsumer.download_failed(...)`, which removes the temp dir.
- If an owned external trainer dies after its lazy result envelope is accepted, its CJ sends a
  bounded acknowledged failure notice for the exact source FQCN/reference IDs to the declared
  terminal receiver. Matching active downloads are interrupted immediately; a bounded tombstone
  also covers failure notices that arrive just before download startup. FedAvg and Swarm then use
  their existing materialization/task-error paths instead of retrying the dead source for the
  normal 600-second streaming timeout.
- Mid-transfer receiver cancellation is best effort. If the producer advertised support, it
  promptly settles the abandoned receiver as failed; otherwise the transaction's existing
  receiver/transaction timeout remains the cleanup backstop.
- Invalid safetensors payload/header parsing fails fast and bubbles up as a download-consume error.
- Existing in-memory download path remains unchanged when offload is disabled.

## Design-Relevant Files

- `nvflare/app_opt/pt/decomposers.py`
- `nvflare/app_opt/pt/lazy_tensor_dict.py`
- `nvflare/app_opt/pt/tensor_downloader.py`
- `nvflare/app_opt/pt/lazy_aggregation.py`
- `nvflare/app_opt/pt/safetensors_model_persistor.py`
- `nvflare/fuel/utils/fobs/decomposers/via_downloader.py`
- `nvflare/app_common/workflows/fedavg.py`
- `nvflare/app_common/ccwf/swarm_client_ctl.py`
- `nvflare/app_common/ccwf/ccwf_job.py`
- `nvflare/recipe/fedavg.py`
- `nvflare/app_opt/pt/recipes/swarm.py`

## Test Coverage

- `tests/unit_test/app_common/workflow/fedavg_test.py`
- `tests/unit_test/app_common/ccwf/test_swarm_tensor_disk_offload.py`
- `tests/unit_test/recipe/swarm_recipe_test.py`
- `tests/unit_test/app_opt/pt/test_lazy_tensor_dict.py`
- `tests/unit_test/app_opt/pt/test_disk_tensor_consumer.py`
- `tests/unit_test/app_opt/pt/lazy_aggregation_test.py`
- `tests/unit_test/app_opt/pt/lazy_fedavg_test.py`
- `tests/unit_test/app_opt/pt/safetensors_model_persistor_test.py`
- `tests/unit_test/app_common/aggregators/weighted_aggregation_helper_test.py`
- `tests/unit_test/private/fed/server/server_runner_test.py`
- `tests/stress_test/fedavg_large_model/fedavg_stress_test.py`
