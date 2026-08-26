# Client/server job-outcome protocol

This model checks the contract between the client process lifecycle and the
server completion lifecycle. It is intentionally not their Cartesian product:
one job, one participating client, at most two copies of one terminal report,
and an abstract server timeout are enough to exercise the cross-boundary rules.

```mermaid
sequenceDiagram
    participant C as Client process
    participant N as CellNet
    participant S as Server completion
    C->>C: ProcessExited(outcome)
    C->>N: REPORT_JOB_FAILURE
    alt authenticated report is delivered
        N->>S: RecordClientOutcome(participant, outcome)
        S->>S: settle once; failure overrides success
    else report is lost or a stale copy is rejected
        S->>S: missing-client / timeout fallback
        S->>S: settle as failure
    end
    S->>S: publish only after settlement
    Note over C: resources may be released after the report attempt
    opt server requests abort
        S-->>C: heartbeat ABORT_JOBS
        C->>C: RequestStop then ProcessExited
    end
```

## Rules checked

- A completed publication requires a completed server candidate and an
  authenticated participant `NO_OVERRIDE` report. A missing report is
  resolved by fallback as failure, never as completion.
- A participant failure dominates candidate server success.
- A first delivered report from an authenticated, tracked participant is
  accepted. This is distinct from allowing transport loss or rejecting a stale
  duplicate after the obligation was already settled.
- Duplicate delivery can settle the server obligation only once.
- The client releases resources only after attempting its terminal report. It
  does **not** claim that the server accepted the report; the fallback exists
  precisely because transport can fail.
- Under weak fairness for delivery, timeout, publication, and process exit, a
  waiting server completes, an exited client releases resources, and a pending
  abort eventually moves the client out of process ownership.
- An attacker report cannot satisfy a participant's completion obligation.

Each rule has an unsafe configuration that disables it. The acceptance mutation
recreates the historical failure shape where a legitimate report was delivered
but the server used an incompatible authentication contract and rejected it.
The checker requires TLC to find the corresponding counterexample or temporal
failure; this guards
against a model whose invariants pass only because the bad behavior was never
represented.

## Mapping to production

| Protocol step | Production boundary |
|---|---|
| `ClientExited`, `StartParticipantReport` | `JobExecutor._wait_child_process_finish()` and `_report_outcome()` |
| `FinishParticipantReport`, `ReleaseClient` | `ClientProcessDriver.outcome_settled()` and `resources_released()` |
| authenticated delivery | `FederatedServer.process_job_failure()` token lookup and registered client name |
| settle once / outcome precedence | `JobRunner.record_client_outcome()` and `JobCompletionDriver.record_client_outcome()` |
| missing-report fallback | `FederatedServer._resolve_missing_client_outcome()` and completion timeout |
| publication | `job_completion.py` plus `JobCompletionDriver.advance()` |
| abort delivery | heartbeat `ABORT_JOBS`, `Communicator._clean_up_runs()`, and `ClientProcessDriver.request_stop()` |

The two local lifecycle models are exact graph comparisons against their pure
Python transition functions. This protocol model is different: it checks the
architecture-level composition, while focused boundary tests verify that the
listed Python entry points implement the modeled messages and classifications.
It is design assurance, not an exact refinement proof of those entry points.

## Deliberate limits

This bounded spike does not prove CellNet, thread scheduling, job artifact
copying, job-store transactions, or multi-client quorum behavior. Those are
separate protocols. In particular, artifact commit and deletion should be a
future cross-machine slice rather than extra Boolean state added here.
