# Lifecycle formalization evaluation

## Verdict

The spike is useful, but the evidence does not yet justify describing all 717
new runtime lines as formally verified.

The strongest result is narrow and real: the two pure Python transition
functions have exactly the same bounded state graphs as their TLA+ models, and
TLC rejects every deliberately unsafe model configuration. This makes ordering,
terminal-result precedence, retry liveness, and client process ownership much
harder to change accidentally.

The main limitation is equally important: only 276 of the 717 physical runtime
lines are in those exact graph comparisons. The 388 driver lines and 53 lines
of return-code classification and merge policy are conventional Python. The
cross-client/server TLA+ model is an architectural model with focused boundary
tests, not an exact refinement check.

The historical replay below found that many lifecycle defects entered through
those unverified boundaries: a launcher failed to produce an event, an adapter
classified an event incorrectly, an authenticated report was rejected, or a
caller supplied the wrong participant set. TLC cannot detect a fact that is not
represented in the model.

## Frozen subject

This evaluation freezes the implementation before selecting the retrospective
cases. The repository base is `59aa4d3bb1cec3d1227589c53ca6150b38deb1de`;
the evaluated lifecycle files were uncommitted work on top of that base.

| File | Physical lines | SHA-256 |
|---|---:|---|
| `nvflare/private/fed/server/job_completion.py` | 217 | `1fa6302d5daab1bb8db662fa713ebf635159d65664583050748ffbf150460144` |
| `nvflare/private/fed/server/job_completion_driver.py` | 227 | `2d64667cd6cbab234796676c44ad1fc23e3b846b15e62c8decbeb1338c2f90cc` |
| `nvflare/private/fed/client/client_process.py` | 112 | `a921f1f0f3ef8ce7b20fb3dd56c2574fa8b01f93184febe3e62a299bc77844ec` |
| `nvflare/private/fed/client/client_process_driver.py` | 161 | `44019d7bf5b097f853e6f24fb15b430018a3cbbed9246377159073b9973fc073` |
| **Runtime total** | **717** | |

The exact graph comparison covers the server machine through
`merge_terminal_status` (164 physical lines) and all 112 lines of the client
machine: 276/717, or 38.5%. This physical-line ratio is only a scope indicator;
it is not a code-quality metric.

The frozen TLA+ model hashes are:

- `JobCompletion.tla`: `d86e0fbec4257273b495bc725a5201fdc1872306b4d4570e647539bae873d9b2`
- `ClientProcess.tla`: `c8f21ec2d5b7baad25340f2bb2d0a246ba935fa35286af4445a224a2fe4576c3`
- `JobOutcomeProtocol.tla`: `799528598090352ceeb2526eb7011a11e164967e741c3fe3bf8736d42fc13d33`

## Scoring rules

A replay is scored at the first layer that would automatically reject the
historical bug in its natural form:

1. **TLC**: an invariant or temporal property fails.
2. **Exact graph**: production `transition()` no longer matches the safe TLA+
   graph.
3. **Architecture guard**: the recoding makes the old path unavailable or a
   static boundary test rejects it.
4. **Python contract test**: a driver, adapter, or integration test rejects it.
5. **Miss**: the current package accepts it or has no representation of it.

A test added by the historical bug fix proves that the present repository does
not regress that exact case. It does not count as evidence that this formal
redesign would have found the original defect.

## Training-set mutation result

All 15 deliberately unsafe configurations are killed:

| Model | Safe graph/result | Mutations killed | Families exercised |
|---|---|---:|---|
| Server completion | 41 states, 160 transitions | 4/4 | publish-before-archive, failure precedence, re-archive, infinite retry |
| Client process | 24 states, 79 transitions | 5/5 | lost pre-attach stop, stop precedence, early removal, dirty completion, infinite retry |
| Client/server outcome protocol | safety and liveness pass | 6/6 | identity, premature publication/release, failure masking, duplicate settlement, lost abort |

This is a necessary sensitivity test: each stated rule has a represented bad
behavior and TLC finds it. It is not independent evidence of predictive value,
because these mutations were chosen while designing the models.

## Post-freeze historical replay

These cases were selected after freezing the hashes above. They are a
retrospective challenge set, not a statistically random sample and not a claim
that every historical lifecycle bug was reviewed.

| Historical defect | Natural failure shape | First current rejecting layer | What the replay says |
|---|---|---|---|
| [PR #4633](https://github.com/NVIDIA/NVFlare/pull/4633), aborted-status publication race | `fed_server.py` wrote `FINISHED_ABORTED` directly while completion could write another status | Architecture guard | TLC would not see the direct `set_status`. The recoding routes live completion publication through `JobCompletionDriver`, and the AST boundary test rejects a reintroduced direct write. This is a redesign win, not a TLC-only win. |
| [PR #1627](https://github.com/NVIDIA/NVFlare/pull/1627), abort status updated before process completion | abort handling persisted terminal state and removed ownership before the completion path settled | Architecture guard | The new vocabulary has no early publication transition and the status-write boundary rejects the old side door. Again, the protection depends on funneling all writes through the machine. |
| [PR #1615](https://github.com/NVIDIA/NVFlare/pull/1615), config error produced the wrong terminal status | child return code was not propagated and classified correctly | Python contract test | TLA accepts any valid terminal status supplied by `ServerExited`; it cannot know that the adapter supplied the wrong one. The new single disposition table and existing mapping tests help, but they are ordinary Python assurance. |
| [PR #4552](https://github.com/NVIDIA/NVFlare/pull/4552), pending launcher timeout remained `RUNNING` | the launcher/engine failed to turn a resource timeout into a server-exit outcome | Python integration test | `JobCompletion` deliberately has no progress property out of `waiting_for_server`. The current K8s pipeline test catches this; the formal model does not. |
| [PR #4432](https://github.com/NVIDIA/NVFlare/pull/4432), legitimate failure reports were rejected | the report lacked `PROJECT_NAME`, so the wrong authentication API rejected an authorized token | Miss | The protocol permits a report to be lost or rejected and proves that fallback fails closed. It does not require an authorized report to be accepted. The new auth tests check rejection and identity binding, but do not recreate the missing-header acceptance contract. |
| [PR #4209](https://github.com/NVIDIA/NVFlare/pull/4209), deployment timeouts were treated as successful participants | `None` replies were omitted from `failed_clients`, then the wrong client set was started | Python integration test | `ParticipantsSelected(active)` trusts the supplied set. TLC cannot infer which deployments actually succeeded. The deploy/start tests catch the adapter error. |
| [PR #630](https://github.com/NVIDIA/NVFlare/pull/630), server start succeeded while client start failed | partial startup was not compensated by stopping the already-started server and cleaning ownership | Miss for this slice | This happens before either checked completion machine owns the run. It needs a start/compensation protocol, not another completion invariant. |

Result for this challenge set:

- TLC or exact graph alone rejects **0/7** natural historical implementations.
- The recoded architecture makes **2/7** old direct-publication paths illegal.
- A centralized but non-formal return-code table improves **1/7**.
- Ordinary integration tests remain the first detector for **2/7**.
- **2/7** remain outside this slice's automatic contract.

That does not contradict the 15/15 mutation result. The two measurements answer
different questions: the mutation result asks whether the model enforces the
rules it states; the replay asks how often real defects enter through those
rules rather than around them.

## Boundary challenge results

The three proposed challenges were implemented after the frozen replay:

1. **Authorized report acceptance:** `AcceptValidReport` and
   `DeliveredValidReportIsAccepted` now distinguish a delivered valid report
   from transport loss or a stale duplicate. The unsafe acceptance configuration
   recreates PR #4432 and TLC rejects it. The production test uses a real
   `ClientManager`, omits `PROJECT_NAME`, demonstrates that `validate_client()`
   would reject the request, and verifies that the token-only report path accepts
   it. This closes the previous replay miss at both the model and mapping
   boundary.
2. **Participant evidence:** no state was added to TLA+. The machine continues
   to trust the confirmed set supplied by `JobRunner`. Strict and non-strict
   reply-path tests now both assert that a timed-out client is absent from the
   completion obligation. These tests would fail if the historical "all
   requested clients are active" mutation returned.
3. **Launcher delivery:** no launcher state machine was added. `JobCompletion`
   remains explicitly conditional on `ServerExited`, because a TLA+ wrapper
   cannot prove that an arbitrary `JobHandleSpec.wait()` implementation returns.
   A successful-engine-handoff test complements the existing real K8s
   pending-timeout pipeline test.

The challenge added no production runtime lines. The updated cross-protocol
model passes safety and liveness, and all seven deliberate protocol mutations
are rejected. Its post-challenge SHA-256 is
`c68d80c2b46bbfc3f5df1499295e95e39b9a495e8ec953fb78d01980200ac3b7`.

## Recommendation

Use a **hybrid assurance pattern**, not a formal model of the entire FLARE
lifecycle:

- Use TLA+/TLC for small, closed concurrency and ordering cores only when the
  production transition function can be compared automatically with the full
  bounded TLA+ graph. Keep the server completion and client process machines in
  this category.
- Treat side-effect drivers as conventional concurrent Python. Keep their
  single-owner locking structure and failure-injection tests, but do not call
  them formally verified.
- Verify authentication, participant derivation, launcher return codes,
  filesystem operations, and job-store writes with concrete Python contract and
  integration tests. Modeling those adapters would mostly restate assumptions
  that only executable tests can validate.
- Keep `JobOutcomeProtocol.tla` as a design/exploration artifact for the spike,
  not as evidence that the mapped production entry points are refinement-proven.
  Do not expand it until there is an automated mapping check.

I would therefore **stop the broad formal-lifecycle rewrite here**. The pilot
has shown a useful technique for the two pure cores, but not a credible path to
proving the surrounding FLARE lifecycle. A production proposal should be
smaller than this spike and state its guarantee as: "exact checked transition
graphs plus tested adapters," never "the lifecycle is formally verified."

With these boundaries frozen, judge the next ten lifecycle defects or
substantive change requests prospectively. Record which layer finds each issue
before the fix is designed. A useful continuation threshold is:

- every change to a checked transition is killed by TLC or exact graph
  comparison;
- at least half of lifecycle defects in the declared boundary are rejected by
  the formal/refinement layer before ordinary integration tests; and
- escapes are not dominated by the same unmodeled adapter boundary.

Only prospective evidence should justify another formal slice. If the next
changes continue to fail in adapters, retain the cleaner Python ownership design
without expanding TLA+. If they fail inside checked transitions and TLC finds
them before ordinary tests, the narrow formal core is earning its maintenance
cost.

## Commands run

The three TLC checkers were run with the worktree-local `jdk4py` Java runtime
and `/private/tmp/tla2tools.jar`. A focused Python replay suite covering status
write boundaries, both drivers, report authentication, deployment/start
selection, K8s timeout propagation, and download gating passed with:

```text
45 passed, 119 deselected in 8.26s
```

The post-freeze boundary challenge validation additionally passed seven
protocol mutations, nine initial focused Python tests, and a broader 143-test
server/client lifecycle selection.
