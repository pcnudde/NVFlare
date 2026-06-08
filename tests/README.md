# NVIDIA Flare Test


This file introduces how the tests in NVIDIA FLARE are organized.

We divide tests into unit tests and integration tests.

```commandline
tests:
  - unit_test
  - integration_test
  - agent_skill_eval
```

## Unit Tests

### Structure

The structure of unit test is organized as parallel directories of the production code.

Each directory in `test/unit_test` maps to its counterpart in `nvflare`.

For example, we have `test/unit_test/app_common/job_schedulers/job_scheduler_test.py`
that tests `nvflare/app_common/job_schedulers/job_scheduler.py`.

### Run

To run unit test: `./runtest.sh`.

### Develop a test case

We use pytest to run our unit tests.
So please follow pytest test case style.

## Integration Tests

Please refer to [integration tests README](./integration_test/README.md).

## Agent Skill Eval

`tests/agent_skill_eval` contains an optional Docker-backed harness for
evaluating coding agents against Markdown testcases. It is not part of the
normal unit or integration test suites; see
[`tests/agent_skill_eval/README.md`](./agent_skill_eval/README.md) for setup and
execution details.


