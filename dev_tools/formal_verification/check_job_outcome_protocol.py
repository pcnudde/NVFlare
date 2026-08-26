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

"""Check the bounded client/server terminal-outcome protocol."""

from pathlib import Path

from tlc import run_cli, verify_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "formal_models" / "job_outcome_protocol"
EXPECTED_MUTATIONS = {
    "JobOutcomeProtocolUnsafeIdentity.cfg": "Invariant AcceptedReportIsAuthenticated is violated.",
    "JobOutcomeProtocolUnsafeAcceptance.cfg": "Invariant DeliveredValidReportIsAccepted is violated.",
    "JobOutcomeProtocolUnsafePublication.cfg": "Invariant CompletedPublicationIsSafe is violated.",
    "JobOutcomeProtocolUnsafeRelease.cfg": "Invariant ClientReleaseFollowsReportAttempt is violated.",
    "JobOutcomeProtocolUnsafeFailure.cfg": "Invariant FailureDominatesPublication is violated.",
    "JobOutcomeProtocolUnsafeDuplicate.cfg": "Invariant AtMostOneSettlement is violated.",
    "JobOutcomeProtocolUnsafeAbort.cfg": "Temporal properties were violated.",
}


def check(java: str, jar: Path) -> None:
    verify_model(
        java,
        jar,
        MODEL_DIR,
        "JobOutcomeProtocol",
        "JobOutcomeProtocol.cfg",
        "JobOutcomeProtocolLiveness.cfg",
        EXPECTED_MUTATIONS,
    )

    print("PASS client/server outcome safety and liveness model")
    for config, expected in EXPECTED_MUTATIONS.items():
        print(f"PASS {config} fails with {expected}")


if __name__ == "__main__":
    run_cli(check)
