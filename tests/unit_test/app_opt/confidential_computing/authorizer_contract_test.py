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

import base64
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.app_opt.confidential_computing import aci_authorizer as aci
from nvflare.app_opt.confidential_computing import snp_authorizer as snp
from nvflare.app_opt.confidential_computing import tdx_authorizer as tdx

REPORT_TEXT = (
    "Chip ID: "
    + "AA " * 15
    + "AA\nReported TCB:\nTCB Version:\nMicrocode: 12\nSNP: 3\nTEE: 4\nBoot Loader: 5\nFMC: None\n"
)


def test_snp_report_parser_extracts_chip_and_tcb_for_cache_key():
    assert snp.parse_chip_id(REPORT_TEXT) == "aa" * 16
    assert snp.parse_reported_tcb(REPORT_TEXT) == {"Microcode": 12, "SNP": 3, "TEE": 4, "Boot Loader": 5, "FMC": None}
    assert snp.parse_chip_id("truncated") == ""
    assert snp.parse_reported_tcb("truncated") == {}
    assert snp.parse_reported_tcb(REPORT_TEXT.replace("FMC: None", "FMC: 7"))["FMC"] == "7"


def test_snp_retry_preserves_command_timeout_and_exponential_backoff(monkeypatch):
    run = MagicMock(
        side_effect=[
            subprocess.CompletedProcess([], 1, stderr=b"busy"),
            subprocess.TimeoutExpired([], 60),
            subprocess.CompletedProcess([], 0),
        ]
    )
    monkeypatch.setattr(snp.subprocess, "run", run)
    sleep = MagicMock()
    monkeypatch.setattr(snp, "time", SimpleNamespace(sleep=sleep))
    authorizer = snp.SNPAuthorizer(max_retries=3, retry_interval=2)
    assert authorizer._run_with_retry(["snpguest", "report"], "report").returncode == 0
    assert [c.args[0] for c in sleep.call_args_list] == [2, 4]
    assert all(c.kwargs == {"capture_output": True, "timeout": 60} for c in run.call_args_list)
    run.side_effect = None
    run.return_value = subprocess.CompletedProcess([], 1, stderr=b"bad")
    with pytest.raises(RuntimeError, match="Failed after 3 attempts"):
        authorizer._run_with_retry([], "verify")


def test_snp_generation_encodes_report_and_records_nonce(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(snp.secrets, "token_bytes", lambda n: b"n" * n)
    authorizer = snp.SNPAuthorizer()

    def write_report(command, action):
        assert command == ["snpguest", "report", "report.bin", "request.bin"]
        assert Path("request.bin").read_bytes() == b"n" * 64
        Path("report.bin").write_bytes(b"report-data")

    monkeypatch.setattr(authorizer, "_run_with_retry", write_report)
    assert authorizer.generate() == base64.b64encode(b"report-data")
    assert authorizer.my_nonce_history.add(bytearray(b"n" * 64)) is False
    assert authorizer.get_namespace() == snp.SNP_NAMESPACE


@pytest.mark.parametrize("accepted", [False, True])
def test_snp_verify_cleans_temporary_report_on_success_or_nonce_rejection(monkeypatch, tmp_path, accepted):
    monkeypatch.chdir(tmp_path)
    authorizer = snp.SNPAuthorizer()
    monkeypatch.setattr(authorizer, "_ensure_amd_ca_certs", MagicMock())
    monkeypatch.setattr(authorizer, "_parse_report", MagicMock(return_value="cache-key"))
    monkeypatch.setattr(authorizer, "_ensure_amd_vcek", MagicMock())
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(authorizer, "_run_with_retry", run)
    monkeypatch.setattr(authorizer, "_check_nonce", MagicMock(return_value=accepted))
    assert authorizer.verify(base64.b64encode(b"binary-report")) is accepted
    assert list(tmp_path.iterdir()) == []
    assert run.call_args.args[1] == "verify_attestation"
    authorizer._ensure_amd_ca_certs.side_effect = RuntimeError("unavailable")
    assert authorizer.verify(b"broken") is False
    assert list(tmp_path.iterdir()) == []


def test_snp_certificates_fetch_once_and_reuse_vcek_cache(monkeypatch, tmp_path):
    authorizer = snp.SNPAuthorizer(amd_certs_dir=str(tmp_path))
    run = MagicMock()
    monkeypatch.setattr(authorizer, "_run_with_retry", run)
    authorizer._ensure_amd_ca_certs()
    assert run.call_args.args[1] == "fetch_ca_certs"
    for name in (snp.AMD_ASK, snp.AMD_ARK):
        (tmp_path / name).write_text("certificate")
    run.reset_mock()
    authorizer._ensure_amd_ca_certs()
    run.assert_not_called()

    def fetch(command, action):
        assert action == "fetch_vcek"
        (tmp_path / snp.AMD_VCEK).write_text("vcek-certificate")

    run.side_effect = fetch
    authorizer._ensure_amd_vcek("cache.pem", "report.bin")
    assert (tmp_path / "cache.pem").read_text() == "vcek-certificate"
    assert (tmp_path / snp.AMD_VCEK).read_text() == "vcek-certificate"
    run.reset_mock()
    authorizer._ensure_amd_vcek("cache.pem", "report.bin")
    run.assert_not_called()
    (tmp_path / snp.AMD_VCEK).unlink()
    run.side_effect = None
    with pytest.raises(RuntimeError, match="VCEK file not generated"):
        authorizer._ensure_amd_vcek("new-cache.pem", "report.bin")


def test_snp_display_rejects_bad_reports_and_detects_replayed_nonce(monkeypatch):
    authorizer = snp.SNPAuthorizer()
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout=REPORT_TEXT.encode()))
    monkeypatch.setattr(snp.subprocess, "run", run)
    assert authorizer._parse_report("report.bin") == "a" * 32 + "-12-3-4-5"
    run.return_value.stdout = b"missing tcb"
    with pytest.raises(RuntimeError, match="Reported TCB"):
        authorizer._parse_report("report.bin")
    run.return_value.stdout = b"Report Data:\nAA BB\nCC DD\nEE FF\n00 11\n22 33"
    assert authorizer._check_nonce("report.bin") is True
    assert authorizer._check_nonce("report.bin") is False
    run.return_value.returncode = 1
    assert authorizer._check_nonce("report.bin") is False
    with pytest.raises(RuntimeError, match="display SNP report"):
        authorizer._parse_report("report.bin")


def test_aci_retries_transport_failure_and_verifies_signed_claims(monkeypatch):
    authorizer = aci.ACIAuthorizer(maa_endpoint="issuer", retry_count=2, retry_sleep=0.25)
    post = MagicMock(
        side_effect=[OSError("offline"), SimpleNamespace(status_code=200, json=lambda: {"token": "token"})]
    )
    monkeypatch.setattr(aci.requests, "post", post)
    sleep = MagicMock()
    monkeypatch.setattr(aci, "time", SimpleNamespace(sleep=sleep))
    assert authorizer.generate() == "token"
    sleep.assert_called_once_with(0.25)
    client = MagicMock()
    client.get_signing_key_from_jwt.return_value.key = "public-key"
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(aci.jwt, "PyJWKClient", factory)
    monkeypatch.setattr(aci.jwt, "get_unverified_header", lambda token: {"alg": "RS256"})
    decode = MagicMock(return_value={"valid": True})
    monkeypatch.setattr(aci.jwt, "decode", decode)
    assert authorizer.verify("token") is True
    factory.assert_called_once_with("https://issuer/certs")
    decode.assert_called_once_with("token", "public-key", algorithms=["RS256"])
    decode.side_effect = ValueError("signature")
    assert authorizer.verify("tampered") is False
    post.side_effect = OSError("offline")
    assert authorizer.generate() == ""
    assert authorizer.get_namespace() == aci.ACI_NAMESPACE


@pytest.mark.parametrize("error,expected", [("", True), ("Error: no evidence", False)])
def test_tdx_process_boundary_uses_config_and_rejects_cli_errors(monkeypatch, tmp_path, error, expected):
    authorizer = tdx.TDXAuthorizer("tdx", str(tmp_path))
    commands = []

    def run(command, stdout, stderr):
        commands.append(command)
        stdout.write("token-value\n")
        stdout.flush()
        stderr.write(error)
        stderr.flush()

    monkeypatch.setattr(tdx, "run_in_new_session", run)
    assert authorizer.generate() == ("token-value\n" if expected else "")
    assert authorizer.verify("given-token") is expected
    assert commands[0] == ["sudo", "tdx", "-c", str(tmp_path / "config.json"), "token", "--no-eventlog"]
    assert commands[1][-2:] == ["--token", "given-token"]
    assert authorizer.get_namespace() == tdx.TDX_NAMESPACE
