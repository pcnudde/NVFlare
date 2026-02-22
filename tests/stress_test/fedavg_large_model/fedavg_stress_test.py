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

"""FedAvg stress test (POC) with server RSS comparison.

By default, this script runs both modes and verifies:
1) Disk streaming uses less server RSS than memory mode.
2) Final model results are the same.

Usage:
    python fedavg_stress_test.py --model-size-gb 0.5 --num-clients 3
    python fedavg_stress_test.py --no-compare --no-disk-streaming
"""

import argparse
import glob
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

import psutil
import torch

from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils import fobs
from nvflare.recipe.poc_env import PocEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_FILENAME = "FL_global_model.pt"
FALLBACK_SAVE_FILENAMES = ("best_FL_global_model.pt",)
MODEL_LOOKUP_TIMEOUT_SEC = 30.0
MODEL_LOOKUP_POLL_SEC = 0.5
MODEL_DOWNLOAD_RETRY_COUNT = 3
MODEL_DOWNLOAD_RETRY_DELAY_SEC = 2.0


def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def _checksum_state_dict(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _max_abs_diff(state_dict_a: dict, state_dict_b: dict) -> float:
    if set(state_dict_a.keys()) != set(state_dict_b.keys()):
        return float("inf")
    max_diff = 0.0
    for k in sorted(state_dict_a.keys()):
        va = state_dict_a[k]
        vb = state_dict_b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            diff = (va - vb).abs().max().item()
            if diff > max_diff:
                max_diff = diff
        elif va != vb:
            return float("inf")
    return max_diff


def _extract_tensor_state_dict(payload) -> Optional[dict]:
    if isinstance(payload, dict):
        if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload

        for key in ("model", "state_dict", "weights", "params"):
            candidate = payload.get(key)
            state_dict = _extract_tensor_state_dict(candidate)
            if state_dict:
                return state_dict

        for v in payload.values():
            state_dict = _extract_tensor_state_dict(v)
            if state_dict:
                return state_dict

    return None


def _load_tensor_state_dict(model_path: str) -> dict:
    payload = torch.load(model_path, weights_only=False)
    state_dict = _extract_tensor_state_dict(payload)
    if not state_dict:
        raise RuntimeError("no tensor weights found in saved model")
    return state_dict


def _find_result_model(
    result_dir: str,
    save_filename: str,
    extra_roots: Optional[list[str]] = None,
    fallback_filenames: Optional[tuple[str, ...]] = None,
    timeout_sec: float = MODEL_LOOKUP_TIMEOUT_SEC,
    poll_sec: float = MODEL_LOOKUP_POLL_SEC,
) -> str:
    roots = [result_dir]
    if extra_roots:
        roots.extend(extra_roots)
    roots = list(dict.fromkeys(roots))

    candidate_filenames = [save_filename]
    for name in fallback_filenames or ():
        if name and name not in candidate_filenames:
            candidate_filenames.append(name)

    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        for root in roots:
            for candidate in candidate_filenames:
                pattern = os.path.join(root, "**", candidate)
                matches = sorted(glob.glob(pattern, recursive=True))
                if matches:
                    return matches[0]

        if time.monotonic() >= deadline:
            break
        time.sleep(poll_sec)

    discovered_pt_files = []
    for root in roots:
        pt_matches = sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))
        for m in pt_matches:
            discovered_pt_files.append(m)
            if len(discovered_pt_files) >= 20:
                break
        if len(discovered_pt_files) >= 20:
            break

    raise RuntimeError(
        f"cannot find saved model candidates={candidate_filenames} under roots={roots}; "
        f"discovered_pt_files={discovered_pt_files}"
    )


def _locate_result_model_with_retries(run, initial_result_dir: str, save_filename: str, workspace_root: str) -> tuple[str, str]:
    result_dir = initial_result_dir
    last_err = None

    for attempt in range(1, MODEL_DOWNLOAD_RETRY_COUNT + 1):
        try:
            model_path = _find_result_model(
                result_dir=result_dir,
                save_filename=save_filename,
                extra_roots=[workspace_root],
                fallback_filenames=FALLBACK_SAVE_FILENAMES,
            )
            return model_path, result_dir
        except RuntimeError as e:
            last_err = e
            if attempt >= MODEL_DOWNLOAD_RETRY_COUNT:
                break
            fprint(
                f"Model artifact not found (attempt {attempt}/{MODEL_DOWNLOAD_RETRY_COUNT}); "
                "retrying result download..."
            )
            try:
                redownload_dir = run.get_result(timeout=30.0)
                if redownload_dir:
                    result_dir = redownload_dir
            except Exception as re:
                fprint(f"Warning: get_result retry failed on attempt {attempt}: {re}")
            time.sleep(MODEL_DOWNLOAD_RETRY_DELAY_SEC)

    raise RuntimeError(str(last_err))


def _find_server_pid(workspace_hint: str, started_after: float, fallback_unscoped: bool) -> tuple[Optional[int], Optional[str]]:
    def _scan(scope_workspace: bool):
        candidates = []
        for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                cmdline = proc.info["cmdline"] or []
                if not cmdline:
                    continue
                cmdline_str = " ".join(cmdline)
                if "server_train" not in cmdline_str or "python" not in cmdline_str.lower():
                    continue
                if scope_workspace and workspace_hint and workspace_hint not in cmdline_str:
                    continue
                create_time = proc.info.get("create_time") or 0.0
                if create_time + 5.0 < started_after:
                    continue
                candidates.append((create_time, proc.info["pid"], cmdline_str))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not candidates:
            return None, None
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, pid, cmd = candidates[0]
        return pid, cmd

    pid, cmd = _scan(scope_workspace=True)
    if pid is not None:
        return pid, cmd
    if fallback_unscoped:
        return _scan(scope_workspace=False)
    return None, None


def get_server_tree_rss_gb(server_pid: int) -> float:
    """Get total RSS of server process + all its children (job workers)."""
    try:
        parent = psutil.Process(server_pid)
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024**3)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


class ServerRSSMonitor:
    def __init__(self, workspace_hint: str, interval: float = 0.1):
        self.workspace_hint = workspace_hint
        self.interval = interval
        self.peak_gb = 0.0
        self.samples = 0
        self._server_pid = None
        self._server_cmd = None
        self._started_at = time.time()
        self._warned_fallback = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            if self._server_pid is None:
                pid, cmd = _find_server_pid(
                    workspace_hint=self.workspace_hint,
                    started_after=self._started_at,
                    fallback_unscoped=not self._warned_fallback,
                )
                if pid is not None:
                    self._server_pid = pid
                    self._server_cmd = cmd
                    if self.workspace_hint and self.workspace_hint not in (cmd or "") and not self._warned_fallback:
                        fprint("  [monitor] workspace-scoped server not found; using unscoped fallback")
                        self._warned_fallback = True
                    fprint(f"  [monitor] tracking server pid={pid}")
                    fprint(f"  [monitor] cmd={(cmd or '')[:200]}")

            if self._server_pid is not None:
                rss = get_server_tree_rss_gb(self._server_pid)
                if rss > 0:
                    prev_peak = self.peak_gb
                    if rss > self.peak_gb:
                        self.peak_gb = rss
                    self.samples += 1
                    if self.samples <= 5 or self.samples % 25 == 0 or self.peak_gb > prev_peak:
                        fprint(f"  [monitor] rss={rss:.3f} GB peak={self.peak_gb:.3f}")
                else:
                    self._server_pid = None
                    self._server_cmd = None

            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()


@dataclass
class RunMetrics:
    mode: str
    peak_gb: float
    samples: int
    result_dir: str
    model_path: str
    checksum: str
    sample_value: float
    artifact_dirs_seen: int
    artifact_files_seen: int
    artifact_peak_live_files: int
    log_markers: dict[str, int]


class DiskArtifactMonitor:
    """Track lazy tensor disk artifacts generated under temp roots."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.dir_patterns = self._build_dir_patterns()
        self._baseline_dirs = set()
        self._baseline_files = set()
        self.seen_dirs = set()
        self.seen_files = set()
        self.peak_live_dirs = 0
        self.peak_live_files = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _build_dir_patterns() -> list[str]:
        roots = {
            tempfile.gettempdir(),
            os.getenv("TMPDIR", ""),
            "/tmp",
            "/private/tmp",
        }
        patterns = [os.path.join(r, "nvflare_tensors_*") for r in roots if r]
        # On macOS, server subprocesses can use per-user temp roots under /var/folders.
        patterns.extend(
            [
                "/var/folders/*/*/*/T/nvflare_tensors_*",
                "/private/var/folders/*/*/*/T/nvflare_tensors_*",
            ]
        )
        return patterns

    def _list_current(self):
        dirs = set()
        for pattern in self.dir_patterns:
            for d in glob.glob(pattern):
                if os.path.isdir(d):
                    dirs.add(os.path.realpath(d))
        files = set()
        for d in dirs:
            files.update(os.path.realpath(f) for f in glob.glob(os.path.join(d, "*.safetensors")))
        return dirs, files

    def _scan_once(self):
        dirs, files = self._list_current()
        new_dirs = dirs.difference(self._baseline_dirs)
        new_files = files.difference(self._baseline_files)

        self.seen_dirs.update(new_dirs)
        self.seen_files.update(new_files)
        self.peak_live_dirs = max(self.peak_live_dirs, len(new_dirs))
        self.peak_live_files = max(self.peak_live_files, len(new_files))

    def _run(self):
        while not self._stop.is_set():
            self._scan_once()
            self._stop.wait(self.interval)

    def start(self):
        self._baseline_dirs, self._baseline_files = self._list_current()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        # one last scan for race-free final accounting
        self._scan_once()


def _count_log_markers(workspace: str, markers: list[str]) -> dict[str, int]:
    counts = {m: 0 for m in markers}
    for log_file in sorted(glob.glob(os.path.join(workspace, "**", "log*.txt"), recursive=True)):
        try:
            with open(log_file, "r") as f:
                for line in f:
                    for m in markers:
                        if m in line:
                            counts[m] += 1
        except OSError:
            continue
    return counts


def _copy_model_artifact(model_path: str, mode: str) -> str:
    dst_dir = tempfile.mkdtemp(prefix=f"fedavg_stress_{mode.lower()}_")
    dst_path = os.path.join(dst_dir, os.path.basename(model_path))
    shutil.copy2(model_path, dst_path)
    return dst_path


def _cleanup_run_artifacts(*runs: Optional[RunMetrics]):
    for run in runs:
        if not run:
            continue
        model_path = getattr(run, "model_path", None)
        if not model_path or model_path == "<skipped>":
            continue
        artifact_dir = os.path.dirname(model_path)
        try:
            shutil.rmtree(artifact_dir, ignore_errors=True)
        except Exception as e:
            fprint(f"Warning: failed to cleanup artifact dir {artifact_dir}: {e}")


def _build_recipe(args, disk_streaming: bool, name_suffix: str) -> FedAvgRecipe:
    recipe = FedAvgRecipe(
        name=f"fedavg_stress_{name_suffix}",
        model={"class_path": "net.StressNet", "args": {"size_gb": args.model_size_gb}},
        min_clients=args.num_clients,
        num_rounds=args.num_rounds,
        train_script=os.path.join(SCRIPT_DIR, "fedavg_train.py"),
        server_expected_format=ExchangeFormat.PYTORCH,
        enable_tensor_disk_offload=disk_streaming,
    )
    recipe.add_server_file(os.path.join(SCRIPT_DIR, "net.py"))
    return recipe


def _print_relevant_logs(workspace: str):
    keywords = [
        "use_disk",
        "enable_tensor_disk_offload",
        "native_recompose",
        "pre-processing datum",
        "ViaDownloader",
        "_set_enable_tensor_disk_offload",
    ]
    max_matches = 80
    matches = 0
    for log_file in sorted(glob.glob(os.path.join(workspace, "**", "log*.txt"), recursive=True)):
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if any(k in line for k in keywords):
                        fprint(f"  [log] {os.path.basename(log_file)}: {line.rstrip()}")
                        matches += 1
                        if matches >= max_matches:
                            fprint("  [log] ... truncated ...")
                            return
        except OSError:
            continue


def _run_one(args, disk_streaming: bool) -> RunMetrics:
    mode = "DISK" if disk_streaming else "MEMORY"
    fprint("\n" + "=" * 70)
    fprint(f"Mode: {mode}")
    fprint(f"Model: {args.model_size_gb} GB | Clients: {args.num_clients} | Rounds: {args.num_rounds}")
    fprint("=" * 70)

    fobs.register(TensorDecomposer)
    env = PocEnv(num_clients=args.num_clients)
    mon = ServerRSSMonitor(workspace_hint=env.poc_workspace, interval=args.sample_interval)
    artifact_mon = DiskArtifactMonitor(interval=args.sample_interval)
    recipe = _build_recipe(args=args, disk_streaming=disk_streaming, name_suffix=mode.lower())
    fprint(f"Configured server_expected_format={ExchangeFormat.PYTORCH}, enable_tensor_disk_offload={disk_streaming}")

    result_dir = None
    run = None
    mon.start()
    artifact_mon.start()
    try:
        run = recipe.execute(env)
        job_id = run.get_job_id()
        fprint(f"Job submitted: {job_id}")
        result_dir = run.get_result(timeout=args.timeout)
        if not result_dir:
            raise RuntimeError(f"job did not complete within timeout={args.timeout}s")
        if not os.path.exists(result_dir):
            raise RuntimeError(f"result path does not exist: {result_dir}")
    finally:
        mon.stop()
        artifact_mon.stop()
        fprint(f"Peak server RSS: {mon.peak_gb:.2f} GB ({mon.samples} samples)")
        fprint(
            "Disk artifacts: "
            f"seen_dirs={len(artifact_mon.seen_dirs)} seen_files={len(artifact_mon.seen_files)} "
            f"peak_live_files={artifact_mon.peak_live_files}"
        )
        if args.print_logs:
            _print_relevant_logs(env.poc_workspace)

    try:
        if mon.samples == 0:
            raise RuntimeError("server monitor captured 0 samples; cannot compare memory usage")

        marker_counts = _count_log_markers(
            workspace=env.poc_workspace,
            markers=["TensorDownloadable", "enable_tensor_disk_offload", "use_disk", "ViaDownloader", "pre-processing datum"],
        )
        fprint(f"Streaming markers: {marker_counts}")
        if args.require_streaming_markers and marker_counts["TensorDownloadable"] <= 0:
            raise RuntimeError("no TensorDownloadable marker found - streaming path not evidenced in logs")
        if disk_streaming and args.require_disk_artifacts and len(artifact_mon.seen_files) <= 0:
            raise RuntimeError("disk streaming enabled but no nvflare_tensors_*/chunk_*.safetensors artifacts observed")

        if args.skip_model_check:
            model_path = "<skipped>"
            checksum = "<skipped>"
            sample_value = float("nan")
            copied_model_path = model_path
            fprint("Model check skipped (large-scale memory/streaming-only run).")
        else:
            model_path, result_dir = _locate_result_model_with_retries(
                run=run,
                initial_result_dir=result_dir,
                save_filename=args.save_filename,
                workspace_root=env.poc_workspace,
            )
            state_dict = _load_tensor_state_dict(model_path)
            checksum = _checksum_state_dict(state_dict)

            first_tensor = next((v for v in state_dict.values() if isinstance(v, torch.Tensor)), None)
            if first_tensor is None:
                raise RuntimeError("no tensor weights found in saved model")
            sample_value = float(first_tensor.reshape(-1)[0].item())
            expected_value = 1.0 + float(args.num_rounds)
            if abs(sample_value - expected_value) > args.atol:
                raise RuntimeError(
                    f"unexpected model value for {mode}: got {sample_value}, expected {expected_value} (atol={args.atol})"
                )

            fprint(f"Result dir: {result_dir}")
            fprint(f"Model file: {model_path}")
            fprint(f"Checksum: {checksum}")
            fprint(f"Sample value: {sample_value}")
            copied_model_path = _copy_model_artifact(model_path=model_path, mode=mode)

        return RunMetrics(
            mode=mode,
            peak_gb=mon.peak_gb,
            samples=mon.samples,
            result_dir=result_dir,
            model_path=copied_model_path,
            checksum=checksum,
            sample_value=sample_value,
            artifact_dirs_seen=len(artifact_mon.seen_dirs),
            artifact_files_seen=len(artifact_mon.seen_files),
            artifact_peak_live_files=artifact_mon.peak_live_files,
            log_markers=marker_counts,
        )
    finally:
        try:
            env.stop(clean_poc=True)
        except Exception as e:
            fprint(f"Warning: failed to stop POC env cleanly: {e}")


def _compare_results(args, disk_run: RunMetrics, mem_run: RunMetrics):
    if args.skip_model_check:
        max_diff = float("nan")
        same_result = True
    else:
        disk_state = _load_tensor_state_dict(disk_run.model_path)
        mem_state = _load_tensor_state_dict(mem_run.model_path)
        keys_equal = set(disk_state.keys()) == set(mem_state.keys())
        max_diff = _max_abs_diff(disk_state, mem_state)
        same_result = keys_equal and max_diff <= args.atol

    mem_saved_gb = mem_run.peak_gb - disk_run.peak_gb
    memory_improved = mem_saved_gb > args.min_memory_reduction_gb

    fprint("\n" + "=" * 70)
    fprint("Comparison Summary")
    fprint("=" * 70)
    fprint(f"Disk peak RSS:   {disk_run.peak_gb:.3f} GB")
    fprint(f"Memory peak RSS: {mem_run.peak_gb:.3f} GB")
    fprint(f"Saved:           {mem_saved_gb:.3f} GB")
    fprint(f"Disk checksum:   {disk_run.checksum}")
    fprint(f"Memory checksum: {mem_run.checksum}")
    fprint(f"Max abs diff:    {max_diff}")
    fprint(
        "Disk artifacts: "
        f"dirs={disk_run.artifact_dirs_seen} files={disk_run.artifact_files_seen} "
        f"peak_live_files={disk_run.artifact_peak_live_files}"
    )
    fprint(
        "Memory artifacts: "
        f"dirs={mem_run.artifact_dirs_seen} files={mem_run.artifact_files_seen} "
        f"peak_live_files={mem_run.artifact_peak_live_files}"
    )
    fprint(f"Disk markers:    {disk_run.log_markers}")
    fprint(f"Memory markers:  {mem_run.log_markers}")

    if not same_result:
        raise RuntimeError("result mismatch: disk and memory mode produced different final models")
    if not memory_improved:
        raise RuntimeError(
            "memory check failed: disk streaming did not reduce server RSS enough "
            f"(saved={mem_saved_gb:.3f} GB, min_required={args.min_memory_reduction_gb:.3f} GB)"
        )

    if args.skip_model_check:
        fprint("PASS: disk streaming reduced server memory (model equality check skipped).")
    else:
        fprint("PASS: disk streaming reduced server memory and produced identical model results.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size-gb", type=float, default=0.5)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--save-filename", type=str, default=DEFAULT_SAVE_FILENAME)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--min-memory-reduction-gb", type=float, default=0.0)
    parser.add_argument("--print-logs", action="store_true")
    parser.add_argument("--no-disk-streaming", action="store_true")
    parser.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-disk-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-streaming-markers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-model-check", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    try:
        if args.compare:
            disk_run = None
            mem_run = None
            try:
                if args.no_disk_streaming:
                    fprint("Warning: --no-disk-streaming is ignored in compare mode")
                disk_run = _run_one(args, disk_streaming=True)
                mem_run = _run_one(args, disk_streaming=False)
                _compare_results(args, disk_run, mem_run)
            finally:
                _cleanup_run_artifacts(disk_run, mem_run)
        else:
            run = None
            try:
                run = _run_one(args, disk_streaming=not args.no_disk_streaming)
                fprint(f"PASS: {run.mode} run completed successfully.")
            finally:
                _cleanup_run_artifacts(run)
    except Exception as e:
        fprint(f"FAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
