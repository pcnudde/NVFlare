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

"""Swarm learning stress test with large PyTorch models.

Exercises the DXO aggregation path (CCWF/Swarm -> DXOAggregator ->
WeightedAggregationHelper) with PyTorch tensors to validate disk-streamed
tensor aggregation.

This test now supports:
1) configurable number of clients,
2) configurable concurrent submissions to stress the aggregation client,
3) deterministic aggregator-site selection,
4) explicit streaming marker and disk-artifact checks,
5) aggregator-site RSS tracking in addition to full process-tree RSS.

Usage:
    # Smoke test (200 MB model)
    python swarm_stress_test.py --model-size-gb 0.2 --num-rounds 1 --num-clients 3

    # Compare disk vs memory with concurrent submissions
    python swarm_stress_test.py --model-size-gb 1 --num-rounds 1 --num-clients 5 \
      --max-concurrent-submissions 5 --compare

    # Baseline run without disk streaming
    python swarm_stress_test.py --model-size-gb 1 --num-rounds 1 --no-disk-streaming
"""

import argparse
import gc
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Optional

import psutil
import torch

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.ccwf_job import CCWFJob, SwarmClientConfig, SwarmServerConfig
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs
from nvflare.security.logging import secure_format_exception

NUM_LAYERS = 50
RESULT_FILENAME = "stress_test_final_model.pt"
AGGR_RSS_FILENAME = "stress_test_aggregator_rss.json"
DEFAULT_AGGREGATOR_SITE = "site-1"


class ProcessTreeRSSTracker:
    """Track peak tree RSS plus per-site process RSS for simulator child processes."""

    def __init__(self, interval: float = 0.1, site_names: Optional[list[str]] = None):
        self._interval = interval
        self._proc = psutil.Process()
        self._site_names = site_names or []
        self._site_patterns = {
            s: re.compile(rf"(^|[\\/\\s]){re.escape(s)}([\\/\\s]|$)") for s in self._site_names
        }
        self._site_cache = {}  # (pid, create_time) -> site or ""
        self._peak_total_bytes = 0
        self._baseline_total_bytes = 0
        self._peak_site_bytes = {s: 0 for s in self._site_names}
        self._baseline_site_bytes = {s: 0 for s in self._site_names}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    @staticmethod
    def _proc_key(proc: psutil.Process):
        try:
            return proc.pid, proc.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return proc.pid, None

    def _detect_site(self, proc: psutil.Process) -> Optional[str]:
        if not self._site_patterns:
            return None

        key = self._proc_key(proc)
        cached = self._site_cache.get(key)
        if cached is not None:
            return cached or None

        text_parts = []
        try:
            cmdline = proc.cmdline() or []
            if cmdline:
                text_parts.append(" ".join(cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        try:
            cwd = proc.cwd()
            if cwd:
                text_parts.append(cwd)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

        text = " ".join(text_parts)
        site = ""
        if text:
            for s, pattern in self._site_patterns.items():
                if pattern.search(text):
                    site = s
                    break

        self._site_cache[key] = site
        return site or None

    def _tree_rss(self) -> tuple[int, dict[str, int]]:
        total = 0
        per_site = {s: 0 for s in self._site_names}
        try:
            procs = [self._proc]
            procs.extend(self._proc.children(recursive=True))
            for p in procs:
                try:
                    rss = p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                total += rss
                site = self._detect_site(p)
                if site:
                    per_site[site] += rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return total, per_site

    def _sample(self):
        while not self._stop.is_set():
            rss, site_rss = self._tree_rss()
            if rss > self._peak_total_bytes:
                self._peak_total_bytes = rss
            for s, v in site_rss.items():
                if v > self._peak_site_bytes[s]:
                    self._peak_site_bytes[s] = v
            self._stop.wait(self._interval)

    def start(self):
        self._baseline_total_bytes, self._baseline_site_bytes = self._tree_rss()
        self._peak_total_bytes = self._baseline_total_bytes
        self._peak_site_bytes = dict(self._baseline_site_bytes)
        self._thread.start()

    def stop(self) -> tuple[float, float, dict[str, float], dict[str, float]]:
        self._stop.set()
        self._thread.join(timeout=2.0)

        to_gb = 1024**3
        peak_sites = {k: v / to_gb for k, v in self._peak_site_bytes.items()}
        base_sites = {k: v / to_gb for k, v in self._baseline_site_bytes.items()}
        return (
            self._peak_total_bytes / to_gb,
            self._baseline_total_bytes / to_gb,
            peak_sites,
            base_sites,
        )


class DiskArtifactMonitor:
    """Track lazy tensor disk artifacts generated under temp roots."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.dir_patterns = self._build_dir_patterns()
        self._baseline_dirs = set()
        self._baseline_files = set()
        self.seen_dirs = set()
        self.seen_files = set()
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
        self._scan_once()


def _site_names(num_clients: int) -> list[str]:
    return [f"site-{i}" for i in range(1, num_clients + 1)]


def _build_state_dict(size_gb: float) -> dict[str, torch.Tensor]:
    total_elements = int(size_gb * (1024**3) / 4)
    per_layer = total_elements // NUM_LAYERS
    remainder = total_elements - per_layer * NUM_LAYERS

    # Fixed seed so both modes start with identical model (for result comparison).
    # Random data defeats macOS transparent memory compression, giving accurate RSS.
    torch.manual_seed(42)
    state_dict = {}
    for i in range(NUM_LAYERS):
        n = per_layer + (1 if i < remainder else 0)
        state_dict[f"layer_{i}.weight"] = torch.randn(n, dtype=torch.float32)

    actual_gb = sum(t.nelement() * 4 for t in state_dict.values()) / (1024**3)
    print(f"  State dict: {len(state_dict)} tensors, {total_elements:,} elements, {actual_gb:.2f} GB")
    return state_dict


def _checksum_state_dict(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _count_log_markers(workdir: str, markers: list[str]) -> dict[str, int]:
    counts = {m: 0 for m in markers}
    for log_file in sorted(glob.glob(os.path.join(workdir, "**", "log*.txt"), recursive=True)):
        try:
            with open(log_file, "r") as f:
                for line in f:
                    for m in markers:
                        if m in line:
                            counts[m] += 1
        except OSError:
            continue
    return counts


class LargePTModelPersistor(ModelPersistor):
    def __init__(self, size_gb: float):
        super().__init__()
        self.size_gb = size_gb

    def load_model(self, fl_ctx: FLContext):
        fobs.register(TensorDecomposer)
        self.log_info(fl_ctx, f"Creating initial PT model (~{self.size_gb:.1f} GB)")
        state_dict = _build_state_dict(self.size_gb)
        return make_model_learnable(weights=state_dict, meta_props={})

    def save_model(self, model_learnable, fl_ctx: FLContext):
        weights = model_learnable.get(ModelLearnableKey.WEIGHTS, {})
        if not weights:
            self.log_info(fl_ctx, "No weights to save")
            return

        # Resolve any lazy refs before saving
        for k, v in list(weights.items()):
            if hasattr(v, "materialize"):
                weights[k] = v.materialize()

        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        result_dir = engine.get_workspace().get_run_dir(job_id)
        save_path = os.path.join(result_dir, RESULT_FILENAME)

        checksum = _checksum_state_dict(weights)
        self.log_info(fl_ctx, f"Saving final model: {len(weights)} keys, checksum={checksum}")
        torch.save(weights, save_path)
        self.log_info(fl_ctx, f"Saved to {save_path}")


class LargePTTrainer(Executor):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self._delta = delta
        fobs.register(TensorDecomposer)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != AppConstants.TASK_TRAIN:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        try:
            dxo = from_shareable(shareable)
        except Exception as e:
            self.system_panic(f"Cannot convert shareable: {secure_format_exception(e)}", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)

        if dxo.data_kind != DataKind.WEIGHTS:
            self.system_panic("Expected DataKind.WEIGHTS", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)

        # Resolve lazy refs from disk-streamed download
        for k, v in list(dxo.data.items()):
            if hasattr(v, "materialize"):
                dxo.data[k] = v.materialize()

        total_elements = sum(t.nelement() for t in dxo.data.values() if isinstance(t, torch.Tensor))
        total_gb = total_elements * 4 / (1024**3)
        self.log_info(
            fl_ctx,
            f"Round {current_round}/{total_rounds} – "
            f"{len(dxo.data)} tensors, {total_elements:,} elements ({total_gb:.1f} GB)",
        )

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        for t in dxo.data.values():
            if isinstance(t, torch.Tensor):
                t.add_(self._delta)

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=dxo.data,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
        return outgoing_dxo.to_shareable()


class RSSAwareInTimeAggregator(InTimeAccumulateWeightedAggregator):
    """In-time aggregator that records its own process RSS for stress-test reporting."""

    def __init__(
        self,
        exclude_vars=None,
        aggregation_weights=None,
        expected_data_kind=DataKind.WEIGHT_DIFF,
        weigh_by_local_iter: bool = True,
    ):
        super().__init__(
            exclude_vars=exclude_vars,
            aggregation_weights=aggregation_weights,
            expected_data_kind=expected_data_kind,
            weigh_by_local_iter=weigh_by_local_iter,
        )
        self._baseline_rss_bytes = 0
        self._peak_rss_bytes = 0
        self._accepted = 0

    @staticmethod
    def _get_rss_bytes() -> int:
        try:
            return psutil.Process().memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _sample_rss(self):
        rss = self._get_rss_bytes()
        if rss <= 0:
            return
        if self._baseline_rss_bytes <= 0:
            self._baseline_rss_bytes = rss
        if rss > self._peak_rss_bytes:
            self._peak_rss_bytes = rss

    def _write_metrics(self, fl_ctx: FLContext):
        if self._accepted <= 0:
            return

        baseline_gb = self._baseline_rss_bytes / (1024**3)
        peak_gb = self._peak_rss_bytes / (1024**3)
        self.log_info(fl_ctx, f"STRESS_AGGR_RSS baseline_gb={baseline_gb:.4f} peak_gb={peak_gb:.4f} accepted={self._accepted}")

        engine = fl_ctx.get_engine()
        run_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        if not engine or not run_id:
            return

        run_dir = engine.get_workspace().get_run_dir(run_id)
        metrics = {
            "baseline_gb": baseline_gb,
            "peak_gb": peak_gb,
            "accepted": self._accepted,
        }
        out_path = os.path.join(run_dir, AGGR_RSS_FILENAME)
        try:
            with open(out_path, "w") as f:
                json.dump(metrics, f)
        except OSError:
            pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.START_RUN:
            self._baseline_rss_bytes = self._get_rss_bytes()
            self._peak_rss_bytes = self._baseline_rss_bytes
            self._accepted = 0
        elif event_type == EventType.END_RUN:
            self._sample_rss()
            self._write_metrics(fl_ctx)

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        self._sample_rss()
        accepted = super().accept(shareable, fl_ctx)
        if accepted:
            self._accepted += 1
        self._sample_rss()
        return accepted

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        self._sample_rss()
        result = super().aggregate(fl_ctx)
        self._sample_rss()
        self._write_metrics(fl_ctx)
        return result


def build_job(
    model_size_gb: float,
    num_rounds: int,
    num_clients: int,
    max_concurrent_submissions: int,
    aggregator_site: str,
    disk_streaming: bool,
) -> CCWFJob:
    sites = _site_names(num_clients)
    if aggregator_site not in sites:
        raise ValueError(f"aggregator_site {aggregator_site} not in participating sites {sites}")

    job = CCWFJob(name="swarm_pt_stress_test", min_clients=num_clients)
    job.add_swarm(
        server_config=SwarmServerConfig(
            num_rounds=num_rounds,
            start_task_timeout=3600,
            progress_timeout=600.0,
            participating_clients=sites,
            result_clients=sites,
            starting_client=sites[0],
            aggr_clients=[aggregator_site],
            train_clients=sites,
        ),
        client_config=SwarmClientConfig(
            executor=LargePTTrainer(delta=1.0),
            aggregator=RSSAwareInTimeAggregator(expected_data_kind=DataKind.WEIGHTS),
            persistor=LargePTModelPersistor(size_gb=model_size_gb),
            shareable_generator=SimpleModelShareableGenerator(),
            learn_task_ack_timeout=3600,
            final_result_ack_timeout=3600,
            min_responses_required=num_clients,
            wait_time_after_min_resps_received=30.0,
            max_concurrent_submissions=max_concurrent_submissions,
            enable_tensor_disk_offload=disk_streaming,
        ),
    )
    return job


def _find_result_file(workdir: str) -> str | None:
    pattern = os.path.join(workdir, "**", RESULT_FILENAME)
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def _read_aggregator_rss_metrics(workdir: str, aggregator_site: str) -> tuple[float, float]:
    # Primary: explicit metrics artifact written by RSSAwareInTimeAggregator.
    pattern = os.path.join(workdir, aggregator_site, "**", AGGR_RSS_FILENAME)
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        pass
    else:
        try:
            with open(matches[0], "r") as f:
                d = json.load(f)
            return float(d.get("peak_gb", 0.0)), float(d.get("baseline_gb", 0.0))
        except (OSError, ValueError, TypeError):
            pass

    # Fallback: parse STRESS_AGGR_RSS marker from aggregator-site logs.
    marker = re.compile(r"STRESS_AGGR_RSS baseline_gb=([0-9.]+) peak_gb=([0-9.]+) accepted=([0-9]+)")
    best_peak = 0.0
    best_base = 0.0
    for log_file in glob.glob(os.path.join(workdir, aggregator_site, "**", "log*.txt"), recursive=True):
        try:
            with open(log_file, "r") as f:
                for line in f:
                    m = marker.search(line)
                    if not m:
                        continue
                    base = float(m.group(1))
                    peak = float(m.group(2))
                    if peak > best_peak:
                        best_peak = peak
                        best_base = base
        except OSError:
            continue
    return best_peak, best_base


def run_one(args, disk_streaming: bool) -> dict:
    mode = "DISK" if disk_streaming else "MEMORY"
    print(f"\n{'='*70}")
    print(f"Mode: {mode}")
    print(f"Model: {args.model_size_gb} GB, Rounds: {args.num_rounds}")
    print(
        f"Clients: {args.num_clients}, max_concurrent_submissions: {args.max_concurrent_submissions}, "
        f"aggregator_site: {args.aggregator_site}"
    )
    print(f"{'='*70}\n")

    workdir = args.workdir
    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    gc.collect()
    tracker = ProcessTreeRSSTracker(interval=args.sample_interval, site_names=_site_names(args.num_clients))
    artifact_mon = DiskArtifactMonitor(interval=args.sample_interval)
    tracker.start()
    artifact_mon.start()
    try:
        job = build_job(
            model_size_gb=args.model_size_gb,
            num_rounds=args.num_rounds,
            num_clients=args.num_clients,
            max_concurrent_submissions=args.max_concurrent_submissions,
            aggregator_site=args.aggregator_site,
            disk_streaming=disk_streaming,
        )
        job.simulator_run(workdir, n_clients=args.num_clients)
    finally:
        artifact_mon.stop()
        peak_rss, baseline_rss, site_peaks, site_baselines = tracker.stop()

    # Prefer direct aggregator-process metrics emitted by RSSAwareInTimeAggregator.
    # Fallback to process-tagged per-site estimate when metrics file is unavailable.
    aggr_peak, aggr_baseline = _read_aggregator_rss_metrics(workdir, args.aggregator_site)
    if aggr_peak <= 0.0:
        aggr_peak = site_peaks.get(args.aggregator_site, 0.0)
        aggr_baseline = site_baselines.get(args.aggregator_site, 0.0)

    print(f"  Baseline RSS (tree):           {baseline_rss:.2f} GB")
    print(f"  Peak RSS (tree):               {peak_rss:.2f} GB")
    print(f"  Baseline RSS ({args.aggregator_site}): {aggr_baseline:.2f} GB")
    print(f"  Peak RSS ({args.aggregator_site}):     {aggr_peak:.2f} GB")
    print(
        "  Disk artifacts: "
        f"dirs={len(artifact_mon.seen_dirs)} files={len(artifact_mon.seen_files)} "
        f"peak_live_files={artifact_mon.peak_live_files}"
    )

    markers = ["TensorDownloadable", "enable_tensor_disk_offload", "use_disk", "ViaDownloader", "pre-processing datum"]
    marker_counts = _count_log_markers(workdir, markers)
    print(f"  Streaming markers: {marker_counts}")

    if args.require_streaming_markers and marker_counts["TensorDownloadable"] <= 0:
        raise RuntimeError("no TensorDownloadable marker found - streaming path not evidenced in logs")
    if disk_streaming and args.require_disk_artifacts and len(artifact_mon.seen_files) <= 0:
        raise RuntimeError("disk mode enabled but no nvflare_tensors_*/chunk_*.safetensors artifacts were observed")

    result_file = _find_result_file(workdir)
    checksum = None
    if result_file:
        sd = torch.load(result_file, weights_only=True)
        checksum = _checksum_state_dict(sd)
        print(f"  Result: {result_file}")
        print(f"  Keys: {len(sd)}, Checksum: {checksum}")

    # Emit machine-readable summary for subprocess parsing.
    print(
        "STRESS_RESULT|"
        f"peak={peak_rss:.4f}|"
        f"baseline={baseline_rss:.4f}|"
        f"aggr_peak={aggr_peak:.4f}|"
        f"aggr_baseline={aggr_baseline:.4f}|"
        f"checksum={checksum or 'NONE'}|"
        f"result={result_file or 'NONE'}|"
        f"artifact_dirs={len(artifact_mon.seen_dirs)}|"
        f"artifact_files={len(artifact_mon.seen_files)}|"
        f"marker_tensor_downloadable={marker_counts['TensorDownloadable']}"
    )

    return {
        "peak_gb": peak_rss,
        "baseline_gb": baseline_rss,
        "aggr_peak_gb": aggr_peak,
        "aggr_baseline_gb": aggr_baseline,
        "checksum": checksum,
        "result_file": result_file,
        "artifact_dirs": len(artifact_mon.seen_dirs),
        "artifact_files": len(artifact_mon.seen_files),
        "marker_tensor_downloadable": marker_counts["TensorDownloadable"],
    }


def _parse_stress_result(output: str) -> dict | None:
    for line in output.splitlines():
        if line.startswith("STRESS_RESULT|"):
            parts = dict(p.split("=", 1) for p in line.split("|")[1:])
            return {
                "peak_gb": float(parts["peak"]),
                "baseline_gb": float(parts["baseline"]),
                "aggr_peak_gb": float(parts["aggr_peak"]),
                "aggr_baseline_gb": float(parts["aggr_baseline"]),
                "checksum": parts["checksum"] if parts["checksum"] != "NONE" else None,
                "result_file": parts["result"] if parts["result"] != "NONE" else None,
                "artifact_dirs": int(parts.get("artifact_dirs", 0)),
                "artifact_files": int(parts.get("artifact_files", 0)),
                "marker_tensor_downloadable": int(parts.get("marker_tensor_downloadable", 0)),
            }
    return None


def _run_subprocess(args, workdir: str, disk_streaming: bool) -> dict:
    """Run one mode in a fresh subprocess for clean memory measurement."""

    cmd = [
        sys.executable,
        __file__,
        "--model-size-gb",
        str(args.model_size_gb),
        "--num-rounds",
        str(args.num_rounds),
        "--num-clients",
        str(args.num_clients),
        "--max-concurrent-submissions",
        str(args.max_concurrent_submissions),
        "--aggregator-site",
        args.aggregator_site,
        "--sample-interval",
        str(args.sample_interval),
        "--workdir",
        workdir,
    ]
    if not disk_streaming:
        cmd.append("--no-disk-streaming")
    if not args.require_disk_artifacts:
        cmd.append("--no-require-disk-artifacts")
    if not args.require_streaming_markers:
        cmd.append("--no-require-streaming-markers")

    mode = "DISK" if disk_streaming else "MEMORY"
    print(f"\n--- Launching {mode} mode in subprocess ---")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"{mode} subprocess failed with rc={result.returncode}")

    parsed = _parse_stress_result(result.stdout)
    if not parsed:
        raise RuntimeError(f"Could not parse STRESS_RESULT from {mode} output")
    return parsed


def compare_mode(args) -> bool:
    workdir_disk = args.workdir + "_disk"
    workdir_mem = args.workdir + "_mem"

    r_disk = _run_subprocess(args, workdir_disk, disk_streaming=True)
    r_mem = _run_subprocess(args, workdir_mem, disk_streaming=False)

    tree_saved = r_mem["peak_gb"] - r_disk["peak_gb"]
    aggr_saved = r_mem["aggr_peak_gb"] - r_disk["aggr_peak_gb"]

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    print(
        "  Tree peak RSS (disk):   "
        f"{r_disk['peak_gb']:.2f} GB (baseline {r_disk['baseline_gb']:.2f}, delta {r_disk['peak_gb'] - r_disk['baseline_gb']:.2f})"
    )
    print(
        "  Tree peak RSS (memory): "
        f"{r_mem['peak_gb']:.2f} GB (baseline {r_mem['baseline_gb']:.2f}, delta {r_mem['peak_gb'] - r_mem['baseline_gb']:.2f})"
    )
    print(f"  Tree memory saved:       {tree_saved:.2f} GB")

    print(
        f"  {args.aggregator_site} peak RSS (disk):   "
        f"{r_disk['aggr_peak_gb']:.2f} GB (baseline {r_disk['aggr_baseline_gb']:.2f}, delta {r_disk['aggr_peak_gb'] - r_disk['aggr_baseline_gb']:.2f})"
    )
    print(
        f"  {args.aggregator_site} peak RSS (memory): "
        f"{r_mem['aggr_peak_gb']:.2f} GB (baseline {r_mem['aggr_baseline_gb']:.2f}, delta {r_mem['aggr_peak_gb'] - r_mem['aggr_baseline_gb']:.2f})"
    )
    print(f"  Aggregator memory saved: {aggr_saved:.2f} GB")

    metric_name = "aggregator" if args.memory_metric == "aggregator" else "tree"
    if metric_name == "aggregator":
        metric_disk = r_disk["aggr_peak_gb"]
        metric_mem = r_mem["aggr_peak_gb"]
    else:
        metric_disk = r_disk["peak_gb"]
        metric_mem = r_mem["peak_gb"]
    metric_saved = metric_mem - metric_disk

    cksum_disk = r_disk["checksum"]
    cksum_mem = r_mem["checksum"]

    if not cksum_disk:
        print("  ERROR: disk mode result not found")
        return False
    if not cksum_mem:
        print("  ERROR: memory mode result not found")
        return False

    f_disk = r_disk["result_file"]
    f_mem = r_mem["result_file"]

    sd_disk = torch.load(f_disk, weights_only=True)
    sd_mem = torch.load(f_mem, weights_only=True)

    if set(sd_disk.keys()) != set(sd_mem.keys()):
        print("  FAIL: different keys")
        return False

    max_diff = 0.0
    for k in sorted(sd_disk.keys()):
        diff = (sd_disk[k] - sd_mem[k]).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"  Disk checksum:   {cksum_disk}")
    print(f"  Memory checksum: {cksum_mem}")
    print(f"  Max diff:        {max_diff}")

    if cksum_disk == cksum_mem:
        same_result = True
    else:
        same_result = max_diff < 1e-6

    if not same_result:
        print("  RESULT: MISMATCH")
        return False

    if args.require_memory_reduction and metric_saved <= args.min_memory_reduction_gb:
        print(
            "  FAIL: memory reduction check failed: "
            f"metric={metric_name}, saved={metric_saved:.3f} GB, "
            f"min_required={args.min_memory_reduction_gb:.3f} GB"
        )
        return False

    print(
        "  PASS: "
        f"results equivalent and {metric_name} memory reduced by {metric_saved:.3f} GB"
    )
    return True


def _validate_args(parser: argparse.ArgumentParser, args):
    if args.num_clients < 3:
        parser.error("--num-clients must be >= 3 for this swarm stress test")
    if args.max_concurrent_submissions < 1:
        parser.error("--max-concurrent-submissions must be >= 1")
    sites = _site_names(args.num_clients)
    if args.aggregator_site not in sites:
        parser.error(f"--aggregator-site must be one of: {', '.join(sites)}")


def main():
    parser = argparse.ArgumentParser(description="Swarm stress test with large PyTorch models")
    parser.add_argument("--model-size-gb", type=float, default=5.0, help="Model size in GB (default: 5)")
    parser.add_argument("--num-rounds", type=int, default=2, help="Number of swarm rounds (default: 2)")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients/sites (default: 3)")
    parser.add_argument(
        "--max-concurrent-submissions",
        type=int,
        default=3,
        help="Max concurrent submissions allowed on aggregation site (default: 3)",
    )
    parser.add_argument(
        "--aggregator-site",
        type=str,
        default=DEFAULT_AGGREGATOR_SITE,
        help="Fixed aggregation site name (default: site-1)",
    )
    parser.add_argument("--sample-interval", type=float, default=0.1, help="RSS/artifact sample interval (default: 0.1s)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Subprocess timeout seconds in compare mode")
    parser.add_argument(
        "--workdir",
        type=str,
        default="/tmp/nvflare/swarm_pt_stress",
        help="Simulator working directory",
    )
    parser.add_argument("--no-disk-streaming", action="store_true", help="Disable disk-based tensor streaming (baseline)")
    parser.add_argument("--compare", action="store_true", help="Run both modes and compare results")
    parser.add_argument("--require-disk-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-streaming-markers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-metric", choices=["aggregator", "tree"], default="aggregator")
    parser.add_argument("--require-memory-reduction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-memory-reduction-gb", type=float, default=0.0)
    args = parser.parse_args()

    _validate_args(parser, args)

    try:
        if args.compare:
            ok = compare_mode(args)
            sys.exit(0 if ok else 1)

        disk_streaming = not args.no_disk_streaming
        run_one(args=args, disk_streaming=disk_streaming)
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
