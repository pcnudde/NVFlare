#!/usr/bin/env python3
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

"""Generate an HTML dashboard from agent skill eval run artifacts."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any


MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "agent_skill_eval_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
XDG_CACHE_HOME = Path(os.environ.get("TMPDIR", "/tmp")) / "agent_skill_eval_cache"
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

MODEL_COSTS_FILE = Path(__file__).with_name("model_costs.yaml")
TOKEN_KEYS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
    "cached_tokens",
    "reasoning_output_tokens",
    "reasoning_tokens",
}
COST_KEYS = {"cost_usd", "total_cost_usd", "cost_usd_estimate"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an HTML report from agent_skill_eval run artifacts.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directory or directories, for example agent_skill_eval/runs/<timestamp>.",
    )
    parser.add_argument("--output", type=Path, help="Output HTML path.")
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in args.run_dirs]
    missing = [path for path in run_dirs if not path.exists()]
    if missing:
        raise SystemExit(f"Run directory does not exist: {missing[0]}")

    output = args.output.resolve() if args.output else default_output(run_dirs)
    rows = load_runs(run_dirs)
    aggregates = aggregate_rows(rows)
    output.write_text(render_report(run_dirs, rows, aggregates))
    print(f"Wrote {output}")
    return 0


def default_output(run_dirs: list[Path]) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0] / "report.html"
    return run_dirs[0].parent / "combined_report.html"


def load_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    model_costs = load_model_costs()
    for run_dir in run_dirs:
        for result_path in sorted(run_dir.glob("*/*/run_*/result.json")):
            result = read_json(result_path)
            analysis = read_nested_json(result, "analysis", "analysis_file") or result.get("analysis", {}).get("parsed") or {}
            grade = read_nested_json(result, "grade", "grade_file") or result.get("grade", {}).get("parsed") or {}
            evidence = result.get("evidence", [])
            agent_usage = merged_agent_usage(result, result_path)
            agent_cost = agent_cost_value(result, agent_usage, model_costs)
            row = {
                "run_dir": run_dir,
                "path": result_path,
                "cohort_label": cohort_label(run_dir, result),
                "testcase_id": result.get("testcase_id", ""),
                "agent_id": result.get("agent", {}).get("id", ""),
                "agent_label": result.get("agent", {}).get("label", ""),
                "run_index": result.get("run_index", ""),
                "score": grade.get("score", ""),
                "score_before_caps": grade.get("score_before_caps", ""),
                "caps_applied": grade.get("caps_applied", []),
                "summary": grade.get("summary", ""),
                "flare_version_used": active_flare_version(evidence),
                "achieved_accuracy": analysis.get("achieved_accuracy", ""),
                "run_summary_bullets": analysis.get("run_summary_bullets", []),
                "testcase_improvement_recommendations": analysis.get(
                    "testcase_improvement_recommendations", []
                ),
                "interesting_observations": analysis.get("interesting_observations", []),
                "agent_duration_seconds": result.get("agent_duration_seconds", ""),
                "total_duration_seconds": result.get("duration_seconds", ""),
                "agent_tokens": token_total(agent_usage),
                "agent_input_tokens": input_token_total(agent_usage),
                "agent_output_tokens": output_token_total(agent_usage),
                "agent_cache_tokens": cache_token_total(agent_usage),
                "agent_cost_usd": agent_cost,
                "container": result.get("agent_container", {}).get("name", ""),
                "kept_container": result.get("agent_container", {}).get("kept", ""),
                "workdir": result.get("workdir", ""),
                "evidence": evidence,
            }
            row["achieved_accuracy"] = row["achieved_accuracy"] or infer_accuracy(row)
            rows.append(row)
    rows = dedupe_latest_rows(rows)
    return sorted(rows, key=lambda row: (row["cohort_label"], row["testcase_id"], row["agent_id"], row["run_index"]))


def dedupe_latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_slot: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("cohort_label") or ""),
            str(row.get("testcase_id") or ""),
            str(row.get("agent_id") or ""),
            str(row.get("run_index") or ""),
        )
        current = by_slot.get(key)
        if current is None or row_sort_time(row) >= row_sort_time(current):
            by_slot[key] = row
    return list(by_slot.values())


def row_sort_time(row: dict[str, Any]) -> tuple[str, float]:
    run_dir = row.get("run_dir")
    path = row.get("path")
    run_dir_name = run_dir.name if isinstance(run_dir, Path) else ""
    mtime = path.stat().st_mtime if isinstance(path, Path) and path.exists() else 0.0
    return run_dir_name, mtime


def cohort_label(run_dir: Path, result: dict[str, Any]) -> str:
    image = docker_image_from_result(result)
    return image or run_dir.name


def docker_image_from_result(result: dict[str, Any]) -> str:
    command = result.get("agent_container", {}).get("start", {}).get("command", [])
    if not isinstance(command, list):
        return ""
    for index, part in enumerate(command):
        if part == "sleep" and index > 0:
            return str(command[index - 1])
    return ""


def read_nested_json(result: dict[str, Any], section: str, file_key: str) -> dict[str, Any] | None:
    path_text = result.get(section, {}).get(file_key)
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    return read_json(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def load_model_costs() -> dict[str, Any]:
    if not MODEL_COSTS_FILE.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return {}
    data = yaml.safe_load(MODEL_COSTS_FILE.read_text()) or {}
    return data if isinstance(data, dict) else {}


def merged_agent_usage(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    usage = dict(result.get("token_usage", {}).get("agent", {}) or {})
    logs_dir = result_path.parent / "logs"
    for log_name in ["agent_stdout.txt", "agent_stderr.txt"]:
        path = logs_dir / log_name
        if path.exists():
            collect_usage_metrics_from_text(path.read_text(errors="replace"), usage)
    return usage


def collect_usage_metrics_from_text(text: str, usage: dict[str, Any]) -> None:
    for value in iter_json_values(text):
        collect_usage_metrics(value, usage)


def iter_json_values(text: str) -> list[Any]:
    values = []
    stripped = text.strip()
    if stripped:
        try:
            values.append(json.loads(stripped))
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] not in "[{":
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values


def collect_usage_metrics(value: Any, usage: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TOKEN_KEYS | COST_KEYS and isinstance(item, (int, float)):
                current = usage.get(key)
                usage[key] = max(float(current), float(item)) if isinstance(current, (int, float)) else item
            collect_usage_metrics(item, usage)
    elif isinstance(value, list):
        for item in value:
            collect_usage_metrics(item, usage)


def token_total(usage: dict[str, Any]) -> int | str:
    for key in ["total_tokens", "total_tokens_estimate"]:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return ""


def input_token_total(usage: dict[str, Any]) -> int | str:
    input_tokens = numeric_usage_value(usage, "input_tokens", "prompt_tokens")
    cached_tokens = numeric_usage_value(usage, "cached_input_tokens", "cached_tokens") or 0
    if input_tokens is None:
        return ""
    return int(max(input_tokens - cached_tokens, 0))


def output_token_total(usage: dict[str, Any]) -> int | str:
    output_tokens = numeric_usage_value(usage, "output_tokens", "completion_tokens")
    return int(output_tokens) if output_tokens is not None else ""


def cache_token_total(usage: dict[str, Any]) -> int | str:
    cache_tokens = (
        (numeric_usage_value(usage, "cached_input_tokens", "cached_tokens") or 0)
        + (numeric_usage_value(usage, "cache_creation_input_tokens") or 0)
        + (numeric_usage_value(usage, "cache_read_input_tokens") or 0)
    )
    return int(cache_tokens) if cache_tokens else ""


def numeric_usage_value(usage: dict[str, Any], *keys: str) -> float | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def cost_value(usage: dict[str, Any]) -> float | str:
    if not isinstance(usage, dict):
        return ""
    for key in ["total_cost_usd", "cost_usd", "cost_usd_estimate"]:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return ""


def total_cost(result: dict[str, Any]) -> float | str:
    total = 0.0
    found = False
    usage_by_role = result.get("token_usage", {})
    if not isinstance(usage_by_role, dict):
        return ""
    for usage in usage_by_role.values():
        value = cost_value(usage)
        if isinstance(value, (int, float)):
            total += float(value)
            found = True
    return total if found else ""


def agent_cost_value(result: dict[str, Any], agent_usage: dict[str, Any], model_costs: dict[str, Any]) -> float | str:
    reported = cost_value(agent_usage)
    if isinstance(reported, (int, float)) and agent_usage.get("cost_source") == "reported":
        return reported

    pricing = find_model_pricing(
        model_costs,
        model_from_command(result.get("agent", {}).get("command", [])) or str(agent_usage.get("cost_model") or ""),
        result.get("agent", {}).get("id"),
    )
    estimate = calculate_configured_cost(agent_usage, pricing)
    if estimate is not None:
        return estimate
    return reported


def model_from_command(command: Any) -> str | None:
    if not isinstance(command, list):
        return None
    for index, part in enumerate(command):
        if part == "--model" and index + 1 < len(command):
            return command[index + 1]
        if isinstance(part, str) and part.startswith("--model="):
            return part.split("=", 1)[1]
    return None


def find_model_pricing(model_costs: dict[str, Any], model_name: str | None, agent_id: str | None) -> dict[str, Any] | None:
    models = model_costs.get("models")
    if not isinstance(models, dict):
        return None
    if model_name in models and isinstance(models[model_name], dict):
        return models[model_name]
    for name, config in models.items():
        if not isinstance(config, dict):
            continue
        aliases = config.get("aliases", [])
        if isinstance(aliases, list) and (model_name in aliases or agent_id in aliases):
            config = dict(config)
            config.setdefault("model", name)
            return config
    return None


def calculate_configured_cost(usage: dict[str, Any], pricing: dict[str, Any] | None) -> float | None:
    if not pricing:
        return None
    rates = pricing.get("rates")
    if not isinstance(rates, dict):
        return None

    input_tokens = numeric_usage_value(usage, "input_tokens", "prompt_tokens") or 0
    output_tokens = numeric_usage_value(usage, "output_tokens", "completion_tokens") or 0
    cached_tokens = numeric_usage_value(usage, "cached_input_tokens", "cached_tokens") or 0
    cache_creation_tokens = numeric_usage_value(usage, "cache_creation_input_tokens") or 0
    cache_read_tokens = numeric_usage_value(usage, "cache_read_input_tokens") or 0

    cost = 0.0
    priced = False
    input_rate = rate_value(rates, "input_tokens", "prompt_tokens")
    cached_input_rate = rate_value(rates, "cached_input_tokens")

    if cached_input_rate is not None and cached_tokens:
        uncached_input_tokens = max(input_tokens - cached_tokens, 0)
        if input_rate is not None:
            cost += uncached_input_tokens * input_rate / 1_000_000
            priced = True
        cost += cached_tokens * cached_input_rate / 1_000_000
        priced = True
    elif input_rate is not None and input_tokens:
        cost += input_tokens * input_rate / 1_000_000
        priced = True

    for token_count, *rate_keys in [
        (output_tokens, "output_tokens", "completion_tokens"),
        (cache_creation_tokens, "cache_creation_input_tokens"),
        (cache_read_tokens, "cache_read_input_tokens"),
    ]:
        rate = rate_value(rates, *rate_keys)
        if rate is not None and token_count:
            cost += token_count * rate / 1_000_000
            priced = True
    return cost if priced else None


def rate_value(rates: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = rates.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def active_flare_version(evidence_records: list[dict[str, Any]]) -> str:
    for evidence in evidence_records:
        version = evidence.get("nvflare_version")
        if version:
            return str(version)
        error = evidence.get("nvflare_version_error")
        if error:
            return f"error: {error}"
        probe = evidence.get("nvflare_version_probe")
        if isinstance(probe, dict):
            version = probe.get("distribution_version") or probe.get("nvflare_version")
            if version:
                return str(version)
            error = probe.get("error")
            if error:
                return f"error: {error}"
    return ""


def infer_accuracy(row: dict[str, Any]) -> str:
    text = "\n".join(evidence_text(evidence) for evidence in row.get("evidence", []))
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    values = []
    for pattern in [
        r"accuracy\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
        r"accuracy\s+([0-9]+(?:\.[0-9]+)?)",
        r"validation metric\s+([0-9]+(?:\.[0-9]+)?)",
    ]:
        values.extend(float(match) for match in re.findall(pattern, text, flags=re.IGNORECASE))
    if not values:
        return ""
    return f"{values[-1]:.4f}"


def evidence_text(evidence: dict[str, Any]) -> str:
    parts = [str(evidence.get("stdout_tail") or ""), str(evidence.get("stderr_tail") or "")]
    for key in ["stdout_path", "stderr_path"]:
        path_text = evidence.get(key)
        if path_text and Path(path_text).exists():
            parts.append(Path(path_text).read_text(errors="replace")[-8000:])
    return "\n".join(parts)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["cohort_label"],
            row["testcase_id"],
            row["agent_id"],
            row.get("agent_label") or row["agent_id"],
        )
        groups.setdefault(key, []).append(row)

    aggregates = []
    multiple_testcases = len({row["testcase_id"] for row in rows}) > 1
    multiple_cohorts = len({row["cohort_label"] for row in rows}) > 1
    for (cohort, testcase_id, agent_id, agent_label), group in groups.items():
        label_parts = []
        if multiple_cohorts:
            label_parts.append(cohort)
        label_parts.append(agent_id)
        if multiple_testcases:
            label_parts.append(testcase_id)
        aggregate = {
            "label": " / ".join(label_parts),
            "cohort_label": cohort,
            "testcase_id": testcase_id,
            "agent_id": agent_id,
            "agent_label": agent_label,
            "runs": len(group),
            "score": stats(group, "score"),
            "duration": stats(group, "agent_duration_seconds"),
            "tokens": stats(group, "agent_tokens"),
            "input_tokens": stats(group, "agent_input_tokens"),
            "output_tokens": stats(group, "agent_output_tokens"),
            "cache_tokens": stats(group, "agent_cache_tokens"),
            "total_cost": stats(group, "agent_cost_usd"),
            "agent_cost": stats(group, "agent_cost_usd"),
            "accuracies": sorted({row.get("achieved_accuracy") for row in group if row.get("achieved_accuracy")}),
            "flare_versions": sorted({row.get("flare_version_used") for row in group if row.get("flare_version_used")}),
        }
        aggregates.append(aggregate)
    return sorted(
        aggregates,
        key=lambda row: (
            row["score"]["avg"] is None,
            -(row["score"]["avg"] or -1),
            row["cohort_label"],
            row["agent_id"],
            row["testcase_id"],
        ),
    )


def stats(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [value for value in (to_float(row.get(field)) for row in rows) if value is not None]
    if not values:
        return {"avg": None, "min": None, "max": None}
    return {"avg": sum(values) / len(values), "min": min(values), "max": max(values)}


def render_report(run_dirs: list[Path], rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    chart = render_chart(aggregates)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Skill Eval Report</title>
  <style>{styles()}</style>
</head>
<body>
  <header>
    <div>
      <h1>Agent Skill Eval Report</h1>
      <div class="meta">
        <span>{len(run_dirs)} run {'directory' if len(run_dirs) == 1 else 'directories'}</span>
        <span>{len(rows)} agent runs</span>
        <span>Generated {escape(generated)}</span>
      </div>
    </div>
  </header>
  <main>
    {render_cards(rows, aggregates)}
    {render_sources(run_dirs)}
    {render_comparison(rows, aggregates)}
    {chart}
    {render_aggregate_table(aggregates)}
    {render_run_cards(rows)}
  </main>
</body>
</html>
"""


def styles() -> str:
    return """
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #667085;
      --line: #d7dce3;
      --accent: #2563eb;
      --soft: #eef2f7;
      --good: #087443;
      --warn: #a15c00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 26px 32px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0 0 6px; font-size: 25px; letter-spacing: 0; }
    h2 { margin: 26px 0 10px; font-size: 18px; letter-spacing: 0; }
    .meta { color: var(--muted); display: flex; gap: 16px; flex-wrap: wrap; }
    main { padding: 22px 32px 42px; max-width: 1480px; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .kpi, .panel, .run-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 7px;
    }
    .kpi { padding: 12px 14px; }
    .kpi b { display: block; font-size: 23px; line-height: 1.15; }
    .kpi span { color: var(--muted); }
    .panel { padding: 14px; margin: 14px 0; }
    .chart img { display: block; width: 100%; height: auto; }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }
    th { color: #344054; background: var(--soft); font-weight: 650; }
    tr:last-child td { border-bottom: 0; }
    .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
    .score { color: var(--accent); font-weight: 750; }
    .delta-good { color: var(--good); font-weight: 700; }
    .delta-bad { color: #b42318; font-weight: 700; }
    .muted { color: var(--muted); }
    .sources { font-size: 12px; color: var(--muted); }
    .sources code, code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: #f0f3f8;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
    }
    .run-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 12px;
    }
    .run-card { padding: 13px 14px; }
    .run-title { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; margin-bottom: 8px; }
    .run-title b { font-size: 15px; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #fbfcfe;
      color: #344054;
      font-size: 12px;
    }
    ul { margin: 8px 0 0; padding-left: 18px; }
    li + li { margin-top: 4px; }
    details { margin-top: 8px; }
    summary { cursor: pointer; color: var(--accent); font-weight: 650; }
    """


def render_cards(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> str:
    scores = [value for value in (to_float(row["score"]) for row in rows) if value is not None]
    durations = [value for value in (to_float(row["agent_duration_seconds"]) for row in rows) if value is not None]
    tokens = [value for value in (to_float(row["agent_tokens"]) for row in rows) if value is not None]
    costs = [value for value in (to_float(row["agent_cost_usd"]) for row in rows) if value is not None]
    best = max(scores) if scores else None
    return f"""
    <section class="kpis">
      <div class="kpi"><b>{len(aggregates)}</b><span>agent groups</span></div>
      <div class="kpi"><b>{len(rows)}</b><span>runs</span></div>
      <div class="kpi"><b>{format_stat(avg(scores))}</b><span>avg score</span></div>
      <div class="kpi"><b>{format_stat(best)}</b><span>best score</span></div>
      <div class="kpi"><b>{format_stat(avg(durations))}</b><span>avg agent sec</span></div>
      <div class="kpi"><b>{format_int(avg(tokens))}</b><span>avg agent tokens</span></div>
      <div class="kpi"><b>{format_money(avg(costs))}</b><span>avg agent cost</span></div>
    </section>
"""


def render_sources(run_dirs: list[Path]) -> str:
    items = "".join(f"<div><code>{escape(str(path))}</code></div>" for path in run_dirs)
    return f'<section class="panel sources"><b>Sources</b>{items}</section>'


def render_chart(aggregates: list[dict[str, Any]]) -> str:
    image = chart_png_base64(aggregates)
    if not image:
        return '<section class="panel chart muted">No chartable aggregate data yet.</section>'
    return f'<section class="panel chart"><img alt="Score, duration, and token charts" src="data:image/png;base64,{image}"></section>'


def render_comparison(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> str:
    cohort_order = []
    for row in rows:
        cohort = row.get("cohort_label")
        if cohort and cohort not in cohort_order:
            cohort_order.append(cohort)
    if len(cohort_order) < 2:
        return ""

    baseline, candidate = cohort_order[:2]
    by_key = {(row["cohort_label"], row["testcase_id"], row["agent_id"]): row for row in aggregates}
    testcase_ids = sorted({row["testcase_id"] for row in aggregates})
    agent_ids = sorted({row["agent_id"] for row in aggregates})
    compare_rows = []
    for testcase_id in testcase_ids:
        for agent_id in agent_ids:
            left = by_key.get((baseline, testcase_id, agent_id))
            right = by_key.get((candidate, testcase_id, agent_id))
            if not left or not right:
                continue
            label = agent_id if len(testcase_ids) == 1 else f"{agent_id} / {testcase_id}"
            compare_rows.append((label, left, right))
    if not compare_rows:
        return ""

    chart = comparison_chart_png_base64(baseline, candidate, compare_rows)
    body = "\n".join(render_comparison_row(agent_id, left, right) for agent_id, left, right in compare_rows)
    chart_html = (
        f'<div class="chart"><img alt="Original versus skills score, duration, and token comparison charts" src="data:image/png;base64,{chart}"></div>'
        if chart
        else ""
    )
    return f"""
    <h2>Run Comparison</h2>
    <section class="panel">
      <div class="muted">Baseline <code>{escape(baseline)}</code> compared with <code>{escape(candidate)}</code>.</div>
      {chart_html}
      <table>
        <thead>
          <tr>
            <th>Agent</th>
            <th>Score avg</th>
            <th>Score delta</th>
            <th>Duration avg</th>
            <th>Duration delta</th>
            <th>Tokens avg</th>
            <th>Tokens delta</th>
            <th>Agent cost avg</th>
            <th>Agent cost delta</th>
          </tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </section>
"""


def render_comparison_row(label: str, left: dict[str, Any], right: dict[str, Any]) -> str:
    left_score = left["score"]["avg"]
    right_score = right["score"]["avg"]
    left_duration = left["duration"]["avg"]
    right_duration = right["duration"]["avg"]
    left_tokens = left["tokens"]["avg"]
    right_tokens = right["tokens"]["avg"]
    left_cost = left["total_cost"]["avg"]
    right_cost = right["total_cost"]["avg"]
    return f"""
        <tr>
          <td><b>{escape(label)}</b></td>
          <td class="num">{format_stat(left_score)} -> <span class="score">{format_stat(right_score)}</span></td>
          <td class="num {delta_class(delta(left_score, right_score))}">{format_delta(delta(left_score, right_score))}</td>
          <td class="num">{format_stat(left_duration)} -> {format_stat(right_duration)}</td>
          <td class="num {delta_class(delta(left_duration, right_duration), lower_is_better=True)}">{format_delta(delta(left_duration, right_duration), suffix='s')}</td>
          <td class="num">{format_int(left_tokens)} -> {format_int(right_tokens)}</td>
          <td class="num {delta_class(delta(left_tokens, right_tokens), lower_is_better=True)}">{format_delta(delta(left_tokens, right_tokens), compact=True)}</td>
          <td class="num">{format_money(left_cost)} -> {format_money(right_cost)}</td>
          <td class="num {delta_class(delta(left_cost, right_cost), lower_is_better=True)}">{format_money_delta(delta(left_cost, right_cost))}</td>
        </tr>
"""


def comparison_chart_png_base64(
    baseline: str, candidate: str, compare_rows: list[tuple[str, dict[str, Any], dict[str, Any]]]
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    labels = [agent_id for agent_id, _left, _right in compare_rows]
    x = list(range(len(labels)))
    width = 0.36

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), dpi=180)
    left_positions = [value - width / 2 for value in x]
    right_positions = [value + width / 2 for value in x]
    metrics = [
        ("score", "Average Score", (0, 100), lambda value, _pos: f"{value:.0f}"),
        ("duration", "Average Agent Duration (s)", None, lambda value, _pos: f"{value:.0f}"),
        ("total_cost", "Average Agent Cost (USD)", None, lambda value, _pos: money_tick(value)),
    ]

    for axis, (field, title, ylim, formatter) in zip(axes[:3], metrics):
        baseline_values = [left[field]["avg"] or 0 for _agent_id, left, _right in compare_rows]
        candidate_values = [right[field]["avg"] or 0 for _agent_id, _left, right in compare_rows]
        bars_left = axis.bar(left_positions, baseline_values, width, label=baseline, color="#64748b", alpha=0.86)
        bars_right = axis.bar(right_positions, candidate_values, width, label=candidate, color="#2563eb", alpha=0.9)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=34, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
        axis.yaxis.set_major_formatter(FuncFormatter(formatter))
        if ylim:
            axis.set_ylim(*ylim)
        else:
            max_value = max(baseline_values + candidate_values) if baseline_values or candidate_values else 0
            axis.set_ylim(0, max(max_value * 1.18, 1))
        axis.bar_label(bars_left, labels=[formatter(value, None) for value in baseline_values], padding=3, fontsize=7)
        axis.bar_label(bars_right, labels=[formatter(value, None) for value in candidate_values], padding=3, fontsize=7)

    axes[0].legend(frameon=False, fontsize=8)
    draw_grouped_stacked_tokens(
        axes[3],
        labels,
        compare_rows,
        baseline,
        candidate,
        left_positions,
        right_positions,
        width,
    )
    fig.suptitle("Original vs Skills: average columns by agent", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def draw_grouped_stacked_tokens(
    axis: Any,
    labels: list[str],
    compare_rows: list[tuple[str, dict[str, Any], dict[str, Any]]],
    baseline: str,
    candidate: str,
    left_positions: list[float],
    right_positions: list[float],
    width: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    colors = {
        "input": "#64748b",
        "output": "#2563eb",
        "cache": "#f59e0b",
    }
    stacks = [
        ("input", "Input", "input_tokens"),
        ("output", "Output", "output_tokens"),
        ("cache", "Cache", "cache_tokens"),
    ]
    max_total = 0.0
    for positions, side, alpha in [(left_positions, 1, 0.72), (right_positions, 2, 0.92)]:
        bottoms = [0.0] * len(compare_rows)
        for key, label, field in stacks:
            values = []
            for _agent_id, left, right in compare_rows:
                aggregate = left if side == 1 else right
                values.append(aggregate[field]["avg"] or 0)
            axis.bar(
                positions,
                values,
                width,
                bottom=bottoms,
                color=colors[key],
                alpha=alpha,
                label=f"{label} ({baseline if side == 1 else candidate})" if key == "input" else None,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        max_total = max(max_total, max(bottoms) if bottoms else 0)
    axis.set_title("Average Agent Tokens", fontsize=11, fontweight="bold")
    axis.set_xticks(list(range(len(labels))))
    axis.set_xticklabels(labels, rotation=34, ha="right")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: compact_number(value)))
    axis.set_ylim(0, max(max_total * 1.18, 1))
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["input"], alpha=0.88),
        plt.Rectangle((0, 0), 1, 1, color=colors["output"], alpha=0.88),
        plt.Rectangle((0, 0), 1, 1, color=colors["cache"], alpha=0.88),
    ]
    axis.legend(legend_handles, ["Input", "Output", "Cache"], frameon=False, fontsize=7, loc="upper right")


def chart_png_base64(aggregates: list[dict[str, Any]]) -> str:
    chart_rows = [row for row in aggregates if row["score"]["avg"] is not None]
    if not chart_rows:
        return ""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    labels = [row["label"] for row in chart_rows]
    x = list(range(len(chart_rows)))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), dpi=180)
    metrics = [
        ("score", "Score", "#2563eb", (0, 100), lambda value, _pos: f"{value:.0f}"),
        ("duration", "Agent Duration (s)", "#0f766e", None, lambda value, _pos: f"{value:.0f}"),
        ("total_cost", "Agent Cost (USD)", "#b45309", None, lambda value, _pos: money_tick(value)),
    ]

    for axis, (field, title, color, xlim, formatter) in zip([axes[0], axes[1], axes[3]], metrics):
        avg_values = [row[field]["avg"] or 0 for row in chart_rows]
        min_values = [row[field]["min"] if row[field]["min"] is not None else row[field]["avg"] or 0 for row in chart_rows]
        max_values = [row[field]["max"] if row[field]["max"] is not None else row[field]["avg"] or 0 for row in chart_rows]
        yerr = [
            [max(avg - min_value, 0) for avg, min_value in zip(avg_values, min_values)],
            [max(max_value - avg, 0) for avg, max_value in zip(avg_values, max_values)],
        ]
        bars = axis.bar(x, avg_values, yerr=yerr, color=color, alpha=0.88, capsize=3, width=0.68)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=34, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
        axis.yaxis.set_major_formatter(FuncFormatter(formatter))
        if xlim:
            axis.set_ylim(*xlim)
        elif avg_values:
            axis.set_ylim(0, max(max(max_values) * 1.18, 1))
        if len(labels) <= 8:
            axis.bar_label(bars, labels=[formatter(value, None) for value in avg_values], padding=3, fontsize=7)

    draw_stacked_tokens(axes[2], labels, chart_rows, x)
    fig.suptitle("Aggregate Results: average columns with min/max whiskers", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def draw_stacked_tokens(axis: Any, labels: list[str], rows: list[dict[str, Any]], x: list[int]) -> None:
    from matplotlib.ticker import FuncFormatter

    from matplotlib.ticker import FuncFormatter

    colors = {
        "input_tokens": "#64748b",
        "output_tokens": "#2563eb",
        "cache_tokens": "#f59e0b",
    }
    stacks = [
        ("input_tokens", "Input"),
        ("output_tokens", "Output"),
        ("cache_tokens", "Cache"),
    ]
    bottoms = [0.0] * len(rows)
    max_total = 0.0
    for field, label in stacks:
        values = [row[field]["avg"] or 0 for row in rows]
        axis.bar(x, values, bottom=bottoms, color=colors[field], alpha=0.88, width=0.68, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        max_total = max(max_total, max(bottoms) if bottoms else 0)
    axis.set_title("Agent Tokens", fontsize=11, fontweight="bold")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=34, ha="right")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: compact_number(value)))
    axis.set_ylim(0, max(max_total * 1.18, 1))
    if len(labels) <= 8:
        axis.bar_label(
            axis.containers[-1],
            labels=[compact_number(value) if value else "" for value in bottoms],
            padding=3,
            fontsize=7,
        )
    axis.legend(frameon=False, fontsize=7, loc="upper right")


def render_aggregate_table(aggregates: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td><b>{escape(row['label'])}</b><br><span class="muted">{escape(row['agent_label'])}</span></td>
          <td class="num">{row['runs']}</td>
          <td class="num score">{format_triplet(row['score'])}</td>
          <td class="num">{format_triplet(row['duration'])}</td>
          <td class="num">{format_triplet(row['tokens'], integer=True)}</td>
          <td class="num">{format_triplet(row['total_cost'], money=True)}</td>
          <td>{escape(', '.join(row['flare_versions']))}</td>
          <td>{escape(', '.join(row['accuracies']))}</td>
        </tr>
"""
        for row in aggregates
    )
    return f"""
    <h2>Overview</h2>
    <section class="panel">
      <table>
        <thead>
          <tr>
            <th>Agent</th>
            <th>Runs</th>
            <th>Score avg/min/max</th>
            <th>Duration avg/min/max</th>
            <th>Tokens avg/min/max</th>
            <th>Agent cost avg/min/max</th>
            <th>FLARE</th>
            <th>Accuracy</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
"""


def render_run_cards(rows: list[dict[str, Any]]) -> str:
    cards = "\n".join(render_run_card(row) for row in rows)
    return f"""
    <h2>Run Notes</h2>
    <section class="run-grid">{cards}</section>
"""


def render_run_card(row: dict[str, Any]) -> str:
    run_name = f"{row['agent_id']} run {row['run_index']}"
    bullets = row.get("run_summary_bullets") or [row.get("summary") or "No analysis summary available."]
    recommendations = row.get("testcase_improvement_recommendations") or []
    observations = row.get("interesting_observations") or []
    caps = row.get("caps_applied") or []
    return f"""
    <article class="run-card">
      <div class="run-title">
        <b>{escape(run_name)}</b>
        <span class="score">{escape(str(row.get('score') or ''))}</span>
      </div>
      <div class="muted">{escape(str(row.get('cohort_label') or ''))} · {escape(str(row.get('testcase_id') or ''))}</div>
      <div class="chips">
        <span class="chip">FLARE {escape(str(row.get('flare_version_used') or 'unknown'))}</span>
        <span class="chip">accuracy {escape(str(row.get('achieved_accuracy') or 'none'))}</span>
        <span class="chip">{escape(str(row.get('agent_duration_seconds') or ''))}s</span>
        <span class="chip">{escape(format_int(to_float(row.get('agent_tokens'))))} tokens</span>
        <span class="chip">agent cost {escape(format_money(to_float(row.get('agent_cost_usd'))))}</span>
      </div>
      <ul>{''.join(f'<li>{escape(str(item))}</li>' for item in bullets[:5])}</ul>
      {render_details('Recommendations', recommendations)}
      {render_details('Observations', observations)}
      {render_details('Caps', caps)}
    </article>
"""


def render_details(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    return f"""
      <details>
        <summary>{escape(title)}</summary>
        <ul>{''.join(f'<li>{escape(str(item))}</li>' for item in items)}</ul>
      </details>
"""


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def to_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def format_stat(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}" if value % 1 else str(int(value))


def format_int(value: float | None) -> str:
    if value is None:
        return ""
    return compact_number(value)


def compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(int(value))


def format_triplet(values: dict[str, float | None], integer: bool = False, money: bool = False) -> str:
    items = [values.get("avg"), values.get("min"), values.get("max")]
    if all(item is None for item in items):
        return ""
    if money:
        return " / ".join(format_money(item) for item in items)
    if integer:
        return " / ".join(format_int(item) for item in items)
    return " / ".join(format_stat(item) for item in items)


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return right - left


def delta_class(value: float | None, lower_is_better: bool = False) -> str:
    if value is None or value == 0:
        return ""
    good = value < 0 if lower_is_better else value > 0
    return "delta-good" if good else "delta-bad"


def format_delta(value: float | None, suffix: str = "", compact: bool = False) -> str:
    if value is None:
        return ""
    sign = "+" if value > 0 else ""
    if compact:
        return f"{sign}{compact_number(value)}"
    return f"{sign}{format_stat(value)}{suffix}"


def format_money(value: float | None) -> str:
    if value is None:
        return ""
    prefix = "-" if value < 0 else ""
    value = abs(value)
    if abs(value) < 0.01:
        return f"{prefix}${value:.4f}"
    return f"{prefix}${value:.2f}"


def format_money_delta(value: float | None) -> str:
    if value is None:
        return ""
    sign = "+" if value > 0 else ""
    return f"{sign}{format_money(value)}"


def money_tick(value: float) -> str:
    if abs(value) < 0.01:
        return f"${value:.3f}"
    return f"${value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
