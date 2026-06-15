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

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Optional

SKILL_NAME = "nvflare"
SKILL_VERSION = "0.1.0"
MIN_FLARE_VERSION = "2.8.0"
SUPPORTED_AGENT_TARGETS = ("codex", "claude")
INSTALL_MANIFEST_FILE_NAME = ".nvflare_skill_install.json"
_SKILL_PACKAGE = "nvflare.tool.agent.skills"
_SKILL_FILE_NAME = "SKILL.md"


def available_skills() -> list[dict]:
    """Return the bundled NVFLARE skills available for installation."""
    return [_skill_metadata()]


def resolve_agent_target_dir(agent: str = "codex", target_dir: Optional[str | Path] = None) -> Path:
    """Resolve a supported agent name to its skill directory."""
    if target_dir:
        return Path(target_dir).expanduser().resolve(strict=False)

    if agent == "codex":
        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            return Path(codex_home).expanduser().joinpath("skills").resolve(strict=False)
        return Path.home().joinpath(".codex", "skills").resolve(strict=False)
    if agent == "claude":
        return Path.home().joinpath(".claude", "skills").resolve(strict=False)
    raise ValueError(f"unsupported agent target: {agent}")


def install_skills(
    agent: str = "codex",
    skill_name: Optional[str] = None,
    dry_run: bool = False,
    target_dir: Optional[str | Path] = None,
) -> dict:
    """Install bundled NVFLARE skills.

    This function is intentionally small and never raises. Existing provision
    and POC flows call it opportunistically, so filesystem failures are reported
    in the returned plan instead of interrupting the caller.
    """
    try:
        return _install_skills(agent=agent, skill_name=skill_name, dry_run=dry_run, target_dir=target_dir)
    except Exception as e:
        return _base_plan(agent, str(target_dir or ""), skill_name, applied=False) | {
            "errors": [_error("install_failed", str(target_dir or ""), e)]
        }


def list_skills(agent: str = "codex", target_dir: Optional[str | Path] = None) -> dict:
    """List the bundled skill and whether it is installed for the target agent."""
    try:
        target = resolve_agent_target_dir(agent, target_dir)
        installed = []
        target_skill_dir = target / SKILL_NAME
        if target_skill_dir.is_dir():
            installed.append(
                {
                    "name": SKILL_NAME,
                    "skill_version": SKILL_VERSION,
                    "target_path": str(target_skill_dir),
                    "status": "current" if _installed_skill_is_current(target_skill_dir) else "modified",
                }
            )
        elif target_skill_dir.exists():
            installed.append(
                {
                    "name": SKILL_NAME,
                    "skill_version": SKILL_VERSION,
                    "target_path": str(target_skill_dir),
                    "status": "conflict",
                }
            )

        return {
            "agent": agent,
            "target_path": str(target),
            "available": available_skills(),
            "installed": installed,
            "errors": [],
        }
    except Exception as e:
        return {
            "agent": agent,
            "target_path": str(target_dir or ""),
            "available": available_skills(),
            "installed": [],
            "errors": [_error("list_failed", str(target_dir or ""), e)],
        }


def _install_skills(agent: str, skill_name: Optional[str], dry_run: bool, target_dir: Optional[str | Path]) -> dict:
    target = resolve_agent_target_dir(agent, target_dir)
    plan = _base_plan(agent, str(target), skill_name, applied=False)

    if skill_name and skill_name != SKILL_NAME:
        plan["missing"].append(skill_name)
        return plan

    skill = _skill_metadata()
    target_skill_dir = target / SKILL_NAME
    entry = {
        "name": SKILL_NAME,
        "skill_version": SKILL_VERSION,
        "source_hash": skill["source_hash"],
        "target_path": str(target_skill_dir),
        "files": [{"target": str(target_skill_dir / _SKILL_FILE_NAME)}],
        "action": _planned_action(target_skill_dir),
        "status": "planned" if dry_run else "pending",
    }
    if entry["action"] == "skip":
        entry["status"] = "skipped"
        entry["reason"] = "already_installed"
        plan["skipped"].append(SKILL_NAME)

    plan["skills"].append(entry)

    if dry_run or plan["missing"]:
        return plan

    if entry["action"] == "skip":
        plan["applied"] = True
        return plan

    try:
        target.mkdir(parents=True, exist_ok=True)
        backup_path = None
        if target_skill_dir.exists():
            if target_skill_dir.is_symlink():
                raise ValueError(f"target skill path must not be a symlink: {target_skill_dir}")
            backup_path = _backup_existing_skill(target_skill_dir, target)
            entry["backup_path"] = str(backup_path)
            plan["backed_up"].append(str(backup_path))
        _write_skill(target_skill_dir, skill)
        entry["status"] = "installed" if entry["action"] == "copy" else "replaced"
        plan["installed"].append(SKILL_NAME)
        plan["applied"] = True
    except Exception as e:
        entry["status"] = "failed"
        error = _error("skill_install_failed", str(target_skill_dir), e)
        entry["error"] = error
        plan["errors"].append(error)
        plan["applied"] = False

    return plan


def _base_plan(agent: str, target_path: str, requested_skill: Optional[str], applied: bool) -> dict:
    return {
        "agent": agent,
        "target_path": target_path,
        "requested_skill": requested_skill,
        "available": available_skills(),
        "skills": [],
        "installed": [],
        "skipped": [],
        "backed_up": [],
        "missing": [],
        "errors": [],
        "applied": applied,
    }


def _skill_metadata() -> dict:
    return {
        "name": SKILL_NAME,
        "skill_version": SKILL_VERSION,
        "min_flare_version": MIN_FLARE_VERSION,
        "source_hash": _skill_hash(),
    }


def _skill_text() -> str:
    return resources.files(_SKILL_PACKAGE).joinpath(SKILL_NAME, _SKILL_FILE_NAME).read_text(encoding="utf-8")


def _skill_hash() -> str:
    return hashlib.sha256(_skill_text().encode("utf-8")).hexdigest()


def _planned_action(target_skill_dir: Path) -> str:
    if not target_skill_dir.exists():
        return "copy"
    if target_skill_dir.is_dir() and _installed_skill_is_current(target_skill_dir):
        return "skip"
    return "replace"


def _installed_skill_is_current(target_skill_dir: Path) -> bool:
    skill_file = target_skill_dir / _SKILL_FILE_NAME
    if not skill_file.is_file():
        return False
    try:
        return skill_file.read_text(encoding="utf-8") == _skill_text()
    except OSError:
        return False


def _backup_existing_skill(target_skill_dir: Path, target_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup_dir = target_root / ".nvflare_skill_bak" / f"{timestamp}-{time.time_ns()}"
    backup_path = backup_dir / target_skill_dir.name
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.move(str(target_skill_dir), str(backup_path))
    return backup_path


def _write_skill(target_skill_dir: Path, skill: dict) -> None:
    target_skill_dir.mkdir(parents=True, exist_ok=False)
    target_skill_dir.joinpath(_SKILL_FILE_NAME).write_text(_skill_text(), encoding="utf-8")
    target_skill_dir.joinpath(INSTALL_MANIFEST_FILE_NAME).write_text(
        json.dumps(
            {
                "schema_version": "1",
                "managed_by": "nvflare",
                "name": SKILL_NAME,
                "skill_version": SKILL_VERSION,
                "source_hash": skill["source_hash"],
                "installed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _error(code: str, target: str, error: Exception) -> dict:
    return {
        "code": code,
        "target": target,
        "type": type(error).__name__,
        "message": str(error),
    }
