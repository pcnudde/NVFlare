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

"""Minimal agent-facing skill installer CLI."""

import argparse
import sys
from typing import Optional

from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.tool.install_skills import SUPPORTED_AGENT_TARGETS, install_skills, list_skills

CMD_AGENT = "agent"
CMD_AGENT_SKILLS = "skills"
CMD_AGENT_SKILLS_INSTALL = "install"
CMD_AGENT_SKILLS_LIST = "list"

_AGENT_EXAMPLES = [
    "nvflare agent skills install --agent codex --dry-run --format json",
    "nvflare agent skills install --agent claude --target /tmp/skills --format json",
    "nvflare agent skills list --agent codex --format json",
]
_AGENT_OUTPUT_MODES = ["json"]
_agent_parser: Optional[argparse.ArgumentParser] = None
_agent_skills_parser: Optional[argparse.ArgumentParser] = None
_agent_skills_sub_cmd_parsers = {}


def def_agent_cli_parser(sub_cmd) -> dict:
    """Register the top-level `nvflare agent` command group."""
    global _agent_parser
    global _agent_skills_parser

    parser = sub_cmd.add_parser(
        CMD_AGENT,
        description="Minimal agent-facing NVFLARE skill installer.",
        help="install NVFLARE agent skills",
    )
    parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    agent_subparser = parser.add_subparsers(title="agent subcommands", metavar="", dest="agent_sub_cmd")

    skills_parser = agent_subparser.add_parser(
        CMD_AGENT_SKILLS,
        description="Install and list NVFLARE-owned agent skills.",
        help="install and list NVFLARE-owned agent skills",
    )
    skills_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    skills_subparser = skills_parser.add_subparsers(
        title="agent skills subcommands", metavar="", dest="agent_skills_sub_cmd"
    )

    install_parser = skills_subparser.add_parser(
        CMD_AGENT_SKILLS_INSTALL,
        description="Install bundled NVFLARE skills into a local agent skill directory.",
        help="install bundled NVFLARE skills",
    )
    _add_agent_target_args(install_parser)
    install_parser.add_argument("--skill", help="install one skill by name; omit to install all bundled skills")
    install_parser.add_argument("--dry-run", action="store_true", help="show the install plan without copying files")
    install_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")

    list_parser = skills_subparser.add_parser(
        CMD_AGENT_SKILLS_LIST,
        description="List bundled and installed NVFLARE skills for an agent target.",
        help="list bundled NVFLARE skills",
    )
    _add_agent_target_args(list_parser)
    list_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")

    _agent_parser = parser
    _agent_skills_parser = skills_parser
    _agent_skills_sub_cmd_parsers[CMD_AGENT_SKILLS_INSTALL] = install_parser
    _agent_skills_sub_cmd_parsers[CMD_AGENT_SKILLS_LIST] = list_parser
    return {CMD_AGENT: parser}


def _add_agent_target_args(parser) -> None:
    parser.add_argument(
        "--agent",
        choices=list(SUPPORTED_AGENT_TARGETS),
        required=True,
        help="agent skill target to manage",
    )
    parser.add_argument("--target", help="override the resolved agent skill directory")


def handle_agent_cmd(args) -> None:
    from nvflare.tool.cli_output import output_error_message
    from nvflare.tool.cli_schema import handle_schema_flag

    agent_sub_cmd = getattr(args, "agent_sub_cmd", None)
    if agent_sub_cmd is None:
        handle_schema_flag(
            _agent_parser,
            "nvflare agent",
            _AGENT_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=False,
            idempotent=True,
        )
        output_error_message(
            "AGENT_SUBCOMMAND_REQUIRED",
            "Agent subcommand required.",
            "Run 'nvflare agent skills --help' or 'nvflare agent --schema'.",
            exit_code=4,
        )
        return

    if agent_sub_cmd == CMD_AGENT_SKILLS:
        _handle_agent_skills_cmd(args, handle_schema_flag)
        return

    raise CLIUnknownCmdException(f"unknown agent subcommand: {agent_sub_cmd}")


def _handle_agent_skills_cmd(args, handle_schema_flag) -> None:
    from nvflare.tool.cli_output import output_error, output_error_message, output_ok

    skills_sub_cmd = getattr(args, "agent_skills_sub_cmd", None)
    if skills_sub_cmd is None:
        handle_schema_flag(
            _agent_skills_parser,
            "nvflare agent skills",
            _AGENT_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=False,
            idempotent=True,
        )
        output_error_message(
            "AGENT_SKILLS_SUBCOMMAND_REQUIRED",
            "Agent skills subcommand required.",
            "Run 'nvflare agent skills install --schema' or 'nvflare agent skills list --schema'.",
            exit_code=4,
        )
        return

    if skills_sub_cmd == CMD_AGENT_SKILLS_INSTALL:
        handle_schema_flag(
            _agent_skills_sub_cmd_parsers[CMD_AGENT_SKILLS_INSTALL],
            "nvflare agent skills install",
            _AGENT_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=True,
            idempotent=True,
        )
        plan = install_skills(
            agent=args.agent,
            skill_name=getattr(args, "skill", None),
            dry_run=getattr(args, "dry_run", False),
            target_dir=getattr(args, "target", None),
        )
        if plan.get("missing"):
            output_error(
                "AGENT_SKILL_NOT_FOUND",
                exit_code=4,
                hint="Run 'nvflare agent skills list --agent <codex|claude> --format json' to inspect skills.",
                data=plan,
                detail=", ".join(plan["missing"]),
            )
            return
        if plan.get("errors"):
            output_error(
                "AGENT_SKILL_INSTALL_FAILED",
                exit_code=1,
                hint="Review data.errors and retry after fixing the filesystem issue.",
                data=plan,
            )
            return
        output_ok(plan)
        return

    if skills_sub_cmd == CMD_AGENT_SKILLS_LIST:
        handle_schema_flag(
            _agent_skills_sub_cmd_parsers[CMD_AGENT_SKILLS_LIST],
            "nvflare agent skills list",
            _AGENT_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=False,
            idempotent=True,
        )
        data = list_skills(agent=args.agent, target_dir=getattr(args, "target", None))
        if data.get("errors"):
            output_error(
                "AGENT_SKILL_LIST_FAILED",
                exit_code=1,
                hint="Review data.errors and retry after fixing the filesystem issue.",
                data=data,
            )
            return
        output_ok(data)
        return

    raise CLIUnknownCmdException(f"unknown agent skills subcommand: {skills_sub_cmd}")
