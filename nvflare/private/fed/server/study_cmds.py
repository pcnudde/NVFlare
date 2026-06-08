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

from copy import deepcopy
from typing import Dict, List, Tuple

from nvflare.apis import study_store
from nvflare.apis.client import ClientPropKey
from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.apis.utils.format_check import name_check
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import ConfirmMethod, MetaStatusValue, make_meta
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.authz import AuthorizationService, AuthzContext, Person
from nvflare.fuel.utils.argument_utils import SafeArgumentParser
from nvflare.private.fed.server.server_engine import ServerEngine

from .cmd_utils import CommandUtil


class _InvalidArgsError(ValueError):
    pass


class _InvalidSiteError(ValueError):
    pass


class _InvalidStudyNameError(ValueError):
    pass


def _study_parser(
    cmd_name: str,
    include_sites: bool = False,
    include_site_org: bool = False,
    include_user: bool = False,
):
    parser = SafeArgumentParser(prog=cmd_name)
    parser.add_argument("study")
    if include_sites:
        parser.add_argument("--sites", required=False)
    if include_site_org:
        parser.add_argument("--site-org", action="append", default=[], dest="site_orgs")
    if include_user:
        parser.add_argument("user")
    return parser


class StudyCommandModule(CommandModule, CommandUtil):
    def get_spec(self):
        return CommandModuleSpec(
            name="study_mgmt",
            cmd_specs=[
                CommandSpec(
                    name=AdminCommandNames.REGISTER_STUDY,
                    description="create or merge a study",
                    usage=f"{AdminCommandNames.REGISTER_STUDY} <study> [--sites s1,s2,...] [--site-org org:s1,s2,...]",
                    handler_func=self.cmd_register_study,
                    authz_func=self.authorize_study_admin,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.ADD_STUDY_SITE,
                    description="add sites to a study",
                    usage=f"{AdminCommandNames.ADD_STUDY_SITE} <study> [--sites s1,s2,...] [--site-org org:s1,s2,...]",
                    handler_func=self.cmd_add_study_site,
                    authz_func=self.authorize_study_admin,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.REMOVE_STUDY_SITE,
                    description="remove sites from a study",
                    usage=f"{AdminCommandNames.REMOVE_STUDY_SITE} <study> [--sites s1,s2,...] [--site-org org:s1,s2,...]",
                    handler_func=self.cmd_remove_study_site,
                    authz_func=self.authorize_study_admin,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.REMOVE_STUDY,
                    description="remove a study",
                    usage=f"{AdminCommandNames.REMOVE_STUDY} <study>",
                    handler_func=self.cmd_remove_study,
                    authz_func=self.must_be_project_admin,
                    visible=True,
                    confirm=ConfirmMethod.AUTH,
                ),
                CommandSpec(
                    name=AdminCommandNames.LIST_STUDIES,
                    description="list visible studies",
                    usage=AdminCommandNames.LIST_STUDIES,
                    handler_func=self.cmd_list_studies,
                    authz_func=self.authorize_list_studies,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.SHOW_STUDY,
                    description="show a study",
                    usage=f"{AdminCommandNames.SHOW_STUDY} <study>",
                    handler_func=self.cmd_show_study,
                    authz_func=self.authorize_study_admin,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.ADD_STUDY_USER,
                    description="add a user to a study",
                    usage=f"{AdminCommandNames.ADD_STUDY_USER} <study> <user>",
                    handler_func=self.cmd_add_study_user,
                    authz_func=self.authorize_study_admin,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.REMOVE_STUDY_USER,
                    description="remove a user from a study",
                    usage=f"{AdminCommandNames.REMOVE_STUDY_USER} <study> <user>",
                    handler_func=self.cmd_remove_study_user,
                    authz_func=self.authorize_study_admin,
                    visible=True,
                ),
            ],
        )

    def authorize_study_admin(self, conn: Connection, args: List[str]):
        role = conn.get_prop(ConnProps.USER_ROLE, "")
        if role in {"project_admin", "org_admin"}:
            return PreAuthzReturnCode.OK
        conn.append_error(f"NOT_AUTHORIZED for {role}", meta=make_meta(MetaStatusValue.NOT_AUTHORIZED))
        return PreAuthzReturnCode.ERROR

    def authorize_list_studies(self, conn: Connection, args: List[str]):
        role = conn.get_prop(ConnProps.USER_ROLE, "")
        # This only authorizes invoking the list command. cmd_list_studies still
        # filters each returned study by caller role, org, and explicit user mapping.
        if role in {"project_admin", "org_admin", "lead", "member"}:
            return PreAuthzReturnCode.OK
        conn.append_error(f"NOT_AUTHORIZED for {role}", meta=make_meta(MetaStatusValue.NOT_AUTHORIZED))
        return PreAuthzReturnCode.ERROR

    @staticmethod
    def _reply(conn: Connection, payload: dict):
        conn.append_dict(payload, meta=make_meta(MetaStatusValue.OK))

    def _error(self, conn: Connection, error_code: str, message: str, hint: str = "", exit_code: int = 1):
        self._reply(
            conn,
            {
                "error_code": error_code,
                "message": message,
                "hint": hint,
                "exit_code": exit_code,
            },
        )

    @staticmethod
    def _parse_sites(sites_arg: str) -> List[str]:
        if sites_arg is None:
            raise _InvalidArgsError("--sites is required")
        sites = []
        seen = set()
        for site in sites_arg.split(","):
            site = site.strip()
            if not site or site in seen:
                continue
            seen.add(site)
            sites.append(site)
        if not sites:
            raise _InvalidArgsError("--sites must contain at least one site")
        return sites

    @staticmethod
    def _parse_site_orgs(site_org_args: List[str]) -> Dict[str, List[str]]:
        if not site_org_args:
            raise _InvalidArgsError("--site-org is required")
        result = {}
        seen_sites = set()
        for item in site_org_args:
            org, sep, raw_sites = item.partition(":")
            if not sep:
                raise _InvalidArgsError(f"invalid --site-org value '{item}'")
            org = org.strip()
            sites = []
            for site in raw_sites.split(","):
                site = site.strip()
                if not site:
                    continue
                if site in seen_sites:
                    raise _InvalidArgsError(f"site '{site}' appears in more than one org group")
                seen_sites.add(site)
                sites.append(site)
            if not sites:
                raise _InvalidArgsError(f"--site-org '{item}' must contain at least one site")
            result.setdefault(org, [])
            result[org].extend(sites)
        return result

    def _validate_study_name(self, study: str):
        invalid, _ = name_check(study, "study")
        if invalid or study == "default":
            raise _InvalidStudyNameError(f"invalid study name '{study}'")

    def _validate_site_names(self, sites: List[str]):
        if not sites:
            raise _InvalidSiteError("sites are required")
        for site in sites:
            invalid, _ = name_check(site, "site")
            if invalid:
                raise _InvalidSiteError(f"invalid site '{site}'")

    def _validate_site_orgs(self, site_orgs: Dict[str, List[str]]):
        for org, sites in site_orgs.items():
            invalid, _ = name_check(org, "org")
            if invalid:
                raise _InvalidArgsError(f"invalid org '{org}'")
            self._validate_site_names(sites)

    @staticmethod
    def _study_payload(study: str, study_def: dict):
        site_orgs = deepcopy(study_def.get("site_orgs", {}))
        sites = []
        for org_sites in site_orgs.values():
            sites.extend(org_sites)
        return {
            "name": study,
            "site_orgs": site_orgs,
            "sites": sorted(sites),
            "users": list(study_def.get("admins", [])),
        }

    @staticmethod
    def _caller_name(conn: Connection) -> str:
        return conn.get_prop(ConnProps.USER_NAME, "")

    @staticmethod
    def _caller_role(conn: Connection) -> str:
        return conn.get_prop(ConnProps.USER_ROLE, "")

    @staticmethod
    def _caller_org(conn: Connection) -> str:
        return conn.get_prop(ConnProps.USER_ORG, "")

    @staticmethod
    def _validate_sites_for_org(engine: ServerEngine, sites: List[str], expected_org: str) -> List[str]:
        """Returns sites that are unknown to the server or whose cert org does not match expected_org."""
        if not expected_org:
            return list(sites)
        bad = []
        for site in sites:
            client = engine.client_manager.get_client_from_name(site)
            if client is None or client.get_prop(ClientPropKey.ORG, "") != expected_org:
                bad.append(site)
        return bad

    def _is_visible_to_caller(self, conn: Connection, study_def: dict) -> bool:
        # Visibility policy:
        # - project_admin can see every study.
        # - org_admin can see studies that include at least one site from the caller's org.
        # - lead/member and other non-admin roles can see only studies where the
        #   caller is explicitly listed in the study admins mapping.
        caller_role = self._caller_role(conn)
        if caller_role == "project_admin":
            return True
        if caller_role != "org_admin":
            return self._caller_name(conn) in set((study_def or {}).get("admins", []))
        caller_org = self._caller_org(conn)
        if not caller_org:
            return False
        site_orgs = (study_def or {}).get("site_orgs", {})
        return caller_org in site_orgs

    def _is_list_visible_to_caller(self, conn: Connection, study_def: dict) -> bool:
        return self._is_visible_to_caller(conn, study_def)

    def _study_list_item(self, conn: Connection, study_name: str) -> dict:
        role = self._caller_role(conn)
        can_submit, reason = self._can_submit_job(conn)
        item = {
            "name": study_name,
            "role": role,
            "capabilities": {"submit_job": can_submit},
            "can_submit_job": can_submit,
        }
        if not can_submit:
            item["reason"] = reason or f"user '{self._caller_name(conn)}' is not authorized for 'submit_job'"
        return item

    def _can_submit_job(self, conn: Connection) -> Tuple[bool, str]:
        user = Person(
            name=self._caller_name(conn),
            org=self._caller_org(conn),
            role=self._caller_role(conn),
        )
        submitter = Person(name="", org="", role="")
        ctx = AuthzContext(user=user, submitter=submitter, right=AdminCommandNames.SUBMIT_JOB)
        return AuthorizationService.authorize(ctx)

    def _caller_identity_payload(self, conn: Connection) -> dict:
        return {
            "name": self._caller_name(conn),
            "org": self._caller_org(conn),
            "role": self._caller_role(conn),
        }

    @staticmethod
    def _normalize_admins(study_def: dict) -> List[str]:
        admins = study_def.setdefault("admins", [])
        if admins is None:
            admins = []
            study_def["admins"] = admins
        if isinstance(admins, list):
            return admins
        raise ValueError("study admins must be a list")

    @staticmethod
    def _normalize_site_orgs(study_def: dict) -> Dict[str, List[str]]:
        site_orgs = study_def.setdefault("site_orgs", {})
        if site_orgs is None:
            site_orgs = {}
            study_def["site_orgs"] = site_orgs
        if not isinstance(site_orgs, dict):
            raise ValueError("study site_orgs must be a mapping")
        return site_orgs

    def _requested_site_orgs(self, conn: Connection, parsed) -> Dict[str, List[str]]:
        caller_role = self._caller_role(conn)
        has_sites = bool(getattr(parsed, "sites", None))
        has_site_org = bool(getattr(parsed, "site_orgs", []))

        if has_sites and has_site_org:
            raise _InvalidArgsError("--sites and --site-org are mutually exclusive; provide only one")
        if caller_role == "org_admin" and has_site_org:
            raise _InvalidArgsError("org_admin must use --sites, not --site-org")
        if caller_role == "project_admin" and has_sites:
            raise _InvalidArgsError("project_admin must use --site-org, not --sites")

        if caller_role == "project_admin":
            site_orgs = self._parse_site_orgs(getattr(parsed, "site_orgs", []))
        else:
            caller_org = self._caller_org(conn)
            if not caller_org:
                raise _InvalidArgsError("caller org is empty — misconfigured certificate")
            sites = self._parse_sites(getattr(parsed, "sites", None))
            site_orgs = {caller_org: sites}
        self._validate_site_orgs(site_orgs)
        return site_orgs

    def _site_mutation_payload(
        self, study: str, study_def: dict, added=None, already_enrolled=None, removed=None, not_enrolled=None
    ):
        payload = {"study": study}
        if added is not None:
            payload["added"] = added
        if already_enrolled is not None:
            payload["already_enrolled"] = already_enrolled
        if removed is not None:
            payload["removed"] = removed
        if not_enrolled is not None:
            payload["not_enrolled"] = not_enrolled
        payload["site_orgs"] = deepcopy(study_def.get("site_orgs", {}))
        payload["sites"] = sorted({s for org_sites in payload["site_orgs"].values() for s in org_sites})
        return payload

    def _run_study_command(self, conn: Connection, command_cb):
        try:
            engine = conn.app_ctx
            if not isinstance(engine, ServerEngine):
                raise TypeError(f"engine must be ServerEngine but got {type(engine)}")

            payload = command_cb(engine)
            if payload is None:
                self._error(conn, "INTERNAL_ERROR", "study command returned no result", exit_code=5)
                return
            if isinstance(payload, dict) and payload.get("error_code"):
                self._reply(conn, payload)
                return
            self._reply(conn, payload)
        except Exception as e:
            self._error(conn, "INTERNAL_ERROR", f"study command failed: {e}", exit_code=5)

    def _study_not_found(self, conn: Connection, study: str):
        self._error(
            conn,
            "STUDY_NOT_FOUND",
            f"Study '{study}' not found.",
            hint="Verify the study name. If the study exists and you expect access, contact a project_admin.",
        )

    def cmd_register_study(self, conn: Connection, args: List[str]):
        parser = _study_parser(AdminCommandNames.REGISTER_STUDY, include_sites=True, include_site_org=True)
        try:
            parsed = parser.parse_args(args[1:])
            self._validate_study_name(parsed.study)
            requested = self._requested_site_orgs(conn, parsed)
        except _InvalidArgsError as e:
            self._error(conn, "INVALID_ARGS", str(e), exit_code=4)
            return
        except _InvalidSiteError as e:
            self._error(conn, "INVALID_SITE", str(e), exit_code=4)
            return
        except _InvalidStudyNameError as e:
            self._error(conn, "INVALID_STUDY_NAME", str(e), exit_code=4)
            return

        def _run(_engine):
            study_def = study_store.get_study(parsed.study)
            is_new_study = study_def is None
            caller = self._caller_name(conn)
            caller_role = self._caller_role(conn)
            caller_org = self._caller_org(conn)
            if is_new_study:
                study_def = {"site_orgs": {}, "admins": []}
            elif caller_role == "org_admin" and caller_org not in self._normalize_site_orgs(study_def):
                return {
                    "error_code": "STUDY_ALREADY_EXISTS",
                    "message": f"Study '{parsed.study}' already exists.",
                    "hint": "The study name is already taken. Contact a project_admin to grant access.",
                    "exit_code": 1,
                }

            for org, sites in requested.items():
                bad_sites = self._validate_sites_for_org(_engine, sites, org)
                if bad_sites:
                    return {
                        "error_code": "INVALID_SITE",
                        "message": f"Sites {bad_sites} are unknown or do not belong to org '{org}'.",
                        "hint": "Each site must be currently connected to the server and provisioned under the specified org.",
                        "exit_code": 4,
                    }

            if is_new_study:
                study_def = study_store.upsert_study(parsed.study, {"site_orgs": requested, "admins": [caller]})
            else:
                site_orgs = self._normalize_site_orgs(study_def)
                all_existing = {site for org_sites in site_orgs.values() for site in org_sites}
                for org, sites in requested.items():
                    current = site_orgs.setdefault(org, [])
                    for site in sites:
                        if site in all_existing:
                            continue
                        current.append(site)
                        all_existing.add(site)
                admins = self._normalize_admins(study_def)
                if caller not in admins:
                    admins.append(caller)
                study_def = study_store.upsert_study(parsed.study, {"site_orgs": site_orgs, "admins": admins})
            return self._study_payload(parsed.study, study_def)

        self._run_study_command(conn, _run)

    def cmd_add_study_site(self, conn: Connection, args: List[str]):
        parser = _study_parser(AdminCommandNames.ADD_STUDY_SITE, include_sites=True, include_site_org=True)
        try:
            parsed = parser.parse_args(args[1:])
            self._validate_study_name(parsed.study)
            requested = self._requested_site_orgs(conn, parsed)
        except _InvalidArgsError as e:
            self._error(conn, "INVALID_ARGS", str(e), exit_code=4)
            return
        except _InvalidSiteError as e:
            self._error(conn, "INVALID_SITE", str(e), exit_code=4)
            return
        except _InvalidStudyNameError as e:
            self._error(conn, "INVALID_STUDY_NAME", str(e), exit_code=4)
            return

        def _run(_engine):
            study_def = study_store.get_study(parsed.study)
            if not study_def or not self._is_visible_to_caller(conn, study_def):
                return {
                    "error_code": "STUDY_NOT_FOUND",
                    "message": f"Study '{parsed.study}' not found.",
                    "hint": "Verify the study name or contact a project_admin.",
                    "exit_code": 1,
                }
            for org, sites in requested.items():
                bad_sites = self._validate_sites_for_org(_engine, sites, org)
                if bad_sites:
                    return {
                        "error_code": "INVALID_SITE",
                        "message": f"Sites {bad_sites} are unknown or do not belong to org '{org}'.",
                        "hint": "Each site must be currently connected to the server and provisioned under the specified org.",
                        "exit_code": 4,
                    }
            site_orgs = self._normalize_site_orgs(study_def)
            added = []
            already_enrolled = []
            all_existing = {site for org_sites in site_orgs.values() for site in org_sites}
            for org, sites in requested.items():
                current = site_orgs.setdefault(org, [])
                existing = set(current)
                for site in sites:
                    if site in all_existing:
                        already_enrolled.append(site)
                    else:
                        current.append(site)
                        existing.add(site)
                        all_existing.add(site)
                        added.append(site)
            study_def = study_store.add_sites(parsed.study, requested)
            return self._site_mutation_payload(parsed.study, study_def, added=added, already_enrolled=already_enrolled)

        self._run_study_command(conn, _run)

    def cmd_remove_study_site(self, conn: Connection, args: List[str]):
        parser = _study_parser(AdminCommandNames.REMOVE_STUDY_SITE, include_sites=True, include_site_org=True)
        try:
            parsed = parser.parse_args(args[1:])
            self._validate_study_name(parsed.study)
            requested = self._requested_site_orgs(conn, parsed)
        except _InvalidArgsError as e:
            self._error(conn, "INVALID_ARGS", str(e), exit_code=4)
            return
        except _InvalidSiteError as e:
            self._error(conn, "INVALID_SITE", str(e), exit_code=4)
            return
        except _InvalidStudyNameError as e:
            self._error(conn, "INVALID_STUDY_NAME", str(e), exit_code=4)
            return

        def _run(_engine):
            study_def = study_store.get_study(parsed.study)
            if not study_def or not self._is_visible_to_caller(conn, study_def):
                return {
                    "error_code": "STUDY_NOT_FOUND",
                    "message": f"Study '{parsed.study}' not found.",
                    "hint": "Verify the study name or contact a project_admin.",
                    "exit_code": 1,
                }
            # No connectivity check: the site may be offline or decommissioned.
            # The operator is explicitly requesting removal, so current connection
            # status is irrelevant.
            site_orgs = self._normalize_site_orgs(study_def)
            removed = []
            not_enrolled = []
            for org, sites in requested.items():
                if org not in site_orgs:
                    not_enrolled.extend(sites)
                    continue
                current = site_orgs[org]
                current_set = set(current)
                new_current = []
                for site in current:
                    if site in sites:
                        removed.append(site)
                    else:
                        new_current.append(site)
                for site in sites:
                    if site not in current_set:
                        not_enrolled.append(site)
                site_orgs[org] = new_current
            study_def = study_store.remove_sites(parsed.study, requested)
            return self._site_mutation_payload(parsed.study, study_def, removed=removed, not_enrolled=not_enrolled)

        self._run_study_command(conn, _run)

    def cmd_remove_study(self, conn: Connection, args: List[str]):
        parser = _study_parser(AdminCommandNames.REMOVE_STUDY)
        try:
            parsed = parser.parse_args(args[1:])
            self._validate_study_name(parsed.study)
        except Exception as e:
            self._error(conn, "INVALID_STUDY_NAME", str(e), exit_code=4)
            return

        def _run(engine):
            study_def = study_store.get_study(parsed.study)
            if study_def is None:
                return {
                    "error_code": "STUDY_NOT_FOUND",
                    "message": f"Study '{parsed.study}' not found.",
                    "hint": "Verify the study name.",
                    "exit_code": 1,
                }

            job_def_manager = engine.job_def_manager
            count = 0
            if isinstance(job_def_manager, JobDefManagerSpec):
                with engine.new_context() as fl_ctx:
                    for job in job_def_manager.get_all_jobs(fl_ctx) or []:
                        if job.meta.get(JobMetaKey.STUDY.value) == parsed.study:
                            count += 1
            if count > 0:
                return {
                    "error_code": "STUDY_HAS_JOBS",
                    "message": f"Study '{parsed.study}' has {count} associated job(s) and cannot be removed.",
                    "hint": "Archive or delete the associated jobs before retrying.",
                    "exit_code": 1,
                }
            study_store.delete_study(parsed.study)
            return {"name": parsed.study, "removed": True}

        self._run_study_command(conn, _run)

    def cmd_list_studies(self, conn: Connection, args: List[str]):
        if len(args) != 1:
            self._error(conn, "INVALID_ARGS", "list_studies does not accept arguments", exit_code=4)
            return
        studies = []
        study_details = []
        for row in study_store.list_studies():
            study_name = row.get("name")
            study_def = row.get("config_json")
            if study_name and self._is_list_visible_to_caller(conn, study_def):
                studies.append(study_name)
                study_details.append(self._study_list_item(conn, study_name))
        self._reply(
            conn,
            {
                "identity": self._caller_identity_payload(conn),
                "studies": sorted(studies),
                "study_details": sorted(study_details, key=lambda item: item["name"]),
            },
        )

    def cmd_show_study(self, conn: Connection, args: List[str]):
        parser = _study_parser(AdminCommandNames.SHOW_STUDY)
        try:
            parsed = parser.parse_args(args[1:])
            self._validate_study_name(parsed.study)
        except Exception as e:
            self._error(conn, "INVALID_STUDY_NAME", str(e), exit_code=4)
            return
        study_def = study_store.get_study(parsed.study)
        if not study_def or not self._is_visible_to_caller(conn, study_def):
            self._study_not_found(conn, parsed.study)
            return
        self._reply(conn, self._study_payload(parsed.study, study_def))

    def cmd_add_study_user(self, conn: Connection, args: List[str]):
        parser = _study_parser(AdminCommandNames.ADD_STUDY_USER, include_user=True)
        try:
            parsed = parser.parse_args(args[1:])
            self._validate_study_name(parsed.study)
        except Exception as e:
            self._error(conn, "INVALID_STUDY_NAME", str(e), exit_code=4)
            return

        def _run(_engine):
            study_def = study_store.get_study(parsed.study)
            if not study_def or not self._is_visible_to_caller(conn, study_def):
                return {
                    "error_code": "STUDY_NOT_FOUND",
                    "message": f"Study '{parsed.study}' not found.",
                    "hint": "Verify the study name or contact a project_admin.",
                    "exit_code": 1,
                }
            admins = self._normalize_admins(study_def)
            if parsed.user in admins:
                return {
                    "error_code": "USER_ALREADY_IN_STUDY",
                    "message": f"User '{parsed.user}' is already in study '{parsed.study}'.",
                    "hint": "Use a different user or remove the existing entry first.",
                    "exit_code": 1,
                }
            study_store.add_user(parsed.study, parsed.user)
            return {"study": parsed.study, "user": parsed.user}

        self._run_study_command(conn, _run)

    def cmd_remove_study_user(self, conn: Connection, args: List[str]):
        parser = _study_parser(AdminCommandNames.REMOVE_STUDY_USER, include_user=True)
        try:
            parsed = parser.parse_args(args[1:])
            self._validate_study_name(parsed.study)
        except Exception as e:
            self._error(conn, "INVALID_STUDY_NAME", str(e), exit_code=4)
            return

        def _run(_engine):
            study_def = study_store.get_study(parsed.study)
            if not study_def or not self._is_visible_to_caller(conn, study_def):
                return {
                    "error_code": "STUDY_NOT_FOUND",
                    "message": f"Study '{parsed.study}' not found.",
                    "hint": "Verify the study name or contact a project_admin.",
                    "exit_code": 1,
                }
            admins = self._normalize_admins(study_def)
            if parsed.user not in admins:
                return {
                    "error_code": "USER_NOT_IN_STUDY",
                    "message": f"User '{parsed.user}' is not in study '{parsed.study}'.",
                    "hint": "Use add-user to add the user first.",
                    "exit_code": 1,
                }
            study_store.remove_user(parsed.study, parsed.user)
            return {"study": parsed.study, "user": parsed.user, "removed": True}

        self._run_study_command(conn, _run)
