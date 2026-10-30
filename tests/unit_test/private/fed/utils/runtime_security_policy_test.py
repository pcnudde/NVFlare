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

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReservedKey, SiteType, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobLauncherSpec
from nvflare.fuel.sec.security_content_service import LoadResult
from nvflare.private.defs import SSLConstants
from nvflare.private.fed.utils import fed_utils as subject


@pytest.fixture
def security(monkeypatch, tmp_path):
    service = SimpleNamespace(
        initialize=Mock(),
        security_content_manager=SimpleNamespace(valid_config=True, signature={}),
        load_json=Mock(),
        load_content=Mock(),
        check_json_files=Mock(return_value=[]),
    )
    audit = Mock()
    authorizer = Mock()
    authorizer.initialize.return_value = (None, "")
    study = Mock()
    monkeypatch.setattr(subject, "SecurityContentService", service)
    monkeypatch.setattr(subject, "AuditService", audit)
    monkeypatch.setattr(subject, "AuthorizationService", authorizer)
    monkeypatch.setattr(subject, "StudyRegistryService", study)
    workspace = Mock()
    workspace.get_startup_kit_dir.return_value = str(tmp_path)
    workspace.get_audit_file_path.return_value = str(tmp_path / "audit.log")
    workspace.get_authorization_file_path.return_value = str(tmp_path / "authorization.json")
    workspace.get_study_registry_file_path.return_value = str(tmp_path / "studies.json")
    return SimpleNamespace(
        service=service, audit=audit, auth=authorizer, study=study, workspace=workspace, path=tmp_path
    )


@pytest.mark.parametrize("site_type,private_key", [(SiteType.SERVER, True), (SiteType.CLIENT, False)])
def test_signed_security_content_checks_site_certificates_and_resource_patterns(security, site_type, private_key):
    env = security
    site = {SSLConstants.CERT: "cert", SSLConstants.ROOT_CERT: "root", SSLConstants.PRIVATE_KEY: "key"}
    config = {"servers": [site]} if site_type == SiteType.SERVER else {"client": site}
    env.service.load_json.side_effect = [(config, "bad"), ({}, "bad")]
    env.service.load_content.side_effect = lambda name: (b"data", "bad" if name == "cert" else LoadResult.OK)
    env.service.security_content_manager.signature = {WorkspaceConstants.AUTHORIZATION_CONFIG: "signature"}
    env.service.check_json_files.return_value = ["resources.json"]
    bad = subject._check_secure_content(site_type, check_private_key=private_key)
    startup = (
        WorkspaceConstants.SERVER_STARTUP_CONFIG
        if site_type == SiteType.SERVER
        else WorkspaceConstants.CLIENT_STARTUP_CONFIG
    )
    assert bad == [startup, "cert", WorkspaceConstants.AUTHORIZATION_CONFIG, "resources.json"]
    assert [c.args[0] for c in env.service.load_content.call_args_list] == (
        ["cert", "root", "key"] if private_key else ["cert", "root"]
    )


def test_security_initialization_loads_policy_and_studies_then_closes_audit(security, monkeypatch):
    env = security
    (env.path / "authorization.json").write_text(json.dumps({"rights": []}))
    (env.path / "studies.json").write_text(json.dumps({"studies": []}))
    monkeypatch.setattr(subject, "_check_secure_content", Mock(return_value=[]))
    factory = Mock()
    registry = Mock()
    appauth = Mock()
    monkeypatch.setattr(subject, "FLAuthorizer", factory)
    monkeypatch.setattr(subject, "StudyRegistry", registry)
    monkeypatch.setattr(subject, "AppAuthzService", appauth)
    validator = Mock()
    subject.security_init(True, "org-a", env.workspace, validator, SiteType.SERVER)
    factory.assert_called_once_with("org-a", {"rights": []})
    env.auth.initialize.assert_called_once_with(factory.return_value)
    registry.assert_called_once_with({"studies": []})
    env.study.initialize.assert_called_once_with(registry.return_value)
    appauth.initialize.assert_called_once_with(validator)
    env.audit.initialize.assert_called_once_with(str(env.path / "audit.log"))
    subject.security_close()
    env.audit.close.assert_called_once_with()


@pytest.mark.parametrize("failure", ["signature", "authorization"])
def test_security_initialization_stops_on_insecure_content_or_authorizer_failure(
    security, monkeypatch, capsys, failure
):
    env = security
    check = Mock(return_value=["tampered.json"] if failure == "signature" else [])
    monkeypatch.setattr(subject, "_check_secure_content", check)
    if failure == "authorization":
        env.auth.initialize.return_value = (None, "bad policy")
    with pytest.raises(SystemExit) as raised:
        subject.security_init(True, "org", env.workspace, None, SiteType.CLIENT)
    assert raised.value.code == 1
    output = capsys.readouterr().out
    assert ("tampered.json" if failure == "signature" else "AuthorizationService error: bad policy") in output
    if failure == "signature":
        env.audit.initialize.assert_not_called()
    env.study.initialize.assert_not_called()


def test_scope_information_is_sorted_and_contains_default_scope(monkeypatch):
    manager = Mock(spec=subject.PrivacyManager)
    manager.name_to_scopes = {"zeta": object(), "alpha": object()}
    manager.default_scope = SimpleNamespace(name="alpha")
    monkeypatch.setattr(subject.PrivacyService, "get_manager", Mock(return_value=manager))
    assert subject.get_scope_info() == (["alpha", "zeta"], "alpha")
    monkeypatch.setattr(subject.PrivacyService, "get_manager", Mock(side_effect=RuntimeError("not initialized")))
    assert subject.get_scope_info() == ([], "processing_error")


@pytest.mark.parametrize("meta", [[], "bad", None])
def test_job_processing_properties_reject_non_mapping_metadata(monkeypatch, meta):
    monkeypatch.setattr(subject, "get_job_meta_from_workspace", Mock(return_value=meta))
    with pytest.raises(RuntimeError, match="job_meta must be dict"):
        subject.create_job_processing_context_properties(Mock(), "job")


def test_external_decomposer_registration_discovers_packages_and_contains_missing_modules(monkeypatch):
    package = SimpleNamespace(__path__=["/modules"], __name__="custom")
    importer = Mock(return_value=package)
    monkeypatch.setattr(subject, "importlib", SimpleNamespace(import_module=importer))
    package_info = SimpleNamespace(ispkg=True, module_finder=SimpleNamespace(path="/modules"), name="custom.serializer")
    module_info = SimpleNamespace(ispkg=False)
    monkeypatch.setattr(
        subject, "pkgutil", SimpleNamespace(walk_packages=Mock(return_value=[package_info, module_info]))
    )
    monkeypatch.setattr(subject, "warnings", SimpleNamespace(filterwarnings=Mock()))
    register = Mock()
    monkeypatch.setattr(subject.fobs, "register_folder", register)
    subject.register_ext_decomposers("custom")
    register.assert_called_once_with("/modules/serializer", "custom.serializer")
    importer.side_effect = ModuleNotFoundError("absent")
    subject.register_ext_decomposers(["missing"])
    assert register.call_count == 1
    with pytest.raises(TypeError, match="str or list"):
        subject.register_ext_decomposers(123)


def test_stats_pool_writer_and_summary_failure_are_reported_without_leaking_writer(monkeypatch):
    workspace = Mock()
    workspace.get_stats_pool_records_path.return_value = "records.csv"
    workspace.get_stats_pool_summary_path.return_value = "summary.json"
    monkeypatch.setattr(
        subject, "get_job_meta_from_workspace", Mock(return_value={JobMetaKey.STATS_POOL_CONFIG: {"save_pools": ["*"]}})
    )
    pools = Mock()
    pools.dump_summary.side_effect = OSError("disk full")
    monkeypatch.setattr(subject, "StatsPoolManager", pools)
    writer = Mock()
    monkeypatch.setattr(subject, "CsvRecordHandler", writer)
    subject.set_stats_pool_config_for_job(workspace, "job", prefix="server")
    pools.set_pool_config.assert_called_once_with({"save_pools": ["*"]})
    writer.assert_called_once_with("records.csv")
    pools.set_record_writer.assert_called_once_with(writer.return_value)
    error = subject.create_stats_pool_files_for_job(workspace, "job", prefix="server")
    assert "summary.json" in error and "disk full" in error
    pools.close.assert_called_once_with()


@pytest.mark.parametrize(
    "entries,expected",
    [
        ("site-a", ["site-a"]),
        ({"targets": ["site-a"]}, ["site-a"]),
        ({JobConstants.SITES: ["site-a"]}, ["site-a"]),
        ([{"targets": ["a"]}, "b"], ["a", "b"]),
    ],
)
def test_participant_selection_accepts_each_documented_deploy_map_form(entries, expected):
    assert subject.extract_participants(entries) == expected


@pytest.mark.parametrize("entries", [{"other": []}, 42, [None], [{"targets": "site-a"}]])
def test_participant_selection_rejects_malformed_entries(entries):
    with pytest.raises(ValueError):
        subject.extract_participants(entries)


@pytest.mark.parametrize("mode", ["missing", "not-list", "multiple", "wrong-type", "valid"])
def test_job_launcher_selection_requires_exactly_one_matching_launcher(mode):
    caller = FLContext()
    launch_ctx = FLContext()
    launch_ctx.set_prop(FLContextKey.JOB_LAUNCHER, ["stale"])
    engine = Mock()
    caller.set_prop(ReservedKey.ENGINE, engine)
    engine.new_context.return_value = launch_ctx
    launcher = Mock(spec=JobLauncherSpec)
    choices = {
        "missing": None,
        "not-list": launcher,
        "multiple": [launcher, launcher],
        "wrong-type": [object()],
        "valid": [launcher],
    }

    def choose(event, ctx):
        assert ctx.get_prop(FLContextKey.JOB_LAUNCHER) is None
        assert ctx.get_prop(FLContextKey.JOB_META) == {"job": "a"}
        if choices[mode] is not None:
            ctx.set_prop(FLContextKey.JOB_LAUNCHER, choices[mode])

    engine.fire_event.side_effect = choose
    if mode == "valid":
        assert subject.get_job_launcher({"job": "a"}, caller) is launcher
    else:
        with pytest.raises(RuntimeError):
            subject.get_job_launcher({"job": "a"}, caller)
