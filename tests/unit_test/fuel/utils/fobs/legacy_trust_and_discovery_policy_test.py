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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.utils.class_loader import get_class_name
from nvflare.fuel.utils.fobs import fobs as subject
from nvflare.fuel.utils.fobs.decomposer import DataClassDecomposer


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(subject, "_decomposers", {})
    monkeypatch.setattr(subject, "_dot_handlers", {})
    monkeypatch.setattr(subject, "_type_name_whitelist", {get_class_name(RunStatus), "builtins.dict"})
    monkeypatch.setattr(subject, "_enum_auto_registration", True)
    monkeypatch.setattr(subject, "_data_auto_registration", True)


def test_legacy_whitelisted_enum_reloads_but_plain_type_has_no_implicit_codec():
    packer = subject.Packer(None)
    restored = packer.unpack({subject.FOBS_TYPE: get_class_name(RunStatus), subject.FOBS_DATA: "RUNNING"})
    assert restored is RunStatus.RUNNING
    with pytest.raises(TypeError, match="has no decomposer registered"):
        packer.unpack({subject.FOBS_TYPE: "builtins.dict", subject.FOBS_DATA: {}})
    assert "builtins.dict" not in subject._decomposers


def test_disabled_legacy_enum_registration_does_not_grant_deserialization_trust():
    subject.auto_register_enum_types(False)
    with pytest.raises(TypeError, match="has no decomposer registered"):
        subject.Packer(None).unpack({subject.FOBS_TYPE: get_class_name(RunStatus), subject.FOBS_DATA: "RUNNING"})
    assert not subject._decomposers
    with pytest.raises(TypeError, match="not a subclass of Enum"):
        subject.register_enum_types(dict)


def test_custom_discovery_skips_import_failures_and_parameterized_codecs(tmp_path, monkeypatch, caplog):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "codec.py").write_text("# scanned module")
    (nested / "missing.py").write_text("# unavailable dependency")
    (tmp_path / "ignored.txt").write_text("not a python module")

    def import_boundary(name):
        if name == "nested.codec":
            return SimpleNamespace(Codec=DataClassDecomposer)
        raise ModuleNotFoundError(name)

    loader = Mock(side_effect=import_boundary)
    monkeypatch.setattr(subject, "importlib", SimpleNamespace(import_module=loader))
    monkeypatch.setattr(subject, "sys", SimpleNamespace(path=list(sys.path)))
    subject.register_custom_folder(str(tmp_path))
    assert str(tmp_path) in subject.sys.path
    assert {call.args[0] for call in loader.call_args_list} == {"nested.codec", "nested.missing"}
    assert "can't have argument" in caplog.text
    assert not subject._decomposers
    subject.register_custom_folder(str(tmp_path))
    assert subject.sys.path.count(str(tmp_path)) == 1
