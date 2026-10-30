# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest

from nvflare.tool.kit import kit_config as subject


def make_admin_kit(folder):
    for relative in subject.ADMIN_STARTUP_KIT_REQUIRED_FILES:
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder")
    (folder / subject.ADMIN_STATIC_CERT_FILE).write_text("invalid certificate")
    (folder / "startup" / "fed_admin.json").write_text('{"admin": {}}')


def test_startup_folder_normalization_preserves_participant_root(tmp_path):
    make_admin_kit(tmp_path)
    assert subject.validate_startup_kit(str(tmp_path / "startup")) == str(tmp_path.resolve())


@pytest.mark.parametrize("path", [None, "   "])
def test_empty_kit_paths_and_unregistered_ids_have_actionable_errors(path):
    with pytest.raises(subject.StartupKitConfigError, match="not specified"):
        subject._as_existing_dir(path)
    with pytest.raises(subject.StartupKitConfigError, match="not registered"):
        subject._validate_registered_path("missing", "")


def test_regular_file_cannot_be_registered_as_kit_directory(tmp_path):
    path = tmp_path / "kit"
    path.write_text("not a directory")
    with pytest.raises(subject.StartupKitConfigError, match="not a valid startup kit"):
        subject._as_existing_dir(str(path))
    with pytest.raises(subject.StartupKitConfigError, match="does not exist"):
        subject._validate_registered_path("gone", str(tmp_path / "absent"))


def test_provider_kit_with_malformed_admin_json_cannot_be_classified(tmp_path):
    make_admin_kit(tmp_path)
    (tmp_path / subject.ADMIN_STATIC_CERT_FILE).unlink()
    (tmp_path / "startup" / "fed_admin.json").write_text("[")
    assert not subject._is_admin_startup_kit(tmp_path)


def test_admin_username_resolution_requires_configuration_and_can_fall_back_to_kit_identity(tmp_path, monkeypatch):
    make_admin_kit(tmp_path)
    monkeypatch.setattr(subject.ConfigFactory, "load_config", Mock(return_value=None))
    with pytest.raises(subject.StartupKitConfigError, match="Unable to locate fed_admin"):
        subject.resolve_admin_user_and_dir_from_startup_kit(str(tmp_path))
    config = Mock()
    config.to_dict.return_value = {"admin": {}}
    monkeypatch.setattr(subject.ConfigFactory, "load_config", Mock(return_value=config))
    username, location = subject.resolve_admin_user_and_dir_from_startup_kit(str(tmp_path))
    assert username == tmp_path.name and location == str(tmp_path)
