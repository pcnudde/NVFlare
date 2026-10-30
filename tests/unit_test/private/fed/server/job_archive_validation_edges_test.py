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

import io
import json
from zipfile import ZipFile

import pytest

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.fl_constant import JobConstants
from nvflare.apis.job_def import ALL_SITES, JobMetaKey
from nvflare.private.fed.server.job_meta_validator import MAX_CLIENTS, JobMetaValidator


def archive(client_config=None, custom=False):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as output:
        output.writestr(
            "job/meta.json", json.dumps({"name": "job", "deploy_map": {"app": ["server", "site"]}, "min_clients": 1})
        )
        output.writestr("job/app/config/", "")
        output.writestr("job/app/config/config_fed_server.json", json.dumps({"format_version": 2, "workflows": []}))
        output.writestr(
            "job/app/config/config_fed_client.json",
            json.dumps(client_config if client_config is not None else {"format_version": 2, "executors": []}),
        )
        if custom:
            output.writestr("job/app/custom/", "")
            output.writestr("job/app/custom/model.py", "MODEL = 7")
    return buffer.getvalue()


@pytest.mark.parametrize("custom", [True, False])
def test_job_archive_path_validation_derives_byoc_from_contents(tmp_path, custom):
    path = tmp_path / "job.zip"
    path.write_bytes(archive(custom=custom))
    valid, error, metadata = JobMetaValidator().validate("job", str(path))
    assert valid and error == ""
    assert metadata.get(AppValidationKey.BYOC, False) is custom


def test_job_archive_rejects_nonmapping_client_config_and_wrong_input_type():
    valid, error, metadata = JobMetaValidator().validate("job", archive(client_config=["invalid"]))
    assert not valid and "client app config must be a dictionary" in error
    assert metadata == {}
    with pytest.raises(TypeError, match="job_data must be bytes or str"):
        JobMetaValidator().validate("job", object())


@pytest.mark.parametrize("deploy", [["server"], {"app": [{JobConstants.SITES: ["server"], "extra": True}]}])
def test_deploy_map_rejects_wrong_mapping_shape_or_unknown_entry_keys(deploy):
    with pytest.raises(ValueError, match="dictionary|Unknown keys"):
        JobMetaValidator._validate_deploy_map("job", {JobMetaKey.DEPLOY_MAP.value: deploy})


@pytest.mark.parametrize("value", ["3", "bad", None])
def test_min_client_integer_conversion_is_explicit_for_strings_and_unconvertible_types(value):
    if value == "3":
        assert JobMetaValidator._convert_value_to_int(value) == 3
    else:
        with pytest.raises(ValueError, match="can't not convert to Int"):
            JobMetaValidator._convert_value_to_int(value)


@pytest.mark.parametrize("minimum", [0, MAX_CLIENTS + 1])
def test_minimum_clients_is_positive_and_has_global_upper_limit(minimum):
    with pytest.raises(ValueError, match="must be positive|must be less"):
        JobMetaValidator()._validate_min_clients("job", {JobMetaKey.MIN_CLIENTS: minimum}, {ALL_SITES})


@pytest.mark.parametrize("launcher", [["process"], {"site": "process"}, {"site": {"process": "invalid"}}])
def test_job_launcher_configuration_requires_mapping_at_each_policy_level(launcher):
    with pytest.raises(ValueError, match="must be a dict"):
        JobMetaValidator._validate_launcher_spec("job", {JobMetaKey.JOB_LAUNCHER_SPEC.value: launcher})
