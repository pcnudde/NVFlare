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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import (
    JobMetaKey,
    get_custom_prop,
    get_custom_props,
    get_job_meta_study,
    is_valid_job_id,
    job_from_meta,
)


def test_metadata_maps_sites_resources_and_reads_application_from_job_manager():
    meta = {
        JobMetaKey.JOB_ID: "job",
        JobMetaKey.DEPLOY_MAP: {"app": ["site", "server"]},
        JobMetaKey.RESOURCE_SPEC: {"site": {"gpu": 1}},
        JobMetaKey.MIN_CLIENTS: 2,
        JobMetaKey.MANDATORY_CLIENTS: ["site"],
    }
    job = job_from_meta(meta)
    assert job.get_application_name("site") == "app"
    assert job.get_application_name("unassigned") is None
    assert job.get_deployment() == {"app": ["site", "server"]}
    assert job.get_resource_requirements() == {"site": {"gpu": 1}}
    assert job.min_sites == 2 and job.required_sites == ["site"]
    manager = Mock()
    manager.get_app.return_value = b"archive"
    engine = Mock()
    engine.get_component.return_value = manager
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    assert job.get_application("app", ctx) == b"archive"
    engine.get_component.assert_called_once_with(SystemComponents.JOB_MANAGER)
    manager.get_app.assert_called_once_with(job, "app", ctx)


@pytest.mark.parametrize("meta", [None, [], {"study": ""}, {"study": 1}])
def test_absent_or_invalid_study_defaults_without_custom_property_leaks(meta):
    assert get_job_meta_study(meta) == "default"
    assert get_custom_prop({}, "key", "fallback") == "fallback"
    assert get_custom_props({}, "fallback") == "fallback"
    assert get_custom_prop({JobMetaKey.CUSTOM_PROPS: {"key": 7}}, "key") == 7
    assert get_custom_props({JobMetaKey.CUSTOM_PROPS: {"key": 7}}) == {"key": 7}


@pytest.mark.parametrize("value", [None, 3, "invalid", "00000000-0000-0000-0000-000000000000"])
def test_non_uuid4_job_id_is_rejected(value):
    assert not is_valid_job_id(value)
