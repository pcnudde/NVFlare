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

import copy

import pytest

from nvflare.app_opt.job_launcher import k8s_pod as module


@pytest.mark.parametrize("spec", [{"containers": {}}, {"containers": ["bad"]}])
def test_template_rejects_invalid_container_structure_without_mutating_caller(spec):
    template = {"spec": spec}
    before = copy.deepcopy(template)
    with pytest.raises(ValueError, match="containers"):
        module._prepare_pod_manifest_template(template)
    assert template == before


@pytest.mark.parametrize("template", [None, [], {"kind": "Deployment"}, {"metadata": []}, {"spec": []}])
def test_manifest_type_and_kind_validation_precedes_pod_materialization(template):
    with pytest.raises(ValueError, match="dictionary|kind: Pod"):
        module._prepare_pod_manifest_template(template)


def test_empty_template_container_is_independent_and_named_merges_preserve_unnamed_extensions():
    template = {"spec": {"containers": []}}
    prepared = module._prepare_pod_manifest_template(template)
    assert prepared["spec"]["containers"] == [{}] and template["spec"]["containers"] == []
    original = [{"name": "weights", "config": {"mode": "ro"}}]
    override = [{"name": "weights", "config": {"mode": "rw"}}, {"mountPath": "/legacy"}]
    merged = module._merge_named_items(original, override, "volumes")
    assert merged == override
    merged[0]["config"]["mode"] = "changed"
    assert original[0]["config"]["mode"] == "ro" and override[0]["config"]["mode"] == "rw"
    with pytest.raises(ValueError, match="must be a list"):
        module._merge_named_items({}, [], "volumes")
    with pytest.raises(ValueError, match="entries must be dictionaries"):
        module._merge_named_items([], ["bad"], "volumes")
    with pytest.raises(ValueError, match="entries must be dictionaries"):
        module._merge_named_items(["bad"], [], "volumes")
