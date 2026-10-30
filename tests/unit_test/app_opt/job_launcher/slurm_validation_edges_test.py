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

import pytest

from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobProcessArgs
from nvflare.app_opt.job_launcher.slurm import config, launcher
from nvflare.app_opt.job_launcher.slurm.config import SlurmLauncherError


@pytest.mark.parametrize(
    "method,args,match",
    [
        (config._require_string, ("nul\x00value", "field"), "forbidden NUL"),
        (config._validate_env_name, ("1INVALID",), "must match"),
        (config._validate_absolute_path, ("relative", "path"), "must be absolute"),
        (config._validate_mount_operand, ("/has:colon", "mount"), "unsupported"),
        (config._validate_mount_destination, ("/a//b", "mount"), "normalized"),
        (config._mapping_or_empty, ([], "spec"), "must be a mapping"),
        (config.normalize_slurm_directives, ({"unknown": 1},), "unsupported"),
        (config.normalize_slurm_directives, ({"partition": True},), "string or integer"),
        (config.normalize_slurm_executables, ([],), "must be a mapping"),
        (config.normalize_slurm_image, (None, "unsupported"), "sandbox must"),
        (config.normalize_slurm_image, (None, "apptainer"), "is required"),
        (config.normalize_multi_node_port_range, (4, 8102), "must have the form"),
        (config.normalize_slurm_workspace_path, ("/workspace,unsafe",), "must not contain"),
    ],
)
def test_slurm_rejects_unsafe_or_ambiguous_site_configuration_before_scheduler_invocation(method, args, match):
    with pytest.raises(SlurmLauncherError, match=match):
        method(*args)


def test_slurm_exec_and_options_normalization_preserve_token_boundaries():
    result = config.normalize_slurm_executables({"srun": "srun"})
    assert result["srun"] == "srun" and result["sbatch"] is None
    assert launcher._module_args(
        {JobProcessArgs.OPTIONS: ("--options", 'key="value with spaces"')}, [JobProcessArgs.OPTIONS]
    ) == ("--options", "key=value with spaces")
    for value, message in [(3, "must be a string"), ("unclosed'", "malformed")]:
        with pytest.raises(SlurmLauncherError, match=message):
            launcher._module_args({JobProcessArgs.OPTIONS: ("--options", value)}, [JobProcessArgs.OPTIONS])


@pytest.mark.parametrize(
    "settings,message", [({"internal_port": 65536}, "at most 65535"), ({"forward_env": "NAME"}, "must be a list")]
)
def test_slurm_launcher_rejects_invalid_control_port_and_environment_shape(settings, message):
    base = dict(
        sandbox="none",
        python_path="/usr/bin/python3",
        executables={},
        image=None,
        internal_port=8102,
        sbatch_directives=None,
        setup=None,
        forward_env=None,
        parent_host=None,
        submit_timeout=5,
        query_timeout=5,
        cancel_timeout=5,
        poll_interval=1,
        pending_timeout=10,
    )
    base.update(settings)
    with pytest.raises(SlurmLauncherError, match=message):
        config.normalize_slurm_launcher_settings(**base)


def test_slurm_gpu_topology_rejects_inconsistent_portable_resource_totals():
    meta = {JobMetaKey.RESOURCE_SPEC.value: {"site": {"num_of_gpus": 3}}}
    with pytest.raises(SlurmLauncherError, match="must equal nodes"):
        launcher._resolve_resources(meta, "site", "none", 10, {"nodes": 2, "gpus_per_node": 2})
    with pytest.raises(SlurmLauncherError, match="require explicit gpus_per_node"):
        launcher._resolve_resources(meta, "site", "none", 10, {"nodes": 2})


def test_slurm_job_directory_must_be_direct_child_and_parent_url_must_parse(tmp_path):
    nested = tmp_path / "run" / "nested"
    nested.mkdir(parents=True)
    with pytest.raises(SlurmLauncherError, match="immediate child"):
        launcher._validate_child_dir(str(tmp_path), str(nested), "run", "workspace")
    with pytest.raises(SlurmLauncherError, match="malformed parent URL"):
        launcher._rewrite_parent_url({JobProcessArgs.PARENT_URL: ("-p", "tcp://[invalid")}, "host", 8102)
