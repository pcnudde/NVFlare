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

import errno
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import EnvironmentKey
from nvflare.app_opt.pt import file_model_persistor as subject


@pytest.fixture
def context(tmp_path, isolated_fobs):
    ctx = FLContext()
    ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path))
    ctx.set_prop(ReservedKey.ENGINE, MagicMock())
    return ctx


def test_environment_config_selects_checkpoint_paths_and_real_initial_weights(context, tmp_path):
    model = torch.nn.Linear(2, 1)
    torch.save(model.state_dict(), tmp_path / "initial.pt")
    (tmp_path / "environment.json").write_text(
        json.dumps({EnvironmentKey.CHECKPOINT_DIR: "checkpoints", EnvironmentKey.CHECKPOINT_FILE_NAME: "initial.pt"})
    )
    context.set_prop(FLContextKey.ARGS, SimpleNamespace(env="environment.json"))
    persistor = subject.PTFileModelPersistor(model=model, load_device="cpu", exclude_vars="bias")
    persistor.handle_event(EventType.START_RUN, context)
    assert persistor.log_dir == str(tmp_path / "checkpoints")
    assert persistor.ckpt_preload_path == str(tmp_path / "initial.pt")
    loaded = persistor.load_model(context)
    assert set(loaded["weights"]) == {"weight"}
    torch.testing.assert_close(torch.from_numpy(loaded["weights"]["weight"]), model.weight.detach())
    assert (tmp_path / "checkpoints").is_dir()


@pytest.mark.parametrize("config", [{}, {"path": "missing.module.Class"}, {"path": "builtins.dict"}, object()])
def test_initialize_rejects_bad_model_definitions_without_creating_manager(context, config):
    persistor = subject.PTFileModelPersistor(model=config)
    persistor.system_panic = MagicMock()
    persistor._initialize(context)
    persistor.system_panic.assert_called_once()
    assert persistor.persistence_manager is None


def test_initialize_constructs_dictionary_model_and_resolves_component_ids(context):
    persistor = subject.PTFileModelPersistor(
        model={"path": "torch.nn.Linear", "args": {"in_features": 2, "out_features": 1}}
    )
    persistor._initialize(context)
    assert isinstance(persistor.model, torch.nn.Linear)
    assert persistor.model.weight.shape == (1, 2)
    context.get_engine().get_component.return_value = persistor.model
    component = subject.PTFileModelPersistor(model="network")
    component._initialize(context)
    assert component.model is persistor.model
    component.system_panic = MagicMock()
    component.model = "missing"
    context.get_engine().get_component.return_value = None
    component._initialize(context)
    component.system_panic.assert_called_once()


def test_load_failures_for_bad_env_missing_checkpoint_and_invalid_checkpoint_are_explicit(context, tmp_path):
    (tmp_path / "bad.json").write_text("invalid")
    context.set_prop(FLContextKey.ARGS, SimpleNamespace(env="bad.json"))
    persistor = subject.PTFileModelPersistor()
    persistor.system_panic = MagicMock()
    persistor._initialize(context)
    assert "error opening env config" in persistor.system_panic.call_args.kwargs["reason"]
    context.set_prop(FLContextKey.ARGS, None)
    persistor.source_ckpt_file_full_name = "missing.pt"
    assert persistor.load_model(context) is None
    assert "Source checkpoint not found" in persistor.system_panic.call_args.kwargs["reason"]
    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"not a checkpoint")
    persistor.source_ckpt_file_full_name = str(invalid)
    assert persistor.load_model(context) is None
    assert persistor.system_panic.call_args.kwargs["reason"] == "cannot load model checkpoint"


def test_model_inventory_describes_initial_current_and_best_checkpoints_and_loads_on_cpu(context, tmp_path):
    model = torch.nn.Linear(2, 1)
    persistor = subject.PTFileModelPersistor(model=model, exclude_vars="bias")
    persistor._initialize(context)
    persistor.load_model(context)
    source = tmp_path / "initial.pt"
    torch.save(model.state_dict(), source)
    persistor.source_ckpt_file_full_name = str(source)
    persistor.save_model_file(persistor._ckpt_save_path)
    persistor.save_model_file(persistor._best_ckpt_save_path)
    inventory = persistor.get_model_inventory(context)
    assert len(inventory) == 3
    assert inventory["initial.pt"].props == {"source": "initial_ckpt"}
    assert inventory[persistor.global_model_file_name].location == persistor._ckpt_save_path
    assert persistor.get_model("initial.pt", context)["weights"]["weight"].shape == (1, 2)
    assert persistor.get_model("missing", context) is None
    assert persistor._get_model_from_location(str(tmp_path / "missing.pt"), context) is None
    empty = subject.PTFileModelPersistor()
    empty.log_dir = str(tmp_path / "empty")
    assert empty.get_model("missing", context) is None


@pytest.mark.parametrize("error,raises", [(errno.EINVAL, False), (errno.EIO, True)])
def test_directory_sync_closes_descriptor_when_filesystem_rejects_fsync(monkeypatch, error, raises):
    close = MagicMock()
    filesystem = SimpleNamespace(
        name="posix",
        O_RDONLY=0,
        open=MagicMock(return_value=42),
        fsync=MagicMock(side_effect=OSError(error, "sync")),
        close=close,
    )
    monkeypatch.setattr(subject, "os", filesystem)
    if raises:
        with pytest.raises(OSError):
            subject.PTFileModelPersistor._fsync_directory("dir")
    else:
        subject.PTFileModelPersistor._fsync_directory("dir")
    close.assert_called_once_with(42)
