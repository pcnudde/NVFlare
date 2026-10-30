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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from nvflare.app_opt.hf import utils


def _trainer():
    return SimpleNamespace(model=torch.nn.Linear(2, 1, bias=False))


def test_params_exchange_falls_back_to_torch_for_mixed_tensor_and_scalar_payload(tmp_path):
    descriptor = utils.write_params_exchange_file(str(tmp_path), {"weight": torch.tensor([1.0, 2.0]), "step": 7})
    assert descriptor["format"] == utils.PARAMS_EXCHANGE_FORMAT_TORCH
    loaded = utils.read_params_exchange_file(descriptor)
    assert torch.equal(loaded["weight"], torch.tensor([1.0, 2.0]))
    assert loaded["step"] == 7
    assert not list(tmp_path.rglob("*.tmp"))
    utils.cleanup_params_exchange_file(descriptor)
    assert not Path(descriptor["path"]).exists()
    utils.cleanup_params_exchange_file(descriptor)


@pytest.mark.parametrize("backend", ["safetensors", "torch"])
def test_atomic_exchange_writer_removes_partial_file_after_serializer_failure(monkeypatch, tmp_path, backend):
    import safetensors.torch as safe

    def failing_save(params, path):
        Path(path).write_bytes(b"partial")
        raise ValueError("serializer failed")

    monkeypatch.setattr(
        safe if backend == "safetensors" else torch, "save_file" if backend == "safetensors" else "save", failing_save
    )
    writer = utils._write_safetensors_exchange_file if backend == "safetensors" else utils._write_torch_exchange_file
    with pytest.raises(ValueError, match="serializer failed"):
        writer(str(tmp_path), {"w": torch.ones(2)})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("pretrained", [True, False])
def test_checkpoint_injection_loads_weights_before_saving_supported_model_format(monkeypatch, tmp_path, pretrained):
    trainer = _trainer()
    incoming = {"weight": torch.tensor([[3.0, 4.0]])}
    save = MagicMock(side_effect=lambda directory: torch.testing.assert_close(trainer.model.weight, incoming["weight"]))
    if pretrained:
        monkeypatch.setattr(trainer.model, "save_pretrained", save, raising=False)
    utils.write_params_to_checkpoint(trainer, str(tmp_path / "checkpoint"), incoming, "model")
    torch.testing.assert_close(trainer.model.weight, incoming["weight"])
    if pretrained:
        save.assert_called_once_with(str(tmp_path / "checkpoint"))
    else:
        loaded = utils.extract_params_from_checkpoint(str(tmp_path / "checkpoint"), "model")
        torch.testing.assert_close(loaded["weight"], incoming["weight"])


def test_checkpoint_injection_falls_back_to_torch_after_safetensors_failure(monkeypatch, tmp_path):
    import safetensors.torch as safe

    monkeypatch.setattr(safe, "save_file", MagicMock(side_effect=ValueError("unsupported layout")))
    trainer = _trainer()
    incoming = {"weight": torch.tensor([[5.0, 6.0]])}
    utils.write_params_to_checkpoint(trainer, str(tmp_path), incoming, "model")
    assert (tmp_path / "pytorch_model.bin").is_file()
    torch.testing.assert_close(
        utils.extract_params_from_checkpoint(str(tmp_path), "model")["weight"], incoming["weight"]
    )


@pytest.mark.parametrize("index", [{}, {"weight_map": {}}, {"weight_map": {"w": "missing.bin"}}])
def test_sharded_checkpoint_rejects_invalid_index_or_missing_shard(tmp_path, index):
    path = tmp_path / "pytorch_model.bin.index.json"
    path.write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="checkpoint"):
        utils._load_sharded_checkpoint(str(tmp_path), str(path))


def test_torch_checkpoint_loader_supports_old_torch_signature(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.bin"
    torch.save({"w": torch.tensor([2.0])}, checkpoint)
    original = torch.load

    def old_load(path, **kwargs):
        if "weights_only" in kwargs:
            raise TypeError("old Torch signature")
        return original(path, **kwargs)

    load = MagicMock(side_effect=old_load)
    monkeypatch.setattr(torch, "load", load)
    result = utils._load_checkpoint_file(str(checkpoint))
    assert torch.equal(result["w"], torch.tensor([2.0]))
    assert load.call_count == 2
    assert "weights_only" in load.call_args_list[0].kwargs
    assert "weights_only" not in load.call_args_list[1].kwargs


def test_checkpoint_missing_and_unknown_exchange_format_fail_without_loading(tmp_path):
    assert utils.extract_params_from_checkpoint(str(tmp_path), "model") is None
    assert utils.find_checkpoint_for_step(str(tmp_path), None) is None
    assert utils.get_fl_exchange_dir(str(tmp_path)).startswith(str(tmp_path))
    with pytest.raises(RuntimeError, match="Unsupported"):
        utils.read_params_exchange_file({"path": str(tmp_path / "file"), "format": "unknown"})


@pytest.mark.parametrize("operation", [utils.strip_server_key_prefix, utils.apply_server_key_prefix])
def test_prefix_conversion_rejects_non_string_keys_and_preserves_empty_payload(operation):
    assert operation(None, "model.") == {}
    with pytest.raises(TypeError, match="parameter keys"):
        operation({3: torch.ones(1)}, "model.")


def test_model_unwrap_retries_legacy_accelerator_signature_and_rejects_missing_model():
    model = torch.nn.Linear(1, 1)
    unwrap = MagicMock(side_effect=[TypeError("legacy signature"), model])
    assert utils.unwrap_model(SimpleNamespace(model=model, accelerator=SimpleNamespace(unwrap_model=unwrap))) is model
    assert unwrap.call_args.kwargs == {"keep_fp32_wrapper": False}
    with pytest.raises(ValueError, match="model attribute"):
        utils.unwrap_model(SimpleNamespace())
    assert utils.is_peft_model(None) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_rounds", "invalid"),
        ("total_rounds", 0),
        ("dataset_len", "invalid"),
        ("dataset_len", -1),
        ("num_train_epochs", -1),
        ("per_device_train_batch_size", 0),
        ("gradient_accumulation_steps", "invalid"),
    ],
)
def test_training_step_budget_rejects_invalid_counts(field, value):
    kwargs = dict(dataset_len=10, total_rounds=2)
    args = dict(max_steps=0, num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=1)
    if field in kwargs:
        kwargs[field] = value
    else:
        args[field] = value
    with pytest.raises(ValueError, match=field):
        utils.total_train_steps(args=SimpleNamespace(**args), **kwargs)


def test_world_size_invalid_env_falls_back_to_one_and_preserves_non_tensor_parameters(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "invalid")
    args = SimpleNamespace(
        max_steps=0, num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=1
    )
    assert utils.total_train_steps(10, args, 2) == 10
    assert utils.prepare_out_params({"step": 3}, exchange_format="numpy", server_expected_format="numpy") == {"step": 3}
    trainer = _trainer()
    assert utils.load_params(trainer, {}, "model") is None
    loaded = utils._as_tensor_params({"weight": [[2.0, 3.0]]}, trainer.model.state_dict())
    assert loaded["weight"].dtype == torch.float32
    torch.testing.assert_close(loaded["weight"], torch.tensor([[2.0, 3.0]]))


def test_incompatible_key_tuple_and_unknown_result_are_handled(caplog):
    utils._log_incompatible_keys((["missing"], ["unexpected"]))
    assert "Missing keys" in caplog.text and "Unexpected keys" in caplog.text
    utils._log_incompatible_keys(object())
