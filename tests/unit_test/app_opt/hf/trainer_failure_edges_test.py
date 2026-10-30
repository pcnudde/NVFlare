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

import gc
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from nvflare.app_opt.hf import api  # noqa: E402
from nvflare.app_opt.hf.api import utils  # noqa: E402


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setenv(api.STRATEGY_ENV_VAR, api.STRATEGY_IN_MEMORY)
    monkeypatch.setattr(api, "_world_size", lambda: 1)
    monkeypatch.setattr(api.flare_api, "get_config", lambda: {})
    monkeypatch.setattr(api, "_ACTIVE_STATE", None)
    trainer = SimpleNamespace(
        model=SimpleNamespace(),
        args=SimpleNamespace(
            output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=2, gradient_accumulation_steps=1
        ),
    )
    return api._HFTaskState(
        trainer,
        rank=0,
        restore_state=False,
        load_state_dict_strict=True,
        params_scope="model",
        server_key_prefix=None,
        local_epochs=None,
        local_steps=1,
        stream_metrics=False,
    )


def test_hf_callback_fallback_and_active_state_do_not_retain_dead_trainer(monkeypatch):
    trainer = SimpleNamespace()
    marker = object()
    api._add_callback(trainer, marker)
    assert trainer.callbacks == [marker]
    previous = api._ACTIVE_STATE
    monkeypatch.setattr(api, "_ACTIVE_STATE", previous)

    class WeakState:
        pass

    state = WeakState()
    api._set_active_state(state)
    assert api._get_active_state() is state
    del state
    gc.collect()
    assert api._get_active_state() is None and api._ACTIVE_STATE is None
    assert api._task_fl_model_payload(None) is None


@pytest.mark.parametrize(
    "method,kind",
    [
        ("wrapped_train", api.TASK_STOP),
        ("wrapped_train", api.TASK_EVALUATE),
        ("wrapped_train", api.TASK_SUBMIT_MODEL),
        ("wrapped_train", "invalid"),
        ("wrapped_evaluate", "invalid"),
    ],
)
def test_hf_train_dispatch_never_enters_training_for_nontraining_task(state, monkeypatch, method, kind):
    state.pending = True
    monkeypatch.setattr(state, "_ensure_task", lambda call_name: kind)
    state._submit_model = Mock()
    state._complete_task = Mock()
    state.trainer.train = Mock()
    if kind == "invalid":
        with pytest.raises(RuntimeError, match="Unsupported HF task"):
            getattr(state, method)()
        assert state.aborted
    else:
        assert getattr(state, method)() is None
        assert state._submit_model.called is (kind == api.TASK_SUBMIT_MODEL)
        assert state._complete_task.called is (kind == api.TASK_SUBMIT_MODEL)
    state.trainer.train.assert_not_called()


def test_hf_evaluate_can_capture_pretrain_metrics_without_callback_delivery(state, monkeypatch):
    monkeypatch.setattr(state, "_ensure_task", lambda call_name: api.TASK_TRAIN)
    monkeypatch.setattr(state, "_load_global_params_once", Mock())
    setattr(state.trainer, api.ORIGINAL_EVALUATE_ATTR, lambda: {"eval_loss": 2.5})
    assert state.wrapped_evaluate() == {"eval_loss": 2.5}
    assert state.pre_train_metrics == {"eval_loss": 2.5}
    assert not state.capture_evaluation
    state.on_evaluate({"ignored": 1})
    assert state.pre_train_metrics == {"eval_loss": 2.5}
    state._inside_train = True
    state.on_evaluate({"ignored": 2})
    assert state.pre_train_metrics == {"eval_loss": 2.5}


def test_stateless_reset_handles_custom_state_constructor_without_reusing_optimizer(state):
    class CustomState:
        def __init__(self, required):
            self.global_step, self.epoch, self.num_input_tokens_seen = 9, 3, 20

    state.trainer.state = CustomState("required")
    state.trainer.control = CustomState("required")
    state.trainer.optimizer = object()
    state.trainer.lr_scheduler = object()
    state._reset_stateless_trainer_task_state()
    assert state.trainer.optimizer is None and state.trainer.lr_scheduler is None
    assert (
        state.trainer.state.global_step == state.trainer.state.epoch == state.trainer.state.num_input_tokens_seen == 0
    )
    assert state.trainer.control.global_step == 9


def test_hf_metadata_requires_dictionary_and_empty_metadata_uses_real_step_weight(state):
    setattr(state.trainer.model, api.FL_META_KEY, None)
    meta = state._build_meta(end_global_step=3, end_tokens=None)
    assert meta[api.MetaKey.NUM_STEPS_CURRENT_ROUND] == 6
    setattr(state.trainer.model, api.FL_META_KEY, [])
    with pytest.raises(RuntimeError, match="must be a dictionary"):
        state._build_meta(end_global_step=3, end_tokens=None)


def test_checkpoint_staging_failure_falls_back_to_object_broadcast_without_losing_params(state, monkeypatch):
    state.world_size = 2
    state._should_stage_params = lambda params: True
    monkeypatch.setattr(utils, "write_params_exchange_file", Mock(side_effect=OSError("read-only")))
    broadcast = Mock(side_effect=lambda obj, src: obj)
    monkeypatch.setattr(api, "_broadcast_object", broadcast)
    payload = {"task_kind": api.TASK_TRAIN, "params": {"w": [1, 2]}}
    result = state._broadcast_task_payload(payload)
    assert result == payload
    assert result["params"] == {"w": [1, 2]}
    broadcast.assert_called_once_with(payload, src=0)


def test_hf_single_rank_client_error_keeps_original_failure_and_rankless_fallback(state, monkeypatch):
    failure = ValueError("task source failed")
    monkeypatch.setattr(api.flare_api, "is_running", Mock(side_effect=failure))
    with pytest.raises(ValueError) as caught:
        state.is_running()
    assert caught.value is failure
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(api, "environment_declares_single_client_api_process", lambda: False)
    monkeypatch.setattr(api, "_torch_dist", lambda: None)
    assert api._resolve_rank(SimpleNamespace()) == 0
    assert api._all_gather_object({"status": 1}) == [{"status": 1}]
    assert api._extract_metrics(None) == {}
