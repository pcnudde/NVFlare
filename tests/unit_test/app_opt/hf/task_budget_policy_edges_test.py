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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from nvflare.app_opt.hf import api
from nvflare.client.config import ExchangeFormat


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setenv(api.STRATEGY_ENV_VAR, api.STRATEGY_IN_MEMORY)
    monkeypatch.setattr(api, "_world_size", lambda: 1)
    monkeypatch.setattr(api.flare_api, "get_config", lambda: {})
    trainer = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        args=SimpleNamespace(
            output_dir=str(tmp_path),
            max_steps=0,
            num_train_epochs=1.5,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=2,
        ),
        get_train_dataloader=lambda: list(range(5)),
    )
    return api._HFTaskState(
        trainer,
        rank=0,
        restore_state=True,
        load_state_dict_strict=True,
        params_scope="model",
        server_key_prefix=None,
        local_epochs=None,
        local_steps=None,
        stream_metrics=False,
    )


@pytest.mark.parametrize("epochs", [None, 0.5, 2.0])
def test_budget_converts_batches_and_accumulation_into_rounded_optimizer_steps(state, epochs):
    state.local_epochs = epochs
    state._capture_budget_if_needed()
    assert state.per_round_budget_steps == (5 if epochs is None else 2 if epochs == 0.5 else 6)
    assert state.budget_source == ("args.num_train_epochs" if epochs is None else "local_epochs")


@pytest.mark.parametrize("mode", ["missing_loader", "zero_budget"])
def test_budget_requires_real_dataloader_and_positive_steps(state, mode):
    if mode == "missing_loader":
        state.local_epochs = 1
        state.trainer.get_train_dataloader = None
        with pytest.raises(RuntimeError, match="get_train_dataloader"):
            state._capture_budget_if_needed()
    else:
        state.local_steps = 0
        with pytest.raises(ValueError, match="at least one"):
            state._capture_budget_if_needed()


@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_budget_failure_is_propagated_before_training(monkeypatch, state, rank):
    state.world_size = 2
    state.rank = rank
    state._capture_budget_local = MagicMock(side_effect=ValueError("empty training input"))
    broadcast = MagicMock(
        side_effect=lambda payload, src: (
            payload if rank == 0 else {"ok": False, "error": "ValueError: empty training input"}
        )
    )
    monkeypatch.setattr(api, "_broadcast_object", broadcast)
    with pytest.raises(RuntimeError, match="budget capture failed") as caught:
        state._capture_budget_if_needed()
    assert state._capture_budget_local.call_count == int(rank == 0)
    assert isinstance(caught.value.__cause__, ValueError) is (rank == 0)
    assert state.per_round_budget_steps is None


@pytest.mark.parametrize("total_rounds", [None, 2])
def test_cumulative_budget_extends_when_server_schedules_another_round(state, total_rounds):
    state.total_rounds = total_rounds
    state.per_round_budget_steps = 3
    state.train_start_global_step = 4
    state._apply_cumulative_max_steps()
    expected = 7 if total_rounds is None else 6
    assert state.trainer.args.max_steps == expected
    state.train_start_global_step = expected
    state._apply_cumulative_max_steps()
    assert state.trainer.args.max_steps == expected + 3
    assert state.metric_step(7) == 7
    state.restore_state = False
    state.metric_step_offset = 9
    assert state.metric_step(7) == 16
    state._extend_cumulative_max_steps_if_needed()
    assert state.cumulative_max_steps == expected + 3


def test_pending_task_rejects_prefetch_until_expected_trainer_call(state):
    state.pending = True
    state.task_kind = api.TASK_EVALUATE
    with pytest.raises(RuntimeError, match="trainer.evaluate"):
        state.is_running()
    state.task_kind = api.TASK_TRAIN
    with pytest.raises(RuntimeError, match="trainer.train"):
        state.is_running()
    state.task_kind = api.TASK_SUBMIT_MODEL
    with pytest.raises(RuntimeError, match="train.*or.*evaluate"):
        state.is_running()


def test_checkpoint_resume_warns_once_when_round_is_missing(state, tmp_path, caplog):
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    state.last_checkpoint_path = str(checkpoint)
    state.current_round = None
    assert state._resume_checkpoint_path() == str(checkpoint)
    assert state._resume_checkpoint_path() == str(checkpoint)
    assert caplog.text.count("without current_round") == 1
    state.current_round = 0
    assert state._resume_checkpoint_path() is None


def test_config_read_failure_uses_default_exchange_and_evaluation_policy(monkeypatch, state):
    monkeypatch.setattr(api.flare_api, "get_config", MagicMock(side_effect=RuntimeError("client not initialized")))
    assert state._train_with_evaluation_enabled() is False
    assert state._exchange_format() == ExchangeFormat.PYTORCH
    assert state._server_expected_format() == ExchangeFormat.NUMPY


@pytest.mark.parametrize("value", ["garbage", "-1"])
def test_invalid_file_exchange_threshold_uses_default(monkeypatch, value, caplog):
    monkeypatch.setenv(api.PARAMS_FILE_EXCHANGE_MIN_BYTES_ENV_VAR, value)
    assert api._params_file_exchange_min_bytes() == api.DEFAULT_PARAMS_FILE_EXCHANGE_MIN_BYTES
    assert "Invalid" in caplog.text


def test_invalid_exchange_or_weight_strategy_is_rejected_or_falls_back(monkeypatch, state):
    monkeypatch.setenv(api.PARAMS_EXCHANGE_STRATEGY_ENV_VAR, "unknown")
    assert api._params_exchange_strategy() == api.PARAMS_EXCHANGE_STRATEGY_OBJECT
    monkeypatch.setenv(api.STRATEGY_ENV_VAR, "unknown")
    with pytest.raises(ValueError, match=api.STRATEGY_ENV_VAR):
        state._select_weight_override_strategy()


@pytest.mark.parametrize("value", [None, "invalid", object()])
def test_optional_int_ignores_nonconvertible_metadata(value):
    assert api._optional_int(value) is None


def test_callback_registration_updates_existing_fl_callbacks_without_duplicates(state):
    from nvflare.app_opt.hf.callbacks import FLCallback, FLMetricsCallback

    previous = SimpleNamespace()
    callback = FLCallback(previous)
    state.trainer.callbacks = [callback]
    api._register_callbacks(state.trainer, state, stream_metrics=True)
    assert callback.task_state is state
    assert len(state.trainer.callbacks) == 2
    assert isinstance(state.trainer.callbacks[1], FLMetricsCallback)
    api._register_callbacks(state.trainer, state, stream_metrics=True)
    assert len(state.trainer.callbacks) == 2


def test_nested_evaluation_during_train_delegates_without_starting_new_task(state):
    state._inside_train = True
    setattr(state.trainer, api.ORIGINAL_EVALUATE_ATTR, MagicMock(return_value={"eval_loss": 0.25}))
    assert state.wrapped_evaluate(metric_key_prefix="train_eval") == {"eval_loss": 0.25}
    state.original_evaluate.assert_called_once_with(metric_key_prefix="train_eval")
    assert not state.pending
