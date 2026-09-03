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

"""PyTorch model persistor that never holds the whole model in memory."""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils import fobs

from .decomposers import TensorDecomposer
from .lazy_tensor_dict import (
    _LazyRef,
    materialize,
    metadata_of,
    read_safetensors_layout,
    safetensors_refs,
    write_safetensors,
)

HF_INDEX_FILE = "model.safetensors.index.json"
HF_MODEL_FILE = "model.safetensors"


def load_safetensors_refs(path: str) -> dict[str, _LazyRef]:
    """Lazy refs for a .safetensors file, a Hugging Face index, or a directory holding either."""
    path = os.path.realpath(path)
    if os.path.isdir(path):
        index_path = os.path.join(path, HF_INDEX_FILE)
        path = index_path if os.path.isfile(index_path) else os.path.join(path, HF_MODEL_FILE)
    if not os.path.isfile(path):
        raise ValueError(f"safetensors checkpoint not found: {path}")
    if path.endswith(".json"):
        return _index_refs(path)
    return safetensors_refs(path)


def _index_refs(index_path: str) -> dict[str, _LazyRef]:
    with open(index_path) as index_file:
        weight_map = json.load(index_file)["weight_map"]
    refs = {}
    for shard_name in sorted(set(weight_map.values())):
        refs.update(safetensors_refs(os.path.join(os.path.dirname(index_path), shard_name)))
    return refs


def _single_source_file(weights: dict) -> Optional[str]:
    """The one safetensors file all refs point into, when they cover exactly its tensors."""
    if not all(isinstance(ref, _LazyRef) and ref.key == key for key, ref in weights.items()):
        return None
    files = {ref.file_path for ref in weights.values()}
    if len(files) != 1:
        return None
    (file_path,) = files
    return file_path if set(read_safetensors_layout(file_path)) == set(weights) else None


class PTSafetensorsModelPersistor(ModelPersistor):
    """Persist a PyTorch model as safetensors without holding it in memory.

    ``load_model`` returns lazy refs into the source checkpoint: a ``.safetensors``
    file, a Hugging Face ``model.safetensors.index.json``, or a directory containing
    either. ``save_model`` streams tensors one at a time into the checkpoint file.
    When the weights are refs into one complete file, as produced by FedAvg with
    ``enable_tensor_disk_offload``, that file is hard-linked (copied across file
    systems) instead of being rewritten.

    Only tensors are persisted; ``FLModel.meta`` is not. The model inventory used by
    cross-site evaluation is not implemented.

    Args:
        source_ckpt_file_full_name: checkpoint to start from. A relative path is
            resolved against the app's custom folder.
        global_model_file_name: saved global model file name, under the app log dir.
        best_global_model_file_name: file name used for GLOBAL_BEST_MODEL_AVAILABLE.
        filter_id: optional PersistorFilter component id.
    """

    # Lazy refs materialize as PyTorch tensors, so the exchange with clients must stay in that format.
    required_exchange_format = ExchangeFormat.PYTORCH

    def __init__(
        self,
        source_ckpt_file_full_name: str,
        global_model_file_name: str = "FL_global_model.safetensors",
        best_global_model_file_name: str = "best_FL_global_model.safetensors",
        filter_id: Optional[str] = None,
    ):
        super().__init__(filter_id=filter_id)
        if not isinstance(source_ckpt_file_full_name, str) or not source_ckpt_file_full_name:
            raise ValueError("source_ckpt_file_full_name must be a non-empty string")
        self.source_ckpt_file_full_name = source_ckpt_file_full_name
        self.global_model_file_name = global_model_file_name
        self.best_global_model_file_name = best_global_model_file_name
        self.log_dir = None

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)
        elif event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            ml = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            if ml:
                self._save(ml.get(ModelLearnableKey.WEIGHTS), self._path(fl_ctx, self.best_global_model_file_name))

    def _initialize(self, fl_ctx: FLContext):
        if self.log_dir is None:
            app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
            log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
            self.log_dir = os.path.join(app_root, log_dir) if log_dir else app_root
            os.makedirs(self.log_dir, exist_ok=True)
        fobs.register(TensorDecomposer)

    def _path(self, fl_ctx: FLContext, file_name: str) -> str:
        self._initialize(fl_ctx)
        return os.path.join(self.log_dir, file_name)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        self._initialize(fl_ctx)
        path = self.source_ckpt_file_full_name
        if not os.path.isabs(path):
            path = os.path.join(fl_ctx.get_prop(FLContextKey.APP_ROOT), WorkspaceConstants.CUSTOM_FOLDER_NAME, path)
        return make_model_learnable(load_safetensors_refs(path), {})

    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        self._save(ml.get(ModelLearnableKey.WEIGHTS), self._path(fl_ctx, self.global_model_file_name))

    @staticmethod
    def _save(weights, path: str) -> None:
        if not isinstance(weights, dict) or not weights:
            raise ValueError("model weights must be a non-empty dict")
        temp_path = Path(f"{path}.tmp")
        temp_path.unlink(missing_ok=True)
        try:
            source = _single_source_file(weights)
            if source:
                try:
                    os.link(source, temp_path)
                except OSError:  # another file system
                    shutil.copyfile(source, temp_path)
            else:
                metadata = {key: metadata_of(value) for key, value in weights.items()}
                write_safetensors(
                    str(temp_path), metadata, ((key, materialize(value)) for key, value in weights.items())
                )
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
