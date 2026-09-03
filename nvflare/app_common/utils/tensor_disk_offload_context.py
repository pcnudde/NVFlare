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

import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from nvflare.fuel.utils.fobs import FOBSContextKey

_ENABLE_TENSOR_DISK_OFFLOAD = FOBSContextKey.TENSOR_DISK_OFFLOAD
_TENSOR_DISK_OFFLOAD_ROOT_DIR = "tensor_disk_offload_root_dir"


@dataclass
class TensorDiskOffloadContext:
    previous_value: Any = None
    previous_root_dir: Optional[str] = None
    root_dir: Optional[str] = None
    applied: bool = False
    # (cell, previous flag, previous root dir) for every further cell that decodes this job's payloads
    extra_cells: list = field(default_factory=list)


def _get_cell(engine):
    if not engine:
        return None

    run_manager = getattr(engine, "run_manager", None)
    if run_manager and run_manager.cell:
        return run_manager.cell
    return engine.get_cell()


def _get_extra_cells(engine, primary) -> list:
    """Further cells that decode this job's payloads.

    The simulator runs the server app and its job cell in one process: client results are decoded by
    the job cell, while the engine exposes the parent cell. Production job processes expose the job
    cell itself, so nothing is added there.
    """
    job_cell = getattr(getattr(engine, "server", None), "job_cell", None)
    return [job_cell] if job_cell is not None and job_cell is not primary else []


def setup_tensor_disk_offload(engine, enabled: bool, job_id: str = "job") -> TensorDiskOffloadContext:
    """Enable tensor disk offload in the FOBS context of every cell that decodes this job's payloads.

    Args:
        engine: engine that owns the active Cell.
        enabled: whether to prepare disk-backed tensor downloads.
        job_id: identifier used to name the temporary offload root.

    Returns:
      Context needed to restore the prior setting and cleanup temporary files.
    """
    if not enabled:
        return TensorDiskOffloadContext()

    cell = _get_cell(engine)
    if not cell:
        return TensorDiskOffloadContext()

    fobs_ctx = cell.get_fobs_context()
    previous_value = fobs_ctx.get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    previous_root_dir = fobs_ctx.get(_TENSOR_DISK_OFFLOAD_ROOT_DIR)
    root_dir = tempfile.mkdtemp(prefix=f"nvflare_tensor_offload_{job_id}_")
    props = {_ENABLE_TENSOR_DISK_OFFLOAD: True, _TENSOR_DISK_OFFLOAD_ROOT_DIR: root_dir}
    extra_cells = []
    try:
        cell.update_fobs_context(props)
        for extra in _get_extra_cells(engine, cell):
            extra_ctx = extra.get_fobs_context()
            extra_cells.append(
                (extra, extra_ctx.get(_ENABLE_TENSOR_DISK_OFFLOAD, False), extra_ctx.get(_TENSOR_DISK_OFFLOAD_ROOT_DIR))
            )
            extra.update_fobs_context(props)
    except Exception:
        shutil.rmtree(root_dir, ignore_errors=True)
        raise
    return TensorDiskOffloadContext(
        previous_value=previous_value,
        previous_root_dir=previous_root_dir,
        root_dir=root_dir,
        applied=True,
        extra_cells=extra_cells,
    )


def cleanup_tensor_disk_offload(engine, context: TensorDiskOffloadContext) -> None:
    """Restore the prior FOBS context values and remove any temporary offload root."""
    if not context:
        return

    try:
        if context.applied:
            cell = _get_cell(engine)
            if cell:
                cell.update_fobs_context(
                    {
                        _ENABLE_TENSOR_DISK_OFFLOAD: context.previous_value,
                        _TENSOR_DISK_OFFLOAD_ROOT_DIR: context.previous_root_dir,
                    }
                )
            for extra, previous_value, previous_root_dir in context.extra_cells:
                extra.update_fobs_context(
                    {_ENABLE_TENSOR_DISK_OFFLOAD: previous_value, _TENSOR_DISK_OFFLOAD_ROOT_DIR: previous_root_dir}
                )
    finally:
        if context.root_dir:
            shutil.rmtree(context.root_dir, ignore_errors=True)
