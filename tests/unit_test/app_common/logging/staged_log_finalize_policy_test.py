# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamContextKey
from nvflare.app_common.logging import job_log_receiver as subject
from nvflare.app_common.streamers.streamer_base import KEY_FILE_NAME


@pytest.mark.parametrize("outcome", ["moved", "move-failed", "no-workspace", "explicit-destination"])
def test_finalized_log_without_job_manager_moves_to_workspace_or_retains_staged_copy(tmp_path, monkeypatch, outcome):
    receiver = subject.JobLogReceiver(dest_dir=str(tmp_path) if outcome == "explicit-destination" else None)
    log = Mock()
    monkeypatch.setattr(receiver, "log_exception", log)
    engine = Mock()
    engine.get_component.return_value = None
    engine.get_workspace.return_value = (
        None if outcome == "no-workspace" else SimpleNamespace(get_run_dir=lambda job: str(tmp_path / "run"))
    )
    ctx, peer = FLContext(), FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site")
    peer.set_prop(ReservedKey.RUN_NUM, "job")
    ctx.set_peer_context(peer)
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    staged = tmp_path / "staged.log"
    staged.write_bytes(b"log payload")
    stream = {StreamContextKey.RC: ReturnCode.OK, subject._KEY_RECV_PATH: str(staged), KEY_FILE_NAME: "log.txt"}
    if outcome == "move-failed":
        monkeypatch.setattr(subject.shutil, "move", Mock(side_effect=OSError("read only")))
    with staged.open("ab") as handle:
        stream[subject._KEY_RECV_FILE] = handle
        receiver._finalize_stream(stream, ctx)
        assert handle.closed and stream[subject._KEY_RECV_FILE] is None
    assert log.call_count == int(outcome == "move-failed")
    if outcome == "moved":
        assert (tmp_path / "run" / "site" / "log.txt").read_bytes() == b"log payload"
        assert not staged.exists()
    else:
        assert staged.read_bytes() == b"log payload"
