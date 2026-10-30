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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.defs import SysCommandTopic
from nvflare.private.fed.client import sys_cmd as subject


@pytest.mark.parametrize("gpu", ["absent", "present", "library-missing"])
def test_system_report_includes_memory_and_available_gpu_names(monkeypatch, gpu):
    monkeypatch.setattr(
        subject, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(_asdict=lambda: {"total": 1024}))
    )

    class MissingLibrary(Exception):
        pass

    nvml = Mock()
    nvml.nvml = SimpleNamespace(NVMLError_LibraryNotFound=MissingLibrary)
    nvml.nvmlDeviceGetCount.return_value = 2
    nvml.nvmlDeviceGetName.side_effect = [b"GPU-0", b"GPU-1"]
    if gpu == "library-missing":
        nvml.nvmlInit.side_effect = MissingLibrary()
    monkeypatch.setattr(subject, "pynvml", None if gpu == "absent" else nvml)
    processor = subject.SysInfoProcessor()
    assert processor.get_topics() == [SysCommandTopic.SYS_INFO]
    reply = processor.process(Message("info", {}), None)
    expected = {"total": 1024}
    if gpu == "present":
        expected.update(gpu_count=2, gpu_device_0="GPU-0", gpu_device_1="GPU-1")
    assert reply.topic == "reply_info"
    assert json.loads(reply.body) == expected
    assert nvml.nvmlShutdown.call_count == int(gpu == "present")


@pytest.mark.parametrize("connected", [False, True])
def test_environment_report_includes_site_security_workspace_and_root_url(connected):
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-a")
    context.set_prop(FLContextKey.WORKSPACE_ROOT, "/workspace")
    context.set_prop(FLContextKey.SECURE_MODE, True)
    if connected:
        client = Mock()
        client.cell.get_root_url_for_child.return_value = "tcp://server:1234"
        context.set_prop(SystemComponents.FED_CLIENT, client)
    engine = Mock()
    engine.new_context.return_value = context
    processor = subject.ReportEnvProcessor()
    assert processor.get_topics() == [SysCommandTopic.REPORT_ENV]
    reply = processor.process(Message("env", {}), engine)
    assert reply.topic == "reply_env"
    assert json.loads(reply.body) == {
        "site_name": "site-a",
        "workspace": "/workspace",
        "secure_mode": True,
        "root_url": "tcp://server:1234" if connected else "",
    }


@pytest.mark.parametrize("failure", [None, "validation", "configuration"])
def test_site_log_config_is_validated_before_reconfiguration(monkeypatch, failure):
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-a")
    workspace = Mock()
    workspace.get_root_dir.return_value = "/workspace"
    workspace.get_log_config_file_path.return_value = "/logging.json"
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    engine = Mock()
    engine.new_context.return_value = context
    validate = Mock(
        return_value={"level": "INFO"}, side_effect=ValueError("bad configuration") if failure == "validation" else None
    )
    configure = Mock(side_effect=OSError("write failed") if failure == "configuration" else None)
    monkeypatch.setattr(subject, "validate_site_log_config", validate)
    monkeypatch.setattr(subject, "dynamic_log_config", configure)
    processor = subject.ConfigureSiteLogProcessor()
    assert processor.get_topics() == [SysCommandTopic.CONFIGURE_SITE_LOG]
    reply = processor.process(Message("log", "INFO"), engine)
    validate.assert_called_once_with("INFO")
    if failure is None:
        configure.assert_called_once_with(config={"level": "INFO"}, dir_path="/workspace", reload_path="/logging.json")
        assert reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.OK
        assert reply.body == "successfully configured site-a log"
    else:
        assert reply.get_header(MsgHeader.RETURN_CODE) != ReturnCode.OK
        assert ("bad configuration" if failure == "validation" else "write failed") in reply.body
    assert configure.call_count == int(failure != "validation")
