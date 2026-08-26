# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import threading
import time
from abc import ABC, abstractmethod

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import AdminCommandNames, ConnPropKey, FLContextKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobProcessArgs, JobReturnCode
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.exit_codes import PROCESS_EXIT_REASON, ProcessExitCode
from nvflare.fuel.f3.cellnet.core_cell import FQCN
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.defs import CellChannel, CellChannelTopic, JobFailureMsgKey, new_cell_message
from nvflare.private.fed.utils.fed_utils import get_job_launcher, get_return_code
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .client_process import Phase, StopIntent
from .client_process_driver import ClientProcessDriver
from .client_status import ClientStatus, get_status_message

REPORTABLE_JOB_FAILURES = {
    ProcessExitCode.EXCEPTION: PROCESS_EXIT_REASON[ProcessExitCode.EXCEPTION],
    ProcessExitCode.UNSAFE_COMPONENT: PROCESS_EXIT_REASON[ProcessExitCode.UNSAFE_COMPONENT],
    ProcessExitCode.CONFIG_ERROR: PROCESS_EXIT_REASON[ProcessExitCode.CONFIG_ERROR],
    ProcessExitCode.INFRASTRUCTURE_ERROR: PROCESS_EXIT_REASON[ProcessExitCode.INFRASTRUCTURE_ERROR],
    JobReturnCode.ABORTED: "aborted",
}


class ClientExecutor(ABC):
    @abstractmethod
    def start_app(
        self,
        client,
        job_id,
        job_meta,
        args,
        allocated_resource,
        token,
        resource_manager,
        fl_ctx: FLContext,
    ):
        """Starts the client app.

        Args:
            client: the FL client object
            job_id: the job_id
            args: admin command arguments for starting the FL client training
            allocated_resource: allocated resources
            token: token from resource manager
            resource_manager: resource manager
            fl_ctx: FLContext
        """
        pass

    @abstractmethod
    def check_status(self, job_id) -> str:
        """Checks the status of the running client.

        Args:
            job_id: the job_id

        Returns:
            A client status message
        """
        pass

    @abstractmethod
    def abort_app(self, job_id):
        """Aborts the running app.

        Args:
            job_id: the job_id
        """
        pass

    @abstractmethod
    def abort_task(self, job_id):
        """Aborts the client executing task.

        Args:
            job_id: the job_id
        """
        pass

    @abstractmethod
    def get_run_info(self, job_id):
        """Gets the run information.

        Args:
            job_id: the job_id

        Returns:
            A dict of run information.
        """

    @abstractmethod
    def get_errors(self, job_id):
        """Get the error information.

        Returns:
            A dict of error information.

        """

    @abstractmethod
    def reset_errors(self, job_id):
        """Resets the error information.

        Args:
            job_id: the job_id
        """


class JobExecutor(ClientExecutor):
    """Run the Client executor in a child process."""

    def __init__(self, client, startup):
        """To init the ProcessExecutor.

        Args:
            startup: startup folder
        """
        self.client = client
        self.logger = get_obj_logger(self)
        self.startup = startup
        self.processes = ClientProcessDriver()

        self.job_query_timeout = ConfigService.get_float_var(
            name="job_query_timeout", conf=SystemConfigs.APPLICATION_CONF, default=5.0
        )

    def start_app(
        self,
        client,
        job_id,
        job_meta,
        args,
        allocated_resource,
        token,
        resource_manager: ResourceManagerSpec,
        fl_ctx: FLContext,
    ):
        """Starts the app.

        Args:
            client: the FL client object
            job_id: the job_id
            job_meta: job metadata
            args: admin command arguments for starting the worker process
            allocated_resource: allocated resources
            token: token from resource manager
            resource_manager: resource manager
            fl_ctx: FLContext
        """
        # use a deep copy of the args for operation since its content will be changed!
        args = copy.deepcopy(args)

        # update the job meta
        workspace = Workspace(args.workspace, site_name=client.client_name)
        meta_file = workspace.get_job_meta_path(job_id)
        if not os.path.exists(meta_file):
            raise RuntimeError(f"missing deployed job metadata file for job '{job_id}': {meta_file}")
        with open(meta_file) as f:
            deployed_job_meta = json.load(f)
        for meta_key in (
            JobMetaKey.JOB_ID.value,
            JobMetaKey.RESOURCE_SPEC.value,
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
            JobMetaKey.SCOPE.value,
            JobMetaKey.STUDY.value,
        ):
            deployed_value = deployed_job_meta.get(meta_key)
            if deployed_value != job_meta.get(meta_key):
                raise RuntimeError(f"START_JOB metadata differs from deployed job metadata for '{meta_key}'")
            if meta_key in deployed_job_meta:
                job_meta[meta_key] = copy.deepcopy(deployed_value)
        if deployed_job_meta.get(AppValidationKey.BYOC, False):
            job_meta[AppValidationKey.BYOC] = True
        else:
            job_meta.pop(AppValidationKey.BYOC, None)

        # Preserve deploy-time launch metadata while recording scheduler-maintained start metadata.
        with open(meta_file, "w") as f:
            json.dump(job_meta, f, indent=4)

        job_launcher: JobLauncherSpec = get_job_launcher(job_meta, fl_ctx)

        # prepare command args for the job process
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        server_config = fl_ctx.get_prop(FLContextKey.SERVER_CONFIG)
        if not server_config:
            raise RuntimeError(f"missing {FLContextKey.SERVER_CONFIG} in FL context")
        service = server_config[0].get("service", {})
        if not isinstance(service, dict):
            raise RuntimeError(f"expect server config data to be dict but got {type(service)}")
        command_options = ""
        for t in args.set:
            command_options += " " + t
        command_options += " print_conf=True"
        args.set.append("print_conf=True")

        # Job process args are the same for all job launchers! Letting each job launcher compute the job
        # args would be error-prone and would require access to internal server components (e.g. cell).
        # We prepare job process args here and save the prepared result in the fl_ctx.
        # This way, the job launcher won't need to compute these args again.
        # The job launcher will only need to use the args properly to launch the job process!
        #
        # Each arg is a tuple of (arg_option, arg_value).
        # Note that the arg_option is fixed for each arg, and is not launcher specific!
        job_args = {
            JobProcessArgs.EXE_MODULE: ("-m", "nvflare.private.fed.app.client.worker_process"),
            JobProcessArgs.JOB_ID: ("-n", job_id),
            JobProcessArgs.CLIENT_NAME: ("-c", client.client_name),
            JobProcessArgs.AUTH_TOKEN: ("-t", client.token),
            JobProcessArgs.TOKEN_SIGNATURE: ("-ts", client.token_signature),
            JobProcessArgs.SSID: ("-d", client.ssid),
            JobProcessArgs.WORKSPACE: ("-m", args.workspace),
            JobProcessArgs.STARTUP_DIR: ("-w", workspace_obj.get_startup_kit_dir()),
            JobProcessArgs.PARENT_URL: ("-p", str(client.cell.get_internal_listener_url())),
            JobProcessArgs.SCHEME: ("-scheme", service.get("scheme", "grpc")),
            JobProcessArgs.TARGET: ("-g", service.get("target")),
            JobProcessArgs.STARTUP_CONFIG_FILE: ("-s", "fed_client.json"),
            JobProcessArgs.OPTIONS: ("--set", command_options),
        }

        params = client.cell.get_internal_listener_params()
        if params:
            parent_conn_sec = params.get(ConnPropKey.CONNECTION_SECURITY)
            if parent_conn_sec:
                job_args[JobProcessArgs.PARENT_CONN_SEC] = ("--parent_conn_sec", parent_conn_sec)

        fl_ctx.set_prop(key=FLContextKey.JOB_PROCESS_ARGS, value=job_args, private=True, sticky=False)
        self.processes.register(job_id)
        try:
            job_handle = job_launcher.launch_job(job_meta, fl_ctx)
            if job_handle is None:
                raise RuntimeError(f"job launcher returned no job handle for job '{job_id}'")
        except BaseException:
            self.processes.launch_failed(job_id)
            raise

        stop_intent = self.processes.attach_handle(job_id, job_handle)
        if stop_intent != StopIntent.NONE:
            self.abort_app(job_id, heartbeat_cleanup=stop_intent == StopIntent.HEARTBEAT_CLEANUP)

        self.logger.info(f"Launched job {job_id} with job launcher: {type(job_launcher)} ")

        fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
        engine = fl_ctx.get_engine()
        engine.fire_event(EventType.AFTER_JOB_LAUNCH, fl_ctx)

        client.multi_gpu = False

        thread = threading.Thread(
            target=self._wait_child_process_finish,
            args=(client, job_id, allocated_resource, token, resource_manager, args.workspace, fl_ctx),
        )
        thread.start()

    def _get_job_launcher(self, job_meta: dict, fl_ctx: FLContext) -> JobLauncherSpec:
        engine = fl_ctx.get_engine()
        fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
        engine.fire_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

        job_launcher = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
        if not (job_launcher and isinstance(job_launcher, list)):
            raise RuntimeError(f"There's no job launcher can handle this job: {job_meta}.")

        return job_launcher[0]

    def notify_job_status(self, job_id, job_status):
        self.processes.record_worker_status(job_id, job_status)

    def _job_fqcn(self, job_id: str):
        return FQCN.join([self.client.cell.get_fqcn(), job_id])

    def check_status(self, job_id):
        """Checks the status of the running client.

        Args:
            job_id: the job_id

        Returns:
            A client status message
        """
        try:
            process_status = self.processes.status(job_id, ClientStatus.NOT_STARTED)
            return get_status_message(process_status)
        except Exception as e:
            self.logger.error(f"check_status execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()
            return "execution exception. Please try again."

    def get_run_info(self, job_id):
        """Gets the run information.

        Args:
            job_id: the job_id

        Returns:
            A dict of run information.
        """
        try:
            data = {}
            request = new_cell_message({}, data)
            return_data = self.client.cell.send_request(
                target=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.SHOW_STATS,
                request=request,
                optional=True,
                timeout=self.job_query_timeout,
            )
            return_code = return_data.get_header(MessageHeaderKey.RETURN_CODE)
            if return_code == ReturnCode.OK:
                run_info = return_data.payload
                return run_info
            else:
                return {}
        except Exception as e:
            self.logger.error(f"get_run_info execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()
            return {"error": "no info collector. Please try again."}

    def get_errors(self, job_id):
        """Get the error information.

        Args:
            job_id: the job_id

        Returns:
            A dict of error information.
        """
        try:
            data = {"command": AdminCommandNames.SHOW_ERRORS, "data": {}}
            request = new_cell_message({}, data)
            return_data = self.client.cell.send_request(
                target=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.SHOW_ERRORS,
                request=request,
                optional=True,
                timeout=self.job_query_timeout,
            )
            return_code = return_data.get_header(MessageHeaderKey.RETURN_CODE)
            if return_code == ReturnCode.OK:
                errors_info = return_data.payload
                return errors_info
            else:
                return None
        except Exception as e:
            self.logger.error(f"get_errors execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()
            return None

    def configure_job_log(self, job_id, config):
        """Configure the job log.

        Args:
            job_id: the job_id
            config: log config

         Returns:
            configure_job_log command message
        """
        try:
            request = new_cell_message({}, config)
            return_data = self.client.cell.send_request(
                target=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.CONFIGURE_JOB_LOG,
                request=request,
                optional=True,
                timeout=self.job_query_timeout,
            )
            return_code = return_data.get_header(MessageHeaderKey.RETURN_CODE)
            if return_code == ReturnCode.OK:
                return return_data.payload
            else:
                return f"failed to configure_job_log with return code: {return_code}"
        except Exception as e:
            err = f"configure_job_log execution exception: {secure_format_exception(e)}."
            self.logger.error(err)
            secure_log_traceback()
            return err

    def reset_errors(self, job_id):
        """Resets the error information.

        Args:
            job_id: the job_id
        """
        try:
            data = {"command": AdminCommandNames.RESET_ERRORS, "data": {}}
            request = new_cell_message({}, data)
            self.client.cell.fire_and_forget(
                targets=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.RESET_ERRORS,
                message=request,
                optional=True,
            )

        except Exception as e:
            self.logger.error(f"reset_errors execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()

    def abort_app(self, job_id, heartbeat_cleanup=False):
        """Aborts the running app.

        Args:
            job_id: the job_id
            heartbeat_cleanup: whether heartbeat cleanup requested the abort rather than a user or administrator
        """
        intent = StopIntent.HEARTBEAT_CLEANUP if heartbeat_cleanup else StopIntent.USER_ABORT
        for retry in range(1, -1, -1):
            try:
                stop = self.processes.request_stop(job_id, intent)
                if not stop:
                    self.logger.info(f"Client worker process for run: {job_id} was already terminated.")
                    break
                if stop.handle is None:
                    break
                if stop.phase == Phase.LAUNCHING:
                    self._terminate_handle(stop.handle, stop.intent)
                    break
                if stop.phase == Phase.RUNNING:
                    request = new_cell_message({}, {})
                    self.client.cell.fire_and_forget(
                        targets=self._job_fqcn(job_id),
                        channel=CellChannel.CLIENT_COMMAND,
                        topic=AdminCommandNames.ABORT,
                        message=request,
                        optional=True,
                    )
                    self.logger.debug("abort sent to worker")
                self._terminate_job(stop.handle, job_id)
                break
            except Exception as e:
                if retry == 0:
                    self.logger.error(
                        f"abort_worker_process execution exception: {secure_format_exception(e)} for run: {job_id}."
                    )
                    secure_log_traceback()
                else:
                    time.sleep(5.0)

        self.logger.info("Client worker process is terminated.")

    @staticmethod
    def _terminate_handle(job_handle: JobHandleSpec, intent: StopIntent):
        if intent == StopIntent.HEARTBEAT_CLEANUP:
            terminate_for_cleanup = getattr(job_handle, "_terminate_for_heartbeat_cleanup", None)
            if terminate_for_cleanup:
                return terminate_for_cleanup()
        return job_handle.terminate()

    def _terminate_job(self, job_handle, job_id):
        max_wait = 10.0
        start = time.time()
        while self.processes.process_may_be_running(job_id):
            if time.time() - start > max_wait:
                break
            time.sleep(0.05)
        else:
            return

        intent = self.processes.stop_intent(job_id)
        if intent == StopIntent.NONE:
            return
        self._terminate_handle(job_handle, intent)
        self.logger.info(f"run ({job_id}): child worker process terminated")

    def send_to_job(
        self,
        job_id,
        channel: str,
        topic: str,
        msg: CellMessage,
        timeout: float,
        optional=False,
    ) -> CellMessage:
        """Send a message to CJ

        Args:
            job_id: id of the job
            channel: message channel
            topic: message topic
            msg: the message to be sent
            timeout: how long to wait for reply
            optional: whether the message is optional

        Returns: reply from CJ

        """
        # send any serializable data to the job cell
        return self.client.cell.send_request(
            target=self._job_fqcn(job_id),
            channel=channel,
            topic=topic,
            request=msg,
            timeout=timeout,
            optional=optional,
        )

    def abort_task(self, job_id):
        """Aborts the client executing task.

        Args:
            job_id: the job_id
        """
        process_status = self.processes.status(job_id, ClientStatus.NOT_STARTED)
        if process_status == ClientStatus.STARTED:
            data = {"command": AdminCommandNames.ABORT_TASK, "data": {}}
            request = new_cell_message({}, data)
            self.client.cell.fire_and_forget(
                targets=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.ABORT_TASK,
                message=request,
                optional=True,
            )
            self.logger.debug("abort_task sent")

    def _wait_child_process_finish(
        self, client, job_id, allocated_resource, token, resource_manager, workspace, fl_ctx
    ):
        self.logger.info(f"run ({job_id}): waiting for child worker process to finish.")
        job_handle = self.processes.handle(job_id)
        job_handle.wait()
        exit_phase = self.processes.process_exited(job_id)
        return_code = get_return_code(job_handle, job_id, workspace, self.logger)
        if return_code == JobReturnCode.EXECUTION_ERROR and exit_phase == Phase.LAUNCHING:
            return_code = ProcessExitCode.INFRASTRUCTURE_ERROR

        self.logger.info(f"run ({job_id}): child worker process finished with RC {return_code}")
        self._report_outcome(job_id, return_code)
        self.processes.outcome_settled(job_id)

        if allocated_resource:
            resource_manager.free_resources(
                resources=allocated_resource, token=token, fl_ctx=client.engine.new_context()
            )
        self.processes.resources_released(job_id)
        self.processes.unregister(job_id)
        self.logger.debug(f"run ({job_id}): child worker resources freed.")

        engine = fl_ctx.get_engine()
        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job_id, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.CLIENT_NAME, client.client_name, private=True, sticky=False)
        engine.fire_event(EventType.JOB_COMPLETED, fl_ctx)
        self.processes.completion_published(job_id)
        self.logger.debug(f"Fired event JOB_COMPLETED {EventType.JOB_COMPLETED}")

    def _report_outcome(self, job_id, return_code):
        try:
            request = new_cell_message(
                headers={},
                payload={
                    JobFailureMsgKey.JOB_ID: job_id,
                    JobFailureMsgKey.CODE: return_code,
                    JobFailureMsgKey.REASON: REPORTABLE_JOB_FAILURES.get(return_code),
                },
            )
            reply = self.client.send_request_before_shutdown(
                target=FQCN.ROOT_SERVER,
                channel=CellChannel.SERVER_MAIN,
                topic=CellChannelTopic.REPORT_JOB_FAILURE,
                request=request,
                timeout=self.job_query_timeout,
                optional=True,
            )
            if reply is None:
                self.logger.info(f"not reporting terminal outcome of job {job_id}: client communication has stopped")
            elif reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK:
                self.logger.error(f"could not report terminal outcome of job {job_id}")
        except Exception as e:
            self.logger.error(f"could not report terminal outcome of job {job_id}: {secure_format_exception(e)}")

    def get_status(self, job_id):
        return self.processes.status(job_id, ClientStatus.STOPPED)

    def get_run_processes_keys(self):
        return self.processes.registered_job_ids()
