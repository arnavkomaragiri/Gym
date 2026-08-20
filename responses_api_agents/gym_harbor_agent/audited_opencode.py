# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCode agent variants that record agent-process filesystem accesses."""

import base64
import shlex
from pathlib import PurePath
from typing import Any, cast, override

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step, Trajectory

from responses_api_agents.gym_harbor_agent.alerts import (
    AlertContext,
    AlertScheduleConfig,
    build_time_remaining_scheduler,
)


POLICY_FILE_TRACE_FILENAME = ".policy-fs.trace"
JUDGE_FILE_TRACE_FILENAME = ".judge-fs.trace"
_OPENCODE_PORT = 4096
_PREINSTALLED_OPENCODE_CHECK = (
    'set -euo pipefail; test -r "$HOME/.nvm/nvm.sh"; '
    '. "$HOME/.nvm/nvm.sh"; command -v opencode >/dev/null; opencode --version'
)


def _is_policy_command(command: str) -> bool:
    return "opencode --model=" in command and " run " in command and "--format=json" in command


def _run_with_opencode_server(command: str, port: int) -> str:
    """Run the policy through a loopback OpenCode server for live user turns."""
    if not _is_policy_command(command):
        return command

    server_url = f"http://127.0.0.1:{port}"
    attached_command = command.replace(
        " run ",
        f" run --attach={server_url} ",
        1,
    )
    readiness_script = f"""
import http.client
import time

deadline = time.monotonic() + 60
last_error = None
while True:
    connection = http.client.HTTPConnection('127.0.0.1', {port}, timeout=1)
    try:
        connection.request('GET', '/global/health')
        response = connection.getresponse()
        response.read()
        if response.status == 200:
            break
        last_error = RuntimeError(f'OpenCode health returned HTTP {{response.status}}')
    except (OSError, http.client.HTTPException) as error:
        last_error = error
    finally:
        connection.close()
    if time.monotonic() >= deadline:
        raise RuntimeError('OpenCode server did not become healthy') from last_error
    time.sleep(0.1)
"""
    return (
        'set -e; . "$HOME/.nvm/nvm.sh"; '
        f"opencode serve --hostname=127.0.0.1 --port={port} "
        ">/logs/agent/opencode-server.log 2>&1 & "
        "__nemo_opencode_server_pid=$!; "
        'trap \'kill "$__nemo_opencode_server_pid" 2>/dev/null || true; '
        'wait "$__nemo_opencode_server_pid" 2>/dev/null || true\' EXIT; '
        f"python3 -c {shlex.quote(readiness_script)}; "
        f"{attached_command}"
    )


def _alert_delivery_command(*, port: int, message: str) -> str:
    """Build a sandbox-local request without putting the message in shell syntax."""
    encoded_message = base64.b64encode(message.encode()).decode()
    script = f"""
import base64
import json
import pathlib
import urllib.parse
import urllib.request

events = []
for line in pathlib.Path('/logs/agent/opencode.txt').read_text().splitlines():
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass
session_id = next(
    (event.get('sessionID') for event in reversed(events) if event.get('sessionID')),
    None,
)
if not session_id:
    raise RuntimeError('OpenCode session id is not available yet')
message = base64.b64decode('{encoded_message}').decode()
payload = json.dumps({{
    'parts': [{{'type': 'text', 'text': message}}],
    'noReply': False,
}}).encode()
request = urllib.request.Request(
    'http://127.0.0.1:{port}/session/' + urllib.parse.quote(session_id, safe='') + '/prompt_async',
    data=payload,
    headers={{
        'Content-Type': 'application/json',
        'x-opencode-directory': str(pathlib.Path.cwd()),
    }},
    method='POST',
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
with opener.open(request, timeout=30) as response:
    response.read()
"""
    encoded_script = base64.b64encode(script.encode()).decode()
    return f"python3 -c \"import base64; exec(base64.b64decode('{encoded_script}'))\""


def _trace_command(command: str, trace_filename: str) -> str:
    trace_path = f"/logs/agent/{trace_filename}"
    return (
        "strace -f -qq -yy -s 4096 "
        f"-e trace=%file -o {shlex.quote(trace_path)} "
        f"bash -o pipefail -c {shlex.quote(command)} >/dev/null"
    )


class _OpenCodeTracingEnvironment:
    """Delegate a Harbor environment while wrapping one OpenCode run."""

    def __init__(self, environment: BaseEnvironment, trace_filename: str):
        self._environment = environment
        self._trace_filename = trace_filename
        self.command_wrapped = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    async def exec(self, command: str, **kwargs: Any) -> Any:
        if _is_policy_command(command):
            if self.command_wrapped:
                raise RuntimeError("OpenCode run command was issued more than once")
            self.command_wrapped = True
            command = _trace_command(command, self._trace_filename)
        return await self._environment.exec(command=command, **kwargs)


class _OpenCodeAlertingEnvironment:
    """Run OpenCode with its local API enabled and deliver scheduled user turns."""

    def __init__(
        self,
        environment: BaseEnvironment,
        alert_schedule: AlertScheduleConfig,
        port: int,
    ) -> None:
        self._environment = environment
        self._alert_schedule = alert_schedule
        self._port = port
        self.scheduler = None
        self.command_wrapped = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    async def exec(self, command: str, **kwargs: Any) -> Any:
        if not _is_policy_command(command):
            return await self._environment.exec(command=command, **kwargs)
        if self.command_wrapped:
            raise RuntimeError("OpenCode run command was issued more than once")
        self.command_wrapped = True

        async def deliver(message: str, _context: AlertContext) -> None:
            result = await self._environment.exec(
                command=_alert_delivery_command(port=self._port, message=message),
                timeout_sec=45,
            )
            if result.return_code != 0:
                outputs = [output.strip() for output in (result.stdout, result.stderr) if output and output.strip()]
                output = "\n".join(outputs) or "no output"
                raise RuntimeError(f"OpenCode alert delivery failed: {output}")

        self.scheduler = build_time_remaining_scheduler(
            self._alert_schedule,
            deliver,
        )
        return await self.scheduler.run_until_complete(
            self._environment.exec(
                command=_run_with_opencode_server(command, self._port),
                **kwargs,
            )
        )


class _PolicyTracingEnvironment(_OpenCodeTracingEnvironment):
    """Backward-compatible policy tracing environment used by older callers."""

    def __init__(self, environment: BaseEnvironment):
        super().__init__(environment, POLICY_FILE_TRACE_FILENAME)

    @property
    def policy_command_wrapped(self) -> bool:
        return self.command_wrapped


class PreinstalledOpenCode(OpenCode):
    """Use the OpenCode installation baked into the sandbox image."""

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(environment, command=_PREINSTALLED_OPENCODE_CHECK)


class AlertedOpenCode(OpenCode):
    """Inject configured time-left warnings as native OpenCode user turns."""

    def __init__(
        self,
        *args: Any,
        alert_schedule: dict[str, Any] | AlertScheduleConfig | None = None,
        opencode_port: int = _OPENCODE_PORT,
        use_preinstalled: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._use_preinstalled = use_preinstalled
        self._alert_schedule = (
            AlertScheduleConfig.model_validate(alert_schedule) if alert_schedule is not None else None
        )
        if not 1 <= opencode_port <= 65535:
            raise ValueError("opencode_port must be between 1 and 65535")
        self._opencode_port = opencode_port
        self._runtime_alert_status: list[dict[str, Any]] | None = None

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self._use_preinstalled:
            await self.exec_as_agent(
                environment,
                command=_PREINSTALLED_OPENCODE_CHECK,
            )
            return
        await super().install(environment)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._runtime_alert_status = None
        if self._alert_schedule is None:
            await super().run(instruction, environment, context)
            return

        alerting_environment = _OpenCodeAlertingEnvironment(
            environment,
            self._alert_schedule,
            self._opencode_port,
        )
        try:
            await super().run(
                instruction,
                cast(BaseEnvironment, alerting_environment),
                context,
            )
        finally:
            if alerting_environment.scheduler is not None:
                # Harbor only invokes populate_context_post_run() while the
                # context is empty, so attach metadata in that hook instead.
                self._runtime_alert_status = alerting_environment.scheduler.status()

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        if self._runtime_alert_status is None:
            return
        metadata = dict(context.metadata or {})
        metadata["runtime_alerts"] = self._runtime_alert_status
        context.metadata = metadata

    @override
    def _convert_events_to_trajectory(
        self,
        events: list[dict[str, Any]],
    ) -> Trajectory | None:
        trajectory = super()._convert_events_to_trajectory(events)
        if trajectory is None or self._alert_schedule is None or self._instruction is None:
            return trajectory

        assistant_steps = [step for step in trajectory.steps if step.source == "agent"]
        rebuilt_steps = [Step(step_id=1, source="user", message=self._instruction)]
        assistant_index = 0
        turn_open = False
        deferred_alerts: list[Step] = []
        alert_messages = {alert.message for alert in self._alert_schedule.alerts}
        for event in events:
            event_type = event.get("type")
            if event_type == "step_start":
                turn_open = True
            elif event_type == "user":
                message = self._user_event_text(event)
                if message in alert_messages:
                    user_step = Step(
                        step_id=1,
                        timestamp=self._millis_to_iso(event.get("timestamp")),
                        source="user",
                        message=message,
                    )
                    if turn_open:
                        deferred_alerts.append(user_step)
                    else:
                        rebuilt_steps.append(user_step)
            elif event_type == "step_finish":
                if assistant_index < len(assistant_steps):
                    rebuilt_steps.append(assistant_steps[assistant_index])
                    assistant_index += 1
                rebuilt_steps.extend(deferred_alerts)
                deferred_alerts.clear()
                turn_open = False

        rebuilt_steps.extend(assistant_steps[assistant_index:])
        rebuilt_steps.extend(deferred_alerts)
        for step_id, step in enumerate(rebuilt_steps, start=1):
            step.step_id = step_id
        trajectory.steps = rebuilt_steps
        if trajectory.final_metrics is not None:
            trajectory.final_metrics.total_steps = len(rebuilt_steps)
        return trajectory


class AuditedOpenCode(AlertedOpenCode):
    """Trace actual file accesses made by OpenCode and its child processes.

    The trace records generic file syscalls. It intentionally has no knowledge
    of which paths are sensitive; classification happens after the sandbox has
    stopped so the audited path is never exposed to the policy process.
    """

    def __init__(
        self,
        *args: Any,
        use_preinstalled: bool = False,
        trace_filename: str = POLICY_FILE_TRACE_FILENAME,
        **kwargs: Any,
    ):
        super().__init__(*args, use_preinstalled=use_preinstalled, **kwargs)
        if not trace_filename or PurePath(trace_filename).name != trace_filename:
            raise ValueError("trace_filename must be a filename, not a path")
        self._trace_filename = trace_filename

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self._use_preinstalled:
            await self.exec_as_agent(
                environment,
                command=(f"set -euo pipefail; command -v strace >/dev/null; {_PREINSTALLED_OPENCODE_CHECK}"),
            )
            return
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y strace",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await super().install(environment)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        tracing_environment = _OpenCodeTracingEnvironment(environment, self._trace_filename)
        await super().run(
            instruction,
            cast(BaseEnvironment, tracing_environment),
            context,
        )
        if not tracing_environment.command_wrapped:
            raise RuntimeError("OpenCode run command was not traced")
