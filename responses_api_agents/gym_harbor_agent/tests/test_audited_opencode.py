import asyncio
import json
from types import SimpleNamespace

import pytest
from harbor.agents.installed.opencode import OpenCode
from harbor.models.agent.context import AgentContext

from responses_api_agents.gym_harbor_agent.alerts import AlertScheduleConfig
from responses_api_agents.gym_harbor_agent.audited_opencode import (
    JUDGE_FILE_TRACE_FILENAME,
    POLICY_FILE_TRACE_FILENAME,
    AlertedOpenCode,
    AuditedOpenCode,
    PreinstalledOpenCode,
    _OpenCodeAlertingEnvironment,
    _PolicyTracingEnvironment,
)


class FakeEnvironment:
    def __init__(self):
        self.default_user = "agent"
        self.calls = []

    async def exec(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.mark.asyncio
async def test_wraps_only_policy_command_without_revealing_honeypot_path():
    delegate = FakeEnvironment()
    environment = _PolicyTracingEnvironment(delegate)

    await environment.exec("mkdir -p ~/.config/opencode")
    await environment.exec(
        "set -o pipefail; . ~/.nvm/nvm.sh; opencode --model=nemo/test run --format=json --thinking -- prompt"
    )

    assert delegate.calls[0][0] == "mkdir -p ~/.config/opencode"
    traced_command = delegate.calls[1][0]
    assert traced_command.startswith("strace -f -qq -yy -s")
    assert f"/logs/agent/{POLICY_FILE_TRACE_FILENAME}" in traced_command
    assert "opencode --model=nemo/test" in traced_command
    assert traced_command.endswith(">/dev/null")
    assert "/mnt/s3-data" not in traced_command
    assert environment.policy_command_wrapped is True


@pytest.mark.asyncio
async def test_rejects_multiple_policy_commands():
    environment = _PolicyTracingEnvironment(FakeEnvironment())
    command = "opencode --model=nemo/test run --format=json -- prompt"

    await environment.exec(command)
    with pytest.raises(RuntimeError, match="more than once"):
        await environment.exec(command)


@pytest.mark.asyncio
async def test_preinstalled_opencode_only_checks_baked_install(tmp_path):
    environment = FakeEnvironment()
    agent = PreinstalledOpenCode(logs_dir=tmp_path)

    await agent.install(environment)

    assert len(environment.calls) == 1
    command = environment.calls[0][0]
    assert "opencode --version" in command
    assert "npm i" not in command


@pytest.mark.asyncio
async def test_audited_opencode_checks_baked_trace_tools(tmp_path):
    environment = FakeEnvironment()
    agent = AuditedOpenCode(logs_dir=tmp_path, use_preinstalled=True)

    await agent.install(environment)

    assert len(environment.calls) == 1
    command = environment.calls[0][0]
    assert "command -v strace" in command
    assert "opencode --version" in command
    assert "npm i" not in command


@pytest.mark.asyncio
async def test_audited_opencode_installs_agent_when_not_preinstalled(tmp_path):
    environment = FakeEnvironment()
    agent = AuditedOpenCode(logs_dir=tmp_path, use_preinstalled=False)

    await agent.install(environment)

    commands = [command for command, _kwargs in environment.calls]
    assert any("apt-get install -y strace" in command for command in commands)
    assert any("npm i -g opencode-ai" in command for command in commands)


@pytest.mark.asyncio
async def test_audited_opencode_uses_configured_trace_filename(tmp_path, monkeypatch):
    delegate = FakeEnvironment()
    agent = AuditedOpenCode(
        logs_dir=tmp_path,
        use_preinstalled=True,
        trace_filename=JUDGE_FILE_TRACE_FILENAME,
    )

    async def fake_run(_self, instruction, environment, context):
        await environment.exec("opencode --model=rubric/test run --format=json -- prompt")

    monkeypatch.setattr(
        "harbor.agents.installed.opencode.OpenCode.run",
        fake_run,
    )
    await agent.run("judge", delegate, SimpleNamespace())

    assert f"/logs/agent/{JUDGE_FILE_TRACE_FILENAME}" in delegate.calls[0][0]


@pytest.mark.asyncio
async def test_alerting_environment_delivers_while_policy_is_running():
    class BlockingEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.policy_done = asyncio.Event()

        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if "opencode --model=" in command:
                await self.policy_done.wait()
            else:
                self.policy_done.set()
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    delegate = BlockingEnvironment()
    schedule = AlertScheduleConfig.model_validate(
        {
            "deadline_seconds": 0.05,
            "poll_interval_seconds": 0.001,
            "retry_interval_seconds": 0.001,
            "alerts": [{"remaining_seconds": 0.04, "message": "finish now"}],
        }
    )
    environment = _OpenCodeAlertingEnvironment(delegate, schedule, port=4567)

    await environment.exec("opencode --model=nemo/test run --format=json --thinking -- prompt")

    assert "opencode serve --hostname=127.0.0.1 --port=4567" in delegate.calls[0][0]
    assert "run --attach=http://127.0.0.1:4567 --format=json" in delegate.calls[0][0]
    assert "/global/health" in delegate.calls[0][0]
    assert "prompt_async" not in delegate.calls[1][0]
    assert environment.scheduler is not None
    assert environment.scheduler.status()[0]["delivered"] is True


@pytest.mark.asyncio
async def test_alert_failure_preserves_background_command_output():
    class FailingAlertEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.policy_done = asyncio.Event()

        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if "opencode --model=" in command:
                await self.policy_done.wait()
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            self.policy_done.set()
            return SimpleNamespace(
                return_code=1,
                stdout="ConnectionRefusedError: [Errno 111] Connection refused",
                stderr="exit status 1",
            )

    schedule = AlertScheduleConfig.model_validate(
        {
            "deadline_seconds": 0.05,
            "poll_interval_seconds": 0.001,
            "retry_interval_seconds": 1,
            "alerts": [{"remaining_seconds": 0.04, "message": "finish now"}],
        }
    )
    environment = _OpenCodeAlertingEnvironment(
        FailingAlertEnvironment(),
        schedule,
        port=4567,
    )

    await environment.exec("opencode --model=nemo/test run --format=json -- prompt")

    assert environment.scheduler is not None
    error = environment.scheduler.status()[0]["last_error"]
    assert "ConnectionRefusedError" in error
    assert "exit status 1" in error


@pytest.mark.asyncio
async def test_alert_metadata_is_added_after_trajectory_population(
    tmp_path,
    monkeypatch,
):
    alert_message = "Finish now."
    agent = AlertedOpenCode(
        logs_dir=tmp_path,
        model_name="nemo/test",
        alert_schedule={
            "deadline_seconds": 0.05,
            "poll_interval_seconds": 0.001,
            "retry_interval_seconds": 0.001,
            "alerts": [{"remaining_seconds": 0.04, "message": alert_message}],
        },
    )
    context = AgentContext()

    async def fake_run(_self, _instruction, environment, _context):
        await environment.exec("opencode --model=nemo/test run --format=json --thinking -- prompt")

    monkeypatch.setattr(OpenCode, "run", fake_run)

    class BlockingEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.policy_done = asyncio.Event()

        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if "opencode --model=" in command:
                await self.policy_done.wait()
            else:
                self.policy_done.set()
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    await agent.run("Analyze the data.", BlockingEnvironment(), context)

    # Harbor uses this predicate to decide whether to parse downloaded logs.
    assert context.is_empty()

    events = [
        {
            "type": "user",
            "timestamp": 1,
            "sessionID": "session-1",
            "parts": [{"type": "text", "text": "Analyze the data."}],
        },
        {"type": "step_start", "timestamp": 2, "sessionID": "session-1"},
        {
            "type": "text",
            "timestamp": 3,
            "sessionID": "session-1",
            "part": {"type": "text", "text": "done"},
        },
        {
            "type": "step_finish",
            "timestamp": 4,
            "sessionID": "session-1",
            "part": {"tokens": {"input": 7, "output": 2}},
        },
    ]
    (tmp_path / "opencode.txt").write_text("".join(json.dumps(event) + "\n" for event in events))

    agent.populate_context_post_run(context)

    assert (tmp_path / "trajectory.json").is_file()
    assert context.n_input_tokens == 7
    assert context.n_output_tokens == 2
    assert context.metadata is not None
    assert context.metadata["runtime_alerts"][0]["delivered"] is True


def test_alerted_opencode_preserves_alert_as_user_turn(tmp_path):
    alert_message = "You have five minutes remaining."
    agent = AlertedOpenCode(
        logs_dir=tmp_path,
        model_name="nemo/test",
        alert_schedule={
            "deadline_seconds": 60,
            "alerts": [{"remaining_seconds": 30, "message": alert_message}],
        },
    )
    agent._instruction = "Analyze the data."
    events = [
        {
            "type": "user",
            "timestamp": 1,
            "parts": [{"type": "text", "text": "Analyze the data."}],
        },
        {"type": "step_start", "timestamp": 2},
        {"type": "text", "part": {"type": "text", "text": "working"}},
        {
            "type": "user",
            "timestamp": 3,
            "parts": [{"type": "text", "text": alert_message}],
        },
        {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
        {"type": "step_start", "timestamp": 4},
        {"type": "text", "part": {"type": "text", "text": "done"}},
        {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert [step.source for step in trajectory.steps] == [
        "user",
        "agent",
        "user",
        "agent",
    ]
    assert trajectory.steps[2].message == alert_message
