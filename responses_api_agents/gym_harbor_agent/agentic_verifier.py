# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor verifier that delegates evaluation to an independently configured agent."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal, override

from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.verifier.result import VerifierResult
from harbor.utils.env import resolve_env_vars
from harbor.verifier.base import BaseVerifier
from pydantic import BaseModel, Field, field_validator


_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
_OPENCODE_API_PACKAGES = {
    "chat_completions": "@ai-sdk/openai-compatible",
    "responses": "@ai-sdk/openai",
}


class OpenCodeProviderConfig(BaseModel, extra="forbid"):
    """External model provider used by an OpenCode judge."""

    api_mode: Literal["chat_completions", "responses"]
    base_url: str
    provider_name: str = "rubric"
    api_key_env_var: str = "OPENAI_API_KEY"

    @field_validator("base_url", "provider_name", "api_key_env_var")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value


class AgenticVerifierConfig(BaseModel, extra="forbid"):
    """Configuration passed through ``JobConfig.verifier.kwargs.config``."""

    judge_agent: AgentConfig
    judge_opencode_provider: OpenCodeProviderConfig | None = None
    judge_env_aliases: dict[str, str] = Field(default_factory=dict)
    instruction_path: str = "judge_instruction.md"
    judge_workdir: str = "/judge"
    reward_path: str = "/logs/verifier/reward.json"
    judge_logs_subdir: str = "judge"
    setup_timeout_sec: float | None = Field(default=None, gt=0)
    run_timeout_sec: float | None = Field(default=None, gt=0)

    @field_validator("instruction_path", "judge_logs_subdir")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("path must be non-empty, relative, and remain within its root")
        return value

    @field_validator("judge_workdir", "reward_path")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be absolute and must not contain '..'")
        return value

    @field_validator("reward_path")
    @classmethod
    def validate_reward_path(cls, value: str) -> str:
        verifier_dir = EnvironmentPaths.verifier_dir
        path = PurePosixPath(value)
        if path != verifier_dir and verifier_dir not in path.parents:
            raise ValueError("reward_path must remain under /logs/verifier")
        return value


class _WorkingDirectoryEnvironment:
    """Duck-typed environment view that supplies a judge-only default cwd."""

    def __init__(self, environment: BaseEnvironment, workdir: str) -> None:
        self._environment = environment
        self._workdir = workdir

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        return await self._environment.exec(
            command,
            cwd=cwd or self._workdir,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )


def _resolve_template(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            return value
        name, default = match.groups()
        if name in variables:
            return variables[name]
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Environment variable '{name}' is required by judge_agent")
    if isinstance(value, dict):
        return {key: _resolve_template(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_template(item, variables) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_template(item, variables) for item in value)
    return value


class AgenticVerifier(BaseVerifier):
    """Run a Harbor agent as a judge in Harbor's separate verifier environment."""

    def __init__(
        self,
        *,
        task: Task,
        trial_paths: TrialPaths,
        environment: BaseEnvironment,
        config: AgenticVerifierConfig | dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task=task,
            trial_paths=trial_paths,
            environment=environment,
            **kwargs,
        )
        self.config = (
            config if isinstance(config, AgenticVerifierConfig) else AgenticVerifierConfig.model_validate(config)
        )

    def _instruction(self) -> str:
        relative = Path(self.config.instruction_path)
        candidates = []
        if self.step_name is not None:
            candidates.append(self.task.paths.step_tests_dir(self.step_name) / relative)
        candidates.append(self.task.paths.tests_dir / relative)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.read_text()
        raise FileNotFoundError(f"Judge instruction {self.config.instruction_path!r} was not found in task tests")

    def _resolved_judge_agent(self) -> AgentConfig:
        verifier_templates = {
            **self.task.config.verifier.env,
            **(self.verifier_env or {}),
            **self.override_env,
        }
        variables = resolve_env_vars(verifier_templates) if verifier_templates else {}
        raw = self.config.judge_agent.model_dump(
            mode="python",
            exclude_none=True,
            context={"redact_sensitive_env": False},
        )
        judge_agent = AgentConfig.model_validate(_resolve_template(raw, variables))
        env = dict(judge_agent.env)
        for target, source in self.config.judge_env_aliases.items():
            if source not in variables:
                raise ValueError(
                    f"Verifier environment variable {source!r} required for "
                    f"judge environment variable {target!r} is missing"
                )
            env[target] = variables[source]
        judge_agent = judge_agent.model_copy(update={"env": env})
        return self._configure_opencode_provider(judge_agent)

    def _configure_opencode_provider(self, judge_agent: AgentConfig) -> AgentConfig:
        provider_config = self.config.judge_opencode_provider
        if provider_config is None:
            return judge_agent
        if judge_agent.name != "opencode":
            raise ValueError("judge_opencode_provider requires judge_agent.name='opencode'")
        if not judge_agent.model_name:
            raise ValueError("judge_opencode_provider requires judge_agent.model_name")

        kwargs = copy.deepcopy(judge_agent.kwargs)
        opencode_config = kwargs.setdefault("opencode_config", {})
        providers = opencode_config.setdefault("provider", {})
        provider = providers.setdefault(provider_config.provider_name, {})
        provider["npm"] = _OPENCODE_API_PACKAGES[provider_config.api_mode]
        provider.setdefault("name", "Rubric model")
        options = provider.setdefault("options", {})
        options["baseURL"] = provider_config.base_url
        options.setdefault("apiKey", f"{{env:{provider_config.api_key_env_var}}}")
        model = provider.setdefault("models", {}).setdefault(judge_agent.model_name, {})
        model.setdefault("name", judge_agent.model_name)
        model.setdefault("reasoning", True)
        model.setdefault("tool_call", True)

        selected_model = f"{provider_config.provider_name}/{judge_agent.model_name}"
        opencode_config.setdefault("small_model", selected_model)
        return judge_agent.model_copy(update={"model_name": selected_model, "kwargs": kwargs})

    async def _run_with_timeout(self, awaitable, timeout_sec: float | None) -> None:
        if timeout_sec is None:
            await awaitable
        else:
            await asyncio.wait_for(awaitable, timeout=timeout_sec)

    async def _download_judge_logs(self, judge_logs_dir: Path) -> None:
        env_paths = EnvironmentPaths.for_os(self.environment.os)
        try:
            await self.environment.download_dir(
                source_dir=str(env_paths.agent_dir),
                target_dir=judge_logs_dir,
            )
        except Exception:
            self.logger.exception("Failed to download agentic judge logs")

        if not self.environment.capabilities.mounted:
            await self.environment.download_dir(
                source_dir=str(env_paths.verifier_dir),
                target_dir=self.trial_paths.verifier_dir,
            )

    def _parse_rewards(self) -> dict[str, float | int]:
        reward_relative = PurePosixPath(self.config.reward_path).relative_to(EnvironmentPaths.verifier_dir)
        reward_path = self.trial_paths.verifier_dir.joinpath(*reward_relative.parts)
        if not reward_path.is_file():
            raise FileNotFoundError(f"Agentic judge did not produce {self.config.reward_path}")
        raw = json.loads(reward_path.read_text())
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Agentic judge reward must be a non-empty JSON object")
        rewards: dict[str, float | int] = {}
        for key, value in raw.items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Agentic judge rewards must map non-empty strings to finite numbers")
            rewards[key] = value
        return rewards

    @override
    async def verify(self) -> VerifierResult:
        if "__verifier__" not in self.environment.session_id:
            raise ValueError("AgenticVerifier requires Harbor verifier.environment_mode='separate'")

        await self.environment.ensure_dirs([self.config.judge_workdir], chmod=True)

        judge_logs_dir = self.trial_paths.verifier_dir / self.config.judge_logs_subdir
        judge_logs_dir.mkdir(parents=True, exist_ok=True)
        judge_config = self._resolved_judge_agent()
        agent_kwargs: dict[str, Any] = {}
        if judge_config.mcp_servers:
            agent_kwargs["mcp_servers"] = judge_config.mcp_servers
        judge = AgentFactory.create_agent_from_config(
            judge_config,
            logs_dir=judge_logs_dir,
            logger=self.logger,
            **agent_kwargs,
        )
        judge.session_id = f"{self.environment.session_id}__judge"
        judge.context_id = self.environment.context_id
        context = AgentContext()
        judge_environment = _WorkingDirectoryEnvironment(
            self.environment,
            self.config.judge_workdir,
        )

        run_error: BaseException | None = None
        try:
            with self.environment.scoped_exec_env(judge.extra_env):
                await self._run_with_timeout(
                    judge.setup(environment=judge_environment),
                    self.config.setup_timeout_sec,
                )
                await self._run_with_timeout(
                    judge.run(
                        instruction=self._instruction(),
                        environment=judge_environment,
                        context=context,
                    ),
                    self.config.run_timeout_sec,
                )
        except BaseException as exc:
            run_error = exc
        finally:
            await self._download_judge_logs(judge_logs_dir)
            judge.populate_context_post_run(context)
            (judge_logs_dir / "context.json").write_text(context.model_dump_json(indent=2))

        if run_error is not None:
            raise run_error
        return VerifierResult(rewards=self._parse_rewards())
