# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import json
import logging
import sys
import time
from pathlib import Path

import ray
from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.trial.config import (
    AgentConfig,
    ArtifactConfig,
    EnvironmentConfig,
    VerifierConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY


logger = logging.getLogger(__name__)

NUM_SAMPLES_IN_PARALLEL_KEY_NAME = "num_samples_in_parallel"

_RAY_WORKER_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={"py_executable": sys.executable},
)
def harbor_job_worker(job_config_dict: dict) -> str:
    global _RAY_WORKER_EVENT_LOOP
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    return _RAY_WORKER_EVENT_LOOP.run_until_complete(HarborAgent.run_job(job_config_dict))


class HarborAgentConfig(BaseResponsesAPIAgentConfig):
    jobs_dir: Path
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)
    model_server: ModelServerRef
    model_base_url_env_var: str = "OPENAI_BASE_URL"
    model_api_key_env_var: str = "OPENAI_API_KEY"
    model_api_key: str
    context_window: int = 262144
    max_output_tokens: int = 131072
    reasoning_field: str = "reasoning"
    environment_build_timeout_multiplier: float | None = None

    @field_validator("jobs_dir", mode="after")
    @classmethod
    def normalize_jobs_dir(cls, jobs_dir: Path) -> Path:
        jobs_dir = jobs_dir.resolve()
        if jobs_dir.suffix.lower() == ".jsonl":
            return jobs_dir.parent / "harbor"
        return jobs_dir

    @model_validator(mode="after")
    def validate_opencode_provider(self) -> "HarborAgentConfig":
        if self.agent.name == "opencode" and not (self.agent.model_name or "").startswith("nemo/"):
            raise ValueError("OpenCode must use a nemo/<model> name when routed through the Gym model server")
        return self

    def agent_for_model_server(
        self,
        base_url: str,
        auxiliary_base_url: str | None = None,
    ) -> AgentConfig:
        env = dict(self.agent.env)
        env[self.model_base_url_env_var] = base_url
        env[self.model_api_key_env_var] = self.model_api_key

        kwargs = copy.deepcopy(self.agent.kwargs)
        if self.agent.name == "opencode":
            model_name = (self.agent.model_name or "").removeprefix("nemo/")
            opencode_config = kwargs.setdefault("opencode_config", {})
            provider = opencode_config.setdefault("provider", {})
            nemo_provider = provider.setdefault("nemo", {})
            nemo_provider.setdefault("npm", "@ai-sdk/openai-compatible")
            options = nemo_provider.setdefault("options", {})
            options["baseURL"] = base_url
            options.setdefault("apiKey", "EMPTY")  # pragma: allowlist secret
            model = nemo_provider.setdefault("models", {}).setdefault(model_name, {})
            model.setdefault("name", model_name)
            model.setdefault("reasoning", True)
            model.setdefault("tool_call", True)
            model.setdefault("interleaved", {"field": self.reasoning_field})
            limits = model.setdefault("limit", {})
            limits.setdefault("context", self.context_window)
            limits.setdefault("output", self.max_output_tokens)

            # OpenCode generates session titles with its small model. Keep those
            # utility calls on the same Gym model server, but outside the rollout
            # correlation route so they are not trained as policy turns.
            if auxiliary_base_url and auxiliary_base_url != base_url and "small_model" not in opencode_config:
                auxiliary_provider_name = "nemo-auxiliary"
                auxiliary_provider = provider.setdefault(auxiliary_provider_name, {})
                auxiliary_provider.setdefault("npm", "@ai-sdk/openai-compatible")
                auxiliary_options = auxiliary_provider.setdefault("options", {})
                auxiliary_options["baseURL"] = auxiliary_base_url
                auxiliary_options.setdefault("apiKey", "EMPTY")  # pragma: allowlist secret
                auxiliary_model = auxiliary_provider.setdefault("models", {}).setdefault(model_name, {})
                auxiliary_model.setdefault("name", model_name)
                opencode_config["small_model"] = f"{auxiliary_provider_name}/{model_name}"

        return self.agent.model_copy(update={"env": env, "kwargs": kwargs})

    def build_job_config(self, task_name: str, job_name: str, agent: AgentConfig) -> JobConfig:
        return JobConfig(
            job_name=job_name,
            jobs_dir=self.jobs_dir,
            n_attempts=1,
            n_concurrent_trials=1,
            quiet=True,
            retry=RetryConfig(max_retries=0),
            environment_build_timeout_multiplier=self.environment_build_timeout_multiplier,
            environment=self.environment.model_copy(update={"delete": True}),
            verifier=self.verifier,
            artifacts=self.artifacts,
            agents=[agent],
            datasets=[
                self.dataset.model_copy(update={"task_names": [task_name]}),
            ],
        )


class HarborRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_name: str
    task_index: int = Field(alias=TASK_INDEX_KEY_NAME)
    rollout_index: int = Field(alias=ROLLOUT_INDEX_KEY_NAME)


class HarborVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class HarborAgent(SimpleResponsesAPIAgent):
    config: HarborAgentConfig

    _sem: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context) -> None:
        num_samples_in_parallel = get_global_config_dict().get(NUM_SAMPLES_IN_PARALLEL_KEY_NAME, 1)
        self._sem = asyncio.Semaphore(num_samples_in_parallel)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        # Harbor owns the full run() lifecycle.
        raise NotImplementedError

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        async with self._sem:
            try:
                rollout_id = self.rollout_id_from_run(body)
                model_base_url = self.resolve_model_base_url(
                    self.config.model_server.name,
                    rollout_id,
                )
                auxiliary_model_base_url = self.resolve_model_base_url(self.config.model_server.name)
                agent = self.config.agent_for_model_server(
                    model_base_url,
                    auxiliary_base_url=auxiliary_model_base_url,
                )
                job_name = f"t{body.task_index}-r{body.rollout_index}"
                if rollout_id is not None:
                    # The routed model URL is part of Harbor's job config. NRL
                    # assigns a new rollout id when it redispatches a failed
                    # sample, so each dispatch needs its own job directory.
                    job_name = f"{job_name}-{rollout_id}"
                job_config = self.config.build_job_config(
                    task_name=body.task_name,
                    job_name=job_name,
                    agent=agent,
                )

                trial_dir = Path(await harbor_job_worker.remote(job_config.model_dump(mode="json")))

                return self.success_response(body, trial_dir)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                logger.exception(
                    "Harbor rollout failed: task_index=%s rollout_index=%s",
                    body.task_index,
                    body.rollout_index,
                )
                return self.failure_response(body, err)

    def success_response(self, body: HarborRunRequest, trial_dir: Path) -> HarborVerifyResponse:
        # Reuse the established Harbor trajectory-to-Responses conversion.
        from responses_api_agents.harbor_agent.utils import HarborAgentUtils

        trial_paths = TrialPaths(trial_dir)
        trial = TrialResult.model_validate_json(trial_paths.result_path.read_text())
        if trial.step_results:
            trajectory_paths = [
                trial_paths.step_agent_dir(step.step_name) / "trajectory.json" for step in trial.step_results
            ]
        else:
            trajectory_paths = [trial_paths.agent_dir / "trajectory.json"]

        trajectories = [json.loads(path.read_text()) for path in trajectory_paths if path.is_file()]
        if not trajectories:
            raise FileNotFoundError(f"No Harbor trajectory found in {trial_dir}")

        output = [item for trajectory in trajectories for item in HarborAgentUtils.trajectory_to_responses(trajectory)]
        input_messages = next(
            (
                messages
                for trajectory in trajectories
                if (messages := HarborAgentUtils.extract_input_from_trajectory(trajectory))
            ),
            [],
        )
        n_input_tokens, n_cache_tokens, n_output_tokens, _ = trial.compute_token_cost_totals()
        response = NeMoGymResponse(
            id=f"harbor-{trial.id}",
            created_at=(trial.finished_at.timestamp() if trial.finished_at is not None else time.time()),
            model=self.config.agent.model_name,
            object="response",
            output=output,
            parallel_tool_calls=False,
            temperature=body.responses_create_params.temperature,
            tool_choice="auto",
            tools=[],
            top_p=body.responses_create_params.top_p,
            status="completed",
            usage={
                "input_tokens": n_input_tokens or 0,
                "input_tokens_details": {
                    "cached_tokens": n_cache_tokens or 0,
                    "cache_write_tokens": 0,
                },
                "output_tokens": n_output_tokens or 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": (n_input_tokens or 0) + (n_output_tokens or 0),
            },
        )
        verifier_result = trial.verifier_result.model_dump() if trial.verifier_result is not None else None
        reward = HarborAgentUtils.extract_reward(verifier_result)

        return HarborVerifyResponse.model_validate(
            body.model_dump(by_alias=True)
            | {
                "responses_create_params": body.responses_create_params.model_copy(update={"input": input_messages}),
                "response": response,
                "reward": reward,
            }
        )

    def failure_response(self, body: HarborRunRequest, err: Exception) -> HarborVerifyResponse:
        response = NeMoGymResponse(
            id=f"harbor-error-t{body.task_index}-r{body.rollout_index}",
            created_at=time.time(),
            model=self.config.agent.model_name,
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            status="failed",
        )
        return HarborVerifyResponse.model_validate(
            body.model_dump(by_alias=True)
            | {
                "response": response.model_dump(mode="json"),
                "reward": 0.0,
                NG_FAILURE_CLASS_KEY: "harbor_failed",
                "error": f"{type(err).__name__}: {err}",
            }
        )

    @staticmethod
    async def run_job(job_config_dict: dict) -> str:
        job_config = JobConfig.model_validate(job_config_dict)
        job_err: Exception | None = None

        try:
            job = await Job.create(job_config)
            await job.run()
        except Exception as err:  # noqa: BLE001 - recover Harbor's partial trial artifacts
            job_err = err

        job_dir = job_config.jobs_dir / job_config.job_name
        if job_dir.exists():
            for trial_dir in job_dir.iterdir():
                result_path = trial_dir / "result.json"
                if not trial_dir.is_dir() or not result_path.is_file():
                    continue

                trial_result = TrialResult.model_validate_json(result_path.read_text())
                if trial_result.exception_info is not None:
                    exception_info = trial_result.exception_info
                    # Deleting result.json forces Harbor to replace the failed trial on Gym retry.
                    result_path.unlink()
                    raise RuntimeError(
                        f"Harbor trial failed with {exception_info.exception_type}: {exception_info.exception_message}"
                    )

                return str(trial_dir.resolve())

        if job_err is not None:
            raise job_err

        raise FileNotFoundError(f"No Harbor trial result found in {job_dir}")


if __name__ == "__main__":
    HarborAgent.run_webserver()
