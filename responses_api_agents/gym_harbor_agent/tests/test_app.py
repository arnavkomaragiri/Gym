from pathlib import Path
from unittest.mock import MagicMock, patch

from harbor.models.trial.config import AgentConfig

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.gym_harbor_agent.app import (
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
)


def make_config(tmp_path: Path) -> HarborAgentConfig:
    return HarborAgentConfig(
        name="gym_harbor_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        token_id_capture=True,
        jobs_dir=tmp_path / "rollouts.jsonl",
        dataset={"path": tmp_path / "dataset"},
        agent={
            "name": "opencode",
            "model_name": "openai/test-model",
            "env": {"EXISTING": "value"},
        },
        environment={"type": "docker"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_api_key="test-key",  # pragma: allowlist secret
    )


def test_jobs_dir_uses_sibling_harbor_directory_for_jsonl(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert config.jobs_dir == tmp_path.resolve() / "harbor"


def test_agent_for_model_server_injects_route_without_mutating_config(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    agent = config.agent_for_model_server("http://model/ng-rollout/t0-r1/v1")

    assert agent.env == {
        "EXISTING": "value",
        "OPENAI_BASE_URL": "http://model/ng-rollout/t0-r1/v1",
        "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
    }
    assert agent.kwargs == {
        "opencode_config": {"provider": {"openai": {"options": {"baseURL": "http://model/ng-rollout/t0-r1/v1"}}}}
    }
    assert config.agent.env == {"EXISTING": "value"}
    assert config.agent.kwargs == {}


def test_build_job_config_scopes_task_and_forces_environment_cleanup(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AgentConfig(name="opencode", model_name="openai/test-model")

    job = config.build_job_config("bbh-task", "t0-r1", agent)

    assert job.job_name == "t0-r1"
    assert job.jobs_dir == tmp_path.resolve() / "harbor"
    assert job.datasets[0].task_names == ["bbh-task"]
    assert job.agents == [agent]
    assert job.environment.delete is True
    assert job.n_attempts == 1
    assert job.n_concurrent_trials == 1


def test_nrl_rollout_id_routes_to_prefixed_gym_model_url(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    server_client = MagicMock()
    server_client.global_config_dict = {
        "token_id_capture": {"enabled": True},
        "policy_model": {"responses_api_models": {"vllm_model": {"host": "model-host", "port": 9000}}},
    }
    server_client._build_server_base_url.return_value = "http://model-host:9000"
    with patch(
        "responses_api_agents.gym_harbor_agent.app.get_global_config_dict",
        return_value={},
    ):
        agent = HarborAgent.model_construct(config=config, server_client=server_client)
    body = HarborRunRequest(
        task_name="bbh-task",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        **{
            "_ng_task_index": 3,
            "_ng_rollout_index": 1,
            "_ng_rollout_id": "nrl-step7-sample3",
        },
    )

    rollout_id = agent.rollout_id_from_run(body)
    base_url = agent.resolve_model_base_url(config.model_server.name, rollout_id)

    assert rollout_id == "nrl-step7-sample3"
    assert base_url == "http://model-host:9000/ng-rollout/nrl-step7-sample3/v1"
