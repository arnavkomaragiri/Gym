from pathlib import Path
from unittest.mock import MagicMock, patch

from harbor.models.trial.config import AgentConfig
from omegaconf import OmegaConf

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


def test_opensandbox_config_separates_requests_from_limits(monkeypatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    dataset_path = "/datasets/bbh-harbor-rl-v0/val"
    monkeypatch.setenv("HARBOR_DATASET_PATH", dataset_path)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-key")
    monkeypatch.setenv("OPENSANDBOX_PROTOCOL", "https")
    monkeypatch.setenv("OPENSANDBOX_USE_SERVER_PROXY", "false")
    monkeypatch.setenv("OPENSANDBOX_CA_BUNDLE", "/certs/opensandbox.pem")
    monkeypatch.setenv("HARBOR_IMAGE_OVERRIDE", "docker.io/example/task@sha256:1234")
    monkeypatch.setenv(
        "HARBOR_SANDBOX_ENTRYPOINT",
        '["sh", "/opt/entrypoint.sh", "tail", "-f", "/dev/null"]',
    )

    config = OmegaConf.merge(
        OmegaConf.load(gym_root / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"),
        OmegaConf.load(gym_root / "responses_api_agents/gym_harbor_agent/configs/harbor_agent.yaml"),
        OmegaConf.load(gym_root / "responses_api_agents/gym_harbor_agent/configs/harbor_agent_opensandbox.yaml"),
    )
    agent_config = config.gym_harbor_agent.responses_api_agents.gym_harbor_agent
    environment = OmegaConf.to_container(agent_config.environment, resolve=True)

    assert OmegaConf.to_container(agent_config.dataset, resolve=True)["path"] == dataset_path
    assert environment["import_path"].endswith(":NemoGymSandboxEnvironment")
    assert environment["override_cpus"] == 4
    assert environment["override_memory_mb"] == 65536
    assert environment["kwargs"]["sandbox_provider_options"] == {"resource_requests": {"cpu": 0.25, "memory_mib": 512}}
    assert environment["kwargs"]["sandbox_provider"]["opensandbox"]["connection"]["api_key"] == "test-key"
    assert environment["kwargs"]["sandbox_provider"]["opensandbox"]["connection"]["protocol"] == "https"
    assert environment["kwargs"]["sandbox_provider"]["opensandbox"]["connection"]["use_server_proxy"] is False
    assert (
        environment["kwargs"]["sandbox_provider"]["opensandbox"]["connection"]["ca_bundle_path"]
        == "/certs/opensandbox.pem"
    )
    assert environment["kwargs"]["image_override"] == "docker.io/example/task@sha256:1234"
    assert environment["kwargs"]["entrypoint"] == [
        "sh",
        "/opt/entrypoint.sh",
        "tail",
        "-f",
        "/dev/null",
    ]
