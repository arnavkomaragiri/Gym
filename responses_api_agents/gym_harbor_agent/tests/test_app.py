from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from harbor.models.trial.config import AgentConfig, ArtifactConfig, VerifierConfig
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
            "model_name": "nemo/test-model",
            "env": {"EXISTING": "value"},
        },
        environment={"type": "docker"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_api_key="test-key",  # pragma: allowlist secret
    )


def test_jobs_dir_uses_sibling_harbor_directory_for_jsonl(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert config.jobs_dir == tmp_path.resolve() / "harbor"


def test_jobs_dir_defaults_to_ray_tmpdir(monkeypatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    monkeypatch.delenv("HARBOR_JOBS_DIR", raising=False)
    monkeypatch.setenv("RAY_TMPDIR", "/tmp/ray-test")

    config = OmegaConf.merge(
        {"policy_model_name": "test-model"},
        OmegaConf.load(gym_root / "responses_api_agents/gym_harbor_agent/configs/harbor_agent.yaml"),
    )

    assert config.gym_harbor_agent.responses_api_agents.gym_harbor_agent.jobs_dir == (
        "/tmp/ray-test/gym_harbor_agent_jobs"
    )
    assert config.gym_harbor_agent.responses_api_agents.gym_harbor_agent.agent.model_name.startswith("nemo/")
    assert config.gym_harbor_agent.responses_api_agents.gym_harbor_agent.context_window == 131072
    assert config.policy_model.responses_api_models.vllm_model.reasoning_response_field == "reasoning_content"


def test_agent_for_model_server_injects_route_without_mutating_config(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    agent = config.agent_for_model_server(
        "http://model/ng-rollout/t0-r1/v1",
        auxiliary_base_url="http://model/v1",
    )

    assert agent.env == {
        "EXISTING": "value",
        "OPENAI_BASE_URL": "http://model/ng-rollout/t0-r1/v1",
        "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
    }
    assert agent.kwargs == {
        "opencode_config": {
            "provider": {
                "nemo": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": "http://model/ng-rollout/t0-r1/v1",
                        "apiKey": "EMPTY",  # pragma: allowlist secret
                    },
                    "models": {
                        "test-model": {
                            "name": "test-model",
                            "reasoning": True,
                            "tool_call": True,
                            "interleaved": {"field": "reasoning"},
                            "limit": {"context": 262144, "output": 131072},
                        }
                    },
                },
                "nemo-auxiliary": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": "http://model/v1",
                        "apiKey": "EMPTY",  # pragma: allowlist secret
                    },
                    "models": {"test-model": {"name": "test-model"}},
                },
            },
            "small_model": "nemo-auxiliary/test-model",
        }
    }
    assert config.agent.env == {"EXISTING": "value"}
    assert config.agent.kwargs == {}


def test_build_job_config_scopes_task_and_forces_environment_cleanup(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.environment_build_timeout_multiplier = 3.0
    config.verifier = VerifierConfig(
        import_path=("responses_api_agents.gym_harbor_agent.agentic_verifier:AgenticVerifier"),
        kwargs={"config": {"judge_agent": {"name": "nop"}}},
    )
    config.artifacts = [ArtifactConfig(source="/app", exclude=["data"])]
    agent = AgentConfig(name="opencode", model_name="nemo/test-model")

    job = config.build_job_config("bbh-task", "t0-r1", agent)

    assert job.job_name == "t0-r1"
    assert job.jobs_dir == tmp_path.resolve() / "harbor"
    assert job.datasets[0].task_names == ["bbh-task"]
    assert job.agents == [agent]
    assert job.environment.delete is True
    assert job.verifier == config.verifier
    assert job.artifacts == config.artifacts
    assert job.environment_build_timeout_multiplier == 3.0
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


@patch("responses_api_agents.gym_harbor_agent.app.harbor_job_worker")
@pytest.mark.asyncio
async def test_run_scopes_harbor_job_dir_to_rollout_id(harbor_job_worker, tmp_path: Path) -> None:
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
    harbor_job_worker.remote.side_effect = RuntimeError("stop after config capture")

    response = await agent.run(body)

    job_config = harbor_job_worker.remote.call_args.args[0]
    assert job_config["job_name"] == "t3-r1-nrl-step7-sample3"
    assert job_config["agents"][0]["kwargs"]["opencode_config"]["small_model"] == ("nemo-auxiliary/test-model")
    assert (
        job_config["agents"][0]["kwargs"]["opencode_config"]["provider"]["nemo-auxiliary"]["options"]["baseURL"]
        == "http://model-host:9000/v1"
    )
    assert response.reward == 0.0


def test_explicit_opencode_small_model_is_preserved(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.agent.kwargs = {"opencode_config": {"small_model": "custom/title-model"}}

    agent = config.agent_for_model_server(
        "http://model/ng-rollout/t0-r1/v1",
        auxiliary_base_url="http://model/v1",
    )

    opencode_config = agent.kwargs["opencode_config"]
    assert opencode_config["small_model"] == "custom/title-model"
    assert "nemo-auxiliary" not in opencode_config["provider"]


def test_opensandbox_config_separates_requests_from_limits(monkeypatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    dataset_path = "/datasets/bbh-harbor-rl-v0/val"
    jobs_dir = "/shared/harbor/jobs"
    monkeypatch.setenv("HARBOR_DATASET_PATH", dataset_path)
    monkeypatch.setenv("HARBOR_JOBS_DIR", jobs_dir)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-key")
    monkeypatch.setenv("OPENSANDBOX_PROTOCOL", "https")
    monkeypatch.setenv("OPENSANDBOX_USE_SERVER_PROXY", "false")
    monkeypatch.setenv("OPENSANDBOX_CA_BUNDLE", "/certs/opensandbox.pem")
    monkeypatch.setenv("HARBOR_IMAGE_OVERRIDE", "docker.io/example/task@sha256:1234")
    monkeypatch.setenv(
        "HARBOR_SANDBOX_ENTRYPOINT",
        '["sh", "/opt/entrypoint.sh", "tail", "-f", "/dev/null"]',
    )
    monkeypatch.setenv(
        "HARBOR_SANDBOX_PROBE_COMMAND",
        "python -c \"import bbh_mcp.server; print('nemo-gym-sandbox-ready')\"",
    )
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_EXPECTED_STDOUT", "nemo-gym-sandbox-ready")
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_TIMEOUT_S", "180")
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_DEADLINE_S", "240")
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_STABLE_COUNT", "1")
    monkeypatch.setenv("HARBOR_SANDBOX_PROBE_STABLE_DELAY_S", "0")

    config = OmegaConf.merge(
        OmegaConf.load(gym_root / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"),
        OmegaConf.load(gym_root / "responses_api_agents/gym_harbor_agent/configs/harbor_agent.yaml"),
        OmegaConf.load(gym_root / "responses_api_agents/gym_harbor_agent/configs/harbor_agent_opensandbox.yaml"),
    )
    agent_config = config.gym_harbor_agent.responses_api_agents.gym_harbor_agent
    environment = OmegaConf.to_container(agent_config.environment, resolve=True)

    assert agent_config.jobs_dir == jobs_dir
    assert OmegaConf.to_container(agent_config.dataset, resolve=True)["path"] == dataset_path
    assert environment["import_path"].endswith(":NemoGymSandboxEnvironment")
    assert environment["override_cpus"] == 4
    assert environment["override_memory_mb"] == 65536
    assert agent_config.environment_build_timeout_multiplier == 3.0
    assert environment["kwargs"]["sandbox_provider_options"] == {
        "resource_requests": {"cpu": 0.25, "memory_mib": 512},
        "volumes": [],
    }
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
    assert environment["kwargs"]["sandbox_provider"]["opensandbox"]["probe"] == {
        "command": "python -c \"import bbh_mcp.server; print('nemo-gym-sandbox-ready')\"",
        "expected_stdout": "nemo-gym-sandbox-ready",
        "timeout_s": 180,
        "deadline_s": 240,
        "stable_count": 1,
        "stable_delay_s": 0,
    }


def test_agentic_verifier_config_keeps_policy_and_judge_agents_independent(
    monkeypatch,
) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("RUBRIC_MODEL", "judge-model")
    monkeypatch.setenv("RUBRIC_MODEL_API_BASE", "https://judge.test/v1")
    monkeypatch.setenv("RUBRIC_MODEL_API_MODE", "responses")
    monkeypatch.setenv("HARBOR_POLICY_WORKSPACE_EXCLUDES", "[data]")

    config = OmegaConf.merge(
        {"policy_model_name": "policy-model", "policy_api_key": "policy-key"},
        OmegaConf.load(gym_root / "responses_api_agents/gym_harbor_agent/configs/harbor_agent.yaml"),
        OmegaConf.load(gym_root / "responses_api_agents/gym_harbor_agent/configs/harbor_agent_agentic_verifier.yaml"),
    )
    agent_config = OmegaConf.to_container(
        config.gym_harbor_agent.responses_api_agents.gym_harbor_agent,
        resolve=True,
    )

    assert agent_config["agent"]["model_name"] == "nemo/policy-model"
    verifier = agent_config["verifier"]
    assert verifier["import_path"].endswith(":AgenticVerifier")
    judge = verifier["kwargs"]["config"]["judge_agent"]
    assert judge["name"] == "opencode"
    assert judge["model_name"] == "judge-model"
    assert verifier["kwargs"]["config"]["judge_opencode_provider"] == {
        "api_mode": "responses",
        "base_url": "https://judge.test/v1",
    }
    assert verifier["kwargs"]["config"]["judge_env_aliases"] == {
        "OPENAI_API_KEY": "RUBRIC_MODEL_API_KEY",
        "OPENAI_BASE_URL": "RUBRIC_MODEL_API_BASE",
    }
    assert agent_config["artifacts"] == [{"source": "/app", "exclude": ["data"]}]
