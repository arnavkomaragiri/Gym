# Gym Harbor Agent

This agent runs a Harbor 0.20 task as a Gym Responses API agent. Harbor's
OpenCode model traffic is sent to the configured Gym model server, including
Gym's per-rollout correlation prefix when token capture is enabled.

The included configuration expects a materialized Harbor dataset at
`$HOME/store/data/edison/harbor-bbh/val` and a Docker-capable worker.

## OpenSandbox

Add these config paths to run any prebuilt-image Harbor dataset through Gym's
OpenSandbox provider:

```yaml
config_paths:
  - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
  - nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
  - responses_api_agents/gym_harbor_agent/configs/harbor_agent.yaml
  - responses_api_agents/gym_harbor_agent/configs/harbor_agent_opensandbox.yaml
```

Set `HARBOR_DATASET_PATH` to the Harbor task root and configure
`OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for the target service. The
OpenSandbox layer requests 0.25 CPU and 512 MiB per sandbox, with burstable
limits of 4 CPU and 64 GiB. The OpenSandbox runtime must be authorized to pull
the prebuilt images referenced by the task configs. `HARBOR_BENCHMARK_NAME`
optionally sets the per-sandbox benchmark metadata label.

`OPENSANDBOX_PROTOCOL` defaults to `http`, and
`OPENSANDBOX_USE_SERVER_PROXY` defaults to `true`. The BBH async smoke launcher
at `examples/nemo_gym/launch_bbh_harbor_async_smoke.sh` accepts all of these as
environment variables; `OPENSANDBOX_API_KEY_FILE` can be used instead of
putting the key directly in the launcher's environment.

## Test

```bash
uv sync --project responses_api_agents/gym_harbor_agent
uv run --project responses_api_agents/gym_harbor_agent \
  pytest responses_api_agents/gym_harbor_agent/tests
```
