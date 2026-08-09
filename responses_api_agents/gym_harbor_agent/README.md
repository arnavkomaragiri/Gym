# Gym Harbor Agent

This agent runs a Harbor 0.20 task as a Gym Responses API agent. Harbor's
OpenCode model traffic is sent to the configured Gym model server, including
Gym's per-rollout correlation prefix when token capture is enabled.

The included configuration expects a materialized Harbor dataset at
`$HOME/store/data/edison/harbor-bbh/val` and a Docker-capable worker.

## Test

```bash
uv sync --project responses_api_agents/gym_harbor_agent
uv run --project responses_api_agents/gym_harbor_agent \
  pytest responses_api_agents/gym_harbor_agent/tests
```
