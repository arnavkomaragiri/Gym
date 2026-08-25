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
  - responses_api_agents/gym_harbor_agent/configs/harbor_agent_opencode_compaction.yaml
  - nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
  - responses_api_agents/gym_harbor_agent/configs/harbor_agent.yaml
  - responses_api_agents/gym_harbor_agent/configs/harbor_agent_opensandbox.yaml
```

The OpenCode compaction overlay preserves vLLM's HTTP context-overflow error
instead of translating it into an empty `finish_reason=length` completion.
OpenCode uses that typed error to compact the session and retry; ordinary
nonempty output-length truncation is unchanged.

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

OpenSandbox volume and sandbox-copy string values support
`{context_id}`, `{environment_name}`, `{task_name}`, `{task_id}`, and
`{session_id}`. `{context_id}` is Harbor's trial UUID and is shared by the
policy and separate-verifier environments. `{task_name}` aliases Harbor's
`environment_name`. Unknown placeholders are rejected during Gym agent
configuration and again before the provider create call; they are never passed
through literally.

Before increasing rollout concurrency, create one sandbox using the same
image, entrypoint, provider metadata, and rendered volumes as the intended
training run. Confirm the task-specific mount exists and is readable inside
the sandbox, then scale through a small batch before using the full queue.

The direct EFS workspace path is
`<EFS host>/<owner>/nemo-gym-harbor-artifacts/<run>/{context_id}`. The policy
mounts that physical source read-write at `/app`; the separate verifier mounts
the exact same source read-only at `/app`. Task data is an independent
task-specific S3 volume mounted read-only at `/data` in both roles. The policy
creates `/app/data -> /data`; the verifier validates the inherited link against
its independent clean-data mount without modifying read-only `/app`. Harbor
passes only an integrity marker between roles, so policy files are never
downloaded, uploaded, archived, or reconstructed. After verifier termination,
a short-lived cleanup sandbox mounts only the run parent and removes the exact
context-ID directory.

## Test

```bash
uv sync --project responses_api_agents/gym_harbor_agent
uv run --project responses_api_agents/gym_harbor_agent \
  pytest responses_api_agents/gym_harbor_agent/tests
```
