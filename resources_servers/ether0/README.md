# ether0 benchmark environment

[Benchmark](https://huggingface.co/datasets/futurehouse/ether0-benchmark) and [paper](https://arxiv.org/pdf/2506.17238).

325 chemistry reasoning questions across 14 task types. All answers are a molecule. Around 25 questions per task, including:

- Completing SMILES fragments
- Designing molecules adhering to molecular formula and functional group constraints
- Predicting reaction outcomes
- Proposing one-step synthesis pathways
- Editing the solubility of a molecule
- Converting IUPAC name to SMILES
- Answering multiple-choice questions about safety, ADME properties, BBB permeability, toxicity, scent, and pKa

Retro-synthesis and oracle-solubility use an Ether0 remotes sidecar managed by this
resources server. The sidecar is process-local, loads each model once, and is not a
separately configured Gym server.

## Remotes setup

The legacy OpenNMT runtime does not support Gym's Python 3.13 environment. Build its
isolated Python 3.12 environment ahead of launch:

```bash
cd resources_servers/ether0
python setup_remotes.py \
  --venv /path/to/ether0-remotes-venv \
  --runtime-home /path/to/ether0-runtime-home
```

The setup command also installs its Python 3.12 runtime beside the requested
venv. Keep that directory on storage mounted at the same absolute path on every
Gym node; the resulting venv does not depend on the submit host's home directory.

RDKit's drawing extension requires a small X11 runtime closure even though the
Ether0 verifier does not open a display. If the training container does not ship
those libraries, extract them once from a compatible Enroot image:

```bash
python resources_servers/ether0/scripts/prepare_native_libs.py \
  --container /path/to/compatible-image.sqsh \
  --output-dir /shared/ether0/native-libs
```

Set `LD_LIBRARY_PATH=/shared/ether0/native-libs:${LD_LIBRARY_PATH:-}` before
starting Gym. The NeMo-RL Ether0 smoke launcher does this automatically from
`$ETHER0_ASSET_ROOT/native-libs` and accepts `ETHER0_NATIVE_LIBRARY_DIR` as an
override.

Download the Molecular Transformer checkpoint outside the Git worktree and verify it:

```bash
curl --location --output /path/to/USPTO480k_model_step_400000.pt \
  "https://drive.usercontent.google.com/download?id=1Rjd3wXg2oLeCpNUofFRvVvQoOcgWd6vf&export=download&confirm=t"
echo "1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587  /path/to/USPTO480k_model_step_400000.pt" \
  | sha256sum --check
```

Then configure the resources server:

```yaml
ether0:
  resources_servers:
    ether0:
      remotes:
        python_executable: /path/to/ether0-remotes-venv/bin/python
        model_path: /path/to/USPTO480k_model_step_400000.pt
        runtime_home: /path/to/ether0-runtime-home
```

The setup command also prefetches MolBloom's 2.1 GB ZINC20 catalog under the configured
runtime home. The resources server verifies required assets, starts the sidecar during
FastAPI startup, waits until all three model resources are loaded, and terminates the
child on shutdown. Configure `num_workers: 1` when remotes are enabled; multiple workers
would load duplicate model copies.

## Quickstart 

Create `env.yaml`:
```
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: futurehouse/ether0
```

Start servers and collect rollouts
```bash
# start vllm and nemo gym servers
vllm serve futurehouse/ether0 & 
gym env start \
    --resources-server ether0 \
    --model-type vllm_model &

# wait for above to be ready
gym eval run --no-serve \
    --agent ether0_simple_agent \
    --input resources_servers/ether0/data/example.jsonl \
    --output resources_servers/ether0/data/ether0_rollouts.jsonl

tail -n 1 resources_servers/ether0/data/ether0_rollouts.jsonl | jq | less
```

See `scripts/prepare_ether0.py` to prepare the full dataset.
