# SkyRL + OpenReward / single B300 / Slurm / Apptainer

This is a starter kit for a **single GPU training smoke test** using the current SkyRL OpenReward integration. It is **not a validated end-to-end run on a B300**; your cluster and hosted environment must be tested. The Codex/OpenCode harness route requires SkyRL's separate `harbor_skycap` integration and a Harbor-compatible sandbox provider.

## Requirements
- NVIDIA B300 Slurm allocation; host driver >= R580 for SkyRL CUDA 13 image
- Apptainer, outbound networking for OpenReward API and model/dependency downloads, adequate storage
- A shared writable PROJECT_ROOT, and an OpenReward API key
- A usable /dev/shm allocation for Ray (prefer 8+ GiB)

## Stage files
```bash
export PROJECT_ROOT=/path/to/shared/skyrl-b300
mkdir -p "$PROJECT_ROOT"/{images,scripts,data,cache,logs}
git clone https://github.com/skyrl-project/SkyRL.git "$PROJECT_ROOT/SkyRL"
# Pin this git checkout to a known commit for reproducibility.
apptainer pull "$PROJECT_ROOT/images/skyrl-cu13.sif" \
  docker://novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0
cp scripts/*.sh scripts/*.py "$PROJECT_ROOT/scripts/"
mkdir -p "$PROJECT_ROOT/slurm"
cp slurm/*.sbatch "$PROJECT_ROOT/slurm/"
cd "$PROJECT_ROOT/SkyRL"
python "$PROJECT_ROOT/scripts/patch_openreward_terminal.py"
```

CAUTION: the finalization patch is **provisional**: the OpenReward TerminalHorizon page documents a hidden `submit_answer` tool, but the SDK call through the SkyRL adapter needs a live smoke test. The patch raises on failures instead of treating them as a successful zero-reward episode. The upstream integration otherwise returns reward 0 on a final answer without a tool call, which is inappropriate for terminal tasks.

## Prepare task prompts
```bash
export OPENREWARD_API_KEY='...'
export PROJECT_ROOT=/path/to/shared/skyrl-b300
bash "$PROJECT_ROOT/scripts/prepare_data.sh"
```

The selected environment is `GeneralReasoning/TerminalHorizon` (remote hosted terminal tasks). This is not a host-local terminal sandbox; the GPU and trainer are local on the B300 while the task environment runs at OpenReward.

## Submit job
```bash
sbatch --export=ALL \
  "$PROJECT_ROOT/slurm/train_openreward.sbatch"
```

For strict secret handling, use a restricted-permission credentials file read at job startup rather than exported environment variables. Do not embed keys in git, shell history, job names, or logs.

## Constraints
- Depending on site policy, Slurm GPU request may use `--gres=gpu:1` instead of `--gpus=1`.
- `uv run --isolated` may need outbound access on compute nodes unless all resolved wheels are cached. Prefetch dependencies into a cache while online; the SIF is a base image, not a fully frozen Python environment.
- Ray, UV and vLLM may need adaptations to cluster sockets, ports and /dev/shm configuration. Do not launch on login nodes.
- For terminal tasks, a small model and short turn limit may produce all-zero rewards, in which case GRPO cannot meaningfully improve the policy.
- This does not run Codex or OpenCode. For that use `examples/train_integrations/harbor_skycap`, set `harbor_trial_config.agent.name=codex` or `opencode`, `generator.step_wise_trajectories=true`, `generator.merge_stepwise_output=false`, and configure an accessible Harbor sandbox provider and endpoint exposure.

## Core references
- https://github.com/skyrl-project/SkyRL/tree/main/examples/train_integrations/openreward
- https://github.com/skyrl-project/SkyRL/tree/main/examples/train_integrations/harbor_skycap
- https://docs.skyrl.ai/docs/getting-started/installation
- https://openreward.ai/GeneralReasoning/TerminalHorizon

## Stage 2: actual Codex/OpenCode rollout harness (design sketch, not preflight-tested)

SkyRL's `harbor_skycap` generator, rather than the `openreward` generator, is the integration that supports a full installed coding agent. A **separate** Harbor sandbox provider is mandatory. Harbor 0.14.0 supports `docker` (if admins provide a usable Docker daemon) or cloud sandbox providers such as `daytona` or `modal`. It does not list `apptainer` as an environment backend.

Indicative overrides when using the `main_harbor_skycap` entrypoint:

```text
trainer.placement.colocate_all=true
trainer.strategy=fsdp
trainer.placement.policy_num_gpus_per_node=1
trainer.placement.ref_num_gpus_per_node=1
generator.inference_engine.num_engines=1
generator.inference_engine.tensor_parallel_size=1
generator.inference_engine.backend=vllm
generator.inference_engine.served_model_name=policy
generator.inference_engine.gpu_memory_utilization=0.35
generator.step_wise_trajectories=true
generator.merge_stepwise_output=false
trainer.algorithm.max_seq_len=8192
trainer.algorithm.loss_reduction=token_mean
generator.n_samples_per_prompt=2
trainer.train_batch_size=2
trainer.policy_mini_batch_size=2
trainer.micro_train_batch_size_per_gpu=1
harbor_trial_config.agent.name=codex
harbor_trial_config.environment.type=daytona
skycap.exposure.type=cloudflare
generator.rate_limit.enabled=true
generator.rate_limit.max_concurrency=2
```

Use `harbor_trial_config.agent.name=opencode` for OpenCode, but test its OpenAI-compatible provider configuration; it is not validated by this bundle. Use a valid Harbor task-directory dataset for `data.train_data`. The guide's original OpenReward Parquet dataset is **not** interchangeable with Harbor's task-directory dataset.

On restrictive Slurm systems with no Docker daemon and no cloud sandbox provider, a new Harbor backend implementing an Apptainer task sandbox must be developed and security-reviewed. Merely changing the trainer container to Apptainer does **not** make Harbor's environment Apptainer-compatible.