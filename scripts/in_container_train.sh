#!/usr/bin/env bash
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT missing in container}"
cd "$PROJECT_ROOT/SkyRL"
export PATH="/home/ray/.local/bin:$PATH"
export RAY_RUNTIME_ENV_HOOK='ray._private.runtime_env.uv_runtime_env_hook.hook'
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/cache/uv}"
export TMPDIR="$PROJECT_ROOT/tmp"
export RAY_TMPDIR="$PROJECT_ROOT/tmp/ray-${SLURM_JOB_ID:-manual}-$$"
mkdir -p "$RAY_TMPDIR" "$PROJECT_ROOT/logs"

command -v uv
command -v ray
nvidia-smi
python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'GPU not visible inside Apptainer'
print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))
PY

# Ray is private to this one Slurm allocation; do not connect to a shared head.
ray start --head --num-gpus=1 --num-cpus="${SLURM_CPUS_PER_TASK:-16}" \
  --include-dashboard=false --temp-dir="$RAY_TMPDIR"
trap 'ray stop --force || true' EXIT

export NUM_GPUS=1
export LOGGER=console
export DATA_DIR="$PROJECT_ROOT/data"
export CKPT_DIR="$PROJECT_ROOT/ckpts"
export EXPORT_DIR="$PROJECT_ROOT/exports"
export RUN_NAME="${RUN_NAME:-openreward-terminalhorizon-b300}"
export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# Intentional tiny smoke run: 4 prompts * 2 GRPO samples = 8 trajectories / step.
# Increase the batch, trajectory turns, and dataset only after observing real reward variation.
bash examples/train_integrations/openreward/run_openreward.sh \
  trainer.epochs=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=768 \
  generator.n_samples_per_prompt=2 \
  generator.max_turns=16 \
  environment.skyrl_gym.max_env_workers=4 \
  generator.inference_engine.gpu_memory_utilization=0.35 \
  trainer.eval_interval=0 \
  trainer.eval_before_train=false \
  trainer.ckpt_interval=5 \
  trainer.resume_mode=none \
  "$@"
