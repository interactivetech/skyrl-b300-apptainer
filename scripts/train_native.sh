#!/usr/bin/env bash
# Train directly on the GPU assigned to the current interactive Slurm allocation.
# Unlike the upstream run_openreward.sh, this uses the already-installed venv
# and does NOT call uv run --isolated or launch Apptainer.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/native_env.sh"

: "${OPENREWARD_API_KEY:?Export OPENREWARD_API_KEY before training}"
[[ -x "$SKYRL_VENV/bin/python" ]] || { echo "Run scripts/install_native.sh first" >&2; exit 1; }
[[ -s "$PROJECT_ROOT/data/train.parquet" ]] || { echo "Run scripts/prepare_data.sh first" >&2; exit 1; }
[[ -d "$SKYRL_ROOT" ]] || { echo "Missing SkyRL checkout: $SKYRL_ROOT" >&2; exit 1; }

source "$SKYRL_VENV/bin/activate"
cd "$SKYRL_ROOT"
export PYTHONUNBUFFERED=1
export OPENREWARD_UPLOAD_ROLLOUT="${OPENREWARD_UPLOAD_ROLLOUT:-false}"
export OPENREWARD_RUN_NAME="${OPENREWARD_RUN_NAME:-skyrl-terminal-b300}"
export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export RUN_NAME="${RUN_NAME:-openreward-terminal-b300}"
export TMPDIR="$PROJECT_ROOT/tmp"
export RAY_TMPDIR="$PROJECT_ROOT/tmp/ray-${SLURM_JOB_ID:-manual}-$$"
mkdir -p "$RAY_TMPDIR"
export RAY_RUNTIME_ENV_HOOK="ray._private.runtime_env.uv_runtime_env_hook.hook"

nvidia-smi
python - <<'PY'
import torch, ray, vllm, openreward
assert torch.cuda.is_available(), "PyTorch cannot see the Slurm GPU"
assert torch.cuda.device_count() == 1, f"Expected exactly 1 visible GPU, got {torch.cuda.device_count()}"
print("GPU:", torch.cuda.get_device_name(0))
print("Torch/CUDA:", torch.__version__, torch.version.cuda)
print("Ray/vLLM:", ray.__version__, vllm.__version__)
PY

# Unique per-job Ray port, on the SAME allocated compute node.
# If the port is already occupied, set RAY_HEAD_PORT explicitly.
JOB_NUM="${SLURM_JOB_ID:-$$}"
[[ "$JOB_NUM" =~ ^[0-9]+$ ]] || JOB_NUM=$$
RAY_HEAD_PORT="${RAY_HEAD_PORT:-$((15000 + JOB_NUM % 10000))}"
export RAY_ADDRESS="127.0.0.1:$RAY_HEAD_PORT"

ray start --head --port="$RAY_HEAD_PORT" \
  --num-gpus=1 --num-cpus="${SLURM_CPUS_PER_TASK:-16}" \
  --include-dashboard=false --temp-dir="$RAY_TMPDIR"
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
ray status --address="$RAY_ADDRESS"

TRAIN_DATA="['$PROJECT_ROOT/data/train.parquet']"
# Initial smoke run: 4 prompts x 2 samples = 8 local inference trajectories.
# Increase after verifying graded rewards; this is not a production recipe.
python -m examples.train_integrations.openreward.entrypoints.main_openreward \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$TRAIN_DATA" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.35 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=768 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.n_samples_per_prompt=2 \
  generator.max_turns=16 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</tool_call>"]' \
  environment.env_class=openreward \
  environment.skyrl_gym.max_env_workers=4 \
  trainer.logger=console \
  trainer.project_name=skyrl-openreward \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=5 \
  trainer.max_ckpts_to_keep=3 \
  trainer.resume_mode=none \
  trainer.ckpt_path="$PROJECT_ROOT/ckpts" \
  trainer.export_path="$PROJECT_ROOT/exports" \
  trainer.log_path="$PROJECT_ROOT/logs" \
  trainer.eval_interval=0 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</tool_call>"]' \
  generator.eval_sampling_params.max_generate_length=768 \
  "$@"
