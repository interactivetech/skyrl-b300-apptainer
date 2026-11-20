#!/usr/bin/env bash
#SBATCH --job-name=skyrl-openreward-b300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=skyrl-%j.log
set -Eeuo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to a shared writable directory containing SkyRL and images/skyrl-cu13.sif}"
: "${OPENREWARD_API_KEY:?Export OPENREWARD_API_KEY before sbatch}"

ROOT="$(realpath "$PROJECT_ROOT")"
IMG="$ROOT/images/skyrl-cu13.sif"
test -f "$IMG"
test -f "$ROOT/SkyRL/examples/train_integrations/openreward/run_openreward.sh"
test -f "$ROOT/data/train.parquet" || { echo 'Prepare data/train.parquet first'; exit 1; }
mkdir -p "$ROOT"/{cache/hf,cache/uv,tmp,logs,ckpts,exports}

# Apptainer --cleanenv passes through only explicit APPTAINERENV_* variables.
# Preserve Slurm's assigned GPU visibility through --cleanenv.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
export APPTAINERENV_SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
export APPTAINERENV_OPENREWARD_API_KEY="$OPENREWARD_API_KEY"
export APPTAINERENV_OPENREWARD_UPLOAD_ROLLOUT="${OPENREWARD_UPLOAD_ROLLOUT:-false}"
export APPTAINERENV_RAY_RUNTIME_ENV_HOOK='ray._private.runtime_env.uv_runtime_env_hook.hook'
export APPTAINERENV_HF_HOME="$ROOT/cache/hf"
export APPTAINERENV_UV_CACHE_DIR="$ROOT/cache/uv"
export APPTAINERENV_PROJECT_ROOT="$ROOT"
export APPTAINERENV_RUN_NAME="${RUN_NAME:-openreward-terminalhorizon-b300}"
export APPTAINERENV_MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# No Docker daemon is needed for this trainer container.
apptainer exec --cleanenv --nv \
  --bind "$ROOT:$ROOT" \
  --pwd "$ROOT/SkyRL" \
  "$IMG" \
  bash "$ROOT/scripts/in_container_train.sh"
