#!/usr/bin/env bash
# OPTIONAL batch-mode wrapper. NOT needed when already inside an interactive GPU job.
# Run scripts/install_native.sh first, in a GPU-enabled Slurm allocation.
#SBATCH --job-name=skyrl-b300-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=skyrl-%j.log
set -Eeuo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch1/mendeza/skyrl-b300}"
: "${OPENREWARD_API_KEY:?Export OPENREWARD_API_KEY before submitting batch job}"
[[ -x "$PROJECT_ROOT/venvs/skyrl/bin/python" ]] || { echo "Native venv missing; run scripts/install_native.sh first" >&2; exit 1; }
exec bash "$PROJECT_ROOT/scripts/train_native.sh"
