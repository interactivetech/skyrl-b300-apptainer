#!/usr/bin/env bash
# Fetch OpenReward terminal tasks directly into the native SkyRL virtualenv.
# Run inside an existing GPU-enabled Slurm shell (or another networked node).
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/native_env.sh"
: "${OPENREWARD_API_KEY:?Export OPENREWARD_API_KEY before preparing tasks}"
[[ -x "$SKYRL_VENV/bin/python" ]] || { echo "Run scripts/install_native.sh first" >&2; exit 1; }
[[ -d "$SKYRL_ROOT" ]] || { echo "Missing SkyRL checkout: $SKYRL_ROOT" >&2; exit 1; }

source "$SKYRL_VENV/bin/activate"
cd "$SKYRL_ROOT"
python examples/train_integrations/openreward/prepare_tasks.py \
  --env GeneralReasoning/TerminalHorizon \
  --split train \
  --max-tasks "${MAX_TASKS:-16}" \
  --output "$PROJECT_ROOT/data/train.parquet"
echo "Prepared task file: $PROJECT_ROOT/data/train.parquet"
