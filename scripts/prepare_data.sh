#!/usr/bin/env bash
# Run on a node that has network access to OpenReward, with the SIF present.
set -Eeuo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT}"
: "${OPENREWARD_API_KEY:?Set OPENREWARD_API_KEY}"
ROOT="$(realpath "$PROJECT_ROOT")"
mkdir -p "$ROOT/data" "$ROOT/cache/uv" "$ROOT/cache/hf"
export APPTAINERENV_OPENREWARD_API_KEY="$OPENREWARD_API_KEY"
export APPTAINERENV_UV_CACHE_DIR="$ROOT/cache/uv"
export APPTAINERENV_HF_HOME="$ROOT/cache/hf"
apptainer exec --cleanenv --bind "$ROOT:$ROOT" --pwd "$ROOT/SkyRL" \
 "$ROOT/images/skyrl-cu13.sif" bash -c '
  export PATH="/home/ray/.local/bin:$PATH"
  uv run --isolated --extra fsdp --with openreward --with pyarrow \
    python examples/train_integrations/openreward/prepare_tasks.py \
    --env GeneralReasoning/TerminalHorizon --split train \
    --max-tasks 16 --output "$1/data/train.parquet"
' bash "$ROOT"
