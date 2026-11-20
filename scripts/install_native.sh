#!/usr/bin/env bash
# Run this ONCE (and after upstream upgrades) inside your existing GPU-enabled Slurm session.
# No sudo, container runtime, or new sbatch is required.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/native_env.sh"

for tool in git curl make cc; do
  command -v "$tool" >/dev/null || { echo "Missing $tool: load your site compiler/toolchain module first" >&2; exit 1; }
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "WARNING: SLURM_JOB_ID is unset. Use an allocated GPU compute node; do not build heavy GPU dependencies on a login node." >&2
fi

nvidia-smi || { echo "No accessible GPU. Check Slurm GPU allocation / driver." >&2; exit 1; }
echo "Architecture: $(uname -m); CUDA toolkit (optional):"
if command -v nvcc >/dev/null; then nvcc --version | tail -n 4; else echo "nvcc not on PATH; CUDA 13 toolkit may be needed by source builds"; fi

if ! command -v uv >/dev/null; then
  echo "Installing uv into $PROJECT_ROOT/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PROJECT_ROOT/bin" sh
fi
command -v uv
uv --version

# uv-managed Python lives under $PROJECT_ROOT/python (not ~/.local/share/uv).
uv python install 3.12

if [[ ! -d "$SKYRL_ROOT/.git" ]]; then
  if [[ -e "$SKYRL_ROOT" ]]; then
    echo "$SKYRL_ROOT exists but is not a Git checkout; inspect it first" >&2; exit 1
  fi
  git clone --recurse-submodules https://github.com/skyrl-project/SkyRL.git "$SKYRL_ROOT"
fi

if [[ -n "${SKYRL_COMMIT:-}" ]]; then
  git -C "$SKYRL_ROOT" checkout "$SKYRL_COMMIT"
fi
git -C "$SKYRL_ROOT" rev-parse HEAD | tee "$PROJECT_ROOT/skyrl-source-commit.txt"

# SkyRL recommends the base venv OUTSIDE its source tree for Ray worker shipping.
if [[ ! -x "$SKYRL_VENV/bin/python" ]]; then
  uv venv --python 3.12 "$SKYRL_VENV"
fi
source "$SKYRL_VENV/bin/activate"
cd "$SKYRL_ROOT"

# Use SkyRL's own uv.lock + custom wheel indexes (especially CUDA 13 wheels).
uv sync --active --locked --extra fsdp --no-dev

# OpenReward is supplied with --with in upstream's isolated examples.
# Here we install into the persistent venv and launch Python directly instead.
uv pip install --python "$SKYRL_VENV/bin/python" openreward pyarrow

python - <<'PY'
import sys, torch, ray, vllm, pyarrow, openreward
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Ray:", ray.__version__, "vLLM:", vllm.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU is invisible to PyTorch"
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA capability:", torch.cuda.get_device_capability(0))
PY

echo "Native install complete:"
echo "  source $PROJECT_ROOT/scripts/native_env.sh"
echo "  source $SKYRL_VENV/bin/activate"
echo "  export OPENREWARD_API_KEY=..."
echo "  bash $PROJECT_ROOT/scripts/prepare_data.sh"
echo "  bash $PROJECT_ROOT/scripts/train_native.sh"
