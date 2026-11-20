#!/usr/bin/env bash
# Source this file before native setup, data preparation, or training.
# All persistent environments, caches and outputs live under /scratch1/mendeza.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source scripts/native_env.sh" >&2
  exit 2
fi

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch1/mendeza/skyrl-b300}"
if [[ "$PROJECT_ROOT" != /* ]]; then
  echo "PROJECT_ROOT must be an absolute path (got: $PROJECT_ROOT)" >&2
  return 2
fi

export SKYRL_ROOT="$PROJECT_ROOT/SkyRL"
export SKYRL_VENV="$PROJECT_ROOT/venvs/skyrl"
export UV_PROJECT_ENVIRONMENT="$SKYRL_VENV"
export UV_INSTALL_DIR="$PROJECT_ROOT/bin"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/python"
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export TRITON_CACHE_DIR="$PROJECT_ROOT/cache/triton"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg"
export TMPDIR="$PROJECT_ROOT/tmp"
export RAY_RUNTIME_ENV_HOOK="ray._private.runtime_env.uv_runtime_env_hook.hook"
export PYTHONNOUSERSITE=1
export PATH="$PROJECT_ROOT/bin:$SKYRL_VENV/bin:$PATH"

# If libnuma was built without root privileges, use that local copy.
if [[ -d "$PROJECT_ROOT/opt/numactl/lib" ]]; then
  export LD_LIBRARY_PATH="$PROJECT_ROOT/opt/numactl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LIBRARY_PATH="$PROJECT_ROOT/opt/numactl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export CPATH="$PROJECT_ROOT/opt/numactl/include${CPATH:+:$CPATH}"
fi

mkdir -p "$PROJECT_ROOT"/{bin,venvs,python,data,cache/uv,cache/huggingface,cache/torch,cache/triton,cache/xdg,tmp,logs,ckpts,exports,build}
