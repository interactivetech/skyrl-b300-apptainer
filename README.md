# SkyRL + OpenReward on one NVIDIA B300 (native Slurm install)

**Primary workflow: no Apptainer, Docker, sudo, or new `sbatch` command.**
Assume you are **already inside an interactive shell on a GPU-allocated Slurm compute node**. The repository name is historical; the current workflow installs Python dependencies directly on the host.

This is a **single-GPU GRPO smoke-test starter**, not a production-validated B300 recipe. Training runs locally on your allocated B300; OpenReward terminal task sandboxes are **remote**, not on your cluster. Codex/OpenCode harness training remains a separate, unimplemented second-stage integration.

## 1. Verify the allocated B300 and host software

Run these commands from the interactive Slurm job (not a login node):

```bash
hostname
echo "Slurm job: $SLURM_JOB_ID"
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uname -m
command -v gcc || command -v cc
command -v make
command -v nvcc && nvcc --version || true
df -h /scratch1/mendeza
```

SkyRL's current native install targets **Python 3.12, Ray 2.57.0, CUDA 13.0**, and an NVIDIA **R580+** driver. A CUDA 13 toolkit and working compiler may be needed for package builds. `nvidia-smi`'s "CUDA Version" means the *driver API compatibility*, not that the complete toolkit or `nvcc` is installed.

On clusters, first load the administrator-provided GCC/build and CUDA 13 modules, if needed; **module names vary by site**. `libnuma` development headers are also required. There is a rootless installation option below.

## 2. Clone this repository into /scratch1/mendeza

```bash
mkdir -p /scratch1/mendeza
cd /scratch1/mendeza
git clone https://github.com/interactivetech/skyrl-b300-apptainer.git skyrl-b300
cd /scratch1/mendeza/skyrl-b300
export PROJECT_ROOT=/scratch1/mendeza/skyrl-b300
```

If you already cloned it, run `git pull` instead of cloning again.

All project state stays inside `$PROJECT_ROOT`:

```text
/scratch1/mendeza/skyrl-b300/
├── SkyRL/                   # upstream SkyRL source and uv.lock
├── venvs/skyrl/             # persistent Python 3.12 venv (OUTSIDE SkyRL source)
├── python/                  # uv-managed Python builds
├── bin/                     # uv executable
├── opt/numactl/             # optional rootless libnuma
├── cache/{uv,huggingface,torch,triton,xdg}/
├── tmp/                     # Ray / temporary files
├── data/train.parquet
├── logs/
├── ckpts/
├── exports/
└── scripts/                 # scripts from this repository
```

`scripts/native_env.sh` exports `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_PROJECT_ENVIRONMENT`, `HF_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, `TMPDIR`, and the local paths. It does not change your `~/.bashrc`.

## 3. Install dependencies manually (no container)

If your site lacks the `numa.h` header, build libnuma *without root* in `/scratch1/mendeza/skyrl-b300/opt/numactl` **before** syncing dependencies:

```bash
# Only if libnuma headers/runtime are missing:
bash scripts/build_numactl.sh
```

Now do the native installation:

```bash
bash scripts/install_native.sh
```

The script uses your current Slurm GPU node; it does the following:

1. Installs `uv` to `$PROJECT_ROOT/bin` if missing, and uv-managed Python 3.12 under `$PROJECT_ROOT/python`.
2. Clones `skyrl-project/SkyRL` into `$PROJECT_ROOT/SkyRL` (unless it already exists). Optionally set `SKYRL_COMMIT` before running to pin its source revision.
3. Creates `$PROJECT_ROOT/venvs/skyrl` **outside** the SkyRL source tree, as recommended for Ray workers.
4. Runs `uv sync --active --locked --extra fsdp --no-dev` against SkyRL's own `uv.lock`. This preserves SkyRL's CUDA 13-specific wheel indexes and dependency overrides.
5. Installs `openreward` and `pyarrow` into the **persistent** venv using `uv pip install` (not `uv run --isolated`).
6. Checks imports and verifies PyTorch can see the GPU.

If packages cannot download on your compute node, perform installation on an allowed networked build node with the same architecture and compatible software stack, or prepopulate the cache; consult your HPC administrators. If the cluster lacks CUDA 13 or a compatible driver, this exact SkyRL version may not work.

Use the persistent environment after setup:

```bash
source scripts/native_env.sh
source "$SKYRL_VENV/bin/activate"
which python
python -c "import torch, ray, vllm; print(torch.__version__, ray.__version__, vllm.__version__, torch.cuda.get_device_name(0))"
```

**Important:** Do not call the original upstream `run_openreward.sh` in this workflow. It invokes `uv run --isolated` and may build a second environment. Use `scripts/train_native.sh`, which invokes the installed Python directly. If you rerun `uv sync` manually, reinstall `openreward pyarrow` afterward because sync can remove extra packages.

## 4. Supply the API key and prepare tasks

Get a key from [OpenReward](https://openreward.ai/keys). Avoid writing real keys into the repository, command history, or Slurm logs. In your interactive shell:

```bash
read -r -s -p "OpenReward API key: " OPENREWARD_API_KEY
echo
export OPENREWARD_API_KEY
export MAX_TASKS=16
bash scripts/prepare_data.sh
```

This creates `/scratch1/mendeza/skyrl-b300/data/train.parquet` from `GeneralReasoning/TerminalHorizon`. You need outbound access to OpenReward.

### Verify terminal grading before training

**This step is mandatory for the TerminalHorizon recipe.** The original OpenReward SkyRL adapter ends a final answer with reward zero; this starter includes a *provisional* change to call the environment's hidden `submit_answer` grader. The exact SDK contract is **not independently validated** and could incur API usage costs.

```bash
# Should report finished=True and a numeric reward.
python scripts/smoke_test_terminal_grader.py

# Only after the smoke test succeeds:
cd "$SKYRL_ROOT"
python "$PROJECT_ROOT/scripts/patch_openreward_terminal.py"
cd "$PROJECT_ROOT"
```

If the grader test fails, **do not train on fabricated zero rewards**. Inspect your SDK/environment contract and adjust the adapter. The patch makes a backup file in the SkyRL checkout and refuses to overwrite changed upstream code.

## 5. Train directly in your existing Slurm allocation

You have the GPU already: **do not submit another Slurm job**.

```bash
cd /scratch1/mendeza/skyrl-b300
source scripts/native_env.sh
source "$SKYRL_VENV/bin/activate"
bash scripts/train_native.sh
```

The script starts a one-node, one-GPU Ray head inside the allocated job, runs the OpenReward entrypoint via `python -m`, and cleans up its Ray processes when it exits. It defaults to **Qwen2.5-3B-Instruct**, **4 prompts/step**, **2 samples/prompt**, and **vLLM GPU memory utilization 0.35** for an initial smoke test.

Optional CLI configuration overrides go after the shell script:

```bash
bash scripts/train_native.sh trainer.epochs=2 generator.max_turns=20
```

Outputs land in `$PROJECT_ROOT/logs`, `$PROJECT_ROOT/ckpts`, and `$PROJECT_ROOT/exports`. Ray temp files are under `$PROJECT_ROOT/tmp`, and Hugging Face model downloads use `$PROJECT_ROOT/cache/huggingface`.

The script chooses a Ray head port based on the job ID; if it conflicts with another process, set `RAY_HEAD_PORT` first. Only launch once per allocated node; it is not intended to join an existing shared Ray cluster.

## Optional: submit a future non-interactive Slurm job

Only from outside an existing job, after the native installation and dataset are ready:

```bash
sbatch --export=ALL slurm/train_openreward.sbatch
```

This optional wrapper also uses the **native** environment, not Apptainer. Change the `#SBATCH` directives to match your HPC partition/account policies.

## Limitations / next steps

- **Full coding-agent harness:** This run uses SkyRL's built-in `<tool_call>` loop against OpenReward, **not** the Codex or OpenCode binaries. For that, use SkyRL's separate [Harbor + skycap integration](https://github.com/skyrl-project/SkyRL/tree/main/examples/train_integrations/harbor_skycap) and a compatible sandbox provider. A local Slurm/Apptainer-backed *task* sandbox is a separate integration effort.
- **GPU/runtime compatibility:** The script assumes a single visible B300, host CUDA compatibility, Linux build tools, and sufficient RAM and `/dev/shm`. None of the scripts has been executed on your actual cluster.
- **Reward quality:** All-zero terminal rewards or invalid final grading will not train a useful GRPO policy. Inspect rollout errors, rewards, and optimizer updates before scaling.
- **Reproducibility:** `$PROJECT_ROOT/skyrl-source-commit.txt` records the upstream revision; pin `SKYRL_COMMIT` and keep `uv.lock` with that checkout.
- **Security:** This GitHub repo may be public. Never commit API keys, personal paths to credentials, raw user datasets, or generated checkpoints.

## References

- [SkyRL native installation guide](https://docs.skyrl.ai/docs/getting-started/installation)
- [SkyRL OpenReward example](https://github.com/skyrl-project/SkyRL/tree/main/examples/train_integrations/openreward)
- [SkyRL Harbor + skycap example](https://github.com/skyrl-project/SkyRL/tree/main/examples/train_integrations/harbor_skycap)
- [OpenReward TerminalHorizon](https://openreward.ai/GeneralReasoning/TerminalHorizon)
