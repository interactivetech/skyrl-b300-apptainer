# Repository map — native B300 training inside Slurm

The primary path is **native Python/uv installation**. You are already inside an allocated Slurm GPU shell, so do **not** submit a second job or launch a container.

Default root: `/scratch1/mendeza/skyrl-b300`. Set `PROJECT_ROOT` before sourcing scripts to override it.

| File | Role | State |
|---|---|---|
| [`README.md`](../README.md) | Native setup walkthrough and limitations | Current |
| [`scripts/native_env.sh`](../scripts/native_env.sh) | Exports Python, uv, HF, Triton, Torch, Ray, and output paths under `PROJECT_ROOT` | Ready |
| [`scripts/install_native.sh`](../scripts/install_native.sh) | Installs uv, Python 3.12, SkyRL FSDP deps, OpenReward, pyarrow to persistent scratch venv | Not cluster-tested |
| [`scripts/build_numactl.sh`](../scripts/build_numactl.sh) | Optional rootless libnuma build | Not cluster-tested |
| [`scripts/prepare_data.sh`](../scripts/prepare_data.sh) | Native OpenReward TerminalHorizon dataset preparation | Requires API/network |
| [`scripts/smoke_test_terminal_grader.py`](../scripts/smoke_test_terminal_grader.py) | Verifies hidden final submission grading API | Must be run before RL training |
| [`scripts/patch_openreward_terminal.py`](../scripts/patch_openreward_terminal.py) | Candidate final-reward patch for SkyRL OpenReward adapter | **Provisional** |
| [`scripts/train_native.sh`](../scripts/train_native.sh) | Starts one-GPU local Ray and directly launches SkyRL's Python entrypoint | Not cluster-tested |
| [`slurm/train_openreward.sbatch`](../slurm/train_openreward.sbatch) | Optional separate native batch mode | **Not needed for interactive allocation** |
| [`.env.example`](../.env.example) | Safe configuration sample | No secrets |
| [`.gitignore`](../.gitignore) | Excludes local venv, caches, checkpoints, secrets | Ready |

## Execution path

```text
existing Slurm GPU-allocated shell (1 B300)
   └── /scratch1/mendeza/skyrl-b300/
          ├── bin/uv + python/3.12
          ├── venvs/skyrl/     [persistent native CUDA 13 FSDP stack]
          ├── SkyRL/           [upstream source]
          ├── cache/           [uv, HF models, torch, Triton]
          ├── tmp/             [Ray session]
          └── scripts/train_native.sh
                 ├── Ray local head
                 ├── SkyRL GRPO trainer + vLLM colocated on the B300
                 └── OpenReward TerminalHorizon API (remote tool sandbox)
```

No Apptainer image is necessary. No nested `sbatch` is necessary. The repo name still contains “apptainer” because it was originally created for a container-based starter.

## Not included

A full Codex/OpenCode execution harness is **not** implemented. That requires `examples/train_integrations/harbor_skycap` and Harbor task directories plus a supported sandbox backend; this OpenReward implementation uses the standard `<tool_call>` loop.

## Must validate

1. CUDA 13-compatible driver/toolchain and B300 visibility; `numa.h` and `libnuma.so` accessibility.
2. `uv sync --active --locked --extra fsdp --no-dev` resolves for the GPU node architecture and local compiler.
3. PyTorch, vLLM, Ray imports and single GPU detect successfully.
4. OpenReward hidden `submit_answer` grader produces a finished result and numeric reward before applying the patch.
5. One complete GRPO optimizer step, rewards that are not all identical zero, and no Ray/vLLM startup errors.

This has not been tested on the user's Slurm node. Do not commit any real credentials.
