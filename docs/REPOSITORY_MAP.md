# Repository map and development status

This repo is intentionally a small **starter**, not a fork of SkyRL.
The upstream SkyRL checkout is staged at `$PROJECT_ROOT/SkyRL` at runtime.

| File | Purpose | Status |
|---|---|---|
| [`README.md`](../README.md) | Prerequisites, setup, and caveats | Documented, not cluster-validated |
| [`slurm/train_openreward.sbatch`](../slurm/train_openreward.sbatch) | Single-node/single-GPU job launch | Syntax checked, not cluster-validated |
| [`scripts/prepare_data.sh`](../scripts/prepare_data.sh) | Fetch remote OpenReward terminal tasks into Parquet | Syntax checked, requires API/network validation |
| [`scripts/patch_openreward_terminal.py`](../scripts/patch_openreward_terminal.py) | Candidate fix for terminal final-state reward | **Provisional; verify SDK contract first** |
| [`scripts/in_container_train.sh`](../scripts/in_container_train.sh) | Ray and one-GPU SkyRL training launcher | Syntax checked, not cluster-validated |
| [`.gitignore`](../.gitignore) | Keep credentials, caches, checkpoints and model weights out of Git | Ready |
| [`.env.example`](../.env.example) | Safe environment-variable template | Ready |

## The two separate integration stages

**Stage 1 (files in this repository):** Slurm → Apptainer → SkyRL FSDP/vLLM GRPO → OpenReward remote terminal tasks. This does not execute Codex or OpenCode.

**Stage 2 (not implemented here):** Slurm → Apptainer → SkyRL `harbor_skycap` → Harbor Codex/OpenCode agent → sandbox environment → verifier reward. A Harbor-compatible sandbox provider is required. Harbor 0.14.0 does **not** list Apptainer as a native sandbox backend; a new integration is needed if the cluster offers no Docker-capable or remote sandbox provider.

## Verify before any real training

1. Inspect `nvidia-smi`, architecture (`uname -m`), driver version, and Apptainer visibility of the B300.
2. Confirm the chosen image exists for your CPU architecture and that compute nodes can fetch dependencies or use an offline cache.
3. Verify the OpenReward SDK accepts the terminal `submit_answer` call and returns a real graded terminal reward; the patch is deliberately untested.
4. Check `Ray`, `vLLM`, Torch, and FlashAttention initialize with CUDA 13 on the node.
5. Run a tiny training step and check reward variation and optimizer progression, then scale batch size and turns.

No real API keys or cluster secrets should ever be committed. If you use this repo with a public visibility setting later, review its history and ensure no sensitive outputs are present.
