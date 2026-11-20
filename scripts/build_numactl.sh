#!/usr/bin/env bash
# Optional rootless libnuma installation when the compute node lacks libnuma-dev.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/native_env.sh"
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v make >/dev/null || { echo "make is required (load a compiler module)" >&2; exit 1; }
command -v cc >/dev/null || { echo "C compiler required (load a compiler module)" >&2; exit 1; }

VERSION="2.0.16"
TARBALL="$PROJECT_ROOT/build/numactl-$VERSION.tar.gz"
URL="https://github.com/numactl/numactl/releases/download/v$VERSION/numactl-$VERSION.tar.gz"
curl --fail --location --retry 3 "$URL" --output "$TARBALL"
tar -xzf "$TARBALL" -C "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build/numactl-$VERSION"
./configure --prefix="$PROJECT_ROOT/opt/numactl" --libdir="$PROJECT_ROOT/opt/numactl/lib"
make -j "${SLURM_CPUS_PER_TASK:-4}"
make install
echo "Installed libnuma under $PROJECT_ROOT/opt/numactl"
echo "Source scripts/native_env.sh to set CPATH, LIBRARY_PATH, and LD_LIBRARY_PATH."
