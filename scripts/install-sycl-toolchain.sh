#!/usr/bin/env bash
# install-sycl-toolchain.sh
#
# Provision a recent Intel DPC++/SYCL toolchain (icpx) in an ephemeral
# Claude-Code-on-the-web container, or any Linux box, using only
# repo.anaconda.com.
#
# Strategy: bootstrap a Miniconda *client* from repo.anaconda.com, then pull the
# genuine Intel compiler from conda-forge (BSD-channel): package
# `dpcpp_linux-64`, which also brings `intel-opencl-rt` -- an OpenCL CPU device,
# so SYCL kernels COMPILE AND RUN here (correctness only; no Intel GPU).
#
# Why conda-forge rather than Intel's own apt repo: this route needs one host,
# and the binaries it produces are relocatable with a single -Wl,-rpath, so they
# run under ctest and CI without a sourced environment. (An earlier version of
# this comment asserted apt.repos.intel.com was blocked by the egress policy.
# That is not a fixed property -- egress is configured per environment and it
# changes; it was measured reachable on 2026-09-08. Probe, do not assume.)
#
# Idempotent: re-running is a no-op once `icpx` resolves in the env.
#
# Usage:   bash scripts/install-sycl-toolchain.sh
# Then:    source /opt/conda/etc/profile.d/conda.sh && conda activate sycl
#          icpx -fsycl -O2 -std=c++17 -Wl,-rpath,"$CONDA_PREFIX/lib" foo.cpp -o foo
#          ./foo            # runs on the OpenCL CPU device (sycl-ls to list)
#
# Assisted-by: Claude:claude-opus-4.8
set -euo pipefail

PREFIX="${CONDA_PREFIX_ROOT:-/opt/conda}"
ENV_NAME="${SYCL_ENV_NAME:-sycl}"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
# Honor the session proxy CA bundle when present (Claude Code on the web).
CA="${SSL_CERT_FILE:-/root/.ccr/ca-bundle.crt}"

log() { printf '\n== %s ==\n' "$*"; }

# This script takes no options. Reject them loudly: the skill bundled at
# .claude/skills/intel-sycl-toolchain/ ships a richer installer of the same
# name that does accept --route and --probe, and the two are easy to confuse.
# Silently ignoring --probe here would start a multi-GB install when the caller
# asked only to report reachability.
if [ $# -gt 0 ]; then
    echo "$(basename "$0"): unexpected argument '$1' -- this script takes none." >&2
    echo "For --route/--probe use the skill's installer:" >&2
    echo "  .claude/skills/intel-sycl-toolchain/scripts/install-sycl-toolchain.sh" >&2
    exit 2
fi

# 0. Already provisioned?
if [ -x "$PREFIX/envs/$ENV_NAME/bin/icpx" ]; then
    log "icpx already present at $PREFIX/envs/$ENV_NAME -- nothing to do"
    "$PREFIX/envs/$ENV_NAME/bin/icpx" --version | head -1
    exit 0
fi

# 1. Bootstrap the Miniconda client (not for packages -- just the conda tool).
if [ ! -x "$PREFIX/bin/conda" ]; then
    log "Downloading Miniconda installer"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    curl -fsSL ${CA:+--cacert "$CA"} --max-time 300 -o "$tmp/miniconda.sh" "$MINICONDA_URL"
    log "Installing Miniconda client to $PREFIX"
    rm -rf "$PREFIX"
    bash "$tmp/miniconda.sh" -b -p "$PREFIX"
fi
export PATH="$PREFIX/bin:$PATH"

# 2. Pin conda-forge (reachable + correctly licensed); never use defaults.
log "Configuring conda-forge channel"
conda config --system --remove channels defaults 2>/dev/null || true
conda config --system --add channels conda-forge
conda config --system --set channel_priority strict
[ -f "$CA" ] && conda config --system --set ssl_verify "$CA" || true

# 3. Install the Intel DPC++ compiler + CPU runtime from conda-forge.
# llvm-openmp supplies omp.h: without it find_package(OpenMP) fails and the
# bench_ormqr_sycl_vs_vec target is silently skipped, so the documented install
# path could not build the documented benchmark.
log "Creating env '$ENV_NAME' with dpcpp_linux-64 (the icpx toolchain)"
conda create -y -n "$ENV_NAME" --override-channels -c conda-forge \
    dpcpp_linux-64 llvm-openmp

# 4. Verify: compiler present + a runnable SYCL device.
log "Verifying"
"$PREFIX/envs/$ENV_NAME/bin/icpx" --version | head -1
"$PREFIX/envs/$ENV_NAME/bin/sycl-ls" || true

cat <<EOF

Toolchain ready. To use it:
  source $PREFIX/etc/profile.d/conda.sh && conda activate $ENV_NAME
  icpx -fsycl -O2 -std=c++17 -Wl,-rpath,"\$CONDA_PREFIX/lib" foo.cpp -o foo && ./foo
EOF
