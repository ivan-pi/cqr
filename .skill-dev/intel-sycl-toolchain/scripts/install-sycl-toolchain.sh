#!/usr/bin/env bash
# install-sycl-toolchain.sh
#
# Provision an Intel oneAPI DPC++/SYCL toolchain (icpx) on Linux, choosing
# automatically between the two routes that actually work in sandboxed
# containers. Idempotent: re-running once icpx exists just prints the version.
#
#   Route "apt"   -- Intel's official apt repo (apt.repos.intel.com).
#                    Full oneAPI stack. Binaries need `source setvars.sh`.
#   Route "conda" -- conda-forge dpcpp_linux-64 via a Miniconda client
#                    (repo.anaconda.com). Binaries are relocatable when linked
#                    with -Wl,-rpath,$CONDA_PREFIX/lib.
#
# Usage:
#   bash install-sycl-toolchain.sh                 # auto-detect
#   bash install-sycl-toolchain.sh --route conda   # force a route
#   bash install-sycl-toolchain.sh --route apt
#   bash install-sycl-toolchain.sh --probe         # only report reachability
set -euo pipefail

PREFIX="${CONDA_PREFIX_ROOT:-/opt/conda}"
ENV_NAME="${SYCL_ENV_NAME:-sycl}"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ONEAPI_ROOT="/opt/intel/oneapi"
# Honor a session proxy CA bundle when present (Claude Code on the web).
CA="${SSL_CERT_FILE:-/root/.ccr/ca-bundle.crt}"
[ -f "$CA" ] || CA=""

ROUTE="auto"
PROBE_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --route) ROUTE="$2"; shift 2 ;;
        --probe) PROBE_ONLY=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

log() { printf '\n== %s ==\n' "$*"; }
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"

reachable() {
    local code
    code="$(curl -sS -o /dev/null -w '%{http_code}' ${CA:+--cacert "$CA"} \
            --max-time 10 "https://$1/" 2>/dev/null || echo 000)"
    [ "$code" = "200" ] || [ "$code" = "301" ] || [ "$code" = "302" ]
}

log "Probing package sources"
INTEL_OK=0; CONDA_OK=0
reachable apt.repos.intel.com && INTEL_OK=1
reachable repo.anaconda.com   && CONDA_OK=1
printf '  %-26s %s\n' "apt.repos.intel.com" "$([ $INTEL_OK = 1 ] && echo reachable || echo BLOCKED)"
printf '  %-26s %s\n' "repo.anaconda.com"   "$([ $CONDA_OK = 1 ] && echo reachable || echo BLOCKED)"
[ "$PROBE_ONLY" = 1 ] && exit 0

# ---------------------------------------------------------------- already done?
if [ "$ROUTE" != "conda" ] && [ -x "$ONEAPI_ROOT/setvars.sh" -o -d "$ONEAPI_ROOT/compiler" ]; then
    log "oneAPI already present at $ONEAPI_ROOT -- nothing to do"
    echo "  source $ONEAPI_ROOT/setvars.sh"
    exit 0
fi
if [ "$ROUTE" != "apt" ] && [ -x "$PREFIX/envs/$ENV_NAME/bin/icpx" ]; then
    log "icpx already present at $PREFIX/envs/$ENV_NAME -- nothing to do"
    "$PREFIX/envs/$ENV_NAME/bin/icpx" --version | head -1
    exit 0
fi

# ------------------------------------------------------------------ pick a route
if [ "$ROUTE" = "auto" ]; then
    if [ $CONDA_OK = 1 ]; then ROUTE=conda        # relocatable binaries: safer default
    elif [ $INTEL_OK = 1 ]; then ROUTE=apt
    else
        echo "ERROR: neither apt.repos.intel.com nor repo.anaconda.com is reachable." >&2
        echo "No install route available; check the environment's network policy." >&2
        exit 1
    fi
fi
log "Route: $ROUTE"

# ------------------------------------------------------------------------- apt
if [ "$ROUTE" = "apt" ]; then
    [ $INTEL_OK = 1 ] || { echo "apt.repos.intel.com unreachable" >&2; exit 1; }
    export DEBIAN_FRONTEND=noninteractive
    log "Adding Intel oneAPI apt repository"
    curl -fsSL ${CA:+--cacert "$CA"} \
        https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | $SUDO tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | $SUDO tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
    $SUDO apt-get update -y
    log "Installing intel-oneapi-compiler-dpcpp-cpp (several GB)"
    $SUDO apt-get install -y --no-install-recommends intel-oneapi-compiler-dpcpp-cpp
    log "Verifying"
    # setvars must be sourced; do it in a subshell to check
    bash -lc "source $ONEAPI_ROOT/setvars.sh >/dev/null 2>&1; icpx --version | head -1; sycl-ls || true"
    cat <<EOF

Toolchain ready (apt / oneAPI). Every shell that BUILDS OR RUNS needs:
  source $ONEAPI_ROOT/setvars.sh
  icpx -fsycl -O2 -std=c++17 prog.cpp -o prog && ./prog

Note: binaries from this route are NOT relocatable -- they need the sourced
environment for OpenCL ICD registration, not just for library paths. If your
tests run under ctest/CI without a sourced shell, use --route conda instead.
EOF
    exit 0
fi

# ----------------------------------------------------------------------- conda
[ $CONDA_OK = 1 ] || { echo "repo.anaconda.com unreachable" >&2; exit 1; }
if [ ! -x "$PREFIX/bin/conda" ]; then
    log "Bootstrapping the Miniconda client into $PREFIX"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    curl -fsSL ${CA:+--cacert "$CA"} --max-time 300 -o "$tmp/miniconda.sh" "$MINICONDA_URL"
    # A partial prefix from an interrupted run leaves no conda binary; start clean.
    rm -rf "$PREFIX"
    bash "$tmp/miniconda.sh" -b -p "$PREFIX"
fi
export PATH="$PREFIX/bin:$PATH"

log "Configuring conda-forge (correctly licensed; never use defaults)"
conda config --system --remove channels defaults 2>/dev/null || true
conda config --system --add channels conda-forge
conda config --system --set channel_priority strict
[ -n "$CA" ] && conda config --system --set ssl_verify "$CA" || true

log "Creating env '$ENV_NAME': dpcpp_linux-64 + llvm-openmp"
# dpcpp_linux-64 pulls intel-opencl-rt, which supplies a runnable CPU device.
# llvm-openmp supplies omp.h for -qopenmp; without it OpenMP builds fail.
conda create -y -n "$ENV_NAME" --override-channels -c conda-forge \
    dpcpp_linux-64 llvm-openmp

log "Verifying"
"$PREFIX/envs/$ENV_NAME/bin/icpx" --version | head -1
"$PREFIX/envs/$ENV_NAME/bin/sycl-ls" || true

cat <<EOF

Toolchain ready (conda-forge). To use it:
  source $PREFIX/etc/profile.d/conda.sh && conda activate $ENV_NAME
  icpx -fsycl -O2 -std=c++17 -Wl,-rpath,"\$CONDA_PREFIX/lib" prog.cpp -o prog
  ./prog          # relocatable: runs with no activation, e.g. under ctest

The -Wl,-rpath is what makes the binary standalone. Without it you get
"libsycl.so.9: cannot open shared object file" outside an activated shell.
EOF
