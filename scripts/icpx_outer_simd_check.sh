#!/usr/bin/env bash
#
# icpx_outer_simd_check.sh
#
# Does Intel's oneAPI compiler (icpx) vectorize the *outer-loop* `#pragma omp
# simd` form of the compact geqrf that GCC and Clang refuse? (GCC: "loop nest
# containing two or more consecutive inner loops"; Clang: failed transform.)
#
# Answer (see cqr_geqrf_omp_simd_results.md): it depends on the lowering path.
#   -fopenmp-simd  -> the LLVM auto-vectorizer (icpx is LLVM-based) -> NOT
#                     vectorized, same as Clang.
#   -qopenmp-simd  -> Intel's own OpenMP SIMD code generator -> the outer lane
#                     loop DOES vectorize, ~0.9x of the hand-written vector types.
#
# This script demonstrates both: it prints the scalar-vs-packed instruction
# counts for each flag on the outer-loop kernel (the smoking gun), then builds
# the benchmark through CMake (which auto-selects -qopenmp-simd for icpx) and
# runs the three-way vec-types / omp-outer / omp-inner comparison.
#
# Run from the repo root:  ./scripts/icpx_outer_simd_check.sh
# No MKL needed (the omp-simd kernels and the benchmark are portable).
#
# Assisted-by: Claude:claude-opus-4.8

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 1. Locate icpx, or install the oneAPI DPC++/C++ compiler via apt.
#    The compiler-only package is a few GB; the full toolkit metapackage is
#    intel-basekit if you want MKL etc. too. If apt reaches Intel's CDN through
#    an HTTP(S) proxy, point apt at it first:
#      echo 'Acquire::https::Proxy "http://HOST:PORT";' | sudo tee /etc/apt/apt.conf.d/00proxy
# ---------------------------------------------------------------------------
if ! command -v icpx >/dev/null 2>&1 && [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null
fi
if ! command -v icpx >/dev/null 2>&1; then
    echo ">> icpx not found; installing the oneAPI DPC++/C++ compiler via apt..."
    KEY=/usr/share/keyrings/oneapi-archive-keyring.gpg
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | sudo tee "$KEY" >/dev/null
    echo "deb [signed-by=$KEY] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null
fi
command -v icpx >/dev/null 2>&1 || { echo "!! icpx still not on PATH -- aborting"; exit 1; }
echo ">> using: $(icpx --version | head -1)"
echo

# ---------------------------------------------------------------------------
# 2. The smoking gun: scalar vs packed instruction counts for the OUTER-loop
#    kernel under each lowering path. -xHost tunes for the build host.
# ---------------------------------------------------------------------------
count() { # $1 = flag
    icpx -std=c++17 -O3 -xHost "$1" -Isrc -S -o /tmp/icpx_$1.s \
        src/cqr_geqrf_compact_dispatch_omp.cpp 2>/dev/null
    local sd pd
    sd=$(grep -cE '(mulsd|addsd|subsd|vf[n]*madd[0-9]*sd|sqrtsd)' "/tmp/icpx_$1.s" || true)
    pd=$(grep -cE '(mulpd|addpd|subpd|vf[n]*madd[0-9]*pd|sqrtpd).*[xyz]mm' "/tmp/icpx_$1.s" || true)
    printf "  %-16s scalar *sd = %-4s   packed *pd = %-4s   -> %s\n" \
        "$1" "$sd" "$pd" "$([ "$sd" -gt "$pd" ] && echo 'SCALAR (not vectorized)' || echo 'VECTORIZED')"
}
echo "== outer-loop kernel (geqrf_compact_group_omp), icpx codegen =="
count -fopenmp-simd   # LLVM auto-vectorizer path
count -qopenmp-simd   # Intel OpenMP SIMD codegen path
echo "   (-qopenmp-simd is simd-only: it links no OpenMP runtime.)"
echo

# ---------------------------------------------------------------------------
# 3. Build via CMake (auto-selects -qopenmp-simd for IntelLLVM) and run.
# ---------------------------------------------------------------------------
echo "== building with icpx via CMake (-qopenmp-simd) =="
CC=icx CXX=icpx cmake -S . -B build-icpx -DCQR_WITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -xHost" >/dev/null
cmake --build build-icpx --target bench_geqrf_omp_simd test_cqr_geqrf_compact_omp -j

echo
echo "== drop-in correctness (omp backend, icpx) =="
./build-icpx/test_cqr_geqrf_compact_omp | tail -2
echo
echo "== throughput: vec-types vs omp-outer vs omp-inner (icpx -qopenmp-simd) =="
echo "   omp-outer should now track vec-types (~0.9x), not collapse to ~0.1x."
./build-icpx/bench_geqrf_omp_simd 512 7
