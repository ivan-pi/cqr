#!/usr/bin/env bash
#
# icpx_outer_simd_check.sh
#
# Does Intel's oneAPI compiler (icpx) vectorize the *outer-loop* `#pragma omp
# simd` form of the compact geqrf that GCC and Clang refuse? (GCC: "loop nest
# containing two or more consecutive inner loops"; Clang: failed transform. See
# cqr_geqrf_omp_simd_results.md.) icpx has Intel's own vectorizer and is the most
# likely of the three to do outer-loop vectorization, so this script builds the
# omp-simd geqrf with icpx, extracts whether the outer lane loop vectorized, and
# runs the vec-types vs omp-outer vs omp-inner benchmark.
#
# Run from the repo root:  ./scripts/icpx_outer_simd_check.sh
#
# It does NOT need MKL: the benchmark and the omp-simd kernels are portable.
#
# Assisted-by: Claude:claude-opus-4.8

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# 1. Locate icpx, or install the oneAPI DPC++/C++ compiler via apt.
#    The compiler-only package (intel-oneapi-compiler-dpcpp-cpp) is a few GB;
#    the full toolkit metapackage is intel-basekit if you want MKL etc. too.
# ---------------------------------------------------------------------------
if ! command -v icpx >/dev/null 2>&1; then
    # Pick up an already-installed oneAPI that just is not on PATH.
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        # shellcheck disable=SC1091
        source /opt/intel/oneapi/setvars.sh >/dev/null
    fi
fi

if ! command -v icpx >/dev/null 2>&1; then
    echo ">> icpx not found; installing the oneAPI DPC++/C++ compiler via apt..."
    KEY=/usr/share/keyrings/oneapi-archive-keyring.gpg
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | sudo tee "$KEY" >/dev/null
    echo "deb [signed-by=$KEY] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
    sudo apt-get update
    # A recent (2026.x) compiler; drop the version suffix for the latest.
    sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null
fi

command -v icpx >/dev/null 2>&1 || { echo "!! icpx still not on PATH -- aborting"; exit 1; }
echo ">> using: $(icpx --version | head -1)"
echo

# ---------------------------------------------------------------------------
# 2. Compile just the outer-loop omp-simd dispatch and report whether icpx
#    vectorized the outer `#pragma omp simd` lane loop.
#    -qopt-report=3 writes an Intel vectorization report (*.optrpt);
#    -Rpass=loop-vectorize surfaces the LLVM-style remarks too.
# ---------------------------------------------------------------------------
FLAGS="-std=c++17 -O3 -xHost -fopenmp-simd -Isrc"
echo "== icpx vectorization report for the OUTER-loop kernel =="
echo "   (source: src/cqr_geqrf_compact_omp.hpp, geqrf_compact_group_omp)"
rm -f cqr_geqrf_compact_dispatch_omp.optrpt
icpx $FLAGS -qopt-report=3 -c src/cqr_geqrf_compact_dispatch_omp.cpp -o /tmp/icpx_omp.o \
    -Rpass=loop-vectorize -Rpass-missed=loop-vectorize 2>/tmp/icpx_remarks.txt || true

echo "-- LLVM-style remarks mentioning the outer kernel / simd loop --"
grep -iE 'cqr_geqrf_compact_omp|omp simd|vectoriz' /tmp/icpx_remarks.txt | head -20 \
    || echo "(none captured on stderr; see the .optrpt below)"
echo
echo "-- Intel opt-report (*.optrpt), lane-loop entries --"
OPTRPT=$(ls -1 *.optrpt 2>/dev/null | head -1 || true)
if [ -n "${OPTRPT:-}" ]; then
    grep -iE 'geqrf_compact_group_omp|SIMD|vector|remark #15|LOOP' "$OPTRPT" | head -40 || true
    echo "   (full report: $ROOT/$OPTRPT)"
else
    echo "(no .optrpt produced; the version may write to stderr -- check /tmp/icpx_remarks.txt)"
fi
echo
echo ">> Read the report: if the loop at geqrf_compact_group_omp's '#pragma omp"
echo ">> simd simdlen(V)' is reported vectorized (SIMD LOOP / remark #15300), then"
echo ">> icpx succeeds where GCC and Clang do not."
echo

# ---------------------------------------------------------------------------
# 3. Build and run the three-way benchmark with icpx.
# ---------------------------------------------------------------------------
echo "== building the benchmark with icpx =="
cmake -S . -B build-icpx -DCQR_WITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS="-O3 -xHost" >/dev/null
cmake --build build-icpx --target bench_geqrf_omp_simd test_cqr_geqrf_compact_omp -j

echo
echo "== drop-in correctness (omp backend, icpx) =="
./build-icpx/test_cqr_geqrf_compact_omp | tail -2
echo
echo "== throughput: vec-types vs omp-outer vs omp-inner (icpx, native width) =="
./build-icpx/bench_geqrf_omp_simd 512 7
