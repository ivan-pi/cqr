#!/usr/bin/env bash
# Collect the x86 (Intel MKL Compact) benchmark data for the report.
#
# Runs bench_geqrf_collect with statistical repetitions, saves the raw JSON and
# a provenance record, and converts the JSON into report/data/geqrf_x86.dat.
#
# Knobs (environment variables):
#   BUILD_DIR   cmake build dir            (default: <repo>/build)
#   CQR_NMAT    matrices per pool          (default: 512)
#   REPS        benchmark repetitions      (default: 15)
#   OMP_NUM_THREADS  outer-loop threads    (default: leave as set / all cores)
#
# For STABLE numbers on the final node (do these yourself; they need privilege):
#   * run on an empty, non-shared node;
#   * pin the CPU frequency / disable turbo boost, e.g.
#       sudo cpupower frequency-set -g performance
#       echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
#   * pin threads:  export OMP_PROC_BIND=close OMP_PLACES=cores
#   * build with host-tuned flags so the SIMD kernels use the full width:
#       -DCMAKE_CXX_FLAGS="-O3 -march=native"
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
build="${BUILD_DIR:-$repo/build}"
reps="${REPS:-15}"
export CQR_NMAT="${CQR_NMAT:-512}"

stamp="$(date +%Y%m%d-%H%M%S)"
outdir="$here/results"
mkdir -p "$outdir"
json="$outdir/geqrf_x86_$stamp.json"
prov="$outdir/geqrf_x86_$stamp.provenance.txt"

# --- build (only if the collector is missing) ------------------------------
bin="$build/bench_geqrf_collect"
if [[ ! -x "$bin" ]]; then
  echo ">> configuring/building bench_geqrf_collect in $build"
  cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release \
        -DCQR_BUILD_REPORT_BENCH=ON \
        -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:--O3 -march=native}"
  cmake --build "$build" -j --target bench_geqrf_collect
fi

# --- provenance ------------------------------------------------------------
{
  echo "date:        $(date -u +%FT%TZ)"
  echo "host:        $(hostname)"
  echo "nmat:        $CQR_NMAT"
  echo "reps:        $reps"
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-<unset>}"
  echo "OMP_PROC_BIND:   ${OMP_PROC_BIND:-<unset>}   OMP_PLACES: ${OMP_PLACES:-<unset>}"
  echo "--- lscpu ---";  lscpu 2>/dev/null || true
  echo "--- governor ---"
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
} > "$prov"
echo ">> provenance -> $prov"

# --- run -------------------------------------------------------------------
echo ">> running $reps repetitions (nmat=$CQR_NMAT) -> $json"
"$bin" \
  --benchmark_repetitions="$reps" \
  --benchmark_report_aggregates_only=true \
  --benchmark_format=json \
  --benchmark_out="$json" \
  --benchmark_out_format=json

# --- convert ---------------------------------------------------------------
python3 "$here/json_to_dat.py" "$json" --vendor mkl -o "$here/../data/geqrf_x86.dat"
echo ">> updated report/data/geqrf_x86.dat"
echo ">> rebuild the figures with:  ( cd $repo/report && make figures )"
