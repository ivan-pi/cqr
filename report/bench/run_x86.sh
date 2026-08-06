#!/usr/bin/env bash
# Collect the x86 (Intel MKL Compact) benchmark data for the report.
#
# Builds the SEPARATE report/bench CMake project, runs bench_geqrf_collect with
# statistical repetitions, saves the raw JSON and a provenance record, and
# converts the JSON into report/data/geqrf_x86.dat.
#
# Knobs (environment variables):
#   BUILD_DIR   cmake build dir            (default: <this dir>/build)
#   BLA_VENDOR  MKL BLAS selection         (default: Intel10_64lp_seq)
#   CXXFLAGS_   host tuning flags          (default: -O3 -march=native)
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
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${BUILD_DIR:-$here/build}"
reps="${REPS:-15}"
export CQR_NMAT="${CQR_NMAT:-512}"

stamp="$(date +%Y%m%d-%H%M%S)"
outdir="$here/results"
mkdir -p "$outdir"
json="$outdir/geqrf_x86_$stamp.json"
prov="$outdir/geqrf_x86_$stamp.provenance.txt"

# --- build the separate report/bench project ------------------------------
echo ">> configuring/building the report/bench project in $build"
cmake -S "$here" -B "$build" -DCMAKE_BUILD_TYPE=Release \
      -DBLA_VENDOR="${BLA_VENDOR:-Intel10_64lp_seq}" \
      -DCMAKE_CXX_FLAGS="${CXXFLAGS_:--O3 -march=native}"
cmake --build "$build" -j
bin="$build/bench_geqrf_collect"

# --- provenance ------------------------------------------------------------
{
  echo "date:        $(date -u +%FT%TZ)"
  echo "host:        $(hostname)"
  echo "nmat:        $CQR_NMAT"
  echo "reps:        $reps"
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-<unset>}"
  echo "OMP_PROC_BIND:   ${OMP_PROC_BIND:-<unset>}   OMP_PLACES: ${OMP_PLACES:-<unset>}"
  echo "compiler:    $(${CXX:-c++} --version | head -1)"
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
echo ">> rebuild the figures with:  ( cd $here/.. && make figures )"
