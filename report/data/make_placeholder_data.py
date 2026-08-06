#!/usr/bin/env python3
"""Generate SYNTHETIC placeholder benchmark data for the report figures.

    !!!  THESE NUMBERS ARE NOT MEASURED.  !!!

They exist only so the gnuplot -> LaTeX pipeline renders end to end while the
draft is being written. Replace the .dat files with the real output of
`bench_geqrf_compact` (x86) and the ArmPL interleave-batch harness (Arm) before
the paper is anything but a skeleton. See ./README.md for the column contract
and the one-liner that turns real benchmark stdout into these files.

The curves are shaped to be *plausible*, not authoritative:
  * throughput (matrices/s) falls ~ 1/n^3 (work per matrix is O(n^3));
  * GFLOP/s rises with n and saturates (better vector-unit utilisation);
  * a small wiggle keyed on (n mod V) mimics the SIMD-remainder "staircase";
  * cqr leads at the smallest sizes and converges to the vendor kernel as n grows.

Deterministic: no RNG, so regenerating never perturbs the committed figures.
"""

# Square sizes: the bench_geqrf_compact default list. The non-power sizes
# (30, 45, 60, 105, 168) are the 2-D/3-D RBF-FD stencil sizes; keeping them
# makes the remainder "staircase" visible and ties the axis to the applications.
SIZES = [8, 16, 24, 30, 32, 45, 48, 60, 64, 96, 105, 128, 168, 170, 256, 384, 500]

NMAT = 512          # matrices per pool (matches the bench default)
V_X86 = 8           # AVX-512 double interleave width
V_ARM = 8           # 512-bit SVE double interleave width (illustrative)


def geqrf_gflop(n):
    """Standard LAPACK ?geqrf flop count for a square n x n matrix, in GFLOP."""
    return (2.0 * n * n * n - (2.0 / 3.0) * n * n * n) * 1e-9


def staircase(n, v, depth=0.06):
    """Mild efficiency dip for sizes that are not a multiple of the width v."""
    return 1.0 - depth * ((n % v) / v)


def cqr_efficiency(n, peak):
    """Rising-and-saturating GFLOP/s model, with a gentle large-n roll-off."""
    base = peak * (n ** 1.4) / (n ** 1.4 + 120.0)
    rolloff = 1.0 - 0.10 * max(0.0, (n - 170) / 330.0)
    return base * rolloff


def write_table(path, vendor_col, peak, vendor_ratio, lapack_ratio, v):
    """Emit one benchmark table. `vendor_ratio(n)` = cqr_time / vendor_time model
    expressed as cqr_eff / vendor_eff; `lapack_ratio(n)` likewise for LAPACK."""
    lines = []
    lines.append("# SYNTHETIC PLACEHOLDER DATA -- NOT MEASURED. Regenerate real")
    lines.append("# numbers with bench_geqrf_compact; see report/data/README.md.")
    lines.append(f"# nmat={NMAT}  interleave_width_V={v}")
    lines.append("# n  cqr_gflops  cqr_mats_s  {0}_mats_s  lapack_mats_s  "
                 "sp_cqr_lap  sp_{0}_lap  sp_cqr_{0}  relerr".format(vendor_col))
    for n in SIZES:
        cqr_eff = cqr_efficiency(n, peak) * staircase(n, v)
        f = vendor_ratio(n)                           # cqr_eff / vendor_eff
        g = lapack_ratio(n)                           # cqr_eff / lapack_eff
        # matrices/s = eff / gflop_per_matrix  (the batch-size factor cancels)
        cqr_mats = cqr_eff / geqrf_gflop(n)
        vend_mats = cqr_mats * f
        lap_mats = cqr_mats * g
        lines.append(
            f"{n:4d}  {cqr_eff:10.3f}  {cqr_mats:12.4e}  {vend_mats:12.4e}  "
            f"{lap_mats:12.4e}  {1.0/g:7.3f}  {f/g:7.3f}  {1.0/f:7.3f}  {1.2e-14:.2e}"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}  ({len(SIZES)} rows)")


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))

    # --- x86 / Intel MKL Compact -------------------------------------------
    # cqr leads at small n (1.38x), converges, MKL edges ahead past ~n=400.
    write_table(
        os.path.join(here, "geqrf_x86.dat"),
        vendor_col="mkl",
        peak=50.0,
        vendor_ratio=lambda n: 0.70 + 0.40 * (n / (n + 120.0)),   # cqr/mkl = 1/this
        lapack_ratio=lambda n: 0.15 + 0.80 * (n / (n + 80.0)),    # cqr/lap = 1/this
        v=V_X86,
    )

    # --- Arm / Arm Performance Libraries interleave-batch ------------------
    # Portable cqr leads at the smallest sizes, ArmPL's SVE tuning pulls
    # slightly ahead through the mid-range, parity by the largest sizes.
    write_table(
        os.path.join(here, "geqrf_arm.dat"),
        vendor_col="armpl",
        peak=40.0,                                                # illustrative Arm core
        vendor_ratio=lambda n: 1.0 / (0.90 + 0.50 * (30.0 / (n + 30.0))),
        lapack_ratio=lambda n: 0.18 + 0.78 * (n / (n + 80.0)),
        v=V_ARM,
    )
