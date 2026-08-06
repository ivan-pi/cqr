# Investigation: the N ≈ 120 performance wall in `cqr_mkl_dgeqrf_compact`

> Assisted-by: Claude:claude-opus-4.8

## Question

For every routine in this library, throughput falls below standard non-batched
LAPACK once the matrix order reaches ~120 -- "roughly the L1/L2 cache limit."
Is the remedy cache-blocking, and does that mean switching to a blocked
(Level-3) algorithm? This note investigates the harder of the compact kernels,
`?geqrf_compact`.

**Short answer.** Yes -- but be precise about *what* is blocking. The wall is
the point where **one interleaved group of `V` matrices stops fitting in L2**.
The compact kernel runs the *unblocked* `geqr2`, which is a Level-2
(bandwidth-bound) computation; once the group spills L2 it goes memory-bound,
while reference `dgeqrf` stays fast because it is already blocked (Level-3
`larfb`/GEMM). The cure is the blocked WY algorithm (a compact `larft`/`larfb`);
a free stopgap is to narrow the interleave width for large `N`.

## Measurements

Machine: Intel Xeon (AVX-512, VNNI), 1 socket / 4 cores, **L1d 32 KiB, L2 1 MiB
per core, L3 33 MiB shared.** GCC 13.3, `-O3 -march=native`, single-threaded
(`OMP_NUM_THREADS=1`), MKL LP64 sequential. Reproduce with
`examples/bench_geqrf_compact`.

### 1. Where the crossover is (`bench_geqrf_compact 512 5`, square, V=8/AVX-512)

| n   | cqr GFLOP/s | cqr/lapack | cqr/mkl |
|----:|------------:|-----------:|--------:|
|  96 |       14.07 |     1.76x  |  1.34x  |
| 105 |       14.16 |     1.73x  |  1.36x  |
| 128 |       12.09 |   **1.08x**|  1.32x  |
| 168 |        9.80 |   **0.60x**|  1.34x  |
| 256 |        7.79 |     0.39x  |  1.36x  |
| 500 |        7.28 |     0.25x  |  1.62x  |

cqr crosses below per-matrix LAPACK at **n ≈ 130**, matching the report. Two
things to note:

* cqr GFLOP/s holds a ~14 plateau up to n≈112, then rolls off from n≈120-128 --
  it does not fall off a cliff, it degrades as the working set leaves cache.
* **cqr stays ~1.3x ahead of `mkl_dgeqrf_compact` across the whole range.** MKL's
  own compact kernel hits the *same* wall (mkl/lapack is already 0.82x at n=128).
  So this is not a cqr defect -- it is intrinsic to the compact-*unblocked* class,
  and MKL did not solve it either. Beating LAPACK above the knee requires
  blocking, which neither compact kernel does.

### 2. The cause: interleaving inflates the cache footprint by V×

Compact format interleaves `V` matrices, so the working set of the group the
kernel is actively factoring is **V times a single matrix**:

```
footprint(N, V) = N^2 * V * 8 bytes         (FP64)
```

Setting this equal to L2 (1 MiB) predicts the roll-off knee:

```
N_L2 = sqrt(L2 / (8*V)) = sqrt(131072 / V)
```

| V | predicted L2 knee | predicted L1 knee |
|--:|------------------:|------------------:|
| 8 |          **128**  |        23         |
| 4 |          **181**  |        32         |
| 2 |          **256**  |        45         |

A *single* matrix (the per-matrix LAPACK path) is 8× smaller, so it does not
reach the 1 MiB L2 until N ≈ 362 -- and reference `dgeqrf` is blocked anyway.
That is why LAPACK does not hit this wall at 120 and the compact kernel does.

The knee tracks **L2, not L1**: the `JB=4` trailing-column blocking already keeps
the innermost sweep L1-friendly, so what spills is the reuse of the *whole group*
across the outer column loop.

### 3. Decisive test: force the width, watch the knee move

If the wall is the per-group L2 footprint, narrowing `V` must slide the roll-off
knee right, tracking `1/sqrt(V)`. It does
(`bench_geqrf_compact --size-sweep=48:288:8 --simdlen=V 256 4`):

| V | predicted L2 knee | measured: plateau holds to → roll-off |
|--:|------------------:|---------------------------------------|
| 8 |            128    | flat to 112, knee **120-128**, → 7.7 @288 |
| 4 |            181    | flat to 184, knee **~200**,     → 8.8 @288 |
| 2 |            256    | flat to 248, knee **~256**,     → 9.5 @288 |

The knee moves exactly as predicted. A direct corollary: **above the V=8 knee,
narrower interleave is faster in absolute GFLOP/s**, because cache residency
outweighs SIMD width:

| n   | V=8   | V=4        | V=2        |
|----:|------:|-----------:|-----------:|
| 192 | 8.64  | **12.05**  | 10.06      |
| 256 | 7.69  | **8.40**   | 7.94       |
| 288 | 7.71  | 8.83       | **9.51**   |

(The dip every implementation shows at n=256 is the classic power-of-two leading
-dimension conflict-miss effect -- `ldap = N = 256` maps columns onto the same
cache sets -- and mildly reinforces that this regime is cache-governed.)

### 4. Roofline / arithmetic intensity

Unblocked `geqr2` applies each reflector as a separate rank-1 update (Level-2).
In the fully-uncached limit its arithmetic intensity is

```
AI_unblocked = (4/3)N^3 flops / ((N^3/3)*8 bytes) = 0.5 flop/byte
```

-- squarely memory-bound. The ~14 GFLOP/s plateau is only ~16% of the ~90
GFLOP/s single-core FP64 peak (that ceiling is the 2-load/1-store rank-1 pattern
while cache-resident); above the knee it falls to ~8% as the group streams from
L3/DRAM. A blocked update raises AI by ~`NB`:

```
AI_blocked ~ 0.5 * NB flop/byte   (NB=32 -> ~16 flop/byte, compute-bound)
```

## Remedy

### The cure: blocked WY (compact `larft` + `larfb`)

This is the textbook Level-2 → Level-3 transition, and it is exactly what
per-matrix `dgeqrf` does to pull ahead above 128:

1. Factor a panel of `NB` columns with the existing unblocked `geqr2` (touches
   only the `NB`-wide panel -- cache-resident regardless of `N`).
2. Accumulate the block reflector `T` (`larft`), so `H(kk)...H(kk+NB-1) = I - V T V^T`.
3. Apply the whole block to the trailing matrix **in one sweep** (`larfb`),
   instead of `NB` separate rank-1 sweeps.

This cuts the number of streaming passes over the out-of-cache trailing matrix
from `O(N)` to `O(N/NB)`, which is what turns the kernel back into a
compute-bound one (AI ~ `0.5*NB`).

Notes specific to the compact/interleaved format:

* **You cannot merely tile the current unblocked kernel.** `H(kk)` must be
  applied to *all* trailing columns before column `kk+1`'s reflector is built
  (data dependency), so column tiling alone cannot defer work. The block
  reflector `T` is precisely the mechanism that lifts that dependency -- that is
  why blocking is a genuine algorithm change, not a loop reorder.
* **The `larfb` "GEMM" is `V` lane-wise small GEMMs**, reusing the existing
  `pack<T,V>` machinery -- "the same math, blocked." `T` is `NB×NB` per lane.
* **Size the block with the V-inflation in mind.** The whole point is cache
  residency, so choose `NB` and the trailing tile so `(panel + active tile) * V`
  fits in L2/L1 -- the tiling that a per-matrix BLAS does internally must here
  account for the extra `V` factor.

### Free stopgap: narrow the interleave for large N

Section 3 shows V=4 beats V=8 by up to ~39% above the knee, and V=2 wins at 288.
A width heuristic (drop to V=4 once `N^2*V*8 > L2`, etc.) recovers a good part of
the tail with **no new algorithm** -- the dispatcher already supports every
width. It is a palliative, not a cure: it moves the knee but keeps the unblocked
`O(N)` passes, so it still cannot beat blocked LAPACK (at n=256 the best compact
width is ~3.8e2 mat/s vs LAPACK's 8.9e2).

## Scope recommendation

Below n ≈ 110 the current unblocked kernel is **1.7-10× faster than per-matrix
LAPACK** and is the right design -- everything is cache-resident and blocking
would only add `larft`/`larfb` overhead. The wall bites only at n ≳ 128, and its
exact location is L2-specific (≈64 on a 256 KiB L2, ≈90 on 512 KiB, ≈128 here).

So the decision is workload-driven:

* If the target sizes are genuinely small (the RBF-FD stencils cited in the
  design docs: 30, 45, 60, 105, 168), the current kernel already covers most of
  the range; document the >L2 regime as a known boundary and optionally add the
  narrow-V stopgap for the 168+ tail.
* If sizes `N ≳ 128` matter, implement the blocked WY path (compact
  `larft`/`larfb`). This is the only change that beats blocked LAPACK above the
  knee, and it generalizes: the same Level-2→Level-3 argument applies to
  `potrf` (`syrk`/`trsm`) once its group leaves L2.

## Suggested next steps

1. Prototype a compact `larfb` (block reflector apply) and a `larft`; wire a
   blocked driver that falls back to the unblocked kernel for `N ≤ ~2*NB`.
   Sweep `NB ∈ {8,16,32}` and confirm the knee disappears (GFLOP/s should stop
   rolling off and approach the compact-GEMM ceiling).
2. As a cheap first win, add the width-selection heuristic and re-run the sweep.
3. Repeat the `--simdlen` knee-shift sweep on a 256 KiB-L2 machine to confirm the
   crossover moves to N≈64 as the model predicts (portability of the diagnosis).
