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

## Prototype: the blocked WY path, implemented and measured

The blocked path is implemented in `cqr_geqrf_compact.hpp` and benchmarked by
`examples/bench_geqrf_blocked.cpp` (portable, no MKL). Three internal functions,
each running V matrices at a time:

* `larft_forward_compact` -- builds the `jb x jb` block-reflector factor `T`,
* `larfb_forward_left_compact` -- applies `C := (I - V T^T V^T) C` to the trailing
  block, trailing-column-tiled by `NC` and register-blocked 4 columns at a time
  (so each reflector load feeds 4 FMAs -- the same reuse the unblocked kernel's
  `JB=4` trailing update already exploits),
* `geqrf_blocked_compact` -- the driver: factor each `NB`-column panel with the
  existing unblocked `geqr2`, then one Level-3 `larft`+`larfb` on the trailing
  columns.

### Correctness

The blocked factor reproduces the unblocked one to rounding -- norm-scaled
elementwise difference `<= 3.9e-16` across `n`, and **bit-identical** (`0.0`) for
`n <= NB` (a single panel, no `larfb`). Checked directly against per-matrix
`LAPACKE_dgeqrf` as well (elementwise, norm-scaled): `relerr` `2e-16 .. 3e-15`,
under the `20 n eps` contract the MKL suite uses, including a padded batch
(`nm` not a multiple of `V`) and off-width tile sizes. The CTest
`geqrf_blocked_prototype` gates blocked-vs-unblocked on every run.

### Throughput (V=8 AVX-512, 1 MiB L2, single-thread, nmat=128; GFLOP/s)

| n   | unblocked | blocked NB=8 | blocked NB=16 | best / unblk |
|----:|----------:|-------------:|--------------:|-------------:|
|  80 |      14.9 |     **19.0** |          17.6 |       1.28x  |
| 112 |      14.1 |     **17.0** |          16.7 |       1.20x  |
| 128 |      11.7 |     **15.2** |          15.0 |       1.30x  |
| 160 |      10.3 |         13.7 |          13.4 |       1.33x  |
| 192 |       8.8 |         12.1 |      **12.3** |       1.40x  |
| 256 |       7.8 |         10.9 |      **11.0** |       1.41x  |
| 384 |       7.3 |         10.0 |      **10.3** |       1.42x  |

**The roll-off knee is gone.** The unblocked kernel decays `14.9 -> 7.3` from
n=80 to 384; the blocked kernel *peaks higher* (19.0 at n=80, vs the unblocked
~14 ceiling) and holds `>= 10` GFLOP/s out to n=384 -- a **+30-42%** win over
unblocked for every `n >= 128`, and a tie-or-better even below the knee. `NB=8`
is best in the small/mid range, `NB=16` at `n >= 192`; `NB=32` trails (a wider
panel spends more in the Level-2 panel factor). This confirms the diagnosis end
to end: the wall was the algorithm's Level-2 memory traffic, and converting the
trailing update to Level-3 removes it.

### What remains

The blocked tail plateaus at ~10-11 GFLOP/s, below the ~19 mid-range peak,
because `larfb` is register-blocked but not yet a fully tiled GEMM microkernel --
its `V^T C` / `V W` passes still re-stream the trailing tile ~`jb` times rather
than once. Closing that (a compact GEMM microkernel: accumulate a small `jb x nr`
register tile, stream `V` and `C` once) is the lever to approach the compact-GEMM
ceiling and match optimized per-matrix LAPACK at large `n` -- where MKL's tuned
`dgemm`-backed `dgeqrf` still leads, since at large `n` a single matrix already
vectorizes well and the compact batch's edge is at *small* `n`. The blocking here
pushes the compact kernel's break-even with per-matrix LAPACK out from `n ~ 128`
to roughly `n ~ 150-160` and turns the large-`n` collapse (0.25x LAPACK at
n=500) into a graceful, flat ~0.4-0.5x.

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
* If sizes `N ≳ 128` matter, use the blocked WY path (compact `larft`/`larfb`,
  now prototyped -- see the section above). It removes the compact kernel's
  roll-off and is 30-42% faster than unblocked past the knee; a tiled GEMM
  microkernel in `larfb` is the further step to match per-matrix LAPACK at large
  `n`. The same Level-2→Level-3 argument generalizes to `potrf`
  (`syrk`/`trsm`) once its group leaves L2.

## Suggested next steps

1. **Done (this note):** compact `larft`/`larfb` + blocked driver in
   `cqr_geqrf_compact.hpp`, benchmarked by `examples/bench_geqrf_blocked.cpp`.
   The knee disappears and blocked beats unblocked by 30-42% for `n >= 128`.
2. **Register-tiled compact GEMM microkernel** inside `larfb` (accumulate a
   `jb x nr` register tile, stream `V` and `C` once) to lift the ~10-11 GFLOP/s
   tail toward the mid-range ~19 peak and match per-matrix LAPACK at large `n`.
3. **Productionize:** call the blocked path from `cqr_mkl_?geqrf_compact` above a
   size threshold (`n` past the L2 knee, `~2*NB`), unblocked below. This changes
   the workspace contract -- the blocked path needs `NB*NB + 2*NB*NC` packs of
   scratch, so the `lwork = -1` query must report it instead of `1`. Auto-tune
   `NB`/`NC` from the runtime cache sizes (`NB ~ 8-16` here).
4. **Generalize:** the same Level-2 -> Level-3 argument applies to `potrf`
   (`syrk`/`trsm`) once its group leaves L2.
5. Repeat the `--simdlen` knee-shift sweep on a 256 KiB-L2 machine to confirm the
   crossover moves to `n ~ 64` as the model predicts (portability of the diagnosis).
