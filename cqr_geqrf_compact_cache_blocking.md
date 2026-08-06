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

The `larfb` trailing update is a 2-D (4x4) register-tiled GEMM via the WY V1/V2
split: the `jb x jb` unit-lower corner is a small direct loop, the `(mm-jb) x jb`
bulk goes through `larfb_tn_acc` / `larfb_nn_sub`, which hold a 4x4 output tile in
registers so each load feeds 16 FMAs and the trailing matrix is streamed once per
4 columns.

| n   | unblocked | blocked NB=8 | blocked NB=16 | best / unblk |
|----:|----------:|-------------:|--------------:|-------------:|
|  64 |      13.0 |     **20.3** |          16.1 |       1.57x  |
|  96 |      13.9 |     **24.0** |          18.8 |       1.72x  |
| 128 |      12.0 |     **21.8** |          19.1 |       1.81x  |
| 160 |      10.3 |     **23.5** |          20.4 |       2.28x  |
| 192 |       8.7 |     **22.4** |          20.7 |       2.56x  |
| 256 |       7.7 |     **22.1** |          21.3 |       2.87x  |
| 288 |       7.8 |     **23.1** |          21.9 |       2.95x  |
| 384 |       7.2 |         18.9 |      **20.2** |       2.80x  |

**The roll-off knee is gone and throughput roughly triples at large n.** The
unblocked kernel decays `13.9 -> 7.2` from n=96 to 384; the blocked kernel peaks
**24.0** (n=96) and holds **~20-24 flat** all the way out -- **2.8-2.95x**
unblocked for `n >= 256`, and a clear win at every size. `NB=8` leads through the
mid range, `NB=16` past n~350. All factors match LAPACK to `<= 4e-15` (well under
`20 n eps`); the portable `geqrf_blocked_prototype` CTest gates it.

### What remains

At **~22 GFLOP/s** the blocked kernel is now ~1.6x MKL's own compact GEMM and ~35%
of a regular `dgemm` (~62). The remaining gap to `dgemm` is the format's -- see
below -- but it is much smaller than the earlier un-tiled `larfb` (~11) suggested,
and enough that the compact QR now beats per-matrix LAPACK across essentially the
whole target range, not just below n~128.

## What the compact-format ceiling actually is (mkl_dgemm_compact vs 2-D tiling)

Two natural moves to lift `larfb`: route its update through MKL's batched compact
GEMM, or suspect MKL's compact GEMM is under-optimized and beat it. Both were
tested (`examples/bench_gemm_compact.cpp`, a hand 4x4 register-blocked GEMM, and
the 2-D tiled `larfb` above). Square compact GEMM vs a regular one, single-thread:

| GEMM (m=n=k=N), GFLOP/s        | N=32 | N=64 | N=128 | N=192 | N=256 |
|--------------------------------|-----:|-----:|------:|------:|------:|
| regular `cblas_dgemm` (1 mat)  | 50.2 | 58.3 |  58.2 |  61.5 |  63.5 |
| `mkl_dgemm_compact`            | 22.1 | 17.1 |  14.3 |  12.5 |  13.2 |
| hand 4x4 register-blocked GEMM | 26.5 | 20.1 |  15.4 |  12.5 |  12.5 |
| + A-panel packing              | 26.4 | 19.9 |  18.0 |  14.7 |  15.7 |

**MKL's compact GEMM is under-optimized** -- the point you get from suspecting it.
A hand 4x4 register-blocked kernel already beats it 10-25%, and A-panel packing
(making the strided `k`-reduction contiguous) lifts the large-N square GEMM to
~15-18 (+20% over MKL). More to the point, the **2-D tiled `larfb` sustains ~22
GFLOP/s** -- ~1.6x MKL's compact GEMM -- because its reductions run down the
already-contiguous row axis (packing unneeded) and a 4x4 register tile gives 16
FMAs per load. So the earlier "~14 is the ceiling" reading was a *tiling* deficit,
not a format limit; the format's real reach here is ~22.

That said, the format's reach is still **well below a regular GEMM** (~62): even
2-D tiled, one interleaved group is `V x` a single matrix (spills L2 far sooner --
the QR-knee footprint effect) and cross-lane the natural access is strided and
unpacked, so past L2 the `V x` operands stream from L3. ~22 is ~35% of a regular
GEMM; MKL's ~14 was ~22%. A better compact GEMM narrows the gap but does not
close it.

And the format only pays off for genuinely tiny matrices. Batched compact GEMM vs
a per-matrix `cblas_dgemm` loop over the same batch (aggregate GFLOP/s):

| n            | 4    | 8    | 12   | 16   | 32   | 128  |
|--------------|-----:|-----:|-----:|-----:|-----:|-----:|
| compact/loop | 12.6x| 3.0x | 1.5x | 0.86x| 0.59x| 0.34x|

Compact GEMM beats the per-matrix loop only for `n <~ 14`; above that a
register-blocked per-matrix GEMM wins outright. (QR's crossover with per-matrix
LAPACK is far larger, `n ~ 110-130`, because `dgeqrf` carries much more per-call
overhead than a single `dgemm`, so the batch amortizes more -- but the direction
is the same.)

**Consequences.**

* Routing `larfb` through `mkl_dgemm_compact` would now make it *slower*, not
  faster: MKL's compact GEMM (~14 square, 6-8 at the thin `k = NB` shapes the
  update actually has) is below the 2-D tiled `larfb` (~22). It would also add an
  MKL *compute* dependency against the project's MKL-compute-free premise and a
  copy -- the trailing block is a sub-view of the packed matrix and compact GEMM
  derives its group stride as `ld*ncols*V`, so a full-stride sub-block is not
  addressable in place. The portable 2-D tiled kernel is the better answer.
* A-panel packing is worth keeping in mind for a *general* compact GEMM (it lifts
  the square case ~20% over MKL), but `larfb` does not need it -- its reductions
  are already contiguous, so 2-D register tiling is the whole win there.
* Matching a regular `dgemm` (~62) at large `n` is still **not achievable inside
  the compact format** (the `V x` footprint / cross-lane access bounds it to
  ~22 here); the only route to `dgemm` speed is to leave the format -- unpack and
  use a per-matrix register-blocked GEMM / `?geqrf`. But at ~22 the compact QR now
  beats per-matrix LAPACK across the target range, so that hybrid is only for a
  batch dominated by genuinely large `n`.

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
* If sizes `N ≳ 128` matter, use the blocked WY path (compact `larft`/`larfb`
  with a 2-D register-tiled trailing update, now prototyped -- see above). It
  removes the roll-off and is **2.8-2.95x** unblocked at large `n`, sustaining
  ~22 GFLOP/s (~1.6x MKL's compact GEMM), enough to stay ahead of per-matrix
  LAPACK across the target range. Only a batch dominated by large `n` (where a
  regular `dgemm` at ~62 wins) would want the unpack-to-dense hybrid. The same
  Level-2 -> Level-3 argument generalizes to `potrf` (`syrk`/`trsm`) once its
  group leaves L2.

## Suggested next steps

1. **Done (this note):** compact `larft`/`larfb` (2-D register-tiled trailing
   update) + blocked driver in `cqr_geqrf_compact.hpp`, benchmarked by
   `examples/bench_geqrf_blocked.cpp`. The knee disappears and the blocked kernel
   is 2.8-2.95x unblocked at large `n`, sustaining ~22 GFLOP/s.
2. **Measured, not pursued:** `mkl_dgemm_compact` inside `larfb` (~14, now *below*
   the tiled `larfb`) and A-panel packing (helps a general square GEMM ~20% but
   not `larfb`, whose reductions are already contiguous). See the section above.
3. **Optional large-`n` fast path:** above the size where a regular `dgemm` (~62)
   decisively wins, unpack and use per-matrix register-blocked `?geqrf`. A hybrid
   outside the pure-compact design; only for batches dominated by large `n` (the
   tiled compact QR already beats per-matrix LAPACK across the target range).
4. **Productionize:** call the blocked path from `cqr_mkl_?geqrf_compact` above a
   size threshold (`n` past the L2 knee, `~2*NB`), unblocked below. This changes
   the workspace contract -- the blocked path needs `NB*NB + 2*NB*NC` packs of
   scratch, so the `lwork = -1` query must report it instead of `1`. Auto-tune
   `NB`/`NC` from the runtime cache sizes (`NB ~ 8-16` here).
5. **Generalize:** the same Level-2 -> Level-3 argument applies to `potrf`
   (`syrk`/`trsm`) once its group leaves L2.
6. Repeat the `--simdlen` knee-shift sweep on a 256 KiB-L2 machine to confirm the
   crossover moves to `n ~ 64` as the model predicts (portability of the diagnosis).
