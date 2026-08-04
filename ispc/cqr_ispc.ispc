// cqr_ispc.ispc -- Compact (interleaved-batch) QR pipeline in Intel ISPC.
//
// An SPMD prototype of the cqr GNU-vector kernels (../src/cqr_geqrf_compact.hpp,
// ../src/cqr_compact.hpp) built to measure ISPC against the hand-written GNU
// vector_size approach on the very same algorithm and data layout.
//
// The idea that makes this a one-to-one match for the MKL Compact format:
//
//   * MKL Compact stores element (i,j) of the V interleaved matrices of a group
//     CONTIGUOUSLY: ap[g*ldap*ncol*V + (j*ldap + i)*V + v], v = 0..V-1.
//   * In ISPC a `varying double` is, by definition, programCount consecutive
//     doubles in memory -- one per gang lane.
//
// So a compact pack of V matrices' element (i,j) *is* a `varying double` when
// V == programCount. Casting the group's base pointer to `varying double *
// uniform` and indexing by the linear element offset (j*ldap + i) loads/stores
// that pack in a single aligned vector op, lane v holding matrix v. The kernel
// body is then the ordinary *scalar* geqr2 / dorm2r / back-substitution written
// once; ISPC vectorizes it across the V matrices of the group. Contrast the GNU
// version, which spells the vector out by hand with __attribute__((vector_size)).
//
// The gang width fixes the interleave width. Build with --target=avx512skx-x8 so
// programCount == 8 == V, the AVX-512 FP64 compact format MKL selects on an
// AVX-512 host (mkl_get_format_compact() == MKL_COMPACT_AVX512). Column-major,
// double precision -- the tuned path the batched AX=B solver uses.
//
// Exposed (extern "C" via ISPC `export`) as drop-in siblings of the portable C
// API in ../src/cqr_compact.h, plus a compact triangular solve to complete the
// pipeline (the repo leans on mkl_dtrsm_compact for that step):
//
//   cqr_ispc_dgeqrf_compact  QR factorization      A = Q R
//   cqr_ispc_dormqr_compact  apply Q or Q^T        B := op(Q) B   (side = 'L')
//   cqr_ispc_dtrsm_compact   triangular solve      B := alpha op(A)^-1 B
//                                                  (left, upper, no-trans, non-unit)
//
// Assisted-by: Claude:claude-opus-4.8

// ---------------------------------------------------------------------------
// QR factorization (unblocked geqr2): A = Q R, one group of programCount
// matrices at a time. Mirrors cqr::detail::geqrf_compact_group.
//
//   m, n   rows, columns of each matrix
//   ap     compact A (m x n), column-major; overwritten with (R, Householder v)
//   ldap   compact leading dimension (>= m)
//   taup   compact tau output, k = min(m,n) per matrix
//   nm     total number of matrices (a partial final group is padded to a full
//          pack of identities by the caller, so it runs unmasked -> tau = 0)
// ---------------------------------------------------------------------------
export void cqr_ispc_dgeqrf_compact(uniform int m, uniform int n,
                                    uniform double ap[], uniform int ldap,
                                    uniform double taup[], uniform int nm)
{
    uniform int V = programCount;
    uniform int k = (m < n) ? m : n;
    uniform int ngroups = (nm + V - 1) / V;
    uniform int64 stra = (uniform int64)ldap * n * V; // doubles per group in ap
    uniform int64 strt = (uniform int64)k * V;        // doubles per group in tau

    for (uniform int g = 0; g < ngroups; ++g) {
        // A compact pack == one `varying double`: lane v is matrix v of the group.
        varying double * uniform A = (varying double * uniform)(ap + g * stra);
        varying double * uniform tau = (varying double * uniform)(taup + g * strt);

        for (uniform int kk = 0; kk < k; ++kk) {
            uniform int ckk = kk * ldap; // element index of column kk, row 0

            // --- larfg: build reflector H(kk) from column kk, rows kk..m-1 ---
            varying double x0 = A[ckk + kk];
            varying double tail = 0.0d; // squared norm of the sub-diagonal tail
            for (uniform int i = kk + 1; i < m; ++i) {
                varying double a = A[ckk + i];
                tail += a * a;
            }
            // Branch-free, exactly as larfg_pack: the xnorm==0 lane keeps x0 and
            // gets tau = 0, and no lane ever forms 0/0 or 1/0 that survives.
            varying double norm = sqrt(x0 * x0 + tail);
            varying double beta = (x0 >= 0.0d) ? -norm : norm; // -copysign(norm, x0)
            varying bool has = (tail > 0.0d);                  // anything to zero?
            varying double t = has ? (beta - x0) / beta : 0.0d;
            varying double inv = has ? (1.0d / (x0 - beta)) : 0.0d;
            varying double rdg = has ? beta : x0;

            tau[kk] = t;
            for (uniform int i = kk + 1; i < m; ++i)
                A[ckk + i] = A[ckk + i] * inv; // reflector body v(kk)
            A[ckk + kk] = rdg;                 // R diagonal

            // --- apply H(kk) = I - t v v^T to trailing columns, 4 at a time so
            //     each reflector load A[ckk+i] is reused across 4 columns ---
            uniform int j = kk + 1;
            for (; j + 4 <= n; j += 4) {
                uniform int c0 = (j + 0) * ldap, c1 = (j + 1) * ldap;
                uniform int c2 = (j + 2) * ldap, c3 = (j + 3) * ldap;
                varying double w0 = A[c0 + kk], w1 = A[c1 + kk];
                varying double w2 = A[c2 + kk], w3 = A[c3 + kk];
                for (uniform int i = kk + 1; i < m; ++i) {
                    varying double av = A[ckk + i];
                    w0 += av * A[c0 + i];
                    w1 += av * A[c1 + i];
                    w2 += av * A[c2 + i];
                    w3 += av * A[c3 + i];
                }
                A[c0 + kk] -= t * w0;
                A[c1 + kk] -= t * w1;
                A[c2 + kk] -= t * w2;
                A[c3 + kk] -= t * w3;
                w0 *= t;
                w1 *= t;
                w2 *= t;
                w3 *= t; // fold tau into w
                for (uniform int i = kk + 1; i < m; ++i) {
                    varying double av = A[ckk + i];
                    A[c0 + i] -= av * w0;
                    A[c1 + i] -= av * w1;
                    A[c2 + i] -= av * w2;
                    A[c3 + i] -= av * w3;
                }
            }
            // --- remainder columns ---
            for (; j < n; ++j) {
                uniform int cj = j * ldap;
                varying double w = A[cj + kk];
                for (uniform int i = kk + 1; i < m; ++i)
                    w += A[ckk + i] * A[cj + i];
                A[cj + kk] -= t * w;
                w *= t;
                for (uniform int i = kk + 1; i < m; ++i)
                    A[cj + i] -= A[ckk + i] * w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Apply Q (or Q^T) from the left: B := op(Q) B, one group at a time. Mirrors
// cqr::detail::ormqr_compact_group (side = 'L', column-major). Q is the product
// of k reflectors H(0)..H(k-1) from cqr_ispc_dgeqrf_compact.
//
//   trans  0 -> apply Q (H(0)..H(k-1), descending sweep)
//          1 -> apply Q^T (H(k-1)..H(0), ascending sweep; the solve case)
//   m      rows of B (and reflector length axis)
//   nrhs   columns of B
//   k      number of reflectors (min(m,n) of the factorization)
//   ap     compact reflectors, (ldap, k) per matrix
//   taup   compact tau, k per matrix
//   bp     compact B (m x nrhs), overwritten with op(Q) B
// ---------------------------------------------------------------------------
export void cqr_ispc_dormqr_compact(uniform int trans, uniform int m, uniform int nrhs,
                                    uniform int k, uniform double ap[], uniform int ldap,
                                    uniform double taup[], uniform double bp[],
                                    uniform int ldbp, uniform int nm)
{
    uniform int V = programCount;
    uniform int ngroups = (nm + V - 1) / V;
    uniform int64 stra = (uniform int64)ldap * k * V;
    uniform int64 strt = (uniform int64)k * V;
    uniform int64 strb = (uniform int64)ldbp * nrhs * V;
    uniform bool fwd = (trans != 0); // Q^T: ascending; Q: descending

    for (uniform int g = 0; g < ngroups; ++g) {
        varying double * uniform A = (varying double * uniform)(ap + g * stra);
        varying double * uniform tau = (varying double * uniform)(taup + g * strt);
        varying double * uniform B = (varying double * uniform)(bp + g * strb);

        for (uniform int s = 0; s < k; ++s) {
            uniform int kk = fwd ? s : (k - 1 - s);
            uniform int ck = kk * ldap; // reflector column kk
            varying double t = tau[kk];

            uniform int j = 0;
            // main loop: 4 RHS columns at a time; A[ck+i] loaded once, used 4x
            for (; j + 4 <= nrhs; j += 4) {
                uniform int b0 = (j + 0) * ldbp, b1 = (j + 1) * ldbp;
                uniform int b2 = (j + 2) * ldbp, b3 = (j + 3) * ldbp;
                varying double w0 = B[b0 + kk], w1 = B[b1 + kk];
                varying double w2 = B[b2 + kk], w3 = B[b3 + kk];
                for (uniform int i = kk + 1; i < m; ++i) {
                    varying double av = A[ck + i];
                    w0 += av * B[b0 + i];
                    w1 += av * B[b1 + i];
                    w2 += av * B[b2 + i];
                    w3 += av * B[b3 + i];
                }
                B[b0 + kk] -= t * w0;
                B[b1 + kk] -= t * w1;
                B[b2 + kk] -= t * w2;
                B[b3 + kk] -= t * w3;
                w0 *= t;
                w1 *= t;
                w2 *= t;
                w3 *= t;
                for (uniform int i = kk + 1; i < m; ++i) {
                    varying double av = A[ck + i];
                    B[b0 + i] -= av * w0;
                    B[b1 + i] -= av * w1;
                    B[b2 + i] -= av * w2;
                    B[b3 + i] -= av * w3;
                }
            }
            // remainder columns
            for (; j < nrhs; ++j) {
                uniform int bj = j * ldbp;
                varying double w = B[bj + kk];
                for (uniform int i = kk + 1; i < m; ++i)
                    w += A[ck + i] * B[bj + i];
                B[bj + kk] -= t * w;
                w *= t;
                for (uniform int i = kk + 1; i < m; ++i)
                    B[bj + i] -= A[ck + i] * w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Triangular solve B := alpha * op(A)^-1 B, one group at a time. Specialized to
// the QR-solve step: side = 'L', uplo = 'U', trans = 'N', diag = 'N' -- i.e.
// solve the upper-triangular R X = alpha B by back-substitution. This is the
// compact analogue of mkl_dtrsm_compact(LEFT, UPPER, NOTRANS, NONUNIT).
//
//   n      order of A (= rows of B)
//   nrhs   columns of B
//   alpha  scalar (1.0 in the QR solve)
//   ap     compact upper-triangular A (n x n), column-major
//   bp     compact B (n x nrhs), overwritten with the solution X
// ---------------------------------------------------------------------------
export void cqr_ispc_dtrsm_compact(uniform int n, uniform int nrhs, uniform double alpha,
                                   uniform double ap[], uniform int ldap,
                                   uniform double bp[], uniform int ldbp, uniform int nm)
{
    uniform int V = programCount;
    uniform int ngroups = (nm + V - 1) / V;
    uniform int64 stra = (uniform int64)ldap * n * V;
    uniform int64 strb = (uniform int64)ldbp * nrhs * V;

    for (uniform int g = 0; g < ngroups; ++g) {
        varying double * uniform A = (varying double * uniform)(ap + g * stra);
        varying double * uniform B = (varying double * uniform)(bp + g * strb);

        // Back-substitution: x_i = (alpha b_i - sum_{l>i} A(i,l) x_l) / A(i,i).
        // 4 RHS columns at a time so each A(i,l) load is reused across 4 columns
        // and the reciprocal diagonal is formed once per row.
        uniform int j = 0;
        for (; j + 4 <= nrhs; j += 4) {
            uniform int b0 = (j + 0) * ldbp, b1 = (j + 1) * ldbp;
            uniform int b2 = (j + 2) * ldbp, b3 = (j + 3) * ldbp;
            for (uniform int i = n - 1; i >= 0; --i) {
                varying double s0 = alpha * B[b0 + i], s1 = alpha * B[b1 + i];
                varying double s2 = alpha * B[b2 + i], s3 = alpha * B[b3 + i];
                for (uniform int l = i + 1; l < n; ++l) {
                    varying double r = A[l * ldap + i]; // A(i,l), col-major
                    s0 -= r * B[b0 + l];
                    s1 -= r * B[b1 + l];
                    s2 -= r * B[b2 + l];
                    s3 -= r * B[b3 + l];
                }
                varying double dinv = 1.0d / A[i * ldap + i]; // 1 / A(i,i)
                B[b0 + i] = s0 * dinv;
                B[b1 + i] = s1 * dinv;
                B[b2 + i] = s2 * dinv;
                B[b3 + i] = s3 * dinv;
            }
        }
        // remainder columns
        for (; j < nrhs; ++j) {
            uniform int bj = j * ldbp;
            for (uniform int i = n - 1; i >= 0; --i) {
                varying double s = alpha * B[bj + i];
                for (uniform int l = i + 1; l < n; ++l)
                    s -= A[l * ldap + i] * B[bj + l];
                B[bj + i] = s / A[i * ldap + i];
            }
        }
    }
}
