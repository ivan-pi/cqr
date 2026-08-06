// cqr_ispc.ispc -- Compact (interleaved-batch) QR pipeline in Intel ISPC, an SPMD
// prototype of the GNU vector_size kernels in ../src, on the same algorithm and
// MKL Compact layout, to measure ISPC against them.
//
// The mapping that makes it exact: MKL Compact stores element (i,j) of the V
// interleaved matrices contiguously (ap[g*ldap*ncol*V + (j*ldap+i)*V + v]), and an
// ISPC `varying double` IS programCount consecutive doubles -- so when
// V == programCount a compact pack of V matrices' element (i,j) is one `varying
// double`. Cast the group base to `varying double * uniform`, index by (j*ldap+i),
// and each kernel is the plain scalar geqr2 / dorm2r / back-substitution that ISPC
// vectorizes across the V matrices -- no hand-written vectors or masks.
//
// The kernels are gang-generic: they work whenever the data is packed at
// V == the gang width. The MKL-interop test/benchmark use MKL's AVX-512 FP64
// compact format (V = 8, e.g. --target=avx512skx-x8) and check the gang width at
// runtime; the gang sweep (bench_gang) packs at V = gang width for any target.
// Column-major, double. Exported extern "C" as drop-in siblings of
// ../src/cqr_compact.h, plus a compact trsm/potrf and a gang-width query:
//   cqr_ispc_dgeqrf_compact  A = Q R
//   cqr_ispc_dormqr_compact  B := op(Q) B          (side 'L', trans 0=Q / 1=Q^T)
//   cqr_ispc_dtrsm_compact   B := alpha op(A)^-1 B  (left, upper, no-trans, non-unit)
//   cqr_ispc_dpotrf_compact  A = L L^T / U^T U      (Cholesky, lower or upper)
//   cqr_ispc_gang_width      returns the gang width (= the V the data must use)
//
// Assisted-by: Claude:claude-opus-4.8

// The gang width is the interleave width V these kernels expect the data to use.
export uniform int cqr_ispc_gang_width() { return programCount; }

// QR factorization (unblocked geqr2), one group of programCount matrices at a time;
// mirrors cqr::detail::geqrf_compact_group. ap (m x n col-major) -> (R, Householder
// vectors); taup -> k=min(m,n) tau per matrix. A padded final group factors to
// tau=0 (harmless). Column pack (i,j) == the varying at linear index j*ldap+i.
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

// Apply Q (trans=0, descending sweep) or Q^T (trans=1, ascending) from the left:
// B := op(Q) B, one group at a time; mirrors cqr::detail::ormqr_compact_group
// (side 'L', col-major). ap/taup are the k reflectors from the geqrf above,
// (ldap,k) per matrix; bp (m x nrhs) is overwritten.
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

// Triangular solve B := alpha op(A)^-1 B, one group at a time, for the QR-solve
// case only: side 'L', uplo 'U', trans 'N', diag 'N' -- back-substitution on the
// upper-triangular R (n x n col-major). Compact analogue of
// mkl_dtrsm_compact(LEFT, UPPER, NOTRANS, NONUNIT); bp (n x nrhs) -> solution X.
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

// Cholesky factorization (blocked, right-looking), one group of programCount
// matrices at a time; the SPD analogue of mkl_dpotrf_compact. A = L L^T for the
// lower triangle (upper=0), A = U^T U for the upper (upper=1); only the named
// triangle is read or written. Element (i,j) is the varying at linear pack index
// i*rs + j*cs: the factor column is contiguous (rs=1, cs=ldap) for column-major
// lower and its transpose dual row-major upper, and strided (rs=ldap, cs=1) for
// column-major upper / row-major lower -- one kernel over the same math.
//
// Blocking: process NB columns at a time. (1) factor the panel with the plain
// unblocked rank-1 sweep, but bounded to the panel's own columns and run at full
// row height so the sub-panel L21 below the diagonal block is formed; (2) apply
// the deferred cross-panel update to the trailing submatrix as ONE symmetric
// rank-NB update (SYRK) A(i,c) -= sum_p L(i,p) L(c,p). So the O(n^2) trailing
// submatrix is streamed once per panel (n/NB times) rather than once per column
// -- the reuse that keeps a large batch cache-resident. For n <= NB there is no
// trailing block and this reduces to the unblocked kernel. The pivot sqrt is
// unconditional (a non-SPD lane poisons only itself); no gather/scatter.
export void cqr_ispc_dpotrf_compact(uniform int rowmajor, uniform int upper,
                                    uniform int n, uniform double ap[], uniform int ldap,
                                    uniform int nm)
{
    uniform int V = programCount;
    uniform int ngroups = (nm + V - 1) / V;
    uniform int64 stra = (uniform int64)ldap * n * V; // doubles per group in ap

    // A factor column is contiguous (row step = one pack) for column-major lower
    // and its transpose dual row-major upper; the other two are strided (ldap).
    uniform bool contiguous =
        (rowmajor == 0 && upper == 0) || (rowmajor != 0 && upper != 0);
    uniform int rs = contiguous ? 1 : ldap; // element (i,j) at pack index i*rs + j*cs
    uniform int cs = contiguous ? ldap : 1;
    const uniform int NB = 8; // panel width (columns per block)

    for (uniform int g = 0; g < ngroups; ++g) {
        // A compact pack == one `varying double`: lane v is matrix v of the group.
        varying double * uniform A = (varying double * uniform)(ap + g * stra);

        for (uniform int j0 = 0; j0 < n; j0 += NB) {
            uniform int jend = (j0 + NB < n) ? (j0 + NB) : n; // one past panel's last col

            // ---- (1) panel factor: unblocked Cholesky on columns j0..jend-1, at
            // full row height (forming L21); the cross-panel update is deferred to
            // (2). The rank-1 sweep is the unblocked kernel with the column loops
            // bounded by jend instead of n. ----
            for (uniform int j = j0; j < jend; ++j) {
                uniform int cj = j * cs; // pivot column j base

                // pivot: d = sqrt(A(j,j)); scale the sub-diagonal column by 1/d.
                varying double d = sqrt(A[j * rs + cj]);
                A[j * rs + cj] = d;
                varying double invd = 1.0d / d;
                for (uniform int i = j + 1; i < n; ++i)
                    A[i * rs + cj] = A[i * rs + cj] * invd;

                // rank-1 update of the remaining PANEL columns jj in (j, jend), at
                // full height (i >= jj), 4 columns at a time so A(i,j) is reused.
                uniform int jj = j + 1;
                for (; jj + 4 <= jend; jj += 4) {
                    uniform int d0 = (jj + 0) * cs, d1 = (jj + 1) * cs;
                    uniform int d2 = (jj + 2) * cs, d3 = (jj + 3) * cs;
                    varying double w0 = A[(jj + 0) * rs + cj], w1 = A[(jj + 1) * rs + cj];
                    varying double w2 = A[(jj + 2) * rs + cj], w3 = A[(jj + 3) * rs + cj];

                    // near-diagonal triangular corner (rows jj..jj+3), lower only
                    A[(jj + 0) * rs + d0] -= w0 * w0;

                    A[(jj + 1) * rs + d0] -= w1 * w0;
                    A[(jj + 1) * rs + d1] -= w1 * w1;

                    A[(jj + 2) * rs + d0] -= w2 * w0;
                    A[(jj + 2) * rs + d1] -= w2 * w1;
                    A[(jj + 2) * rs + d2] -= w2 * w2;

                    A[(jj + 3) * rs + d0] -= w3 * w0;
                    A[(jj + 3) * rs + d1] -= w3 * w1;
                    A[(jj + 3) * rs + d2] -= w3 * w2;
                    A[(jj + 3) * rs + d3] -= w3 * w3;

                    // rectangular tail below the corner: all 4 columns, A(i,j) reused
                    for (uniform int i = jj + 4; i < n; ++i) {
                        varying double av = A[i * rs + cj];
                        A[i * rs + d0] -= av * w0;
                        A[i * rs + d1] -= av * w1;
                        A[i * rs + d2] -= av * w2;
                        A[i * rs + d3] -= av * w3;
                    }
                }
                // remainder panel columns (fewer than 4 left)
                for (; jj < jend; ++jj) {
                    uniform int dj = jj * cs;
                    varying double w = A[jj * rs + cj];
                    for (uniform int i = jj; i < n; ++i)
                        A[i * rs + dj] -= A[i * rs + cj] * w;
                }
            }

            // ---- (2) deferred trailing update as a symmetric rank-(jend-j0)
            // update: A(i,c) -= sum_{p=j0..jend-1} A(i,p) A(c,p), c in [jend,n),
            // i in [c,n) (lower trapezoid). Dot-product form so each trailing
            // element is written once; 4 trailing columns at a time so each panel
            // load A(i,p) feeds 4 columns. The panel stays hot across the sweep. ----
            uniform int c = jend;
            for (; c + 4 <= n; c += 4) {
                uniform int e0 = (c + 0) * cs, e1 = (c + 1) * cs;
                uniform int e2 = (c + 2) * cs, e3 = (c + 3) * cs;

                // near-diagonal triangular corner (rows c..c+3), lower only
                for (uniform int i = c; i < c + 4; ++i)
                    for (uniform int ct = c; ct <= i; ++ct) {
                        varying double s = 0.0d;
                        for (uniform int p = j0; p < jend; ++p)
                            s += A[i * rs + p * cs] * A[ct * rs + p * cs];
                        A[i * rs + ct * cs] -= s;
                    }

                // rectangular tail below the corner: all 4 columns, A(i,p) reused
                for (uniform int i = c + 4; i < n; ++i) {
                    varying double s0 = 0.0d, s1 = 0.0d, s2 = 0.0d, s3 = 0.0d;
                    for (uniform int p = j0; p < jend; ++p) {
                        uniform int dp = p * cs;
                        varying double av = A[i * rs + dp];
                        s0 += av * A[(c + 0) * rs + dp];
                        s1 += av * A[(c + 1) * rs + dp];
                        s2 += av * A[(c + 2) * rs + dp];
                        s3 += av * A[(c + 3) * rs + dp];
                    }
                    A[i * rs + e0] -= s0;
                    A[i * rs + e1] -= s1;
                    A[i * rs + e2] -= s2;
                    A[i * rs + e3] -= s3;
                }
            }
            // remainder trailing columns (fewer than 4 left)
            for (; c < n; ++c) {
                uniform int ec = c * cs;
                for (uniform int i = c; i < n; ++i) {
                    varying double s = 0.0d;
                    for (uniform int p = j0; p < jend; ++p)
                        s += A[i * rs + p * cs] * A[c * rs + p * cs];
                    A[i * rs + ec] -= s;
                }
            }
        }
    }
}
