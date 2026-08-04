!> Batched unblocked Householder QR (geqr2) and apply-Q^T (orm2r) for many
!> small matrices, written FOUR ways to compare how gfortran/ifx vectorize the
!> batch dimension. All four share one memory layout -- the compact/interleaved
!> layout A(nb,m,n) with the batch index FASTEST-varying -- so the only thing
!> that changes between them is how the SIMD over the batch is *expressed*:
!>
!>   1. inner-loop  : explicit DO loops, batch index innermost (compiler
!>                    vectorizes the inner batch loop of each micro-kernel).
!>   2. outer-loop  : one `!$omp simd` loop over the batch, the whole scalar
!>                    geqr2 for one matrix in the body (outer-loop vectorization,
!>                    one matrix per lane).
!>   3. array-ops   : Fortran whole-array syntax A(:,i,j) over the batch dim.
!>   4. vector-type : a VL-wide derived type (dvec) with overloaded operators,
!>                    processing VL matrices per group -- the "compact" approach.
!>
!> Each matrix in the batch is m x n; we factor A = Q R (Q = H(1)..H(k),
!> k = min(m,n)) and, for orm2r, overwrite B := Q^T B (side='L', trans='T').
!> Padding/rank-deficiency are handled branch-free by the `tail > 0` mask, so
!> every lane runs unmasked -- identical semantics to LAPACK ?geqr2 / ?orm2r.
module batched_qr
   use, intrinsic :: iso_fortran_env, only: dp => real64
   use dvec_mod
   implicit none
   private

   public :: ref_geqr2, ref_orm2r
   public :: geqr2_inner, geqr2_outer, geqr2_array, geqr2_vtype
   public :: orm2r_inner, orm2r_outer, orm2r_array, orm2r_vtype

contains

   ! =====================================================================
   !  Scalar reference (one matrix at a time) -- the correctness oracle.
   ! =====================================================================

   !> Unblocked Householder QR of a single m x n matrix (LAPACK dgeqr2).
   subroutine ref_geqr2(m, n, a, tau)
      integer, intent(in) :: m, n
      real(dp), intent(inout) :: a(m, n)
      real(dp), intent(out) :: tau(min(m, n))
      integer :: j, jj, i, k, p
      real(dp) :: alpha, xnorm, beta, w
      k = min(m, n)
      do j = 1, k
         p = m - j            ! # below-diagonal entries in column j
         alpha = a(j, j)
         xnorm = 0.0_dp
         do i = j + 1, m
            xnorm = xnorm + a(i, j)**2
         end do
         xnorm = sqrt(xnorm)
         if (xnorm == 0.0_dp .or. p == 0) then
            tau(j) = 0.0_dp
         else
            beta = -sign(sqrt(alpha**2 + xnorm**2), alpha)
            tau(j) = (beta - alpha)/beta
            a(j + 1:m, j) = a(j + 1:m, j)/(alpha - beta)
            a(j, j) = beta
         end if
         ! Apply H(j) = I - tau v v^T to the trailing columns (v = [1; a(j+1:m,j)]).
         if (j < n) then
            do jj = j + 1, n
               w = a(j, jj)
               do i = j + 1, m
                  w = w + a(i, j)*a(i, jj)
               end do
               w = tau(j)*w
               a(j, jj) = a(j, jj) - w
               do i = j + 1, m
                  a(i, jj) = a(i, jj) - w*a(i, j)
               end do
            end do
         end if
      end do
   end subroutine ref_geqr2

   !> Apply Q^T to B from the left (LAPACK dorm2r, side='L', trans='T').
   !> A/tau come from ref_geqr2; B is m x nrhs, overwritten with Q^T B.
   subroutine ref_orm2r(m, n, nrhs, a, tau, b)
      integer, intent(in) :: m, n, nrhs
      real(dp), intent(in) :: a(m, n), tau(min(m, n))
      real(dp), intent(inout) :: b(m, nrhs)
      integer :: j, col, i, k
      real(dp) :: w
      k = min(m, n)
      do j = 1, k                 ! Q^T = H(k)..H(1) applied left-to-right
         do col = 1, nrhs
            w = b(j, col)
            do i = j + 1, m
               w = w + a(i, j)*b(i, col)
            end do
            w = tau(j)*w
            b(j, col) = b(j, col) - w
            do i = j + 1, m
               b(i, col) = b(i, col) - w*a(i, j)
            end do
         end do
      end do
   end subroutine ref_orm2r

   ! =====================================================================
   !  Variant 1: explicit loops, BATCH INDEX INNERMOST.
   !  Each micro-kernel is a DO loop over the batch (unit stride) that the
   !  compiler vectorizes. Temporaries alpha/beta/... are length-nb arrays.
   ! =====================================================================

   subroutine geqr2_inner(nb, m, n, a, tau)
      integer, intent(in) :: nb, m, n
      real(dp), intent(inout) :: a(nb, m, n)
      real(dp), intent(out) :: tau(nb, min(m, n))
      real(dp) :: alpha(nb), tail(nb), beta(nb), w(nb), inv(nb)
      integer :: j, jj, i, b, k
      k = min(m, n)
      do j = 1, k
         do b = 1, nb
            alpha(b) = a(b, j, j)
            tail(b) = 0.0_dp
         end do
         do i = j + 1, m
            do b = 1, nb
               tail(b) = tail(b) + a(b, i, j)**2
            end do
         end do
         do b = 1, nb
            beta(b) = -sign(sqrt(alpha(b)**2 + tail(b)), alpha(b))
            if (tail(b) > 0.0_dp) then
               tau(b, j) = (beta(b) - alpha(b))/beta(b)
               inv(b) = 1.0_dp/(alpha(b) - beta(b))
               a(b, j, j) = beta(b)
            else
               tau(b, j) = 0.0_dp
               inv(b) = 0.0_dp
            end if
         end do
         do i = j + 1, m
            do b = 1, nb
               a(b, i, j) = a(b, i, j)*inv(b)
            end do
         end do
         do jj = j + 1, n
            do b = 1, nb
               w(b) = a(b, j, jj)
            end do
            do i = j + 1, m
               do b = 1, nb
                  w(b) = w(b) + a(b, i, j)*a(b, i, jj)
               end do
            end do
            do b = 1, nb
               w(b) = tau(b, j)*w(b)
               a(b, j, jj) = a(b, j, jj) - w(b)
            end do
            do i = j + 1, m
               do b = 1, nb
                  a(b, i, jj) = a(b, i, jj) - w(b)*a(b, i, j)
               end do
            end do
         end do
      end do
   end subroutine geqr2_inner

   ! =====================================================================
   !  Variant 2: one !$omp simd loop over the batch, scalar geqr2 inside.
   !  Outer-loop vectorization -- one matrix per SIMD lane. Body is written
   !  branch-free so the whole per-matrix algorithm predicates cleanly.
   ! =====================================================================

   subroutine geqr2_outer(nb, m, n, a, tau)
      integer, intent(in) :: nb, m, n
      real(dp), intent(inout) :: a(nb, m, n)
      real(dp), intent(out) :: tau(nb, min(m, n))
      integer :: j, jj, i, b, k
      real(dp) :: alpha, tail, beta, w, inv, t, d
      k = min(m, n)
      !$omp simd private(j, jj, i, alpha, tail, beta, w, inv, t, d)
      do b = 1, nb
         do j = 1, k
            alpha = a(b, j, j)
            tail = 0.0_dp
            do i = j + 1, m
               tail = tail + a(b, i, j)**2
            end do
            beta = -sign(sqrt(alpha*alpha + tail), alpha)
            ! branch-free larfg: t=1 if a reflector is generated, else 0
            t = merge(1.0_dp, 0.0_dp, tail > 0.0_dp)
            d = merge(beta, 1.0_dp, tail > 0.0_dp)     ! avoid /0 in dead lane
            tau(b, j) = t*(beta - alpha)/d
            inv = t/merge(alpha - beta, 1.0_dp, tail > 0.0_dp)
            a(b, j, j) = merge(beta, alpha, tail > 0.0_dp)
            do i = j + 1, m
               a(b, i, j) = a(b, i, j)*inv
            end do
            do jj = j + 1, n
               w = a(b, j, jj)
               do i = j + 1, m
                  w = w + a(b, i, j)*a(b, i, jj)
               end do
               w = tau(b, j)*w
               a(b, j, jj) = a(b, j, jj) - w
               do i = j + 1, m
                  a(b, i, jj) = a(b, i, jj) - w*a(b, i, j)
               end do
            end do
         end do
      end do
   end subroutine geqr2_outer

   ! =====================================================================
   !  Variant 3: Fortran whole-array syntax over the batch dimension.
   !  Reads like scalar geqr2 with the batch collapsed into A(:,i,j).
   ! =====================================================================

   subroutine geqr2_array(nb, m, n, a, tau)
      integer, intent(in) :: nb, m, n
      real(dp), intent(inout) :: a(nb, m, n)
      real(dp), intent(out) :: tau(nb, min(m, n))
      real(dp) :: alpha(nb), tail(nb), beta(nb), w(nb), inv(nb)
      logical :: msk(nb)
      integer :: j, jj, i, k
      k = min(m, n)
      do j = 1, k
         alpha = a(:, j, j)
         tail = 0.0_dp
         do i = j + 1, m
            tail = tail + a(:, i, j)**2
         end do
         beta = -sign(sqrt(alpha**2 + tail), alpha)
         msk = tail > 0.0_dp
         tau(:, j) = merge((beta - alpha)/merge(beta, 1.0_dp, msk), 0.0_dp, msk)
         inv = merge(1.0_dp/merge(alpha - beta, 1.0_dp, msk), 0.0_dp, msk)
         a(:, j, j) = merge(beta, alpha, msk)
         do i = j + 1, m
            a(:, i, j) = a(:, i, j)*inv
         end do
         do jj = j + 1, n
            w = a(:, j, jj)
            do i = j + 1, m
               w = w + a(:, i, j)*a(:, i, jj)
            end do
            w = tau(:, j)*w
            a(:, j, jj) = a(:, j, jj) - w
            do i = j + 1, m
               a(:, i, jj) = a(:, i, jj) - w*a(:, i, j)
            end do
         end do
      end do
   end subroutine geqr2_array

   ! =====================================================================
   !  Variant 4: custom VL-wide vector type with overloaded operators.
   !  Process VL matrices per group; the algorithm reads like scalar code
   !  but every op is a lane-wise SIMD instruction. nb must be a multiple
   !  of VL (driver guarantees this).
   ! =====================================================================

   subroutine geqr2_vtype(nb, m, n, a, tau)
      integer, intent(in) :: nb, m, n
      real(dp), intent(inout) :: a(nb, m, n)
      real(dp), intent(out) :: tau(nb, min(m, n))
      type(dvec) :: av(m, n), alpha, tail, beta, w, inv, one, zero
      logical :: msk(VL)
      integer :: g, ng, b0, j, jj, i, k
      k = min(m, n)
      ng = nb/VL
      one = splat(1.0_dp)
      zero = splat(0.0_dp)
      do g = 1, ng
         b0 = (g - 1)*VL
         ! pack this group into the register array
         do j = 1, n
            do i = 1, m
               av(i, j) = load_group(a(b0 + 1:b0 + VL, i, j))
            end do
         end do
         do j = 1, k
            alpha = av(j, j)
            tail = zero
            do i = j + 1, m
               tail = tail + av(i, j)*av(i, j)
            end do
            beta = vsqrt(alpha*alpha + tail)
            beta%x = -sign(beta%x, alpha%x)
            msk = tail%x > 0.0_dp
            ! tau = msk ? (beta-alpha)/beta : 0 ; inv = msk ? 1/(alpha-beta) : 0
            w = dmerge((beta - alpha)/dmerge(beta, one, msk), zero, msk)
            call store_group(tau(b0 + 1:b0 + VL, j), w)
            inv = dmerge(one/dmerge(alpha - beta, one, msk), zero, msk)
            av(j, j) = dmerge(beta, alpha, msk)
            do i = j + 1, m
               av(i, j) = av(i, j)*inv
            end do
            do jj = j + 1, n
               w = av(j, jj)                       ! w = v^T c  (v = [1; av(j+1:,j)])
               do i = j + 1, m
                  w = w + av(i, j)*av(i, jj)
               end do
               w = load_group(tau(b0 + 1:b0 + VL, j))*w   ! w = tau * (v^T c)
               av(j, jj) = av(j, jj) - w
               do i = j + 1, m
                  av(i, jj) = av(i, jj) - w*av(i, j)
               end do
            end do
         end do
         ! unpack
         do j = 1, n
            do i = 1, m
               call store_group(a(b0 + 1:b0 + VL, i, j), av(i, j))
            end do
         end do
      end do
   end subroutine geqr2_vtype

   ! =====================================================================
   !  orm2r: B := Q^T B, four variants (same taxonomy as above).
   ! =====================================================================

   subroutine orm2r_inner(nb, m, n, nrhs, a, tau, b)
      integer, intent(in) :: nb, m, n, nrhs
      real(dp), intent(in) :: a(nb, m, n), tau(nb, min(m, n))
      real(dp), intent(inout) :: b(nb, m, nrhs)
      real(dp) :: w(nb)
      integer :: j, col, i, bb, k
      k = min(m, n)
      do j = 1, k
         do col = 1, nrhs
            do bb = 1, nb
               w(bb) = b(bb, j, col)
            end do
            do i = j + 1, m
               do bb = 1, nb
                  w(bb) = w(bb) + a(bb, i, j)*b(bb, i, col)
               end do
            end do
            do bb = 1, nb
               w(bb) = tau(bb, j)*w(bb)
               b(bb, j, col) = b(bb, j, col) - w(bb)
            end do
            do i = j + 1, m
               do bb = 1, nb
                  b(bb, i, col) = b(bb, i, col) - w(bb)*a(bb, i, j)
               end do
            end do
         end do
      end do
   end subroutine orm2r_inner

   subroutine orm2r_outer(nb, m, n, nrhs, a, tau, b)
      integer, intent(in) :: nb, m, n, nrhs
      real(dp), intent(in) :: a(nb, m, n), tau(nb, min(m, n))
      real(dp), intent(inout) :: b(nb, m, nrhs)
      integer :: j, col, i, bb, k
      real(dp) :: w
      k = min(m, n)
      !$omp simd private(j, col, i, w)
      do bb = 1, nb
         do j = 1, k
            do col = 1, nrhs
               w = b(bb, j, col)
               do i = j + 1, m
                  w = w + a(bb, i, j)*b(bb, i, col)
               end do
               w = tau(bb, j)*w
               b(bb, j, col) = b(bb, j, col) - w
               do i = j + 1, m
                  b(bb, i, col) = b(bb, i, col) - w*a(bb, i, j)
               end do
            end do
         end do
      end do
   end subroutine orm2r_outer

   subroutine orm2r_array(nb, m, n, nrhs, a, tau, b)
      integer, intent(in) :: nb, m, n, nrhs
      real(dp), intent(in) :: a(nb, m, n), tau(nb, min(m, n))
      real(dp), intent(inout) :: b(nb, m, nrhs)
      real(dp) :: w(nb)
      integer :: j, col, i, k
      k = min(m, n)
      do j = 1, k
         do col = 1, nrhs
            w = b(:, j, col)
            do i = j + 1, m
               w = w + a(:, i, j)*b(:, i, col)
            end do
            w = tau(:, j)*w
            b(:, j, col) = b(:, j, col) - w
            do i = j + 1, m
               b(:, i, col) = b(:, i, col) - w*a(:, i, j)
            end do
         end do
      end do
   end subroutine orm2r_array

   subroutine orm2r_vtype(nb, m, n, nrhs, a, tau, b)
      integer, intent(in) :: nb, m, n, nrhs
      real(dp), intent(in) :: a(nb, m, n), tau(nb, min(m, n))
      real(dp), intent(inout) :: b(nb, m, nrhs)
      type(dvec) :: av(m, n), bv(m, nrhs), tv(min(m, n)), w
      integer :: g, ng, b0, j, col, i, k
      k = min(m, n)
      ng = nb/VL
      do g = 1, ng
         b0 = (g - 1)*VL
         do j = 1, n
            do i = 1, m
               av(i, j) = load_group(a(b0 + 1:b0 + VL, i, j))
            end do
         end do
         do j = 1, k
            tv(j) = load_group(tau(b0 + 1:b0 + VL, j))
         end do
         do col = 1, nrhs
            do i = 1, m
               bv(i, col) = load_group(b(b0 + 1:b0 + VL, i, col))
            end do
         end do
         do j = 1, k
            do col = 1, nrhs
               w = bv(j, col)
               do i = j + 1, m
                  w = w + av(i, j)*bv(i, col)
               end do
               w = tv(j)*w
               bv(j, col) = bv(j, col) - w
               do i = j + 1, m
                  bv(i, col) = bv(i, col) - w*av(i, j)
               end do
            end do
         end do
         do col = 1, nrhs
            do i = 1, m
               call store_group(b(b0 + 1:b0 + VL, i, col), bv(i, col))
            end do
         end do
      end do
   end subroutine orm2r_vtype

end module batched_qr
