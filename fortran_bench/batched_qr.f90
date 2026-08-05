!> Batched unblocked Householder QR (geqr2 / dgeqrf) and apply-Q^T (orm2r /
!> dormqr) for many small matrices, written FOUR ways to compare how the
!> compiler vectorizes the batch dimension.
!>
!> The batch is now interleaved in groups of a FIXED, COMPILE-TIME width
!> VW = 8 (one 512-bit register): layout A(VW, m, n, ng), with the 8-lane batch
!> index FASTEST-varying and ng groups stacked behind it. Every variant loops
!> over the ng groups; within a group the batch extent is the constant 8, so the
!> compiler can lower each lane loop to a single unmasked 8-wide instruction with
!> no remainder. The four variants differ only in how the width-8 SIMD is
!> *expressed*:
!>
!>   1. inner-loop  : explicit DO loops, `do b = 1, VW` innermost.
!>   2. outer-loop  : `!$omp simd` over `do b = 1, VW`, the whole scalar geqr2
!>                    for one matrix in the body (one matrix per lane).
!>   3. array-ops   : Fortran whole-array syntax A(:,i,j,g) over the 8 lanes.
!>   4. vector-type : the vec8 derived type with overloaded operators.
!>
!> Each matrix is m x n; we factor A = Q R (Q = H(1)..H(k), k = min(m,n)) and,
!> for orm2r, overwrite B := Q^T B (side='L', trans='T'). Padding and
!> rank-deficiency are handled branch-free by the `tail > 0` mask, so every lane
!> runs unmasked -- identical semantics to LAPACK ?geqr2 / ?orm2r.
module batched_qr
   use, intrinsic :: iso_fortran_env, only: dp => real64
   use vec8_mod
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
      integer :: j, jj, i, k
      real(dp) :: alpha, xnorm, beta, w
      k = min(m, n)
      do j = 1, k
         alpha = a(j, j)
         xnorm = 0.0_dp
         do i = j + 1, m
            xnorm = xnorm + a(i, j)**2
         end do
         xnorm = sqrt(xnorm)
         if (xnorm == 0.0_dp) then
            tau(j) = 0.0_dp
         else
            beta = -sign(sqrt(alpha**2 + xnorm**2), alpha)
            tau(j) = (beta - alpha)/beta
            a(j + 1:m, j) = a(j + 1:m, j)/(alpha - beta)
            a(j, j) = beta
         end if
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
   subroutine ref_orm2r(m, n, nrhs, a, tau, b)
      integer, intent(in) :: m, n, nrhs
      real(dp), intent(in) :: a(m, n), tau(min(m, n))
      real(dp), intent(inout) :: b(m, nrhs)
      integer :: j, col, i, k
      real(dp) :: w
      k = min(m, n)
      do j = 1, k
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
   !  Variant 1: explicit loops, `do b = 1, VW` (fixed 8) INNERMOST.
   ! =====================================================================

   subroutine geqr2_inner(ng, m, n, a, tau)
      integer, intent(in) :: ng, m, n
      real(dp), intent(inout) :: a(VW, m, n, ng)
      real(dp), intent(out) :: tau(VW, min(m, n), ng)
      real(dp) :: alpha(VW), tail(VW), beta(VW), w(VW), inv(VW)
      integer :: g, j, jj, i, b, k
      k = min(m, n)
      do g = 1, ng
         do j = 1, k
            do b = 1, VW
               alpha(b) = a(b, j, j, g)
               tail(b) = 0.0_dp
            end do
            do i = j + 1, m
               do b = 1, VW
                  tail(b) = tail(b) + a(b, i, j, g)**2
               end do
            end do
            do b = 1, VW
               beta(b) = -sign(sqrt(alpha(b)**2 + tail(b)), alpha(b))
               if (tail(b) > 0.0_dp) then
                  tau(b, j, g) = (beta(b) - alpha(b))/beta(b)
                  inv(b) = 1.0_dp/(alpha(b) - beta(b))
                  a(b, j, j, g) = beta(b)
               else
                  tau(b, j, g) = 0.0_dp
                  inv(b) = 0.0_dp
               end if
            end do
            do i = j + 1, m
               do b = 1, VW
                  a(b, i, j, g) = a(b, i, j, g)*inv(b)
               end do
            end do
            do jj = j + 1, n
               do b = 1, VW
                  w(b) = a(b, j, jj, g)
               end do
               do i = j + 1, m
                  do b = 1, VW
                     w(b) = w(b) + a(b, i, j, g)*a(b, i, jj, g)
                  end do
               end do
               do b = 1, VW
                  w(b) = tau(b, j, g)*w(b)
                  a(b, j, jj, g) = a(b, j, jj, g) - w(b)
               end do
               do i = j + 1, m
                  do b = 1, VW
                     a(b, i, jj, g) = a(b, i, jj, g) - w(b)*a(b, i, j, g)
                  end do
               end do
            end do
         end do
      end do
   end subroutine geqr2_inner

   ! =====================================================================
   !  Variant 2: !$omp simd over the fixed-8 batch, scalar geqr2 inside.
   ! =====================================================================

   subroutine geqr2_outer(ng, m, n, a, tau)
      integer, intent(in) :: ng, m, n
      real(dp), intent(inout) :: a(VW, m, n, ng)
      real(dp), intent(out) :: tau(VW, min(m, n), ng)
      integer :: g, j, jj, i, b, k
      real(dp) :: alpha, tail, beta, w, inv, t, d
      k = min(m, n)
      do g = 1, ng
         !$omp simd private(j, jj, i, alpha, tail, beta, w, inv, t, d)
         do b = 1, VW
            do j = 1, k
               alpha = a(b, j, j, g)
               tail = 0.0_dp
               do i = j + 1, m
                  tail = tail + a(b, i, j, g)**2
               end do
               beta = -sign(sqrt(alpha*alpha + tail), alpha)
               t = merge(1.0_dp, 0.0_dp, tail > 0.0_dp)     ! reflector generated?
               d = merge(beta, 1.0_dp, tail > 0.0_dp)       ! avoid /0 in dead lane
               tau(b, j, g) = t*(beta - alpha)/d
               inv = t/merge(alpha - beta, 1.0_dp, tail > 0.0_dp)
               a(b, j, j, g) = merge(beta, alpha, tail > 0.0_dp)
               do i = j + 1, m
                  a(b, i, j, g) = a(b, i, j, g)*inv
               end do
               do jj = j + 1, n
                  w = a(b, j, jj, g)
                  do i = j + 1, m
                     w = w + a(b, i, j, g)*a(b, i, jj, g)
                  end do
                  w = tau(b, j, g)*w
                  a(b, j, jj, g) = a(b, j, jj, g) - w
                  do i = j + 1, m
                     a(b, i, jj, g) = a(b, i, jj, g) - w*a(b, i, j, g)
                  end do
               end do
            end do
         end do
      end do
   end subroutine geqr2_outer

   ! =====================================================================
   !  Variant 3: Fortran whole-array syntax over the 8 lanes A(:,i,j,g).
   ! =====================================================================

   subroutine geqr2_array(ng, m, n, a, tau)
      integer, intent(in) :: ng, m, n
      real(dp), intent(inout) :: a(VW, m, n, ng)
      real(dp), intent(out) :: tau(VW, min(m, n), ng)
      real(dp) :: alpha(VW), tail(VW), beta(VW), w(VW), inv(VW)
      logical :: msk(VW)
      integer :: g, j, jj, i, k
      k = min(m, n)
      do g = 1, ng
         do j = 1, k
            alpha = a(:, j, j, g)
            tail = 0.0_dp
            do i = j + 1, m
               tail = tail + a(:, i, j, g)**2
            end do
            beta = -sign(sqrt(alpha**2 + tail), alpha)
            msk = tail > 0.0_dp
            tau(:, j, g) = merge((beta - alpha)/merge(beta, 1.0_dp, msk), 0.0_dp, msk)
            inv = merge(1.0_dp/merge(alpha - beta, 1.0_dp, msk), 0.0_dp, msk)
            a(:, j, j, g) = merge(beta, alpha, msk)
            do i = j + 1, m
               a(:, i, j, g) = a(:, i, j, g)*inv
            end do
            do jj = j + 1, n
               w = a(:, j, jj, g)
               do i = j + 1, m
                  w = w + a(:, i, j, g)*a(:, i, jj, g)
               end do
               w = tau(:, j, g)*w
               a(:, j, jj, g) = a(:, j, jj, g) - w
               do i = j + 1, m
                  a(:, i, jj, g) = a(:, i, jj, g) - w*a(:, i, j, g)
               end do
            end do
         end do
      end do
   end subroutine geqr2_array

   ! =====================================================================
   !  Variant 4: the vec8 derived type with overloaded operators.
   ! =====================================================================

   subroutine geqr2_vtype(ng, m, n, a, tau)
      integer, intent(in) :: ng, m, n
      real(dp), intent(inout) :: a(VW, m, n, ng)
      real(dp), intent(out) :: tau(VW, min(m, n), ng)
      type(vec8) :: av(m, n), alpha, tail, beta, w, inv, one, zero
      logical :: msk(VW)
      integer :: g, j, jj, i, k
      k = min(m, n)
      one = splat(1.0_dp)
      zero = splat(0.0_dp)
      do g = 1, ng
         do j = 1, n
            do i = 1, m
               av(i, j) = load8(a(:, i, j, g))
            end do
         end do
         do j = 1, k
            alpha = av(j, j)
            tail = zero
            do i = j + 1, m
               tail = tail + av(i, j)*av(i, j)
            end do
            beta = vsqrt(alpha*alpha + tail)
            beta%v = -sign(beta%v, alpha%v)
            msk = tail%v > 0.0_dp
            w = vmerge((beta - alpha)/vmerge(beta, one, msk), zero, msk)   ! tau
            call store8(tau(:, j, g), w)
            inv = vmerge(one/vmerge(alpha - beta, one, msk), zero, msk)
            av(j, j) = vmerge(beta, alpha, msk)
            do i = j + 1, m
               av(i, j) = av(i, j)*inv
            end do
            do jj = j + 1, n
               w = av(j, jj)                       ! w = v^T c  (v = [1; av(j+1:,j)])
               do i = j + 1, m
                  w = w + av(i, j)*av(i, jj)
               end do
               w = load8(tau(:, j, g))*w           ! w = tau * (v^T c)
               av(j, jj) = av(j, jj) - w
               do i = j + 1, m
                  av(i, jj) = av(i, jj) - w*av(i, j)
               end do
            end do
         end do
         do j = 1, n
            do i = 1, m
               call store8(a(:, i, j, g), av(i, j))
            end do
         end do
      end do
   end subroutine geqr2_vtype

   ! =====================================================================
   !  orm2r: B := Q^T B, four variants (same taxonomy as above).
   ! =====================================================================

   subroutine orm2r_inner(ng, m, n, nrhs, a, tau, b)
      integer, intent(in) :: ng, m, n, nrhs
      real(dp), intent(in) :: a(VW, m, n, ng), tau(VW, min(m, n), ng)
      real(dp), intent(inout) :: b(VW, m, nrhs, ng)
      real(dp) :: w(VW)
      integer :: g, j, col, i, bb, k
      k = min(m, n)
      do g = 1, ng
         do j = 1, k
            do col = 1, nrhs
               do bb = 1, VW
                  w(bb) = b(bb, j, col, g)
               end do
               do i = j + 1, m
                  do bb = 1, VW
                     w(bb) = w(bb) + a(bb, i, j, g)*b(bb, i, col, g)
                  end do
               end do
               do bb = 1, VW
                  w(bb) = tau(bb, j, g)*w(bb)
                  b(bb, j, col, g) = b(bb, j, col, g) - w(bb)
               end do
               do i = j + 1, m
                  do bb = 1, VW
                     b(bb, i, col, g) = b(bb, i, col, g) - w(bb)*a(bb, i, j, g)
                  end do
               end do
            end do
         end do
      end do
   end subroutine orm2r_inner

   subroutine orm2r_outer(ng, m, n, nrhs, a, tau, b)
      integer, intent(in) :: ng, m, n, nrhs
      real(dp), intent(in) :: a(VW, m, n, ng), tau(VW, min(m, n), ng)
      real(dp), intent(inout) :: b(VW, m, nrhs, ng)
      integer :: g, j, col, i, bb, k
      real(dp) :: w
      k = min(m, n)
      do g = 1, ng
         !$omp simd private(j, col, i, w)
         do bb = 1, VW
            do j = 1, k
               do col = 1, nrhs
                  w = b(bb, j, col, g)
                  do i = j + 1, m
                     w = w + a(bb, i, j, g)*b(bb, i, col, g)
                  end do
                  w = tau(bb, j, g)*w
                  b(bb, j, col, g) = b(bb, j, col, g) - w
                  do i = j + 1, m
                     b(bb, i, col, g) = b(bb, i, col, g) - w*a(bb, i, j, g)
                  end do
               end do
            end do
         end do
      end do
   end subroutine orm2r_outer

   subroutine orm2r_array(ng, m, n, nrhs, a, tau, b)
      integer, intent(in) :: ng, m, n, nrhs
      real(dp), intent(in) :: a(VW, m, n, ng), tau(VW, min(m, n), ng)
      real(dp), intent(inout) :: b(VW, m, nrhs, ng)
      real(dp) :: w(VW)
      integer :: g, j, col, i, k
      k = min(m, n)
      do g = 1, ng
         do j = 1, k
            do col = 1, nrhs
               w = b(:, j, col, g)
               do i = j + 1, m
                  w = w + a(:, i, j, g)*b(:, i, col, g)
               end do
               w = tau(:, j, g)*w
               b(:, j, col, g) = b(:, j, col, g) - w
               do i = j + 1, m
                  b(:, i, col, g) = b(:, i, col, g) - w*a(:, i, j, g)
               end do
            end do
         end do
      end do
   end subroutine orm2r_array

   subroutine orm2r_vtype(ng, m, n, nrhs, a, tau, b)
      integer, intent(in) :: ng, m, n, nrhs
      real(dp), intent(in) :: a(VW, m, n, ng), tau(VW, min(m, n), ng)
      real(dp), intent(inout) :: b(VW, m, nrhs, ng)
      type(vec8) :: av(m, n), bv(m, nrhs), tv(min(m, n)), w
      integer :: g, j, col, i, k
      k = min(m, n)
      do g = 1, ng
         do j = 1, n
            do i = 1, m
               av(i, j) = load8(a(:, i, j, g))
            end do
         end do
         do j = 1, k
            tv(j) = load8(tau(:, j, g))
         end do
         do col = 1, nrhs
            do i = 1, m
               bv(i, col) = load8(b(:, i, col, g))
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
               call store8(b(:, i, col, g), bv(i, col))
            end do
         end do
      end do
   end subroutine orm2r_vtype

end module batched_qr
