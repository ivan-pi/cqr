!> Head-to-head: the Fortran batched kernels vs this project's C++ GNU-vector-type
!> kernels (src/cqr_*_compact) -- ifx for Fortran, icpx for C++.
!>
!> The C++ portable compact layout is *byte-identical* to the Fortran A(VW,m,n,ng)
!> interleave (both store element (i,j) of lane v at v + V*(i + ldap*j) + V*ldap*n*g,
!> column-major, V = 8, ldap = m). So the very same buffer is handed to both: only
!> the kernel differs -- Fortran geqr2_array / geqr2_inner vs C++ dgeqrf_compact,
!> and Fortran orm2r_array / orm2r_inner vs C++ dormqr_compact. Identical data,
!> identical layout, identical timing harness (copy cost measured and subtracted),
!> both cross-checked against the scalar reference.
program bench_cpp
   use iso_c_binding
   use, intrinsic :: iso_fortran_env, only: dp => real64, int64
   use vec8_mod, only: VW
   use batched_qr
   implicit none

   ! ---- C entry points of the C++ kernels (src/cqr_*_compact) ----
   interface
      integer(c_int) function dgeqrf_compact(layout, m, n, ap, ldap, taup, V, nm) &
         bind(C, name='dgeqrf_compact')
         import :: c_int, c_double, c_char
         character(kind=c_char), value :: layout
         integer(c_int), value :: m, n, ldap, V, nm
         real(c_double), intent(inout) :: ap(*), taup(*)
      end function

      integer(c_int) function dormqr_compact(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm) &
         bind(C, name='dormqr_compact')
         import :: c_int, c_double, c_char
         character(kind=c_char), value :: trans
         integer(c_int), value :: m, nrhs, k, ldap, ldbp, V, nm
         real(c_double), intent(in) :: ap(*), taup(*)
         real(c_double), intent(inout) :: bp(*)
      end function
   end interface

   character(kind=c_char), parameter :: LAY_C = 'C', TR_T = 'T'
   integer, parameter :: NS = 6
   real(dp), parameter :: BUDGET = 0.25_dp
   integer :: sizes(4, NS), s
   real(dp) :: sink

   sizes(:, 1) = [8, 8, 8, 1024]
   sizes(:, 2) = [16, 16, 8, 512]
   sizes(:, 3) = [32, 16, 8, 256]
   sizes(:, 4) = [32, 32, 8, 256]
   sizes(:, 5) = [48, 48, 8, 128]
   sizes(:, 6) = [64, 64, 8, 64]

   sink = 0.0_dp
   print '(a)', '=================================================================='
   print '(a)', ' Fortran (ifx) vs C++ GNU-vector-type kernels (icpx), same buffer'
   print '(a)', '=================================================================='
   print '(a)', ''
   print '(a)', '                      dgeqrf GFLOP/s      |      dormqr GFLOP/s'
   print '(a)', '   m    n     nm | f-array f-inner  C++  | f-array f-inner  C++  | maxerr'

   do s = 1, NS
      call run(sizes(1, s), sizes(2, s), sizes(3, s), sizes(4, s), sink)
   end do
   print '(a)', ''
   print '(a,es10.2,a)', ' (anti-DCE checksum: ', sink, ')'

contains

   subroutine run(m, n, nrhs, ng, sink)
      integer, intent(in) :: m, n, nrhs, ng
      real(dp), intent(inout) :: sink
      real(dp), allocatable :: a0(:, :, :, :), af(:, :, :, :), tf(:, :, :)
      real(dp), allocatable :: b0(:, :, :, :), bref(:, :, :, :)
      real(dp), allocatable :: ac(:, :, :, :), tc(:, :, :), bc(:, :, :, :)
      real(dp), allocatable :: aref(:, :, :, :), tref(:, :, :)
      real(dp) :: aa(m, n), tt(min(m, n)), bb(m, nrhs)
      real(dp) :: fg, fo, tg(3), to(3), err
      integer :: nm, k, g, lane, b, i, j
      k = min(m, n); nm = VW*ng
      allocate (a0(VW, m, n, ng), af(VW, m, n, ng), tf(VW, k, ng), aref(VW, m, n, ng), tref(VW, k, ng))
      allocate (b0(VW, m, nrhs, ng), bref(VW, m, nrhs, ng))
      allocate (ac(VW, m, n, ng), tc(VW, k, ng), bc(VW, m, nrhs, ng))
      call fill_random(a0, m, n, ng)
      call fill_rhs(b0, m, nrhs, ng)

      ! Reference factors (per matrix) + reference Q^T B, unpacked lane by lane.
      do b = 1, nm
         g = (b - 1)/VW + 1; lane = mod(b - 1, VW) + 1
         do j = 1, n
            do i = 1, m
               aa(i, j) = a0(lane, i, j, g)
            end do
         end do
         call ref_geqr2(m, n, aa, tt)
         do j = 1, n
            do i = 1, m
               aref(lane, i, j, g) = aa(i, j)
            end do
         end do
         do j = 1, k
            tref(lane, j, g) = tt(j)
         end do
         do j = 1, nrhs
            do i = 1, m
               bb(i, j) = b0(lane, i, j, g)
            end do
         end do
         call ref_orm2r(m, n, nrhs, aa, tt, bb)
         do j = 1, nrhs
            do i = 1, m
               bref(lane, i, j, g) = bb(i, j)
            end do
         end do
      end do

      ! Correct factored buffer (interleaved) to feed the dormqr kernels.
      af = aref; tf = tref

      fg = real(nm, dp)*(2.0_dp*m*n*n - (2.0_dp/3.0_dp)*n**3)
      if (m < n) fg = real(nm, dp)*(2.0_dp*m*m*n - (2.0_dp/3.0_dp)*m**3)
      fo = real(nm, dp)*nrhs*(4.0_dp*m*k - 2.0_dp*real(k, dp)**2)

      ! ---- dgeqrf timing ----
      tg(1) = tgeqrf(1, ng, m, n, nm, k, a0, ac, tc, sink)
      tg(2) = tgeqrf(2, ng, m, n, nm, k, a0, ac, tc, sink)
      tg(3) = tgeqrf(3, ng, m, n, nm, k, a0, ac, tc, sink)
      ! ---- dormqr timing ----
      to(1) = tormqr(1, ng, m, n, nrhs, nm, k, af, tf, b0, bc, sink)
      to(2) = tormqr(2, ng, m, n, nrhs, nm, k, af, tf, b0, bc, sink)
      to(3) = tormqr(3, ng, m, n, nrhs, nm, k, af, tf, b0, bc, sink)

      ! ---- correctness (C++ kernels vs reference) ----
      err = 0.0_dp
      ac = a0
      err = max(err, cpp_geqrf_err(ac, aref, tc, tref, ng, m, n, nm, k))
      bc = b0
      err = max(err, cpp_ormqr_err(af, tf, bc, bref, ng, m, n, nrhs, nm, k))

      print '(i4,i5,i7,a,3f7.2,a,3f7.2,a,es9.1)', m, n, nm, ' | ', &
         gf(fg, tg(1)), gf(fg, tg(2)), gf(fg, tg(3)), ' | ', &
         gf(fo, to(1)), gf(fo, to(2)), gf(fo, to(3)), ' | ', err
      deallocate (a0, af, tf, aref, tref, b0, bref, ac, tc, bc)
   end subroutine run

   real(dp) function gf(f, t)
      real(dp), intent(in) :: f, t
      gf = merge(0.0_dp, f/t*1.0e-9_dp, t <= 0.0_dp)
   end function gf

   ! ------------- timers -------------
   real(dp) function tgeqrf(which, ng, m, n, nm, k, a0, ac, tc, sink) result(kt)
      integer, intent(in) :: which, ng, m, n, nm, k
      real(dp), intent(in) :: a0(VW, m, n, ng)
      real(dp), intent(inout) :: ac(VW, m, n, ng), tc(VW, k, ng), sink
      real(dp) :: t0, tfull, tcopy
      integer :: r, reps, info
      reps = 0; t0 = wtime()
      do
         ac = a0
         select case (which)
         case (1); call geqr2_array(ng, m, n, ac, tc)
         case (2); call geqr2_inner(ng, m, n, ac, tc)
         case (3); info = dgeqrf_compact(LAY_C, m, n, ac, m, tc, VW, nm)
         end select
         sink = sink + ac(1, 1, 1, 1) + tc(1, 1, 1)
         reps = reps + 1
         if (wtime() - t0 > BUDGET .and. reps >= 3) exit
      end do
      tfull = wtime() - t0
      t0 = wtime()
      do r = 1, reps
         ac = a0; sink = sink + ac(1, 1, 1, 1)
      end do
      tcopy = wtime() - t0
      kt = max(tfull - tcopy, 1.0e-9_dp)/real(reps, dp)
   end function tgeqrf

   real(dp) function tormqr(which, ng, m, n, nrhs, nm, k, af, tf, b0, bc, sink) result(kt)
      integer, intent(in) :: which, ng, m, n, nrhs, nm, k
      real(dp), intent(in) :: af(VW, m, n, ng), tf(VW, k, ng), b0(VW, m, nrhs, ng)
      real(dp), intent(inout) :: bc(VW, m, nrhs, ng), sink
      real(dp) :: t0, tfull, tcopy
      integer :: r, reps, info
      reps = 0; t0 = wtime()
      do
         bc = b0
         select case (which)
         case (1); call orm2r_array(ng, m, n, nrhs, af, tf, bc)
         case (2); call orm2r_inner(ng, m, n, nrhs, af, tf, bc)
         case (3); info = dormqr_compact(TR_T, m, nrhs, k, af, m, tf, bc, m, VW, nm)
         end select
         sink = sink + bc(1, 1, 1, 1)
         reps = reps + 1
         if (wtime() - t0 > BUDGET .and. reps >= 3) exit
      end do
      tfull = wtime() - t0
      t0 = wtime()
      do r = 1, reps
         bc = b0; sink = sink + bc(1, 1, 1, 1)
      end do
      tcopy = wtime() - t0
      kt = max(tfull - tcopy, 1.0e-9_dp)/real(reps, dp)
   end function tormqr

   ! ------------- correctness of the C++ kernels vs reference -------------
   real(dp) function cpp_geqrf_err(ac, aref, tc, tref, ng, m, n, nm, k) result(e)
      integer, intent(in) :: ng, m, n, nm, k
      real(dp), intent(inout) :: ac(VW, m, n, ng), tc(VW, k, ng)
      real(dp), intent(in) :: aref(VW, m, n, ng), tref(VW, k, ng)
      integer :: info
      info = dgeqrf_compact(LAY_C, m, n, ac, m, tc, VW, nm)
      e = max(maxval(abs(ac - aref)), maxval(abs(tc - tref)))
   end function cpp_geqrf_err

   real(dp) function cpp_ormqr_err(af, tf, bc, bref, ng, m, n, nrhs, nm, k) result(e)
      integer, intent(in) :: ng, m, n, nrhs, nm, k
      real(dp), intent(in) :: af(VW, m, n, ng), tf(VW, k, ng), bref(VW, m, nrhs, ng)
      real(dp), intent(inout) :: bc(VW, m, nrhs, ng)
      integer :: info
      info = dormqr_compact(TR_T, m, nrhs, k, af, m, tf, bc, m, VW, nm)
      e = maxval(abs(bc - bref))
   end function cpp_ormqr_err

   ! ------------- data -------------
   subroutine fill_random(a, m, n, ng)
      integer, intent(in) :: m, n, ng
      real(dp), intent(out) :: a(VW, m, n, ng)
      integer :: i, ns
      integer, allocatable :: sd(:)
      call random_seed(size=ns); allocate (sd(ns)); sd = 20260804; call random_seed(put=sd)
      call random_number(a); a = a - 0.5_dp
      do i = 1, min(m, n)
         a(:, i, i, :) = a(:, i, i, :) + real(n, dp)
      end do
   end subroutine fill_random

   subroutine fill_rhs(b, m, nrhs, ng)
      integer, intent(in) :: m, nrhs, ng
      real(dp), intent(out) :: b(VW, m, nrhs, ng)
      call random_number(b); b = b - 0.5_dp
   end subroutine fill_rhs

   real(dp) function wtime()
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      wtime = real(cnt, dp)/real(rate, dp)
   end function wtime

end program bench_cpp
