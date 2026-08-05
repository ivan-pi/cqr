!> Correctness + throughput benchmark for the four batched-QR expression styles,
!> with the batch interleaved in groups of a FIXED compile-time width VW = 8
!> (layout A(VW,m,n,ng)). For each problem size we:
!>   * build a random, well-conditioned batch,
!>   * establish an oracle with the scalar per-matrix reference, self-checked
!>     via || Q^T A - R ||,
!>   * verify all four geqr2 and all four orm2r variants elementwise vs it,
!>   * time each variant (copy cost measured and subtracted) and report GFLOP/s.
program bench
   use, intrinsic :: iso_fortran_env, only: dp => real64
   use omp_lib, only: omp_get_wtime
   use vec8_mod, only: VW
   use batched_qr
   implicit none

   integer, parameter :: NS = 6
   real(dp), parameter :: BUDGET = 0.25_dp   ! wall-clock seconds per measured kernel
   integer :: sizes(4, NS)   ! rows: m, n, nrhs, ng   (ng = number of 8-wide groups)
   integer :: s
   real(dp) :: sink

   ! m,   n, nrhs,  ng    (total matrices = VW*ng)
   sizes(:, 1) = [8, 8, 8, 1024]
   sizes(:, 2) = [16, 16, 8, 512]
   sizes(:, 3) = [32, 16, 8, 256]    ! tall
   sizes(:, 4) = [32, 32, 8, 256]
   sizes(:, 5) = [48, 48, 8, 128]
   sizes(:, 6) = [64, 64, 8, 64]

   sink = 0.0_dp

   print '(a)', '=================================================================='
   print '(a,i0,a)', ' Batched QR: 4 expression styles   (fixed width VW = ', VW, ')'
   print '(a)', ' layout A(VW,m,n,ng), 8-lane batch fastest -- one AVX-512 register'
   print '(a)', '=================================================================='

   call check_all()

   do s = 1, NS
      call run_size(sizes(1, s), sizes(2, s), sizes(3, s), sizes(4, s), sink)
   end do

   print '(a)', ''
   print '(a,es12.4)', ' (anti-DCE checksum: ', sink, ')'

contains

   ! ------------------------------------------------------------------
   !  Correctness: every variant vs the scalar oracle, at every size.
   ! ------------------------------------------------------------------
   subroutine check_all()
      integer :: s, m, n, nrhs, ng
      print '(a)', ''
      print '(a)', ' CORRECTNESS (max abs elementwise error vs scalar reference)'
      print '(a)', ' -----------------------------------------------------------------'
      print '(a)', '   m    n nrhs     ng | ref-selfchk |  geqr2 (max err)  | orm2r (max err)'
      do s = 1, NS
         m = sizes(1, s); n = sizes(2, s); nrhs = sizes(3, s); ng = sizes(4, s)
         call check_size(m, n, nrhs, ng)
      end do
   end subroutine check_all

   subroutine check_size(m, n, nrhs, ng)
      integer, intent(in) :: m, n, nrhs, ng
      real(dp), allocatable :: a0(:, :, :, :), aref(:, :, :, :), tref(:, :, :)
      real(dp), allocatable :: b0(:, :, :, :), bref(:, :, :, :)
      real(dp), allocatable :: ac(:, :, :, :), tc(:, :, :), bc(:, :, :, :)
      real(dp) :: aa(m, n), tt(min(m, n)), qta(m, n)
      real(dp) :: self_err, anorm, eg, eo
      integer :: g, bb, i, j, k
      k = min(m, n)
      allocate (a0(VW, m, n, ng), aref(VW, m, n, ng), tref(VW, k, ng))
      allocate (b0(VW, m, nrhs, ng), bref(VW, m, nrhs, ng))
      allocate (ac(VW, m, n, ng), tc(VW, k, ng), bc(VW, m, nrhs, ng))

      call fill_random(a0, m, n, ng)
      call fill_rhs(b0, m, nrhs, ng)

      ! Oracle + semantic self-check ||Q^T A - R||.
      self_err = 0.0_dp
      do g = 1, ng
         do bb = 1, VW
            aa = a0(bb, :, :, g)
            call ref_geqr2(m, n, aa, tt)
            aref(bb, :, :, g) = aa
            tref(bb, :, g) = tt
            qta = a0(bb, :, :, g)
            call ref_orm2r(m, n, n, aa, tt, qta)   ! Q^T A0 must equal R = triu(aa)
            anorm = maxval(abs(a0(bb, :, :, g)))
            do j = 1, n
               do i = 1, m
                  if (i <= j) then
                     self_err = max(self_err, abs(qta(i, j) - aa(i, j))/anorm)
                  else
                     self_err = max(self_err, abs(qta(i, j))/anorm)
                  end if
               end do
            end do
            bref(bb, :, :, g) = b0(bb, :, :, g)
            call ref_orm2r(m, n, nrhs, aa, tt, bref(bb, :, :, g))
         end do
      end do

      ! ---- geqr2 variants ----
      eg = 0.0_dp
      ac = a0; call geqr2_inner(ng, m, n, ac, tc); eg = max(eg, gdiff(ac, aref, tc, tref))
      ac = a0; call geqr2_outer(ng, m, n, ac, tc); eg = max(eg, gdiff(ac, aref, tc, tref))
      ac = a0; call geqr2_array(ng, m, n, ac, tc); eg = max(eg, gdiff(ac, aref, tc, tref))
      ac = a0; call geqr2_vtype(ng, m, n, ac, tc); eg = max(eg, gdiff(ac, aref, tc, tref))

      ! ---- orm2r variants (use oracle factors) ----
      eo = 0.0_dp
      bc = b0; call orm2r_inner(ng, m, n, nrhs, aref, tref, bc); eo = max(eo, maxval(abs(bc - bref)))
      bc = b0; call orm2r_outer(ng, m, n, nrhs, aref, tref, bc); eo = max(eo, maxval(abs(bc - bref)))
      bc = b0; call orm2r_array(ng, m, n, nrhs, aref, tref, bc); eo = max(eo, maxval(abs(bc - bref)))
      bc = b0; call orm2r_vtype(ng, m, n, nrhs, aref, tref, bc); eo = max(eo, maxval(abs(bc - bref)))

      print '(i4,i5,i5,i7,a,es11.3,a,es11.3,a,es11.3,a)', &
         m, n, nrhs, ng, ' | ', self_err, ' | ', eg, '       | ', eo, &
         merge('  [PASS]', '  [FAIL]', self_err < 1e-11_dp .and. eg < 1e-9_dp .and. eo < 1e-9_dp)
   end subroutine check_size

   !> Max elementwise diff of (A,tau) vs oracle.
   real(dp) function gdiff(ac, aref, tc, tref)
      real(dp), intent(in) :: ac(:, :, :, :), aref(:, :, :, :), tc(:, :, :), tref(:, :, :)
      gdiff = max(maxval(abs(ac - aref)), maxval(abs(tc - tref)))
   end function gdiff

   ! ------------------------------------------------------------------
   !  Timing.
   ! ------------------------------------------------------------------
   subroutine run_size(m, n, nrhs, ng, sink)
      integer, intent(in) :: m, n, nrhs, ng
      real(dp), intent(inout) :: sink
      real(dp), allocatable :: a0(:, :, :, :), aref(:, :, :, :), tref(:, :, :)
      real(dp), allocatable :: b0(:, :, :, :)
      real(dp), allocatable :: ac(:, :, :, :), tc(:, :, :), bc(:, :, :, :)
      real(dp) :: aa(m, n), tt(min(m, n))
      real(dp) :: fg, fo, tg(4), to(4), nmat
      integer :: g, bb, k
      character(len=7) :: names(4)
      names = ['inner  ', 'outer  ', 'array  ', 'vtype  ']
      k = min(m, n)
      allocate (a0(VW, m, n, ng), aref(VW, m, n, ng), tref(VW, k, ng))
      allocate (b0(VW, m, nrhs, ng), ac(VW, m, n, ng), tc(VW, k, ng), bc(VW, m, nrhs, ng))
      call fill_random(a0, m, n, ng)
      call fill_rhs(b0, m, nrhs, ng)
      do g = 1, ng
         do bb = 1, VW
            aa = a0(bb, :, :, g)
            call ref_geqr2(m, n, aa, tt)
            aref(bb, :, :, g) = aa
            tref(bb, :, g) = tt
         end do
      end do

      nmat = real(VW, dp)*real(ng, dp)
      fg = nmat*(2.0_dp*m*n*n - (2.0_dp/3.0_dp)*n**3)          ! geqr2, m>=n
      if (m < n) fg = nmat*(2.0_dp*m*m*n - (2.0_dp/3.0_dp)*m**3)
      fo = nmat*nrhs*(4.0_dp*m*k - 2.0_dp*real(k, dp)**2)      ! orm2r

      tg(1) = time_geqr2(1, ng, m, n, a0, ac, tc, sink)
      tg(2) = time_geqr2(2, ng, m, n, a0, ac, tc, sink)
      tg(3) = time_geqr2(3, ng, m, n, a0, ac, tc, sink)
      tg(4) = time_geqr2(4, ng, m, n, a0, ac, tc, sink)
      to(1) = time_orm2r(1, ng, m, n, nrhs, b0, aref, tref, bc, sink)
      to(2) = time_orm2r(2, ng, m, n, nrhs, b0, aref, tref, bc, sink)
      to(3) = time_orm2r(3, ng, m, n, nrhs, b0, aref, tref, bc, sink)
      to(4) = time_orm2r(4, ng, m, n, nrhs, b0, aref, tref, bc, sink)

      print '(a)', ''
      print '(a,i0,a,i0,a,i0,a,i0,a,i0,a)', &
         ' m=', m, ' n=', n, ' nrhs=', nrhs, ' ng=', ng, ' (', nint(nmat), ' matrices)'
      print '(a)', '   variant   |  dgeqrf GFLOP/s (time ms) |  dormqr GFLOP/s (time ms)'
      call prow(names(1), fg, tg(1), fo, to(1))
      call prow(names(2), fg, tg(2), fo, to(2))
      call prow(names(3), fg, tg(3), fo, to(3))
      call prow(names(4), fg, tg(4), fo, to(4))
      call winner(names, tg, to)
   end subroutine run_size

   subroutine prow(name, fg, tg, fo, to)
      character(len=*), intent(in) :: name
      real(dp), intent(in) :: fg, tg, fo, to
      print '(3x,a7,a,f9.2,a,f8.3,a,f9.2,a,f8.3,a)', &
         name, ' | ', gflops(fg, tg), ' (', 1.0e3_dp*tg, ')     | ', &
         gflops(fo, to), ' (', 1.0e3_dp*to, ')'
   end subroutine prow

   real(dp) function gflops(f, t)
      real(dp), intent(in) :: f, t
      if (t <= 0.0_dp) then
         gflops = 0.0_dp
      else
         gflops = f/t*1.0e-9_dp
      end if
   end function gflops

   subroutine winner(names, tg, to)
      character(len=7), intent(in) :: names(4)
      real(dp), intent(in) :: tg(4), to(4)
      integer :: ig, io
      ig = minloc(tg, dim=1); io = minloc(to, dim=1)
      print '(a,a7,a,a7)', '   fastest  ->   dgeqrf: ', names(ig), '     dormqr: ', names(io)
   end subroutine winner

   ! ------------------------------------------------------------------
   !  Timers: adaptive reps to a wall budget; per-rep copy cost subtracted.
   ! ------------------------------------------------------------------
   real(dp) function time_geqr2(which, ng, m, n, a0, ac, tc, sink) result(kt)
      integer, intent(in) :: which, ng, m, n
      real(dp), intent(in) :: a0(VW, m, n, ng)
      real(dp), intent(inout) :: ac(VW, m, n, ng), tc(VW, min(m, n), ng), sink
      real(dp) :: t0, tfull, tcopy
      integer :: r, reps
      reps = 0
      t0 = omp_get_wtime()
      do
         ac = a0
         select case (which)
         case (1); call geqr2_inner(ng, m, n, ac, tc)
         case (2); call geqr2_outer(ng, m, n, ac, tc)
         case (3); call geqr2_array(ng, m, n, ac, tc)
         case (4); call geqr2_vtype(ng, m, n, ac, tc)
         end select
         sink = sink + ac(1, 1, 1, 1) + tc(1, 1, 1)
         reps = reps + 1
         if (omp_get_wtime() - t0 > BUDGET .and. reps >= 3) exit
      end do
      tfull = omp_get_wtime() - t0
      t0 = omp_get_wtime()
      do r = 1, reps
         ac = a0
         sink = sink + ac(1, 1, 1, 1)
      end do
      tcopy = omp_get_wtime() - t0
      kt = max(tfull - tcopy, 1.0e-9_dp)/real(reps, dp)
   end function time_geqr2

   real(dp) function time_orm2r(which, ng, m, n, nrhs, b0, aref, tref, bc, sink) result(kt)
      integer, intent(in) :: which, ng, m, n, nrhs
      real(dp), intent(in) :: b0(VW, m, nrhs, ng), aref(VW, m, n, ng), tref(VW, min(m, n), ng)
      real(dp), intent(inout) :: bc(VW, m, nrhs, ng), sink
      real(dp) :: t0, tfull, tcopy
      integer :: r, reps
      reps = 0
      t0 = omp_get_wtime()
      do
         bc = b0
         select case (which)
         case (1); call orm2r_inner(ng, m, n, nrhs, aref, tref, bc)
         case (2); call orm2r_outer(ng, m, n, nrhs, aref, tref, bc)
         case (3); call orm2r_array(ng, m, n, nrhs, aref, tref, bc)
         case (4); call orm2r_vtype(ng, m, n, nrhs, aref, tref, bc)
         end select
         sink = sink + bc(1, 1, 1, 1)
         reps = reps + 1
         if (omp_get_wtime() - t0 > BUDGET .and. reps >= 3) exit
      end do
      tfull = omp_get_wtime() - t0
      t0 = omp_get_wtime()
      do r = 1, reps
         bc = b0
         sink = sink + bc(1, 1, 1, 1)
      end do
      tcopy = omp_get_wtime() - t0
      kt = max(tfull - tcopy, 1.0e-9_dp)/real(reps, dp)
   end function time_orm2r

   ! ------------------------------------------------------------------
   !  Random, well-conditioned data (fixed seed for reproducibility).
   ! ------------------------------------------------------------------
   subroutine fill_random(a, m, n, ng)
      integer, intent(in) :: m, n, ng
      real(dp), intent(out) :: a(VW, m, n, ng)
      integer :: i
      call seed_fixed()
      call random_number(a)
      a = a - 0.5_dp
      do i = 1, min(m, n)      ! diagonal boost -> well conditioned
         a(:, i, i, :) = a(:, i, i, :) + real(n, dp)
      end do
   end subroutine fill_random

   subroutine fill_rhs(b, m, nrhs, ng)
      integer, intent(in) :: m, nrhs, ng
      real(dp), intent(out) :: b(VW, m, nrhs, ng)
      call random_number(b)
      b = b - 0.5_dp
   end subroutine fill_rhs

   subroutine seed_fixed()
      integer :: ns
      integer, allocatable :: sd(:)
      call random_seed(size=ns)
      allocate (sd(ns))
      sd = 20260804
      call random_seed(put=sd)
   end subroutine seed_fixed

end program bench
