!> Head-to-head: the fastest Fortran batched dgeqrf (geqr2_array / geqr2_inner)
!> vs Intel MKL's native `mkl_dgeqrf_compact`, on identical matrices.
!>
!> MKL's compact buffer layout is opaque, so the same random batch is fed to MKL
!> through its own `mkl_dgepack_compact` and to the Fortran kernels through the
!> A(VW,m,n,ng) interleave -- both derived from one master set of matrices. Only
!> the factorization call is timed (pack/copy cost measured and subtracted), and
!> MKL's result is unpacked and checked against the scalar reference so the
!> harness is validated, not just fast.
!>
!> Single-threaded (link sequential MKL; run with MKL_NUM_THREADS=1): the Fortran
!> kernels are single-thread SIMD, so this compares like with like.
!>
!> NOTE: MKL provides compact `geqrf` but no compact `ormqr`, so only the
!> factorization is compared here; the apply-Q^T step (dormqr) has no native-MKL
!> compact counterpart -- which is exactly the gap this project's C++ side fills.
program bench_mkl
   use iso_c_binding
   use, intrinsic :: iso_fortran_env, only: dp => real64, int64
   use vec8_mod, only: VW
   use batched_qr, only: ref_geqr2, geqr2_array, geqr2_inner
   use mkl_compact
   implicit none

   integer, parameter :: NS = 6
   real(dp), parameter :: BUDGET = 0.25_dp
   integer :: sizes(3, NS)   ! m, n, ng
   integer :: s
   real(dp) :: sink

   sizes(:, 1) = [8, 8, 1024]
   sizes(:, 2) = [16, 16, 512]
   sizes(:, 3) = [32, 16, 256]
   sizes(:, 4) = [32, 32, 256]
   sizes(:, 5) = [48, 48, 128]
   sizes(:, 6) = [64, 64, 64]

   sink = 0.0_dp

   print '(a)', '=================================================================='
   print '(a)', ' dgeqrf: fastest Fortran vs MKL mkl_dgeqrf_compact (single core)'
   print '(a,i0,a)', ' MKL compact format = ', mkl_get_format_compact(), &
      '  (183 = AVX-512, V = 8)'
   print '(a)', '=================================================================='
   print '(a)', ''
   print '(a)', '   m    n     nm |   MKL   f-array  f-inner |  err(MKL) err(fort) | array/MKL'

   do s = 1, NS
      call run(sizes(1, s), sizes(2, s), sizes(3, s), sink)
   end do
   print '(a)', ''
   print '(a,es10.2,a)', ' (anti-DCE checksum: ', sink, ')'

contains

   subroutine run(m, n, ng, sink)
      integer, intent(in) :: m, n, ng
      real(dp), intent(inout) :: sink
      integer :: nm, k, b, g, lane, i, j
      real(c_double), allocatable, target :: Aall(:, :, :), Href(:, :, :), Hmkl(:, :, :)
      type(c_ptr), allocatable :: aptr(:), hptr(:)
      real(c_double), allocatable :: ap0(:), ap(:), taup(:), work(:)
      real(dp), allocatable :: A0(:, :, :, :), ac(:, :, :, :), tc(:, :, :)
      real(dp) :: tt(min(m, n)), wq(1)
      integer(c_int) :: fmt, szA, szT, nA, nT, info, lwork
      real(dp) :: t_mkl, t_arr, t_inn, fg, emkl, efort

      k = min(m, n)
      nm = VW*ng
      fmt = mkl_get_format_compact()

      ! ---- master matrices (per-matrix, column-major) + reference factors ----
      allocate (Aall(m, n, nm), Href(m, n, nm), Hmkl(m, n, nm))
      allocate (aptr(nm), hptr(nm))
      call fill_master(Aall, m, n, nm)
      do b = 1, nm
         Href(:, :, b) = Aall(:, :, b)
         call ref_geqr2(m, n, Href(:, :, b), tt)
      end do
      do b = 1, nm
         aptr(b) = c_loc(Aall(1, 1, b))
         hptr(b) = c_loc(Hmkl(1, 1, b))
      end do

      ! ---- MKL: pack pristine input, size workspace ----
      szA = mkl_dget_size_compact(m, n, fmt, nm)      ! bytes
      szT = mkl_dget_size_compact(k, 1, fmt, nm)
      nA = szA/8; nT = szT/8
      allocate (ap0(nA), ap(nA), taup(nT))
      call mkl_dgepack_compact(MKL_COL_MAJOR, m, n, aptr, m, ap0, m, fmt, nm)
      call mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap0, m, taup, wq, -1, info, fmt, nm)
      lwork = max(1, nint(wq(1)))
      allocate (work(lwork))

      ! ---- Fortran: interleaved layout from the same master matrices ----
      allocate (A0(VW, m, n, ng), ac(VW, m, n, ng), tc(VW, k, ng))
      do b = 1, nm
         g = (b - 1)/VW + 1; lane = mod(b - 1, VW) + 1
         do j = 1, n
            do i = 1, m
               A0(lane, i, j, g) = Aall(i, j, b)
            end do
         end do
      end do

      ! ---- time MKL (copy pristine compact buffer each rep, subtract copy) ----
      t_mkl = time_mkl(nA, m, n, ap0, ap, taup, work, lwork, info, fmt, nm, sink)
      ! ---- time the two fastest Fortran kernels ----
      t_arr = time_fort(1, ng, m, n, A0, ac, tc, sink)
      t_inn = time_fort(2, ng, m, n, A0, ac, tc, sink)

      ! ---- correctness (re-factor a clean copy: the timers leave ap pristine) ----
      ap = ap0
      call mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, work, lwork, info, fmt, nm)
      call mkl_dgeunpack_compact(MKL_COL_MAJOR, m, n, hptr, m, ap, m, fmt, nm)
      emkl = maxval(abs(Hmkl - Href))
      ac = A0; call geqr2_array(ng, m, n, ac, tc)
      efort = fort_err(ac, Href, ng, m, n, nm)

      fg = real(nm, dp)*(2.0_dp*m*n*n - (2.0_dp/3.0_dp)*n**3)
      if (m < n) fg = real(nm, dp)*(2.0_dp*m*m*n - (2.0_dp/3.0_dp)*m**3)

      print '(i4,i5,i7,a,f7.2,f8.2,f8.2,a,es9.1,es9.1,a,f7.2,a)', &
         m, n, nm, ' | ', gf(fg, t_mkl), gf(fg, t_arr), gf(fg, t_inn), &
         ' | ', emkl, efort, ' |  ', gf(fg, t_arr)/gf(fg, t_mkl), 'x'

      deallocate (Aall, Href, Hmkl, aptr, hptr, ap0, ap, taup, work, A0, ac, tc)
   end subroutine run

   real(dp) function gf(f, t)
      real(dp), intent(in) :: f, t
      gf = merge(0.0_dp, f/t*1.0e-9_dp, t <= 0.0_dp)
   end function gf

   !> Max error of an interleaved factorization vs the per-matrix reference.
   real(dp) function fort_err(ac, Href, ng, m, n, nm) result(e)
      integer, intent(in) :: ng, m, n, nm
      real(dp), intent(in) :: ac(VW, m, n, ng), Href(m, n, nm)
      integer :: b, g, lane, i, j
      e = 0.0_dp
      do b = 1, nm
         g = (b - 1)/VW + 1; lane = mod(b - 1, VW) + 1
         do j = 1, n
            do i = 1, m
               e = max(e, abs(ac(lane, i, j, g) - Href(i, j, b)))
            end do
         end do
      end do
   end function fort_err

   real(dp) function time_mkl(nA, m, n, ap0, ap, taup, work, lwork, info, fmt, nm, sink) result(kt)
      integer(c_int), intent(in) :: m, n, lwork, fmt, nm
      integer, intent(in) :: nA
      real(c_double), intent(in) :: ap0(nA)
      real(c_double), intent(inout) :: ap(nA), taup(*), work(*)
      integer(c_int), intent(inout) :: info
      real(dp), intent(inout) :: sink
      real(dp) :: t0, tfull, tcopy
      integer :: r, reps
      reps = 0; t0 = wtime()
      do
         ap = ap0
         call mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, work, lwork, info, fmt, nm)
         sink = sink + ap(1)
         reps = reps + 1
         if (wtime() - t0 > BUDGET .and. reps >= 3) exit
      end do
      tfull = wtime() - t0
      t0 = wtime()
      do r = 1, reps
         ap = ap0; sink = sink + ap(1)
      end do
      tcopy = wtime() - t0
      kt = max(tfull - tcopy, 1.0e-9_dp)/real(reps, dp)
   end function time_mkl

   real(dp) function time_fort(which, ng, m, n, A0, ac, tc, sink) result(kt)
      integer, intent(in) :: which, ng, m, n
      real(dp), intent(in) :: A0(VW, m, n, ng)
      real(dp), intent(inout) :: ac(VW, m, n, ng), tc(VW, min(m, n), ng), sink
      real(dp) :: t0, tfull, tcopy
      integer :: r, reps
      reps = 0; t0 = wtime()
      do
         ac = A0
         if (which == 1) then
            call geqr2_array(ng, m, n, ac, tc)
         else
            call geqr2_inner(ng, m, n, ac, tc)
         end if
         sink = sink + ac(1, 1, 1, 1)
         reps = reps + 1
         if (wtime() - t0 > BUDGET .and. reps >= 3) exit
      end do
      tfull = wtime() - t0
      t0 = wtime()
      do r = 1, reps
         ac = A0; sink = sink + ac(1, 1, 1, 1)
      end do
      tcopy = wtime() - t0
      kt = max(tfull - tcopy, 1.0e-9_dp)/real(reps, dp)
   end function time_fort

   subroutine fill_master(a, m, n, nm)
      integer, intent(in) :: m, n, nm
      real(c_double), intent(out) :: a(m, n, nm)
      integer :: ns, i
      integer, allocatable :: sd(:)
      call random_seed(size=ns); allocate (sd(ns)); sd = 20260804
      call random_seed(put=sd)
      call random_number(a)
      a = a - 0.5_dp
      do i = 1, min(m, n)
         a(i, i, :) = a(i, i, :) + real(n, dp)      ! diagonal boost -> well conditioned
      end do
   end subroutine fill_master

   real(dp) function wtime()
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      wtime = real(cnt, dp)/real(rate, dp)
   end function wtime

end program bench_mkl
