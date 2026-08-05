!> Minimal Fortran (iso_c_binding) interface to Intel MKL's Compact LAPACK API,
!> just the pieces needed to benchmark `mkl_dgeqrf_compact` against the Fortran
!> kernels: the format query, the buffer sizing, pack/unpack, and the compact QR
!> factorization itself.
!>
!> Built for an LP64 MKL (`MKL_INT` == 32-bit C int == c_int). `mkl_?get_size_
!> compact` returns the buffer size in BYTES; divide by 8 for the double count.
!> Note MKL ships a compact `geqrf` but NO compact `ormqr`, so only the
!> factorization has a native-MKL counterpart to compare against.
module mkl_compact
   use iso_c_binding, only: c_int, c_double, c_ptr
   implicit none
   public

   integer(c_int), parameter :: MKL_ROW_MAJOR = 101, MKL_COL_MAJOR = 102

   interface
      !> Best compact pack format for this machine (183 = AVX-512 -> V = 8 FP64).
      integer(c_int) function mkl_get_format_compact() bind(C, name='mkl_get_format_compact')
         import :: c_int
      end function

      !> Size (in BYTES) of a compact buffer holding `nm` matrices of `ld` x `sd`.
      integer(c_int) function mkl_dget_size_compact(ld, sd, format, nm) &
         bind(C, name='mkl_dget_size_compact')
         import :: c_int
         integer(c_int), value :: ld, sd, format, nm
      end function

      !> Pack `nm` separate matrices (array of pointers `a`) into compact `ap`.
      subroutine mkl_dgepack_compact(layout, rows, cols, a, lda, ap, ldap, format, nm) &
         bind(C, name='mkl_dgepack_compact')
         import :: c_int, c_double, c_ptr
         integer(c_int), value :: layout, rows, cols, lda, ldap, format, nm
         type(c_ptr), intent(in) :: a(*)
         real(c_double), intent(inout) :: ap(*)
      end subroutine

      !> Unpack compact `ap` back into `nm` separate matrices (pointers `a`).
      subroutine mkl_dgeunpack_compact(layout, rows, cols, a, lda, ap, ldap, format, nm) &
         bind(C, name='mkl_dgeunpack_compact')
         import :: c_int, c_double, c_ptr
         integer(c_int), value :: layout, rows, cols, lda, ldap, format, nm
         type(c_ptr), intent(in) :: a(*)
         real(c_double), intent(in) :: ap(*)
      end subroutine

      !> Batched QR factorization in compact format.
      subroutine mkl_dgeqrf_compact(layout, m, n, ap, ldap, taup, work, lwork, info, format, nm) &
         bind(C, name='mkl_dgeqrf_compact')
         import :: c_int, c_double
         integer(c_int), value :: layout, m, n, ldap, lwork, format, nm
         real(c_double), intent(inout) :: ap(*), taup(*), work(*)
         integer(c_int), intent(out) :: info
      end subroutine
   end interface

end module mkl_compact
