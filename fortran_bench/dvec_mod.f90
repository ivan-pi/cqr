!> A fixed-width SIMD "vector register" as a Fortran derived type.
!>
!> `dvec` holds VL doubles -- one lane per matrix in an interleave group -- and
!> exposes overloaded elementwise arithmetic (+ - * /), a lane-wise sqrt, and a
!> masked select (dmerge). Writing the QR kernels against this type lets the
!> unblocked scalar algorithm lift almost verbatim: every `+ - * /` becomes a
!> VL-wide lane operation over VL independent matrices, and the compiler lowers
!> the internal length-VL loops to AVX/AVX-512 instructions.
!>
!> This is the Fortran analogue of the project's C++ GNU vector types
!> (`__attribute__((vector_size))`) -- the "variant 4" of the benchmark.
module dvec_mod
   use, intrinsic :: iso_fortran_env, only: dp => real64
   implicit none
   private

   !> SIMD width (lanes per group). 8 doubles == one 512-bit register.
   integer, parameter, public :: VL = 8

   public :: dp

   !> One SIMD register: VL interleaved lanes.
   type, public :: dvec
      real(dp) :: x(VL)
   end type dvec

   ! Elementwise arithmetic. All are ELEMENTAL, so they also apply over
   ! arrays of dvec and broadcast a scalar dvec against a dvec array -- exactly
   ! how a `type(dvec) :: A(m,n)` reads in the kernels.
   interface operator(+)
      module procedure vv_add
   end interface
   interface operator(-)
      module procedure vv_sub, v_neg
   end interface
   interface operator(*)
      module procedure vv_mul, vs_mul, sv_mul
   end interface
   interface operator(/)
      module procedure vv_div
   end interface

   public :: operator(+), operator(-), operator(*), operator(/)
   public :: vsqrt, dmerge, load_group, store_group, splat

contains

   elemental function vv_add(a, b) result(c)
      type(dvec), intent(in) :: a, b
      type(dvec) :: c
      c%x = a%x + b%x
   end function

   elemental function vv_sub(a, b) result(c)
      type(dvec), intent(in) :: a, b
      type(dvec) :: c
      c%x = a%x - b%x
   end function

   elemental function v_neg(a) result(c)
      type(dvec), intent(in) :: a
      type(dvec) :: c
      c%x = -a%x
   end function

   elemental function vv_mul(a, b) result(c)
      type(dvec), intent(in) :: a, b
      type(dvec) :: c
      c%x = a%x*b%x
   end function

   !> dvec * scalar
   elemental function vs_mul(a, s) result(c)
      type(dvec), intent(in) :: a
      real(dp), intent(in) :: s
      type(dvec) :: c
      c%x = a%x*s
   end function

   !> scalar * dvec
   elemental function sv_mul(s, a) result(c)
      real(dp), intent(in) :: s
      type(dvec), intent(in) :: a
      type(dvec) :: c
      c%x = s*a%x
   end function

   elemental function vv_div(a, b) result(c)
      type(dvec), intent(in) :: a, b
      type(dvec) :: c
      c%x = a%x/b%x
   end function

   !> Lane-wise sqrt (lowers to a single vsqrtpd).
   elemental function vsqrt(a) result(c)
      type(dvec), intent(in) :: a
      type(dvec) :: c
      c%x = sqrt(a%x)
   end function

   !> Lane-wise select: result(k) = mask(k) ? t(k) : f(k). The branch-free
   !> heart of a vectorized larfg. Operates on a single register (VL lanes),
   !> so it is a plain pure function, not elemental.
   pure function dmerge(t, f, mask) result(c)
      type(dvec), intent(in) :: t, f
      logical, intent(in) :: mask(VL)
      type(dvec) :: c
      c%x = merge(t%x, f%x, mask)
   end function

   !> Broadcast a scalar into every lane.
   elemental function splat(s) result(c)
      real(dp), intent(in) :: s
      type(dvec) :: c
      c%x = s
   end function

   !> Contiguous load of VL lanes from the compact buffer (batch fastest).
   pure function load_group(buf) result(c)
      real(dp), intent(in) :: buf(VL)
      type(dvec) :: c
      c%x = buf
   end function

   pure subroutine store_group(buf, c)
      real(dp), intent(out) :: buf(VL)
      type(dvec), intent(in) :: c
      buf = c%x
   end subroutine

end module dvec_mod
