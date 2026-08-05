!> A fixed-width SIMD "vector register" as a Fortran derived type.
!>
!> The batch/interleave width is now a COMPILE-TIME CONSTANT (VW = 8) -- one
!> 512-bit AVX register of doubles -- rather than a runtime dimension. `vec8`
!> holds those 8 lanes and exposes overloaded elementwise arithmetic (+ - * /),
!> a lane-wise sqrt, and a masked select (vmerge). Writing the QR kernels
!> against it lets the unblocked scalar algorithm lift almost verbatim, with
!> every `+ - * /` a single 8-wide instruction over 8 independent matrices.
!>
!> This is the Fortran analogue of the project's C++ GNU vector types
!> (`__attribute__((vector_size))`) -- the "variant 4" of the benchmark.
module vec8_mod
   use, intrinsic :: iso_fortran_env, only: dp => real64
   implicit none
   private

   !> Fixed SIMD width (lanes per group). 8 doubles == one 512-bit register.
   integer, parameter, public :: VW = 8

   public :: dp

   !> One SIMD register: exactly VW = 8 interleaved lanes.
   type, public :: vec8
      real(dp) :: v(8)
   end type vec8

   ! Elementwise arithmetic. All ELEMENTAL, so they also apply over arrays of
   ! vec8 and broadcast a scalar vec8 against a vec8 array -- exactly how a
   ! `type(vec8) :: A(m,n)` reads in the kernels.
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
   public :: vsqrt, vmerge, load8, store8, splat

contains

   elemental function vv_add(a, b) result(c)
      type(vec8), intent(in) :: a, b
      type(vec8) :: c
      c%v = a%v + b%v
   end function

   elemental function vv_sub(a, b) result(c)
      type(vec8), intent(in) :: a, b
      type(vec8) :: c
      c%v = a%v - b%v
   end function

   elemental function v_neg(a) result(c)
      type(vec8), intent(in) :: a
      type(vec8) :: c
      c%v = -a%v
   end function

   elemental function vv_mul(a, b) result(c)
      type(vec8), intent(in) :: a, b
      type(vec8) :: c
      c%v = a%v*b%v
   end function

   !> vec8 * scalar
   elemental function vs_mul(a, s) result(c)
      type(vec8), intent(in) :: a
      real(dp), intent(in) :: s
      type(vec8) :: c
      c%v = a%v*s
   end function

   !> scalar * vec8
   elemental function sv_mul(s, a) result(c)
      real(dp), intent(in) :: s
      type(vec8), intent(in) :: a
      type(vec8) :: c
      c%v = s*a%v
   end function

   elemental function vv_div(a, b) result(c)
      type(vec8), intent(in) :: a, b
      type(vec8) :: c
      c%v = a%v/b%v
   end function

   !> Lane-wise sqrt (lowers to a single vsqrtpd).
   elemental function vsqrt(a) result(c)
      type(vec8), intent(in) :: a
      type(vec8) :: c
      c%v = sqrt(a%v)
   end function

   !> Lane-wise select: result(k) = mask(k) ? t(k) : f(k). The branch-free
   !> heart of a vectorized larfg. Operates on one register (8 lanes), so it is
   !> a plain pure function, not elemental.
   pure function vmerge(t, f, mask) result(c)
      type(vec8), intent(in) :: t, f
      logical, intent(in) :: mask(VW)
      type(vec8) :: c
      c%v = merge(t%v, f%v, mask)
   end function

   !> Broadcast a scalar into every lane.
   elemental function splat(s) result(c)
      real(dp), intent(in) :: s
      type(vec8) :: c
      c%v = s
   end function

   !> Contiguous load of the 8 lanes from the compact buffer (batch fastest).
   pure function load8(buf) result(c)
      real(dp), intent(in) :: buf(VW)
      type(vec8) :: c
      c%v = buf
   end function

   pure subroutine store8(buf, c)
      real(dp), intent(out) :: buf(VW)
      type(vec8), intent(in) :: c
      buf = c%v
   end subroutine

end module vec8_mod
