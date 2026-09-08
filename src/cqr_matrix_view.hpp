/* cqr_matrix_view.hpp
 *
 * MatrixView<T>: a non-owning, strided 2-D view of one dense matrix -- the
 * host-side counterpart of the kernels' BatchView, for the test suites, the
 * benchmarks and the examples, whose checking and setup code otherwise spells
 * out `A[i + (size_t)j * lda]` (and its row-major mirror) at every access.
 *
 *     auto A = mat_view(a, n, n);                 // column-major, ld = n
 *     auto H = mat_view(h, n, n, ldh, rowmajor);  // either layout, ld given
 *     A(i, j) = 1;                                // element
 *     A.col(j);                                   // contiguous column (col-major)
 *     A.transposed();                             // same storage, (i,j) -> (j,i)
 *     A.ld();                                     // leading dimension, for BLAS
 *
 * It is the minimal slice of what std::mdspan (C++23) offers: extents, one
 * strided accessor, and the transpose. Deliberately not a matrix class -- it
 * owns nothing, allocates nothing, and has no arithmetic; the dense arrays it
 * views belong to MatrixBatch, a benchmark Pool, or a plain std::vector.
 *
 * Layout is carried as runtime strides (si, sj) rather than a template
 * parameter, exactly as BatchView carries it. The MKL suites loop over
 * MKL_COL_MAJOR / MKL_ROW_MAJOR at runtime, so a compile-time layout would
 * force every layout-generic suite body to be duplicated or dispatched behind
 * a runtime switch. Nothing here is timed, and the accessors inline to the
 * same address arithmetic they replace.
 *
 * const-ness rides on T: mat_view() of a `const T *` yields a
 * MatrixView<const T> (spelled ConstMatrixView<T>) whose operator() returns
 * `const T &`, so a read-only operand cannot be written through its view.
 *
 * The strides are int, like the extents -- they are matrix dimensions, which
 * this project keeps in int throughout -- but the offset they form is a
 * std::ptrdiff_t, since a matrix inside a large pool can sit past the 2 GiB
 * mark. That is the `(size_t)j * n` the hand-written indexing spells out, in
 * one place. (BatchView forms its offsets in int on purpose: it addresses one
 * group, whose base offset is applied to the pointer beforehand, and the int
 * arithmetic is what keeps its SIMD sweeps vectorizable.)
 *
 * Assisted-by: Claude
 */

#ifndef CQR_MATRIX_VIEW_HPP
#define CQR_MATRIX_VIEW_HPP

#include <cstddef>
#include <cassert>

namespace cqr::detail {

template <typename T> struct MatrixView {
    T *data;
    int si; /* element stride along the row index i    */
    int sj; /* element stride along the column index j */
    int rows, cols;

    /* Element (i,j). The bounds assert costs nothing in a release build and
     * catches the swapped-index mistakes this view exists to prevent. */
    T &operator()(int i, int j) const noexcept
    {
        assert(i >= 0 && i < rows && j >= 0 && j < cols);
        return data[(std::ptrdiff_t)i * si + (std::ptrdiff_t)j * sj];
    }

    /* Pointer to column j (row i), contiguous only when that axis has unit
     * stride: column-major has si == 1, row-major sj == 1. */
    T *col(int j) const noexcept
    {
        assert(si == 1 && j >= 0 && j < cols);
        return data + (std::ptrdiff_t)j * sj;
    }
    T *row(int i) const noexcept
    {
        assert(sj == 1 && i >= 0 && i < rows);
        return data + (std::ptrdiff_t)i * si;
    }

    /* The leading dimension to hand to BLAS/LAPACK, which take one unit-stride
     * axis and the stride of the other. */
    int ld() const noexcept
    {
        assert(si == 1 || sj == 1);
        return si == 1 ? sj : si;
    }

    /* The same storage read as the transposed matrix: (i,j) -> (j,i). */
    MatrixView transposed() const noexcept { return {data, sj, si, cols, rows}; }
};

template <typename T> using ConstMatrixView = MatrixView<const T>;

/* View a rows x cols matrix with leading dimension ld: column-major has unit
 * row stride, row-major unit column stride. */
template <typename T>
MatrixView<T> mat_view(T *p, int rows, int cols, int ld, bool rowmajor = false) noexcept
{
    return {p, rowmajor ? ld : 1, rowmajor ? 1 : ld, rows, cols};
}

/* The ubiquitous case: column-major, no padding (ld == rows). */
template <typename T> MatrixView<T> mat_view(T *p, int rows, int cols) noexcept
{
    return {p, 1, rows, rows, cols};
}

} /* namespace cqr::detail */

#endif /* CQR_MATRIX_VIEW_HPP */
