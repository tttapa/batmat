#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/elementwise.hpp>
#include <batmat/linalg/micro-kernels/sterf.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>

#include <expected>

namespace batmat::linalg {

using batmat::linalg::micro_kernels::sterf::SterfOptions;

namespace detail {

template <class T, class Abi>
std::expected<index_t, index_t> sterf(view<T, Abi, StorageOrder::ColMajor> diag,
                                      view<T, Abi, StorageOrder::ColMajor> subdiag,
                                      SterfOptions options) {
    static_assert(!std::is_const_v<T>);
    BATMAT_ASSERT(diag.cols() == 1);
    BATMAT_ASSERT(subdiag.cols() == 1);
    const index_t n = diag.rows();
    if (n > 0)
        BATMAT_ASSERT(subdiag.rows() == n - 1);
    else
        BATMAT_ASSERT(subdiag.rows() == 0);
    if (n <= 1)
        return 0;
    return micro_kernels::sterf::sterf<T, Abi>(diag, subdiag, options);
}

} // namespace detail

/// Eigenvalues of a symmetric tridiagonal matrix given by `diag` and `subdiag`, computed in-place
/// using the Pal-Walker-Kahan variant of the implicit QR/QL method with Wilkinson shifts.
/// @return Number of QR steps taken. An unexpected value signals failure to converge within the
///         maximum number of iterations specified in `options`.
template <simdifiable Vd, simdifiable Vs>
    requires simdify_compatible<Vd, Vs>
std::expected<index_t, index_t> sterf(Vd &&diag, Vs &&subdiag, SterfOptions options = {}) {
    static_assert(decltype(simdify(diag))::storage_order == StorageOrder::ColMajor);
    static_assert(decltype(simdify(subdiag))::storage_order == StorageOrder::ColMajor);
    return detail::sterf<simdified_value_t<Vd>, simdified_abi_t<Vd>>(simdify(diag),
                                                                     simdify(subdiag), options);
}

/// Extracts the diagonal and one off-diagonal from a matrix. Pass a triangular view to specify
/// which off-diagonal to extract.
template <simdifiable VA, simdifiable Vd, simdifiable Vs, MatrixStructure SA>
    requires simdify_compatible<VA, Vd, Vs>
void extract_bidiag(Structured<VA, SA> A, Vd &&diag, Vs &&offdiag) {
    BATMAT_ASSERT(rows(A.value) == cols(A.value));
    BATMAT_ASSERT(rows(A.value) == rows(diag));
    BATMAT_ASSERT(cols(diag) == 1);
    copy_diag(A.value, diag);
    if (const index_t n = rows(A.value); n > 1) {
        BATMAT_ASSERT(rows(offdiag) == n - 1);
        BATMAT_ASSERT(cols(offdiag) == 1);
        if constexpr (SA == MatrixStructure::UpperTriangular)
            copy_diag(A.value.top_right(n - 1, n - 1), offdiag);
        else
            copy_diag(A.value.bottom_left(n - 1, n - 1), offdiag);
    }
}

} // namespace batmat::linalg
