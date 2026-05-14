#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/micro-kernels/geqrf.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, micro_kernels::geqrf::KernelConfig Conf, StorageOrder OA,
          StorageOrder OD>
    requires(Conf.struc != MatrixStructure::General)
void geqrf(view<const T, Abi, OA> A, view<T, Abi, OD> D) {
    // Check dimensions
    BATMAT_ASSERT(A.rows() >= A.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols();
    [[maybe_unused]] const auto fc = flops::geqrf(M, N);
    GUANAQO_TRACE_LINALG("geqrf", total(fc) * D.depth());
    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    return micro_kernels::geqrf::geqrf_copy_register<T, Abi, Conf>(A, D);
}
} // namespace detail

/// @addtogroup topic-linalg
/// @{

/// @name QR factorization of batches of matrices
/// @{

/// triu(D) = QR(A) (strict lower part of D contains Householder vectors)
template <simdifiable VA, simdifiable VD>
    requires simdify_compatible<VA, VD>
void geqrf(Structured<VA, MatrixStructure::UpperTriangular> A,
           Structured<VD, MatrixStructure::UpperTriangular> D) {
    detail::geqrf<simdified_value_t<VD>, simdified_abi_t<VD>,
                  {.struc = MatrixStructure::UpperTriangular}>(simdify(A.value).as_const(),
                                                               simdify(D.value));
}

/// triu(D) = QR(D) (strict lower part of D contains Householder vectors)
template <simdifiable VD>
void geqrf(Structured<VD, MatrixStructure::UpperTriangular> D) {
    detail::geqrf<simdified_value_t<VD>, simdified_abi_t<VD>,
                  {.struc = MatrixStructure::UpperTriangular}>(simdify(D.value).as_const(),
                                                               simdify(D.value));
}

/// @}

/// @}

} // namespace batmat::linalg
