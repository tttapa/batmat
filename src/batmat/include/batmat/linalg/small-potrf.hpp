#pragma once

#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/micro-kernels/small-potrf.hpp>
#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, index_t R, StorageOrder OD>
    requires(std::is_same_v<Abi, datapar::scalar_abi<T>> && OD == StorageOrder::ColMajor) // TODO
void small_potrf(view<const T, Abi, OD> A, view<T, Abi, OD> D) {
    // Check dimensions
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == D.cols());
    BATMAT_ASSERT(D.rows() >= D.cols());
    const index_t M = D.rows(), N = D.cols();
    GUANAQO_TRACE_LINALG("small_potrf", total(flops::syrk_potrf(M, N, A.cols())) * C.depth());
    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    return micro_kernels::small_potrf::small_potrf<T, R>(A, D);
}

template <class T, class Abi, index_t R, index_t S, StorageOrder OD>
    requires(std::is_same_v<Abi, datapar::scalar_abi<T>> && OD == StorageOrder::ColMajor) // TODO
void small_potrf_left(view<const T, Abi, OD> A, view<T, Abi, OD> D) {
    // Check dimensions
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == D.cols());
    BATMAT_ASSERT(D.rows() >= D.cols());
    const index_t M = D.rows(), N = D.cols();
    GUANAQO_TRACE_LINALG("small_potrf_left", total(flops::syrk_potrf(M, N, A.cols())) * C.depth());
    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    return micro_kernels::small_potrf::small_potrf_left<T, R, S>(A, D);
}
} // namespace detail

/// @addtogroup topic-linalg
/// @{

/// @name Cholesky factorization of a single matrix
/// @{

/// D = chol(A) with A symmetric, D triangular
template <index_t R = 4, MatrixStructure SD, simdifiable VA, simdifiable VD>
    requires simdify_compatible<VA, VD>
void small_potrf(Structured<VA, SD> A, Structured<VD, SD> D) {
    static_assert(std::is_same_v<simdified_abi_t<VD>, datapar::scalar_abi<simdified_value_t<VD>>>);
    static_assert(SD == MatrixStructure::LowerTriangular);
    static_assert(D.value.storage_order == StorageOrder::ColMajor);
    detail::small_potrf<simdified_value_t<VD>, simdified_abi_t<VD>, R>(simdify(A.value).as_const(),
                                                                       simdify(D.value));
}

/// D = chol(D) with D symmetric as input, triangular as output
template <index_t R = 4, MatrixStructure SD, simdifiable VD>
void small_potrf(Structured<VD, SD> D) {
    small_potrf<R>(D, D);
}

/// D = chol(A) with A symmetric, D triangular
template <index_t R = 4, index_t S = 8, MatrixStructure SD, simdifiable VA, simdifiable VD>
    requires simdify_compatible<VA, VD>
void small_potrf_left(Structured<VA, SD> A, Structured<VD, SD> D) {
    static_assert(std::is_same_v<simdified_abi_t<VD>, datapar::scalar_abi<simdified_value_t<VD>>>);
    static_assert(SD == MatrixStructure::LowerTriangular);
    static_assert(D.value.storage_order == StorageOrder::ColMajor);
    detail::small_potrf_left<simdified_value_t<VD>, simdified_abi_t<VD>, R, S>(
        simdify(A.value).as_const(), simdify(D.value));
}

/// D = chol(D) with D symmetric as input, triangular as output
template <index_t R = 4, index_t S = 8, MatrixStructure SD, simdifiable VD>
void small_potrf_left(Structured<VD, SD> D) {
    small_potrf_left<R, S>(D, D);
}

/// @}

/// @}

} // namespace batmat::linalg
