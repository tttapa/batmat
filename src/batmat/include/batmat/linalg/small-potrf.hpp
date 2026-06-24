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
    GUANAQO_TRACE_LINALG("small_potrf", total(flops::potrf(M, N)) * A.depth());
    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    return micro_kernels::small_potrf::small_potrf<T, R>(A, D);
}

template <class T, class Abi, micro_kernels::small_potrf::KernelConfig Conf, index_t R, index_t S,
          StorageOrder OD>
    requires(std::is_same_v<Abi, datapar::scalar_abi<T>> && OD == StorageOrder::ColMajor) // TODO
void small_potrf_left(view<const T, Abi, OD> A, view<const T, Abi, OD> C, view<T, Abi, OD> D) {
    // Check dimensions
    BATMAT_ASSERT(A.rows() == 0 || A.rows() == D.rows());
    BATMAT_ASSERT(C.rows() == D.rows());
    BATMAT_ASSERT(C.cols() == D.cols());
    BATMAT_ASSERT(D.rows() >= D.cols());
    const index_t M = D.rows(), N = D.cols();
    GUANAQO_TRACE_LINALG("small_potrf_left", total(flops::syrk_potrf(M, N, A.cols())) * C.depth());
    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    return micro_kernels::small_potrf::small_potrf_left<T, Conf, R, S>(A, C, D);
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
    static_assert(decltype(simdify(D.value))::storage_order == StorageOrder::ColMajor);
    detail::small_potrf<simdified_value_t<VD>, simdified_abi_t<VD>, R>(simdify(A.value).as_const(),
                                                                       simdify(D.value));
}

/// D = chol(D) with D symmetric as input, triangular as output
template <index_t R = 4, MatrixStructure SD, simdifiable VD>
void small_potrf(Structured<VD, SD> D) {
    small_potrf<R>(D.ref(), D.ref());
}

/// D = chol(C+AAᵀ) with C symmetric, D triangular
template <index_t R = 4, index_t S = 8, MatrixStructure SD, simdifiable VA, simdifiable VC,
          simdifiable VD>
    requires simdify_compatible<VA, VD>
void small_syrk_add_potrf_left(VA &&A, Structured<VC, SD> C, Structured<VD, SD> D) {
    static_assert(std::is_same_v<simdified_abi_t<VD>, datapar::scalar_abi<simdified_value_t<VD>>>);
    static_assert(SD == MatrixStructure::LowerTriangular);
    static_assert(decltype(simdify(A))::storage_order == StorageOrder::ColMajor);
    static_assert(decltype(simdify(C.value))::storage_order == StorageOrder::ColMajor);
    static_assert(decltype(simdify(D.value))::storage_order == StorageOrder::ColMajor);
    detail::small_potrf_left<simdified_value_t<VD>, simdified_abi_t<VD>, {.negate_A = false}, R, S>(
        simdify(A).as_const(), simdify(C.value).as_const(), simdify(D.value));
}

/// D = chol(C-AAᵀ) with C symmetric, D triangular
template <index_t R = 4, index_t S = 8, MatrixStructure SD, simdifiable VA, simdifiable VC,
          simdifiable VD>
    requires simdify_compatible<VA, VD>
void small_syrk_sub_potrf_left(VA &&A, Structured<VC, SD> C, Structured<VD, SD> D) {
    static_assert(std::is_same_v<simdified_abi_t<VD>, datapar::scalar_abi<simdified_value_t<VD>>>);
    static_assert(SD == MatrixStructure::LowerTriangular);
    static_assert(decltype(simdify(A))::storage_order == StorageOrder::ColMajor);
    static_assert(decltype(simdify(C.value))::storage_order == StorageOrder::ColMajor);
    static_assert(decltype(simdify(D.value))::storage_order == StorageOrder::ColMajor);
    detail::small_potrf_left<simdified_value_t<VD>, simdified_abi_t<VD>, {.negate_A = true}, R, S>(
        simdify(A).as_const(), simdify(C.value).as_const(), simdify(D.value));
}

/// D = chol(D+AAᵀ) with D symmetric as input, triangular as output
template <index_t R = 4, index_t S = 8, MatrixStructure SD, simdifiable VA, simdifiable VD>
    requires simdify_compatible<VA, VD>
void small_syrk_add_potrf_left(VA &&A, Structured<VD, SD> D) {
    small_syrk_add_potrf_left<R, S>(A, D.ref(), D.ref());
}

/// D = chol(D-AAᵀ) with D symmetric as input, triangular as output
template <index_t R = 4, index_t S = 8, MatrixStructure SD, simdifiable VA, simdifiable VD>
    requires simdify_compatible<VA, VD>
void small_syrk_sub_potrf_left(VA &&A, Structured<VD, SD> D) {
    small_syrk_sub_potrf_left<R, S>(A, D.ref(), D.ref());
}

/// D = chol(C) with C symmetric, D triangular
template <index_t R = 4, index_t S = 8, simdifiable VC, MatrixStructure SD, simdifiable VD>
    requires simdify_compatible<VC, VD>
void small_potrf_left(Structured<VC, SD> C, Structured<VD, SD> D) {
    decltype(simdify(D.value).as_const()) null{{.data = nullptr, .rows = 0, .cols = 0}};
    small_syrk_sub_potrf_left<R, S>(null, C.ref(), D.ref());
}

/// D = chol(D) with D symmetric as input, triangular as output
template <index_t R = 4, index_t S = 8, MatrixStructure SD, simdifiable VD>
void small_potrf_left(Structured<VD, SD> D) {
    small_potrf_left<R, S>(D.ref(), D.ref());
}

/// @}

/// @}

} // namespace batmat::linalg
