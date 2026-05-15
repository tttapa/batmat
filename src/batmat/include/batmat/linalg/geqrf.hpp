#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/copy.hpp>
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
void geqrf(view<const T, Abi, OA> A, view<T, Abi, OD> D, view<T, Abi> W) {
    // Check dimensions
    BATMAT_ASSERT(A.rows() >= A.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == D.cols());
    BATMAT_ASSERT(W.rows() == 0 || (W.cols() == 1 && W.rows() == A.cols()) ||
                  std::make_pair(W.rows(), W.cols()) ==
                      (micro_kernels::geqrf::geqrf_W_size<const T, Abi>)(A));
    const index_t M = D.rows(), N = D.cols();
    [[maybe_unused]] const auto fc = flops::geqrf(M, N);
    GUANAQO_TRACE_LINALG("geqrf", total(fc) * D.depth());
    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    return micro_kernels::geqrf::geqrf_copy_register<T, Abi, Conf>(A, D, W);
}

template <class T, class Abi, micro_kernels::geqrf::KernelConfig Conf, StorageOrder OA,
          StorageOrder OD, StorageOrder OB>
void geqrf_apply(view<const T, Abi, OA> A, view<T, Abi, OD> D, view<const T, Abi, OB> B,
                 view<const T, Abi> W, bool transposed) {
    // Check dimensions
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == D.cols());
    BATMAT_ASSERT(B.rows() == A.rows());
    BATMAT_ASSERT(std::make_pair(W.rows(), W.cols()) ==
                  (micro_kernels::geqrf::geqrf_W_size<const T, Abi>)(B));
    const index_t M = D.rows(), N = D.cols(), K = B.cols();
    [[maybe_unused]] const auto fc = flops::geqrf_apply(M, N, K);
    GUANAQO_TRACE_LINALG("geqrf_apply", total(fc) * D.depth());
    // Degenerate case
    if (K == 0 && D.data() != A.data()) [[unlikely]]
        linalg::copy(A, D);
    if (M == 0 || N == 0 || K == 0) [[unlikely]]
        return;

    return micro_kernels::geqrf::geqrf_apply_register<T, Abi, Conf>(A, D, B, W, transposed);
}
} // namespace detail

/// @addtogroup topic-linalg
/// @{

/// @name QR factorization of batches of matrices
/// @{

/// QR factorization. The upper triangular part of D contains the R factor. The Householder vectors
/// are stored in the strict lower triangular part of D.
/// The Householder coefficients are stored in W, which should either be a vector of `A.cols()`
/// elements, or a matrix of size `geqrf_W_size(A)`. If W has zero rows, the coefficients are
/// discarded.
template <simdifiable VA, simdifiable VD, simdifiable VW>
    requires simdify_compatible<VA, VD, VW>
void geqrf(VA &&A, VD &&D, VW &&W) {
    detail::geqrf<simdified_value_t<VD>, simdified_abi_t<VD>, {}>(simdify(A).as_const(), simdify(D),
                                                                  simdify(W));
}

/// QR factorization. The upper triangular part of D contains the R factor. The Householder vectors
/// are stored in the strict lower triangular part of D.
/// The Householder coefficients are stored in W, which should either be a vector of `A.cols()`
/// elements, or a matrix of size `geqrf_W_size(A)`. If W has zero rows, the coefficients are
/// discarded.
template <simdifiable VD, simdifiable VW>
    requires simdify_compatible<VD, VW>
void geqrf(VD &&D, VW &&W) {
    detail::geqrf<simdified_value_t<VD>, simdified_abi_t<VD>, {}>(simdify(D).as_const(), simdify(D),
                                                                  simdify(W));
}

/// Apply the Q factor from @ref geqrf (represented by @p B and @p W) to a matrix @p A, storing
/// either QA or QᵀA in @p D (depending on @p transposed).
template <simdifiable VA, simdifiable VD, simdifiable VB, simdifiable VW>
    requires simdify_compatible<VA, VD, VB, VW>
void geqrf_apply(VA &&A, VD &&D, VB &&B, VW &&W, bool transposed = false) {
    detail::geqrf_apply<simdified_value_t<VD>, simdified_abi_t<VD>, {}>(
        simdify(A).as_const(), simdify(D), simdify(B).as_const(), simdify(W).as_const(),
        transposed);
}

/// Apply the Q factor from @ref geqrf (represented by @p B and @p W) to a matrix @p D, overwriting
/// it with either QA or QᵀA (depending on @p transposed).
template <simdifiable VD, simdifiable VB, simdifiable VW>
    requires simdify_compatible<VD, VB, VW>
void geqrf_apply(VD &&D, VB &&B, VW &&W, bool transposed = false) {
    detail::geqrf_apply<simdified_value_t<VD>, simdified_abi_t<VD>, {}>(
        simdify(D).as_const(), simdify(D), simdify(B).as_const(), simdify(W).as_const(),
        transposed);
}

/// Get the size of the storage for the matrix W returned by
/// @ref geqrf(VA &&A, VD &&D, VW &&W).
template <simdifiable VA>
auto geqrf_size_W(VA &&A) {
    return micro_kernels::geqrf::geqrf_W_size<const simdified_value_t<VA>, simdified_abi_t<VA>>(
        simdify(A).as_const());
}

/// @}

/// @}

} // namespace batmat::linalg
