#pragma once

#include <batmat/linalg/micro-kernels/syomv.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail::syomv {
template <class T, class Abi, micro_kernels::syomv::KernelConfig Conf = {}, StorageOrder OA,
          StorageOrder OB, StorageOrder OD>
void syomv(view<const T, Abi, OA> A, view<const T, Abi, OB> B, view<T, Abi, OD> D) {
    GUANAQO_TRACE("syomv", 0, 0); // TODO
    // Check dimensions
    BATMAT_ASSERT(A.rows() == A.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == B.rows());
    BATMAT_ASSERT(B.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols(), K = A.cols();

    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    if (K == 0) [[unlikely]]
        return;

    return micro_kernels::syomv::syomv_register<T, Abi, Conf>(A, B, D);
}
} // namespace detail::syomv

/// @todo   Describe the operation in detail.
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD>
    requires simdify_compatible<VA, VB, VD>
void syomv(Structured<VA, SA> A, VB &&B, VD &&D) {
    constexpr micro_kernels::syomv::KernelConfig conf{.negate = false, .struc_A = SA};
    detail::syomv::syomv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B).as_const(), simdify(D));
}

/// @todo   Describe the operation in detail.
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD>
    requires simdify_compatible<VA, VB, VD>
void syomv_neg(Structured<VA, SA> A, VB &&B, VD &&D) {
    constexpr micro_kernels::syomv::KernelConfig conf{.negate = true, .struc_A = SA};
    detail::syomv::syomv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B).as_const(), simdify(D));
}

} // namespace batmat::linalg
