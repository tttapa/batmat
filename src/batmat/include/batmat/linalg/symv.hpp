#pragma once

#include <batmat/linalg/micro-kernels/symv.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, micro_kernels::symv::KernelConfig Conf = {}, StorageOrder OA>
    requires(Conf.struc_A == MatrixStructure::LowerTriangular ||
             Conf.struc_A == MatrixStructure::UpperTriangular)
void symv(view<const T, Abi, OA> A, view<const T, Abi> B, std::optional<view<const T, Abi>> C,
          view<T, Abi> D) {
    static_assert(Conf.struc_A == MatrixStructure::LowerTriangular); // TODO
    GUANAQO_TRACE("symv", 0, A.rows() * A.cols() * B.cols() * A.depth());
    // Check dimensions
    const index_t M = D.rows();
    BATMAT_ASSERT(!C || C->rows() == D.rows());
    BATMAT_ASSERT(!C || C->cols() == D.cols());
    BATMAT_ASSERT(A.rows() == M);
    BATMAT_ASSERT(A.cols() == M);
    BATMAT_ASSERT(B.cols() == D.cols());
    BATMAT_ASSERT(B.cols() == 1);

    // Degenerate case
    if (M == 0) [[unlikely]]
        return;
    micro_kernels::symv::symv_copy_register<T, Abi, Conf, OA>(A, B, C, D);
}

} // namespace detail

/// d = A b where A is symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD>
    requires simdify_compatible<VA, VB, VD>
void symv(Structured<VA, SA> A, VB &&B, VD &&D) {
    static constexpr micro_kernels::symv::KernelConfig conf{.negate = false, .struc_A = SA};
    std::optional<decltype(simdify(D).as_const())> null;
    detail::symv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B).as_const(), null, simdify(D));
}

/// d = -A b where A is symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD>
    requires simdify_compatible<VA, VB, VD>
void symv_neg(Structured<VA, SA> A, VB &&B, VD &&D) {
    static constexpr micro_kernels::symv::KernelConfig conf{.negate = true, .struc_A = SA};
    std::optional<decltype(simdify(D).as_const())> null;
    detail::symv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B).as_const(), null, simdify(D));
}

/// d = c + A b where A is symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VC, simdifiable VD>
    requires simdify_compatible<VA, VB, VC, VD>
void symv_add(Structured<VA, SA> A, VB &&B, VC &&C, VD &&D) {
    static constexpr micro_kernels::symv::KernelConfig conf{.negate = false, .struc_A = SA};
    detail::symv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B).as_const(), simdify(C).as_const(), simdify(D));
}
/// d = d + A b where A is symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD>
    requires simdify_compatible<VA, VB, VD>
void symv_add(Structured<VA, SA> A, VB &&B, VD &&D) {
    symv_add(A.ref(), B, D, D);
}

/// d = c - A b where A is symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VC, simdifiable VD>
    requires simdify_compatible<VA, VB, VC, VD>
void symv_sub(Structured<VA, SA> A, VB &&B, VC &&C, VD &&D) {
    static constexpr micro_kernels::symv::KernelConfig conf{.negate = true, .struc_A = SA};
    detail::symv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B).as_const(), simdify(C).as_const(), simdify(D));
}
/// d = d - A b where A is symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD>
    requires simdify_compatible<VA, VB, VD>
void symv_sub(Structured<VA, SA> A, VB &&B, VD &&D) {
    symv_sub(A.ref(), B, D, D);
}

} // namespace batmat::linalg
