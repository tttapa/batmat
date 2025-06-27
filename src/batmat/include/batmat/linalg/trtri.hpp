#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/trtri.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, micro_kernels::trtri::KernelConfig Conf, StorageOrder OA,
          StorageOrder OD>
    requires(Conf.struc != MatrixStructure::General)
void trtri(view<const T, Abi, OA> A, view<T, Abi, OD> D) {
    GUANAQO_TRACE("trtri", 0, 0); // TODO
    // Check dimensions
    BATMAT_ASSERT(D.rows() == D.cols()); // TODO: could be relaxed
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols();

    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;

    return micro_kernels::trtri::trtri_copy_register<T, Abi, Conf>(A, D);
}
} // namespace detail

/// D = A⁻¹ with A, D triangular
template <MatrixStructure SAD, simdifiable VA, simdifiable VD>
    requires simdify_compatible<VA, VD>
void trtri(Structured<VA, SAD> A, Structured<VD, SAD> D) {
    detail::trtri<simdified_value_t<VA>, simdified_abi_t<VA>, {.struc = SAD}>(
        simdify(A.value).as_const(), simdify(D.value));
}

/// D = D⁻¹ with D triangular
template <MatrixStructure SD, simdifiable VD>
void trtri(Structured<VD, SD> D) {
    detail::trtri<simdified_value_t<VD>, simdified_abi_t<VD>, {.struc = SD}>(
        simdify(D.value).as_const(), simdify(D.value));
}

} // namespace batmat::linalg
