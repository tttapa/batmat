#pragma once

#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/micro-kernels/potrf.hpp>
#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, micro_kernels::potrf::KernelConfig Conf, StorageOrder OA,
          StorageOrder OCD>
    requires(Conf.struc_C != MatrixStructure::General)
void potrf(view<const T, Abi, OA> A, view<const T, Abi, OCD> C, view<T, Abi, OCD> D) {
    GUANAQO_TRACE("potrf", 0, 0); // TODO
    // Check dimensions
    BATMAT_ASSERT(D.rows() == D.cols()); // TODO: could be relaxed
    BATMAT_ASSERT(A.cols() == 0 || A.rows() == D.rows());
    BATMAT_ASSERT(C.rows() == D.rows());
    BATMAT_ASSERT(C.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols();

    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;

    return micro_kernels::potrf::potrf_copy_register<T, Abi, Conf>(A, C, D);
}
} // namespace detail

/// D = chol(C + AAᵀ) with C symmetric, D triangular
template <MatrixStructure SC, simdifiable VA, simdifiable VC, simdifiable VD>
    requires simdify_compatible<VA, VC, VD>
void syrk_add_potrf(VA &&A, Structured<VC, SC> C, Structured<VD, SC> D) {
    detail::potrf<simdified_value_t<VA>, simdified_abi_t<VA>, {.negate_A = false, .struc_C = SC}>(
        simdify(A).as_const(), simdify(C.value).as_const(), simdify(D.value));
}

/// D = chol(C - AAᵀ) with C symmetric, D triangular
template <MatrixStructure SC, simdifiable VA, simdifiable VC, simdifiable VD>
    requires simdify_compatible<VA, VC, VD>
void syrk_sub_potrf(VA &&A, Structured<VC, SC> C, Structured<VD, SC> D) {
    detail::potrf<simdified_value_t<VA>, simdified_abi_t<VA>, {.negate_A = true, .struc_C = SC}>(
        simdify(A).as_const(), simdify(C.value).as_const(), simdify(D.value));
}

/// D = chol(C) with C symmetric, D triangular
template <MatrixStructure SC, simdifiable VC, simdifiable VD>
    requires simdify_compatible<VC, VD>
void potrf(Structured<VC, SC> C, Structured<VD, SC> D) {
    decltype(simdify(C.value).as_const()) null{{.data = nullptr, .rows = 0, .cols = 0}};
    detail::potrf<simdified_value_t<VC>, simdified_abi_t<VC>, {.struc_C = SC}>(
        null, simdify(C.value).as_const(), simdify(D.value));
}

/// D = chol(D) with D symmetric/triangular
template <MatrixStructure SC, simdifiable VD>
void potrf(Structured<VD, SC> D) {
    decltype(simdify(D.value).as_const()) null{{.data = nullptr, .rows = 0, .cols = 0}};
    detail::potrf<simdified_value_t<VD>, simdified_abi_t<VD>, {.struc_C = SC}>(
        null, simdify(D.value).as_const(), simdify(D.value));
}

} // namespace batmat::linalg
