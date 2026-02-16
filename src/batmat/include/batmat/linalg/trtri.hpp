#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/flops.hpp>
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
    // Check dimensions
    BATMAT_ASSERT(D.rows() == D.cols()); // TODO: could be relaxed
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols();
    [[maybe_unused]] const auto fc = flops::trtri(M);
    GUANAQO_TRACE_LINALG("trtri", total(fc) * D.depth());
    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    return micro_kernels::trtri::trtri_copy_register<T, Abi, Conf>(A, D);
}
} // namespace detail

/// D = A⁻¹ with A, D lower triangular
template <simdifiable VA, simdifiable VD>
    requires simdify_compatible<VA, VD>
void trtri(Structured<VA, MatrixStructure::LowerTriangular> A,
           Structured<VD, MatrixStructure::LowerTriangular> D) {
    detail::trtri<simdified_value_t<VA>, simdified_abi_t<VA>,
                  {.struc = MatrixStructure::LowerTriangular}>(simdify(A.value).as_const(),
                                                               simdify(D.value));
}

/// D = A⁻¹ with A, D upper triangular
template <simdifiable VA, simdifiable VD>
    requires simdify_compatible<VA, VD>
void trtri(Structured<VA, MatrixStructure::UpperTriangular> A,
           Structured<VD, MatrixStructure::UpperTriangular> D) {
    trtri(A.transposed(), D.transposed());
}

/// D = D⁻¹ with D lower triangular
template <simdifiable VD>
void trtri(Structured<VD, MatrixStructure::LowerTriangular> D) {
    detail::trtri<simdified_value_t<VD>, simdified_abi_t<VD>,
                  {.struc = MatrixStructure::LowerTriangular}>(simdify(D.value).as_const(),
                                                               simdify(D.value));
}

/// D = D⁻¹ with D upper triangular
template <simdifiable VD>
void trtri(Structured<VD, MatrixStructure::UpperTriangular> D) {
    trtri(D.transposed());
}

} // namespace batmat::linalg
