#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/geqrf.hpp>
#include <batmat/linalg/micro-kernels/sytrd.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, micro_kernels::sytrd::KernelConfig Conf, StorageOrder OD>
void sytrd(view<T, Abi, OD> D, view<T, Abi> W, view<T, Abi> Y) {
    // Check dimensions
    BATMAT_ASSERT(D.rows() == D.cols());
    BATMAT_ASSERT(
        W.rows() == 0 || (W.cols() == 1 && W.rows() == std::max<index_t>(D.cols(), 1) - 1) ||
        std::make_pair(W.rows(), W.cols()) == (micro_kernels::sytrd::sytrd_W_size<T, Abi>)(D));
    BATMAT_ASSERT(std::make_pair(Y.rows(), Y.cols()) ==
                  (micro_kernels::sytrd::sytrd_Y_size<T, Abi, OD>)(D));
    const index_t M                = D.rows();
    [[maybe_unused]] const auto fc = flops::sytrd(M);
    GUANAQO_TRACE_LINALG("sytrd", total(fc) * D.depth());
    // Degenerate case
    if (M < 3) [[unlikely]] {
        if (W.rows() > 0 && W.cols() > 0)
            W.set_constant(T{}); // identity
        return;
    }
    return micro_kernels::sytrd::sytrd_register<T, Abi, Conf>(D, W, Y);
}

template <class T, class Abi, micro_kernels::geqrf::KernelConfig Conf, StorageOrder OA,
          StorageOrder OD, StorageOrder OB>
void sytrd_apply(view<const T, Abi, OA> A, view<T, Abi, OD> D, view<const T, Abi, OB> B,
                 view<const T, Abi> W, bool transposed) {
    const index_t k = A.rows();
    if (k == 0)
        return;
    if (A.data() != D.data())
        linalg::copy(A.top_rows(1), D.top_rows(1));
    geqrf_apply<T, Abi, Conf>(A.bottom_rows(k - 1), D.bottom_rows(k - 1),
                              B.bottom_left(k - 1, k - 1), W, transposed, false);
}
} // namespace detail

/// @addtogroup topic-linalg
/// @{

/// @name Tridiagonalization of batches of matrices
/// @{

/// Tridiagonalization. The resulting diagonal and subdiagonal elements overwrite D, and the
/// Householder vectors are stored below the subdiagonal (with the first component implicitly equal
/// to 1). The Householder coefficients are stored in W, which should either be a vector of
/// `A.cols() - 1` elements, or a matrix of size `sytrd_size_W(A)`. If W has zero rows, the
/// coefficients are discarded. The workspace Y is used internally and should have size
/// `sytrd_Y_size(D)`.
template <simdifiable VD, simdifiable VW, simdifiable VY>
    requires simdify_compatible<VD, VW, VY>
void sytrd(Structured<VD, MatrixStructure::LowerTriangular> D, VW &&W, VY &&Y) {
    detail::sytrd<simdified_value_t<VD>, simdified_abi_t<VD>, {}>(simdify(D.value), simdify(W),
                                                                  simdify(Y));
}

template <simdifiable VA, simdifiable VD, simdifiable VB, simdifiable VW>
    requires simdify_compatible<VA, VD, VB, VW>
void sytrd_apply(VA &&A, VD &&D, Structured<VB, MatrixStructure::LowerTriangular> B, VW &&W,
                 bool transposed = false) {
    detail::sytrd_apply<simdified_value_t<VD>, simdified_abi_t<VD>, {}>(
        simdify(A).as_const(), simdify(D), simdify(B.value).as_const(), simdify(W).as_const(),
        transposed);
}

template <simdifiable VD, simdifiable VB, simdifiable VW>
    requires simdify_compatible<VD, VB, VW>
void sytrd_apply(VD &&D, Structured<VB, MatrixStructure::LowerTriangular> B, VW &&W,
                 bool transposed = false) {
    detail::sytrd_apply<simdified_value_t<VD>, simdified_abi_t<VD>, {}>(
        simdify(D).as_const(), simdify(D), simdify(B.value).as_const(), simdify(W).as_const(),
        transposed);
}

/// Get the size of the storage for the matrix W returned by
/// @ref sytrd(Structured<VD, MatrixStructure::LowerTriangular> D, VW &&W, VY &&Y).
template <simdifiable VD>
auto sytrd_size_W(VD &&D) {
    return micro_kernels::sytrd::sytrd_W_size<const simdified_value_t<VD>, simdified_abi_t<VD>>(
        simdify(D).as_const());
}

/// Get the size of the storage for the matrix Y used by
/// @ref sytrd(Structured<VD, MatrixStructure::LowerTriangular> D, VW &&W, VY &&Y).
template <simdifiable VD>
auto sytrd_size_Y(VD &&D) {
    return micro_kernels::sytrd::sytrd_Y_size<const simdified_value_t<VD>, simdified_abi_t<VD>>(
        simdify(D).as_const());
}

/// @}

/// @}

} // namespace batmat::linalg
