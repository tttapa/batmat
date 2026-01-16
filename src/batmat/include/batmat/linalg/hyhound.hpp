#pragma once

#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/micro-kernels/hyhound.hpp>
#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {

template <class T, class Abi, micro_kernels::hyhound::KernelConfig Conf, StorageOrder OL,
          StorageOrder OA>
void hyhound_diag(view<T, Abi, OL> L, view<T, Abi, OA> A, view<const T, Abi> D) {
    const index_t k = A.cols();
    BATMAT_ASSERT(L.rows() >= L.cols());
    BATMAT_ASSERT(L.rows() == A.rows());
    BATMAT_ASSERT(A.cols() == D.rows());
    [[maybe_unused]] const index_t flop_count = total(flops::hyh(L.rows(), L.cols(), k));
    GUANAQO_TRACE("hyhound_diag", 0, flop_count * L.depth());
    if (k == 0) [[unlikely]]
        return;
    return micro_kernels::hyhound::hyhound_diag_register<T, Abi, Conf>(L, A, D);
}

template <class T, class Abi, micro_kernels::hyhound::KernelConfig Conf, StorageOrder OL,
          StorageOrder OA>
void hyhound_diag(view<T, Abi, OL> L, view<T, Abi, OA> A, view<const T, Abi> D, view<T, Abi> W) {
    using namespace micro_kernels::hyhound;
    const index_t k = A.cols();
    BATMAT_ASSERT(L.rows() >= L.cols());
    BATMAT_ASSERT(L.rows() == A.rows());
    BATMAT_ASSERT(A.cols() == D.rows());
    BATMAT_ASSERT(std::make_pair(W.rows(), W.cols()) == (xshhud_W_size<T, Abi>)(L));
    [[maybe_unused]] const index_t flop_count = total(flops::hyh(L.rows(), L.cols(), k));
    GUANAQO_TRACE("hyhound_diag", 0, flop_count * L.depth());
    if (k == 0) [[unlikely]]
        return;
    return hyhound_diag_register<T, Abi, Conf>(L, A, D, W);
}

template <class T, class Abi, micro_kernels::hyhound::KernelConfig Conf, StorageOrder OL,
          StorageOrder OA>
void hyhound_diag_apply(view<T, Abi, OL> L, view<const T, Abi, OA> Ain, view<T, Abi, OA> Aout,
                        view<const T, Abi, OA> B, view<const T, Abi> D, view<const T, Abi> W,
                        index_t k_nonzero_start, index_t k_nonzero_end) {
    using namespace micro_kernels::hyhound;
    const index_t k = Ain.cols();
    BATMAT_ASSERT(Ain.rows() == Aout.rows());
    BATMAT_ASSERT(Ain.cols() == Aout.cols());
    BATMAT_ASSERT(Ain.rows() == L.rows());
    BATMAT_ASSERT(B.rows() == L.cols());
    BATMAT_ASSERT(B.cols() == Ain.cols());
    BATMAT_ASSERT(D.rows() == Ain.cols());
    BATMAT_ASSERT(std::make_pair(W.rows(), W.cols()) == (xshhud_W_size<T, Abi>)(L));
    if (k_nonzero_end == -1)
        k_nonzero_end = k;
    BATMAT_ASSERT(0 <= k_nonzero_start && k_nonzero_start <= k_nonzero_end && k_nonzero_end <= k);
    // Note: ignoring initial zero values of A in the FLOP count for simplicity (for large matrices
    //       this does not matter)
    [[maybe_unused]] const index_t flop_count = total(flops::hyh_apply(L.rows(), L.cols(), k));
    GUANAQO_TRACE("hyhound_diag_apply", 0, flop_count * L.depth());
    if (k == 0) [[unlikely]]
        return;
    return hyhound_diag_apply_register<T, Abi, Conf>(L, Ain, Aout, B, D, W, k_nonzero_start,
                                                     k_nonzero_end);
}

template <class T, class Abi, micro_kernels::hyhound::KernelConfig Conf, StorageOrder OL1,
          StorageOrder OA1, StorageOrder OL2, StorageOrder OA2>
void hyhound_diag_2(view<T, Abi, OL1> L11, view<T, Abi, OA1> A1, view<T, Abi, OL2> L21,
                    view<T, Abi, OA2> A2, view<const T, Abi> D) {
    const index_t k = A1.cols(), m = L11.rows() + L21.rows();
    BATMAT_ASSERT(L11.rows() >= L11.cols());
    BATMAT_ASSERT(L11.rows() == A1.rows());
    BATMAT_ASSERT(A1.cols() == D.rows());
    BATMAT_ASSERT(A2.cols() == A1.cols());
    BATMAT_ASSERT(L21.cols() == L11.cols());
    [[maybe_unused]] const index_t flop_count = total(flops::hyh(m, L11.cols(), k));
    GUANAQO_TRACE("hyhound_diag_2", 0, flop_count * L11.depth());
    if (k == 0) [[unlikely]]
        return;
    return micro_kernels::hyhound::hyhound_diag_2_register<T, Abi, OL1, OA1, OL2, OA2, Conf>(
        L11, A1, L21, A2, D);
}

template <class T, class Abi, micro_kernels::hyhound::KernelConfig Conf, StorageOrder OL,
          StorageOrder OW, StorageOrder OY, StorageOrder OU>
void hyhound_diag_cyclic(view<T, Abi, OL> L11, view<T, Abi, OW> A1, view<T, Abi, OY> L21,
                         view<const T, Abi, OW> A2, view<T, Abi, OW> A2_out, view<T, Abi, OU> L31,
                         view<const T, Abi, OW> A3, view<T, Abi, OW> A3_out, view<const T, Abi> D,
                         index_t split_A) {
    const index_t k = A1.cols(), m = L11.rows() + L21.rows() + L31.rows();
    BATMAT_ASSERT(L11.rows() >= L11.cols());
    BATMAT_ASSERT(L11.rows() == A1.rows());
    BATMAT_ASSERT(L21.rows() == A2.rows());
    BATMAT_ASSERT(L31.rows() == A3.rows());
    BATMAT_ASSERT(A1.cols() == D.rows());
    BATMAT_ASSERT(A2.cols() == A1.cols());
    BATMAT_ASSERT(A3.cols() == A1.cols());
    BATMAT_ASSERT(L21.cols() == L11.cols());
    BATMAT_ASSERT(L31.cols() == L11.cols());
    // Note: ignoring initial zero values of A in the FLOP count for simplicity (for large matrices
    //       this does not matter)
    [[maybe_unused]] const index_t flop_count = total(flops::hyh(m, L11.cols(), k));
    GUANAQO_TRACE("hyhound_diag_cyclic", 0, flop_count * L11.depth());
    if (k == 0) [[unlikely]]
        return;
    return micro_kernels::hyhound::hyhound_diag_cyclic_register<T, Abi, OL, OW, OY, OU, Conf>(
        L11, A1, L21, A2, A2_out, L31, A3, A3_out, D, split_A);
}

template <class T, class Abi, micro_kernels::hyhound::KernelConfig Conf, StorageOrder OL,
          StorageOrder OA, StorageOrder OLu, StorageOrder OAu>
void hyhound_diag_riccati(view<T, Abi, OL> L11, view<T, Abi, OA> A1, view<T, Abi, OL> L21,
                          view<const T, Abi, OA> A2, view<T, Abi, OA> A2_out, view<T, Abi, OLu> Lu1,
                          view<T, Abi, OAu> Au_out, view<const T, Abi> D, bool shift_A_out) {
    const index_t k = A1.cols(), m = L11.rows() + L21.rows() + Lu1.rows();
    BATMAT_ASSERT(L11.rows() >= L11.cols());
    BATMAT_ASSERT(L11.rows() == A1.rows());
    BATMAT_ASSERT(L21.rows() == A2.rows());
    BATMAT_ASSERT(A2_out.rows() == A2.rows());
    BATMAT_ASSERT(A2_out.cols() == A2.cols());
    BATMAT_ASSERT(Lu1.rows() == Au_out.rows());
    BATMAT_ASSERT(A1.cols() == D.rows());
    BATMAT_ASSERT(A2.cols() == A1.cols());
    BATMAT_ASSERT(L21.cols() == L11.cols());
    BATMAT_ASSERT(Lu1.cols() == L11.cols());
    // Note: ignoring upper trapezoidal shape of Lu and initial zero value of Au for simplicity
    //       (for large matrices this does not matter)
    [[maybe_unused]] index_t flop_count = total(flops::hyh(m, L11.cols(), k));
    GUANAQO_TRACE("hyhound_diag_riccati", 0, flop_count * L11.depth());
    if (k == 0) [[unlikely]]
        return;
    return micro_kernels::hyhound::hyhound_diag_riccati_register<T, Abi, OL, OA, OLu, OAu, Conf>(
        L11, A1, L21, A2, A2_out, Lu1, Au_out, D, shift_A_out);
}

} // namespace detail

/// Update Cholesky factor L using low-rank term A diag(d) Aᵀ.
template <MatrixStructure SL, simdifiable VL, simdifiable VA, simdifiable Vd>
    requires simdify_compatible<VL, VA, Vd>
void hyhound_diag(Structured<VL, SL> L, VA &&A, Vd &&d) {
    static_assert(SL == MatrixStructure::LowerTriangular); // TODO
    detail::hyhound_diag<simdified_value_t<VL>, simdified_abi_t<VL>, {}>(
        simdify(L.value), simdify(A), simdify(d).as_const());
}

/// Update Cholesky factor L using low-rank term A diag(d) Aᵀ, with full Householder representation.
template <MatrixStructure SL, simdifiable VL, simdifiable VA, simdifiable Vd, simdifiable VW>
    requires simdify_compatible<VL, VA, Vd, VW>
void hyhound_diag(Structured<VL, SL> L, VA &&A, Vd &&d, VW &&W) {
    static_assert(SL == MatrixStructure::LowerTriangular); // TODO
    detail::hyhound_diag<simdified_value_t<VL>, simdified_abi_t<VL>, {}>(
        simdify(L.value), simdify(A), simdify(d).as_const(), simdify(W));
}

/// Get the size of the storage for the matrix W returned by
/// @ref hyhound_diag(Structured<VL,SL>, VA&&, Vd&&, VW&&).
template <MatrixStructure SL, simdifiable VL>
auto hyhound_size_W(Structured<VL, SL> L) {
    return micro_kernels::hyhound::xshhud_W_size<const simdified_value_t<VL>, simdified_abi_t<VL>>(
        simdify(L.value).as_const());
}

/// Apply Householder transformation generated by hyhound_diag.
/// Only the part of A between columns @p kA_nonzero_start (inclusive) and
/// @p kA_nonzero_end (exclusive) is assumed to be non-zero. A value of -1 for @p kA_nonzero_end
/// indicates that all columns until the end are non-zero.
template <simdifiable VL, simdifiable VA, simdifiable VD, simdifiable VB, simdifiable Vd,
          simdifiable VW>
    requires simdify_compatible<VL, VA, VD, VB, Vd, VW>
void hyhound_diag_apply(VL &&L, VA &&A, VD &&D, VB &&B, Vd &&d, VW &&W,
                        index_t kA_nonzero_start = 0, index_t kA_nonzero_end = -1) {
    detail::hyhound_diag_apply<simdified_value_t<VL>, simdified_abi_t<VL>, {}>(
        simdify(L), simdify(A).as_const(), simdify(D), simdify(B).as_const(), simdify(d).as_const(),
        simdify(W).as_const(), kA_nonzero_start, kA_nonzero_end);
}

/// Update Cholesky factor L using low-rank term A diag(copysign(1, d)) Aᵀ,
/// where d contains only ±0 values.
template <MatrixStructure SL, simdifiable VL, simdifiable VA, simdifiable Vd>
    requires simdify_compatible<VL, VA, Vd>
void hyhound_sign(Structured<VL, SL> L, VA &&A, Vd &&d) {
    detail::hyhound_diag<simdified_value_t<VL>, simdified_abi_t<VL>, {.sign_only = true}>(
        simdify(L.value), simdify(A), simdify(d).as_const());
}

/// Update Cholesky factor L using low-rank term A diag(d) Aᵀ, where L and A are stored as two
/// separate block rows.
/// @f[
///     L = \begin{pmatrix} L_{11} \\ L_{21} \end{pmatrix}, \quad
///     A = \begin{pmatrix} A_{1} \\ A_{2} \end{pmatrix}.
/// @f]
template <MatrixStructure SL, simdifiable VL1, simdifiable VA1, simdifiable VL2, simdifiable VA2,
          simdifiable Vd>
    requires simdify_compatible<VL1, VA1, VL2, VA2, Vd>
void hyhound_diag_2(Structured<VL1, SL> L1, VA1 &&A1, VL2 &&L2, VA2 &&A2, Vd &&d) {
    detail::hyhound_diag_2<simdified_value_t<VL1>, simdified_abi_t<VL1>, {}, StorageOrder::ColMajor,
                           StorageOrder::ColMajor, StorageOrder::ColMajor, StorageOrder::ColMajor>(
        simdify(L1.value), simdify(A1), simdify(L2), simdify(A2), simdify(d).as_const());
}

/// Update structured Cholesky factor L using structured low-rank term A diag(d) Aᵀ,
/// @f[
///     L = \begin{pmatrix} L_{11} \\ L_{21} \\ L_{31} \end{pmatrix}, \quad
///     A = \begin{pmatrix} A_{11} & A_{12} \\ 0 & A_{22} \\ A_{31} & 0 \end{pmatrix}, \quad
///     \tilde A = \begin{pmatrix} 0 \\ \tilde A_{2} \\ \tilde A_{3} \end{pmatrix}.
/// @f]
/// @todo   The matrices A22 and A31 are embedded in A2 and A3 respectively, which are the same
///         width as A1. This is wasteful. We should consider passing in views for A22 and A31 only.
template <MatrixStructure SL, simdifiable VL11, simdifiable VA1, simdifiable VL21, simdifiable VA2,
          simdifiable VA2o, simdifiable VU, simdifiable VA3, simdifiable VA3o, simdifiable Vd>
    requires simdify_compatible<VL11, VA1, VL21, VA2, VA2o, VU, VA3, VA3o, Vd>
void hyhound_diag_cyclic(Structured<VL11, SL> L11, VA1 &&A1, VL21 &&L21, VA2 &&A2, VA2o &&A2_out,
                         VU &&L31, VA3 &&A3, VA3o &&A3_out, Vd &&d, index_t split_A) {
    detail::hyhound_diag_cyclic<simdified_value_t<VL11>, simdified_abi_t<VL11>, {},
                                StorageOrder::ColMajor, StorageOrder::ColMajor,
                                StorageOrder::ColMajor, StorageOrder::ColMajor>(
        simdify(L11.value), simdify(A1), simdify(L21), simdify(A2).as_const(), simdify(A2_out),
        simdify(L31), simdify(A3).as_const(), simdify(A3_out), simdify(d).as_const(), split_A);
}

/// Update structured Cholesky factor L using structured low-rank term A diag(d) Aᵀ,
/// @f[
///     L = \begin{pmatrix} L_{11} \\ L_{21} \\ L_{u} \end{pmatrix}, \quad
///     A = \begin{pmatrix} A_{1} \\ A_{2} \\ 0 \end{pmatrix}, \quad
///     \tilde A = \begin{pmatrix} 0 \\ \tilde A_{2} \\ \tilde A_{u} \end{pmatrix}.
/// @f]
/// The @p shift_A_out parameter indicates whether the output matrix A2_out should be shifted along
/// the batch dimension. This is used in the Cyqlone solver.
template <MatrixStructure SL, simdifiable VL11, simdifiable VA1, simdifiable VL21, simdifiable VA2,
          simdifiable VA2o, simdifiable VLu1, simdifiable VAuo, simdifiable Vd>
    requires simdify_compatible<VL11, VA1, VL21, VA2, VA2o, VLu1, VAuo, Vd>
void hyhound_diag_riccati(Structured<VL11, SL> L11, VA1 &&A1, VL21 &&L21, VA2 &&A2, VA2o &&A2_out,
                          VLu1 &&Lu1, VAuo &&Au_out, Vd &&d, bool shift_A_out = false) {
    detail::hyhound_diag_riccati<simdified_value_t<VL11>, simdified_abi_t<VL11>, {},
                                 StorageOrder::ColMajor, StorageOrder::ColMajor,
                                 StorageOrder::ColMajor, StorageOrder::ColMajor>(
        simdify(L11.value), simdify(A1), simdify(L21), simdify(A2).as_const(), simdify(A2_out),
        simdify(Lu1), simdify(Au_out), simdify(d).as_const(), shift_A_out);
}

} // namespace batmat::linalg
