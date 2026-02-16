#pragma once

#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, MatrixStructure SA, StorageOrder OA>
    requires(SA == MatrixStructure::LowerTriangular)
void symm(view<const T, Abi, OA> A, view<const T, Abi> B, std::optional<view<const T, Abi>> C,
          view<T, Abi> D) {
    GUANAQO_TRACE_LINALG("symm", A.rows() * A.cols() * B.cols() * A.depth());
    // Check dimensions
    BATMAT_ASSERT(!C || C->rows() == D.rows());
    BATMAT_ASSERT(!C || C->cols() == D.cols());
    BATMAT_ASSERT(A.rows() == A.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == B.rows());
    BATMAT_ASSERT(B.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols(), K = A.cols();

    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;

    if (C) {
        uview<const T, Abi, OA> A_                     = A;
        uview<const T, Abi, StorageOrder::ColMajor> B_ = B;
        uview<const T, Abi, StorageOrder::ColMajor> C_ = *C;
        uview<T, Abi, StorageOrder::ColMajor> D_       = D;
        for (index_t j = 0; j < N; ++j)
            for (index_t l = 0; l < K; ++l) {
                auto Blj = B_.load(l, j);
                auto All = A_.load(l, l);
                auto Dlj = All * Blj + C_.load(l, j);
                BATMAT_UNROLLED_IVDEP_FOR (4, index_t i = l + 1; i < M; ++i) {
                    auto Ail = A_.load(i, l);
                    auto Bil = B_.load(i, j);
                    D_.store(Ail * Blj + C_.load(i, j), i, j);
                    Dlj += Ail * Bil;
                }
                D_.store(Dlj, l, j);
            }
    } else {
        BATMAT_ASSERT(!"Not implemented"); // TODO
    }
}

} // namespace detail

#if 0 // Not implemented
/// D = A B with A symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD>
    requires simdify_compatible<VA, VB, VD>
void symm(Structured<VA, SA> A, VB &&B, VD &&D) {
    std::optional<decltype(simdify(D).as_const())> null;
    detail::symm<simdified_value_t<VA>, simdified_abi_t<VA>, SA>(
        simdify(A.value).as_const(), simdify(B).as_const(), null, simdify(D));
}
#endif

/// D = C + A B with A symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VC, simdifiable VD>
    requires simdify_compatible<VA, VB, VC, VD>
void symm_add(Structured<VA, SA> A, VB &&B, VC &&C, VD &&D) {
    detail::symm<simdified_value_t<VA>, simdified_abi_t<VA>, SA>(
        simdify(A.value).as_const(), simdify(B).as_const(), simdify(C).as_const(), simdify(D));
}
/// D = D + A B with A symmetric
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD>
    requires simdify_compatible<VA, VB, VD>
void symm_add(Structured<VA, SA> A, VB &&B, VD &&D) {
    symm_add(A.ref(), B, D, D);
}

} // namespace batmat::linalg
