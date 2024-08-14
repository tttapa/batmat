#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>

#include "koqkatoo/unroll.h"
#include "micro_kernels/xtrsm.tpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrsm_RLTN_ref(single_batch_view L,
                                      mut_single_batch_view H) {
    assert(L.rows() == L.cols());
    assert(H.cols() == L.rows());
    micro_kernels::trsm::trsm_register<simd, {.trans = true}>(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLNN_ref(single_batch_view L,
                                      mut_single_batch_view H) {
    assert(L.rows() == L.cols());
    assert(L.rows() == H.rows());
    micro_kernels::trsm::trsm_register<simd, {.trans = false}>(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLTN_ref(single_batch_view L,
                                      mut_single_batch_view H) {
    assert(L.rows() == L.cols());
    assert(L.rows() == H.rows());
    const auto n = L.rows(), m = H.cols();
    for (index_t j = 0; j < m; ++j) {
        for (index_t i = H.rows(); i-- > 0;) {
            simd Hij = {&H(0, i, j), stdx::vector_aligned};
            KOQKATOO_UNROLLED_IVDEP_FOR (4, index_t k = i + 1; k < n; ++k) {
                simd Hkj = {&H(0, k, j), stdx::vector_aligned};
                Hij -= Hkj * simd{&L(0, k, i), stdx::vector_aligned};
            }
            simd pivot = {&L(0, i, i), stdx::vector_aligned};
            Hij /= pivot;
            aligned_store(&H(0, i, j), Hij);
        }
    }
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrsm_RLTN(batch_view L, mut_batch_view H,
                                  PreferredBackend b) {
    assert(L.depth() == H.depth());
    assert(L.rows() == L.cols());
    assert(H.cols() == L.rows());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xtrsm_compact(MKL_COL_MAJOR, MKL_RIGHT, MKL_LOWER, MKL_TRANS,
                                 MKL_NONUNIT, H.rows(), H.cols(), real_t{1},
                                 L.data, L.rows(), H.data, H.rows(),
                                 vector_format_mkl<real_t, Abi>::format,
                                 L.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return xtrsm_batch_strided(CblasColMajor, CblasRight, CblasLower,
                                       CblasTrans, CblasNonUnit, H.rows(),
                                       H.cols(), real_t{1}, L.data, L.rows(),
                                       L.rows() * L.cols(), H.data, H.rows(),
                                       H.rows() * H.cols(), L.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrsm_RLTN_ref(L.batch(i), H.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLNN(batch_view L, mut_batch_view H,
                                  PreferredBackend b) {
    assert(L.depth() == H.depth());
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER,
                                 MKL_NOTRANS, MKL_NONUNIT, H.rows(), H.cols(),
                                 real_t{1}, L.data, L.rows(), H.data, H.rows(),
                                 vector_format_mkl<real_t, Abi>::format,
                                 L.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return xtrsm_batch_strided(CblasColMajor, CblasLeft, CblasLower,
                                       CblasNoTrans, CblasNonUnit, H.rows(),
                                       H.cols(), real_t{1}, L.data, L.rows(),
                                       L.rows() * L.cols(), H.data, H.rows(),
                                       H.rows() * H.cols(), L.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrsm_LLNN_ref(L.batch(i), H.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLTN(batch_view L, mut_batch_view H,
                                  PreferredBackend b) {
    assert(L.depth() == H.depth());
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_TRANS,
                                 MKL_NONUNIT, H.rows(), H.cols(), real_t{1},
                                 L.data, L.rows(), H.data, H.rows(),
                                 vector_format_mkl<real_t, Abi>::format,
                                 L.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return xtrsm_batch_strided(CblasColMajor, CblasLeft, CblasLower,
                                       CblasTrans, CblasNonUnit, H.rows(),
                                       H.cols(), real_t{1}, L.data, L.rows(),
                                       L.rows() * L.cols(), H.data, H.rows(),
                                       H.rows() * H.cols(), L.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrsm_LLTN_ref(L.batch(i), H.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrsm_RLTN(single_batch_view L, mut_single_batch_view H,
                                  PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(H.cols() == L.rows());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xtrsm(CblasColMajor, CblasRight, CblasLower,
                                 CblasTrans, CblasNonUnit, H.rows(), H.cols(),
                                 real_t{1}, L.data, L.outer_stride(), H.data,
                                 H.outer_stride());
    xtrsm_RLTN_ref(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLNN(single_batch_view L, mut_single_batch_view H,
                                  PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xtrsm(CblasColMajor, CblasLeft, CblasLower,
                                 CblasNoTrans, CblasNonUnit, H.rows(), H.cols(),
                                 real_t{1}, L.data, L.outer_stride(), H.data,
                                 H.outer_stride());
    xtrsm_LLNN_ref(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLTN(single_batch_view L, mut_single_batch_view H,
                                  PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xtrsm(CblasColMajor, CblasLeft, CblasLower,
                                 CblasTrans, CblasNonUnit, H.rows(), H.cols(),
                                 real_t{1}, L.data, L.outer_stride(), H.data,
                                 H.outer_stride());
    xtrsm_LLTN_ref(L, H);
}

} // namespace koqkatoo::linalg::compact
