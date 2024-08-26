#pragma once

#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xgemm.hpp>
#include "util.hpp"

#include <concepts>

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

// TODO: pick optimal values depending on real_t and system's cache sizes
static constexpr index_t GemmBlockSizeRows = 32;
static constexpr index_t GemmBlockSizeCols = 48;

template <class Abi>
void CompactBLAS<Abi>::xgemm_ref(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = 0; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<Abi, {}>(A.block(i, l, ni, nl),
                                                         B.middle_rows(l, nl),
                                                         C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_neg_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = 0; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<Abi, {.negate = true}>(
                A.block(i, l, ni, nl), B.middle_rows(l, nl),
                C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_add_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate = false};
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    const index_t M = C.rows(), N = C.cols(), K = A.cols();
    if (M == 0 || N == 0 || K == 0)
        return;
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    // TODO: cache blocking
    static const index_t L1_cache_size = 48_KiB; // TODO: determine dynamically
    static const index_t L2_cache_size = 512_KiB;
    static const index_t L3_cache_size = 16_MiB;
    static const index_t n_cores       = 10; // TODO: OMP
    static const index_t N_reg         = micro_kernels::gemm::ColsReg;
    static const index_t M_reg         = micro_kernels::gemm::RowsReg;
    static const index_t K_cache =
        L1_cache_size / sizeof(real_t) / simd_stride / 2 / N_reg;
    static const index_t M_cache = L2_cache_size / sizeof(real_t) /
                                   simd_stride / K_cache / 2 / M_reg * M_reg;
    static const index_t N_cache =
        std::max<index_t>(L3_cache_size / sizeof(real_t) / simd_stride /
                              K_cache / n_cores / M_cache,
                          1) *
        M_cache;

    if (M <= M_cache && N <= N_cache && K <= K_cache)
        return micro_kernels::gemm::xgemm_register<Abi, conf>(A, B, C);

    using sto_t                    = aligned_simd_storage<real_t, simd_align>;
    const index_t B_cache_sto_size = B.ceil_depth() * K_cache * N_cache;
    const index_t A_cache_sto_size = A.ceil_depth() * M_cache * K_cache;
    const index_t B_size           = B.ceil_depth() * K * N;
    const index_t A_size           = A.ceil_depth() * M * K;
    const bool pack_B = B_size >= 2 * B_cache_sto_size; // TODO: tune
    const bool pack_A = A_size >= 2 * A_cache_sto_size; // TODO: tune
    sto_t B_cache_sto{pack_B ? static_cast<size_t>(B_cache_sto_size) : 0,
                      uninitialized};
    sto_t A_cache_sto{pack_A ? static_cast<size_t>(A_cache_sto_size) : 0,
                      uninitialized};

    for (index_t j_cache = 0; j_cache < N; j_cache += N_cache) {
        index_t n_cache = std::min(N_cache, N - j_cache);
        for (index_t p_cache = 0; p_cache < K; p_cache += K_cache) {
            index_t k_cache = std::min(K_cache, K - p_cache);
            auto Bkj        = [&] {
                auto Bkj = B.block(p_cache, j_cache, k_cache, n_cache);
                if (pack_B) {
                    mut_single_batch_view B_cache{{
                               .data       = B_cache_sto.data(),
                               .depth      = B.depth(),
                               .rows       = k_cache,
                               .cols       = n_cache,
                               .batch_size = B.batch_size(),
                    }};
                    // TODO: proper packing
                    xcopy(Bkj, B_cache);
                    return B_cache.as_const();
                }
                return Bkj;
            }();
            for (index_t i_cache = 0; i_cache < M; i_cache += M_cache) {
                index_t m_cache = std::min(M_cache, M - i_cache);
                auto Cij        = C.block(i_cache, j_cache, m_cache, n_cache);
                auto Aik        = [&] {
                    auto Aik = A.block(i_cache, p_cache, m_cache, k_cache);
                    if (pack_A) {
                        mut_single_batch_view A_cache{{
                                   .data       = A_cache_sto.data(),
                                   .depth      = A.depth(),
                                   .rows       = m_cache,
                                   .cols       = k_cache,
                                   .batch_size = A.batch_size(),
                        }};
                        // TODO: proper packing
                        xcopy(Aik, A_cache);
                        return A_cache.as_const();
                    }
                    return Aik;
                }();
                micro_kernels::gemm::xgemm_register<Abi, conf>(Aik, Bkj, Cij);
            }
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_sub_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate = true};
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<Abi, conf>(
                A.block(i, l, ni, nl), B.middle_rows(l, nl),
                C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_sub_ref(single_batch_view A,
                                        single_batch_view B,
                                        mut_single_batch_view C) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate  = true,
                                                            .trans_A = true};
    for (index_t l = 0; l < A.rows(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.rows() - l);
        for (index_t i = 0; i < A.cols(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.cols() - i);
            micro_kernels::gemm::xgemm_register<Abi, conf>(
                A.block(l, i, nl, ni), B.middle_rows(l, nl),
                C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_ref(single_batch_view A, single_batch_view B,
                                    mut_single_batch_view C) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate  = false,
                                                            .trans_A = true};
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = 0; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    for (index_t l = 0; l < A.rows(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.rows() - l);
        for (index_t i = 0; i < A.cols(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.cols() - i);
            micro_kernels::gemm::xgemm_register<Abi, conf>(
                A.block(l, i, nl, ni), B.middle_rows(l, nl),
                C.middle_rows(i, ni));
        }
    }
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xgemm(batch_view A, batch_view B, mut_batch_view C,
                             PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.rows(), B.data, B.rows(),
                real_t{0}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{0}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_neg(batch_view A, batch_view B, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.rows(), B.data, B.rows(),
                real_t{0}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{0}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_neg_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_add(batch_view A, batch_view B, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.rows(), B.data, B.rows(),
                real_t{1}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{1}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < A.num_batches(); ++i)
            xgemm_add_ref(A.batch(i), B.batch(i), C.batch(i));
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_add_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_sub(batch_view A, batch_view B, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.rows(), B.data, B.rows(),
                real_t{1}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{1}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_sub_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_sub(batch_view A, batch_view B,
                                    mut_batch_view C, PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.rows(), B.data, B.rows(),
                real_t{1}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{1}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_TN_sub_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN(batch_view A, batch_view B, mut_batch_view C,
                                PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.rows(), B.data, B.rows(),
                real_t{0}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{0}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_TN_ref(A.batch(i), B.batch(i), C.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xgemm(single_batch_view A, single_batch_view B,
                             mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
    xgemm_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_neg(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
    xgemm_neg_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_add(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
    xgemm_add_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_sub(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
    xgemm_sub_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_sub(single_batch_view A, single_batch_view B,
                                    mut_single_batch_view C,
                                    PreferredBackend b) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
    xgemm_TN_sub_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN(single_batch_view A, single_batch_view B,
                                mut_single_batch_view C, PreferredBackend b) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
    xgemm_TN_ref(A, B, C);
}

} // namespace koqkatoo::linalg::compact
