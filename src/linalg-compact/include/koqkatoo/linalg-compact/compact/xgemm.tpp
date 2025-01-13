#pragma once

#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/trace.hpp>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xgemm.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xtrmm.hpp>
#include "util.hpp"

#include <algorithm>
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
    KOQKATOO_TRACE("xgemm", 0, A.rows() * A.cols() * B.cols() * A.depth());
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
void CompactBLAS<Abi>::xgemv_ref(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C) {
    KOQKATOO_TRACE("xgemv", 0, A.rows() * A.cols() * B.cols() * A.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = 0; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<Abi, {}>(
                A.block(i, l, ni, nl), B.middle_rows(l, nl),
                C.middle_rows(i, ni)); // TODO: optimized microkernel
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_neg_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    KOQKATOO_TRACE("xgemm_neg", 0, A.rows() * A.cols() * B.cols() * A.depth());
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
void CompactBLAS<Abi>::xtrmm_RLNN_neg_ref(single_batch_view A,
                                          single_batch_view B,
                                          mut_single_batch_view C) {
    [[maybe_unused]] auto [m, M] = std::minmax({B.rows(), B.cols()});
    KOQKATOO_TRACE("xtrmm_RLNN_neg", 0,
                   (m * (m + 1) / 2 + (M - m) * m) * A.rows() * A.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.rows() >= B.cols());
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = 0; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    static constexpr index_t R = micro_kernels::gemm::RowsReg;
    static_assert(R == micro_kernels::gemm::ColsReg);
    micro_kernels::single_batch_matrix_accessor<Abi> A_ = A, B_ = B;
    micro_kernels::mut_single_batch_matrix_accessor<Abi> C_ = C;
    foreach_chunked(
        0, C.cols(), R,
        [&](index_t c) {
            foreach_chunked(
                0, C.rows(), R,
                [&](index_t r) {
                    micro_kernels::trmm::xtrmm_rlnn_microkernel<
                        Abi, {.negate = true}, R, R>(
                        A_.block(r, c), B_.block(c, c), C_.block(r, c),
                        B.rows() - c);
                },
                [&](index_t r, index_t nr) {
                    micro_kernels::trmm::microkernel_rlnn_lut<
                        Abi, {.negate = true}>[nr - 1][R - 1](
                        A_.block(r, c), B_.block(c, c), C_.block(r, c),
                        B.rows() - c);
                });
        },
        [&](index_t c, index_t nc) {
            foreach_chunked_merged(0, C.rows(), R, [&](index_t r, auto nr) {
                micro_kernels::trmm::microkernel_rlnn_lut<
                    Abi, {.negate = true}>[nr - 1][nc - 1](
                    A_.block(r, c), B_.block(c, c), C_.block(r, c),
                    B.rows() - c);
            });
        });
    // TODO: cache blocking
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_add_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    KOQKATOO_TRACE("xgemm_add", 0, A.rows() * A.cols() * B.cols() * A.depth());
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
    KOQKATOO_TRACE("xgemm_sub", 0, A.rows() * A.cols() * B.cols() * A.depth());
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
void CompactBLAS<Abi>::xgemv_sub_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    KOQKATOO_TRACE("xgemv_sub", 0, A.rows() * A.cols() * B.cols() * A.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate = true};
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<Abi, conf>(
                A.block(i, l, ni, nl), B.middle_rows(l, nl),
                C.middle_rows(i, ni)); // TODO: optimized microkernel
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_sub_ref(single_batch_view A,
                                        single_batch_view B,
                                        mut_single_batch_view C) {
    KOQKATOO_TRACE("xgemm_TN_sub", 0,
                   A.cols() * A.rows() * B.cols() * A.depth());
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
void CompactBLAS<Abi>::xgemv_T_sub_ref(single_batch_view A, single_batch_view B,
                                       mut_single_batch_view C) {
    KOQKATOO_TRACE("xgemv_T_sub", 0,
                   A.cols() * A.rows() * B.cols() * A.depth());
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate  = true,
                                                            .trans_A = true};
    for (index_t l = 0; l < A.rows(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.rows() - l);
        for (index_t i = 0; i < A.cols(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.cols() - i);
            micro_kernels::gemm::xgemm_register<Abi, conf>(
                A.block(l, i, nl, ni), B.middle_rows(l, nl),
                C.middle_rows(i, ni)); // TODO: optimized microkernel
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_ref(single_batch_view A, single_batch_view B,
                                    mut_single_batch_view C) {
    KOQKATOO_TRACE("xgemm_TN", 0, A.cols() * A.rows() * B.cols() * A.depth());
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

template <class Abi>
void CompactBLAS<Abi>::xgemv_T_ref(single_batch_view A, single_batch_view B,
                                   mut_single_batch_view C) {
    KOQKATOO_TRACE("xgemv_T", 0, A.cols() * A.rows() * B.cols() * A.depth());
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate  = false,
                                                            .trans_A = true};
    for (index_t i = 0; i < C.rows(); ++i)
        simd{0}.copy_to(&C(0, i, 0), stdx::vector_aligned);
    for (index_t l = 0; l < A.rows(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.rows() - l);
        for (index_t i = 0; i < A.cols(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.cols() - i);
            micro_kernels::gemm::xgemm_register<Abi, conf>(
                A.block(l, i, nl, ni), B.middle_rows(l, nl),
                C.middle_rows(i, ni)); // TODO: optimized microkernel
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
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemm_mkl_compact", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemm_batched", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), A.layer_stride(),
                B.data, B.outer_stride(), B.layer_stride(), real_t{0}, C.data,
                C.outer_stride(), C.layer_stride(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemv(batch_view A, batch_view B, mut_batch_view C,
                             PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemv_mkl_compact", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemv_batched", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            linalg::xgemv_batch_strided(
                CblasColMajor, CblasNoTrans, A.rows(), A.cols(), real_t{1},
                A.data, A.outer_stride(), A.layer_stride(), B.data, index_t{1},
                B.layer_stride(), real_t{0}, C.data, index_t{1},
                C.layer_stride(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemv_ref(A.batch(i), B.batch(i), C.batch(i));
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
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemm_neg_mkl_compact", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemm_neg_batched", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(),
                A.layer_stride(), B.data, B.outer_stride(), B.layer_stride(),
                real_t{0}, C.data, C.outer_stride(), C.layer_stride(),
                A.depth());
            return;
        }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_neg_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrmm_RLNN_neg(batch_view A, batch_view B,
                                      mut_batch_view C, PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xtrmm_RLNN_neg(A.batch(i), B.batch(i), C.batch(i), b);
    // TODO: batched BLAS version
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
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemm_add_mkl_compact", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemm_add_batched", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), A.layer_stride(),
                B.data, B.outer_stride(), B.layer_stride(), real_t{1}, C.data,
                C.outer_stride(), C.layer_stride(), A.depth());
            return;
        }
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
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemm_sub_mkl_compact", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemm_sub_batched", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(),
                A.layer_stride(), B.data, B.outer_stride(), B.layer_stride(),
                real_t{1}, C.data, C.outer_stride(), C.layer_stride(),
                A.depth());
            return;
        }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_sub_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemv_sub(batch_view A, batch_view B, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemv_sub_mkl_compact", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemv_sub_batched", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            linalg::xgemv_batch_strided(
                CblasColMajor, CblasNoTrans, A.rows(), A.cols(), real_t{-1},
                A.data, A.outer_stride(), A.layer_stride(), B.data, index_t{1},
                B.layer_stride(), real_t{1}, C.data, index_t{1},
                C.layer_stride(), A.depth());
            return;
        }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemv_sub_ref(A.batch(i), B.batch(i), C.batch(i));
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
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemm_TN_sub_mkl_compact", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemm_TN_sub_batched", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            linalg::xgemm_batch_strided(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.outer_stride(),
                A.layer_stride(), B.data, B.outer_stride(), B.layer_stride(),
                real_t{1}, C.data, C.outer_stride(), C.layer_stride(),
                A.depth());
            return;
        }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_TN_sub_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemv_T_sub(batch_view A, batch_view B, mut_batch_view C,
                                   PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemv_T_sub_mkl_compact", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemv_T_sub_batched", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            linalg::xgemv_batch_strided(
                CblasColMajor, CblasTrans, A.rows(), A.cols(), real_t{-1},
                A.data, A.outer_stride(), A.layer_stride(), B.data, index_t{1},
                B.layer_stride(), real_t{1}, C.data, index_t{1},
                C.layer_stride(), A.depth());
            return;
        }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemv_T_sub_ref(A.batch(i), B.batch(i), C.batch(i));
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
        if (use_mkl_compact(b) && A.has_full_layer_stride() &&
            B.has_full_layer_stride() && C.has_full_layer_stride()) {
            KOQKATOO_TRACE("xgemm_TN_mkl_compact", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            xgemm_compact(
                MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xgemm_TN_batched", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            linalg::xgemm_batch_strided(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.outer_stride(), A.layer_stride(),
                B.data, B.outer_stride(), B.layer_stride(), real_t{0}, C.data,
                C.outer_stride(), C.layer_stride(), A.depth());
            return;
        }
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
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemm_blas", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
        }
    xgemm_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemv(single_batch_view A, single_batch_view B,
                             mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemv_blas", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            return linalg::xgemv(CblasColMajor, CblasNoTrans, A.rows(),
                                 A.cols(), real_t{1}, A.data, A.outer_stride(),
                                 B.data, index_t{1}, real_t{0}, C.data,
                                 index_t{1});
        }
    xgemv_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_neg(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemm_neg_blas", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
        }
    xgemm_neg_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xtrmm_RLNN_neg(single_batch_view A, single_batch_view B,
                                      mut_single_batch_view C,
                                      PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.rows() >= B.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            [[maybe_unused]] auto [m, M] = std::minmax({B.rows(), B.cols()});
            KOQKATOO_TRACE("xtrmm_RLNN_neg_blas", 0,
                           (m * (m + 1) / 2 + (M - m) * m) * A.rows() *
                               A.depth());
            auto A1 = A.left_cols(B.cols()), B1 = B.top_rows(B.cols());
            C = A1;
            linalg::xtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                          CblasNonUnit, C.rows(), C.cols(), real_t{-1}, B1.data,
                          B1.outer_stride(), C.data, C.outer_stride());
            if (B.rows() > B.cols()) { // lower trapezoidal
                auto A2 = A.right_cols(A.cols() - B.cols()),
                     B2 = B.bottom_rows(B.rows() - B.cols());
                linalg::xgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                              A2.rows(), B2.cols(), A2.cols(), real_t{-1},
                              A2.data, A2.outer_stride(), B2.data,
                              B2.outer_stride(), real_t{1}, C.data,
                              C.outer_stride());
            }
            return;
        }
    xtrmm_RLNN_neg_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_add(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemm_add_blas", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
        }
    xgemm_add_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_sub(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemm_sub_blas", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
        }
    xgemm_sub_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemv_sub(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    assert(B.cols() == 1);
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemv_sub_blas", 0,
                           A.rows() * A.cols() * B.cols() * A.depth());
            return linalg::xgemv(CblasColMajor, CblasNoTrans, A.rows(),
                                 A.cols(), real_t{-1}, A.data, A.outer_stride(),
                                 B.data, index_t{1}, real_t{1}, C.data,
                                 index_t{1});
        }
    xgemv_sub_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_sub(single_batch_view A, single_batch_view B,
                                    mut_single_batch_view C,
                                    PreferredBackend b) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemm_TN_sub_blas", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            return linalg::xgemm(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
        }
    xgemm_TN_sub_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemv_T_sub(single_batch_view A, single_batch_view B,
                                   mut_single_batch_view C,
                                   PreferredBackend b) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemv_T_sub_blas", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            return linalg::xgemv(CblasColMajor, CblasTrans, A.rows(), A.cols(),
                                 real_t{-1}, A.data, A.outer_stride(), B.data,
                                 index_t{1}, real_t{1}, C.data, index_t{1});
        }
    xgemv_T_sub_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN(single_batch_view A, single_batch_view B,
                                mut_single_batch_view C, PreferredBackend b) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xgemm_TN_blas", 0,
                           A.cols() * A.rows() * B.cols() * A.depth());
            return linalg::xgemm(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
        }
    xgemm_TN_ref(A, B, C);
}

} // namespace koqkatoo::linalg::compact
