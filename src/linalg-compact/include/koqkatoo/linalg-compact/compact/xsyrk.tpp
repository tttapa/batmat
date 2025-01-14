#pragma once

#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/aligned-storage.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xgemm.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/trace.hpp>
#include "util.hpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xsyrk_schur(single_batch_view A, single_batch_view d,
                                   mut_single_batch_view C) {
    KOQKATOO_TRACE("xsyrk_schur", 0); // TODO: FLOP count
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    assert(d.rows() == A.cols());
    constexpr micro_kernels::gemm::KernelConfig conf{.trans_B = true};
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = j; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    // TODO: cache blocking
    micro_kernels::gemm::xgemmt_diag_register<Abi, conf>(A, A, C, d);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_schur_add(single_batch_view A, single_batch_view d,
                                       mut_single_batch_view C) {
    KOQKATOO_TRACE("xsyrk_schur_add", 0); // TODO: FLOP count
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    assert(d.rows() == A.cols());
    constexpr micro_kernels::gemm::KernelConfig conf{.trans_B = true};
    // TODO: cache blocking
    micro_kernels::gemm::xgemmt_diag_register<Abi, conf>(A, A, C, d);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T_schur_copy(single_batch_view C,
                                          single_batch_view Σ,
                                          bool_single_batch_view mask,
                                          single_batch_view H_in,
                                          mut_single_batch_view H_out) {
    [[maybe_unused]] const auto op_cnt_syrk =
                                    C.cols() * (C.cols() + 1) * C.rows() / 2,
                                op_cnt_diag = C.cols() * C.rows();
    KOQKATOO_TRACE("xsyrk_T_schur_copy", 0,
                   (op_cnt_syrk + op_cnt_diag) * C.depth());
    for (index_t j = 0; j < H_out.cols(); ++j)
        for (index_t i = j; i < H_out.rows(); ++i)
            aligned_store(&H_out(0, i, j), aligned_load(&H_in(0, i, j)));
    // TODO: cache blocking
    constexpr micro_kernels::gemm::KernelConfig conf{.trans_A = true};
    micro_kernels::gemm::xgemmt_diag_mask_register<Abi, conf>(C, C, H_out, Σ,
                                                              mask);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T_ref(single_batch_view A,
                                   mut_single_batch_view C) {
    [[maybe_unused]] const auto op_cnt_syrk =
        C.cols() * (C.cols() + 1) * C.rows() / 2;
    KOQKATOO_TRACE("xsyrk_T", 0, op_cnt_syrk * C.depth());
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = j; i < C.rows(); ++i)
            aligned_store(&C(0, i, j), simd{0});
    // TODO: cache blocking
    micro_kernels::gemm::xgemmt_register<Abi, {.trans_A = true}>(A, A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_ref(single_batch_view A, mut_single_batch_view C) {
    [[maybe_unused]] const auto op_cnt_syrk =
        C.rows() * (C.rows() + 1) * C.cols() / 2;
    KOQKATOO_TRACE("xsyrk", 0, op_cnt_syrk * C.depth());
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = j; i < C.rows(); ++i)
            aligned_store(&C(0, i, j), simd{0});
    // TODO: cache blocking
    micro_kernels::gemm::xgemmt_register<Abi, {.trans_B = true}>(A, A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_add_ref(single_batch_view A,
                                     mut_single_batch_view C) {
    [[maybe_unused]] const auto op_cnt_syrk =
        C.rows() * (C.rows() + 1) * C.cols() / 2;
    KOQKATOO_TRACE("xsyrk", 0, op_cnt_syrk * C.depth());
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    // TODO: cache blocking
    micro_kernels::gemm::xgemmt_register<Abi, {.trans_B = true}>(A, A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_sub_ref(single_batch_view A,
                                     mut_single_batch_view C) {
    [[maybe_unused]] const auto op_cnt_syrk =
        C.rows() * (C.rows() + 1) * C.cols() / 2;
    KOQKATOO_TRACE("xsyrk", 0, op_cnt_syrk * C.depth());
    namespace uk = micro_kernels;
    assert(C.rows() >= C.cols());
    assert(C.rows() == A.rows());
    // TODO: cache blocking
    constexpr uk::gemm::KernelConfig conf{.negate = true, .trans_B = true};
    static const index_t L1_cache_size = 48_KiB; // TODO: determine dynamically
    static const index_t L2_cache_size = 512_KiB;
    static const index_t N_reg         = uk::gemm::ColsReg;
    static const index_t M_reg         = uk::gemm::RowsReg;
    static const index_t K_cache =
        L1_cache_size / sizeof(real_t) / simd_stride / 2 / N_reg;
    static const index_t M_cache = L2_cache_size / sizeof(real_t) /
                                   simd_stride / K_cache / 2 / M_reg * M_reg;
    static const index_t N_cache = M_cache;

    const index_t M = C.rows(), N = C.cols(), K = A.cols();
    if (M <= M_cache && N <= N_cache && K <= K_cache) // TODO
        return uk::gemm::xgemmt_register<Abi, conf>(A, A.top_rows(N), C);

    using sto_t                    = aligned_simd_storage<real_t, simd_align>;
    const index_t B_cache_sto_size = A.ceil_depth() * K_cache * N_cache;
    const index_t A_cache_sto_size = A.ceil_depth() * M_cache * K_cache;
    const index_t B_size           = A.ceil_depth() * K * N;
    const index_t A_size           = A.ceil_depth() * M * K;
    const bool pack_B = B_size >= 32 * B_cache_sto_size; // TODO: tune
    const bool pack_A = A_size >= 32 * A_cache_sto_size; // TODO: tune
    sto_t B_cache_sto{static_cast<size_t>(B_cache_sto_size), uninitialized};
    sto_t A_cache_sto{static_cast<size_t>(A_cache_sto_size), uninitialized};

    for (index_t j_cache = 0; j_cache < N; j_cache += N_cache) {
        index_t n_cache = std::min(N_cache, N - j_cache);
        for (index_t p_cache = 0; p_cache < K; p_cache += K_cache) {
            index_t k_cache = std::min(K_cache, K - p_cache);
            auto Bkj        = [&] {
                auto Bkj = A.block(j_cache, p_cache, n_cache, k_cache);
                if (pack_B) {
                    mut_single_batch_view B_cache{{
                               .data       = B_cache_sto.data(),
                               .depth      = A.depth(),
                               .rows       = n_cache,
                               .cols       = k_cache,
                               .batch_size = A.batch_size(),
                    }};
                    // TODO: proper packing
                    xcopy(Bkj, B_cache);
                    return B_cache.as_const();
                }
                return Bkj;
            }();
            for (index_t i_cache = j_cache; i_cache < M; i_cache += M_cache) {
                index_t m_cache = std::min(M_cache, M - i_cache);
                auto Cij        = C.block(i_cache, j_cache, m_cache, n_cache);
                if (i_cache == j_cache && m_cache == n_cache) {
                    uk::gemm::xgemmt_register<Abi, conf>(Bkj, Bkj, Cij);
                } else {
                    auto Aik = [&] {
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
                    if (i_cache == j_cache) [[unlikely]]
                        uk::gemm::xgemmt_register<Abi, conf>(Aik, Bkj, Cij);
                    else
                        uk::gemm::xgemm_register<Abi, conf>(Aik, Bkj, Cij);
                }
            }
        }
    }
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T(batch_view A, mut_batch_view C,
                               PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_syrk =
                C.cols() * (C.cols() + 1) * C.rows() / 2;
            KOQKATOO_TRACE("xsyrk_T_batched", 0, op_cnt_syrk * C.depth());
            linalg::xsyrk_batch_strided(CblasColMajor, CblasLower, CblasTrans,
                                        C.rows(), A.rows(), real_t{1}, A.data,
                                        A.outer_stride(), A.layer_stride(),
                                        real_t{0}, C.data, C.outer_stride(),
                                        C.layer_stride(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_T_ref(A.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_schur(batch_view A, batch_view d, mut_batch_view C,
                                   PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    assert(A.ceil_depth() == d.ceil_depth());
    std::ignore = b; // not supported by BLAS
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_schur(A.batch(i), d.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_schur_add(batch_view A, batch_view d,
                                       mut_batch_view C, PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    assert(A.ceil_depth() == d.ceil_depth());
    std::ignore = b; // not supported by BLAS
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_schur_add(A.batch(i), d.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T_schur_copy(batch_view C, batch_view Σ,
                                          bool_batch_view mask, batch_view H_in,
                                          mut_batch_view H_out,
                                          PreferredBackend b) {
    assert(C.ceil_depth() == Σ.ceil_depth());
    assert(C.ceil_depth() == mask.ceil_depth());
    assert(C.ceil_depth() == H_in.ceil_depth());
    assert(C.ceil_depth() == H_out.ceil_depth());
    std::ignore = b; // not supported by BLAS
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < C.num_batches(); ++i)
        xsyrk_T_schur_copy(C.batch(i), Σ.batch(i), mask.batch(i), H_in.batch(i),
                           H_out.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk(batch_view A, mut_batch_view C,
                             PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_syrk =
                C.rows() * (C.rows() + 1) * C.cols() / 2;
            KOQKATOO_TRACE("xsyrk_batched", 0, op_cnt_syrk * C.depth());
            linalg::xsyrk_batch_strided(CblasColMajor, CblasLower, CblasNoTrans,
                                        C.rows(), A.cols(), real_t{1}, A.data,
                                        A.outer_stride(), A.layer_stride(),
                                        real_t{0}, C.data, C.outer_stride(),
                                        C.layer_stride(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_ref(A.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_add(batch_view A, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_syrk =
                C.rows() * (C.rows() + 1) * C.cols() / 2;
            KOQKATOO_TRACE("xsyrk_add_batched", 0, op_cnt_syrk * C.depth());
            linalg::xsyrk_batch_strided(CblasColMajor, CblasLower, CblasNoTrans,
                                        C.rows(), A.cols(), real_t{1}, A.data,
                                        A.outer_stride(), A.layer_stride(),
                                        real_t{1}, C.data, C.outer_stride(),
                                        C.layer_stride(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_add_ref(A.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_sub(batch_view A, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_syrk =
                C.rows() * (C.rows() + 1) * C.cols() / 2;
            KOQKATOO_TRACE("xsyrk_sub_batched", 0, op_cnt_syrk * C.depth());
            linalg::xsyrk_batch_strided(CblasColMajor, CblasLower, CblasNoTrans,
                                        C.rows(), A.cols(), real_t{-1}, A.data,
                                        A.outer_stride(), A.layer_stride(),
                                        real_t{1}, C.data, C.outer_stride(),
                                        C.layer_stride(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_sub_ref(A.batch(i), C.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T(single_batch_view A, mut_single_batch_view C,
                               PreferredBackend b) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.cols());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_syrk =
                C.cols() * (C.cols() + 1) * C.rows() / 2;
            KOQKATOO_TRACE("xsyrk_T_blas", 0, op_cnt_syrk * C.depth());
            linalg::xsyrk(CblasColMajor, CblasLower, CblasTrans, C.rows(),
                          A.rows(), real_t{1}, A.data, A.outer_stride(),
                          real_t{0}, C.data, C.outer_stride());
            return;
        }
    }
    xsyrk_T_ref(A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk(single_batch_view A, mut_single_batch_view C,
                             PreferredBackend b) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_syrk =
                C.rows() * (C.rows() + 1) * C.cols() / 2;
            KOQKATOO_TRACE("xsyrk_blas", 0, op_cnt_syrk * C.depth());
            linalg::xsyrk(CblasColMajor, CblasLower, CblasNoTrans, C.rows(),
                          A.cols(), real_t{1}, A.data, A.outer_stride(),
                          real_t{0}, C.data, C.outer_stride());
            return;
        }
    }
    xsyrk_ref(A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_add(single_batch_view A, mut_single_batch_view C,
                                 PreferredBackend b) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_syrk =
                C.rows() * (C.rows() + 1) * C.cols() / 2;
            KOQKATOO_TRACE("xsyrk_add_blas", 0, op_cnt_syrk * C.depth());
            linalg::xsyrk(CblasColMajor, CblasLower, CblasNoTrans, C.rows(),
                          A.cols(), real_t{1}, A.data, A.outer_stride(),
                          real_t{1}, C.data, C.outer_stride());
            return;
        }
    }
    xsyrk_add_ref(A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_sub(single_batch_view A, mut_single_batch_view C,
                                 PreferredBackend b) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_syrk =
                C.rows() * (C.rows() + 1) * C.cols() / 2;
            KOQKATOO_TRACE("xsyrk_sub_blas", 0, op_cnt_syrk * C.depth());
            linalg::xsyrk(CblasColMajor, CblasLower, CblasNoTrans, C.rows(),
                          A.cols(), real_t{-1}, A.data, A.outer_stride(),
                          real_t{1}, C.data, C.outer_stride());
            return;
        }
    }
    xsyrk_sub_ref(A, C);
}

} // namespace koqkatoo::linalg::compact
