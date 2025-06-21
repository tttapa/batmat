#pragma once

#include <batmat/kib.hpp>
#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/micro-kernels/gemm.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

enum class PackingSelector : int8_t { Never, Always, Transpose };

struct TilingOptions {
    bool no_tiling         = false;
    PackingSelector pack_A = PackingSelector::Transpose;
    PackingSelector pack_B = PackingSelector::Always;
    index_t n_c = 0, k_c = 0, m_c = 0;
};

template <class T, class Abi, micro_kernels::gemm::KernelConfig Conf, StorageOrder OA,
          StorageOrder OB, StorageOrder OC>
void gemm(view<const T, Abi, OA> A, view<const T, Abi, OB> B, view<T, Abi, OC> C, bool init_zero,
          TilingOptions packing = {}) {
    GUANAQO_TRACE("gemm", 0, A.rows() * A.cols() * B.cols() * A.depth());
    static constexpr micro_kernels::gemm::KernelConfig conf{
        .negate  = Conf.negate,
        .shift_A = Conf.shift_A,
        .shift_B = Conf.shift_B,
        .shift_C = Conf.shift_C,
        .shift_D = Conf.shift_D,
    };
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    const index_t M = C.rows(), N = C.cols(), K = A.cols();
    if (M == 0 || N == 0 || K == 0) [[unlikely]]
        return;
    // TODO: cache blocking
    static const index_t simd_stride   = simd_view_types<T, Abi>::simd_stride;
    static const index_t simd_align    = simd_view_types<T, Abi>::simd_align;
    static const index_t L1_cache_size = 48_KiB; // TODO: determine dynamically
    static const index_t L2_cache_size = 512_KiB;
    static const index_t L3_cache_size = 16_MiB;
    static const index_t n_cores       = 8; // TODO: OMP
    static const index_t N_reg         = micro_kernels::gemm::ColsReg<T, Abi>;
    static const index_t M_reg         = micro_kernels::gemm::RowsReg<T, Abi>;
    // clang-format off
    static const index_t K_cache_default = L1_cache_size / sizeof(T) / simd_stride / N_reg;
    static const index_t M_cache_default = (L2_cache_size / sizeof(T) / simd_stride / K_cache_default / M_reg) * M_reg;
    static const index_t N_cache_default = std::max<index_t>(L3_cache_size / sizeof(T) / simd_stride / K_cache_default / n_cores / M_cache_default, 1) * M_cache_default;
    // clang-format on
    const index_t K_cache = packing.k_c ? packing.k_c : K_cache_default;
    const index_t M_cache = packing.m_c ? packing.m_c : M_cache_default;
    const index_t N_cache = packing.n_c ? packing.n_c : N_cache_default;

    if ((M <= M_cache && N <= N_cache && K <= K_cache) || packing.no_tiling) [[likely]]
        return micro_kernels::gemm::gemm_register<T, Abi, conf>(A, B, C, init_zero);

    using sto_t               = batmat::matrix::aligned_simd_storage<T, simd_align>;
    const index_t B_pack_size = B.ceil_depth() * K_cache * N_cache;
    const index_t A_pack_size = A.ceil_depth() * M_cache * K_cache;
    const index_t B_size      = B.ceil_depth() * K * N;
    const index_t A_size      = A.ceil_depth() * M * K;
    const bool select_pack_B =
        packing.pack_B == PackingSelector::Always ||
        (packing.pack_B == PackingSelector::Transpose && OB == StorageOrder::RowMajor);
    const bool select_pack_A =
        packing.pack_A == PackingSelector::Always ||
        (packing.pack_A == PackingSelector::Transpose && OA == StorageOrder::ColMajor);
    const bool pack_B = select_pack_B && B_size >= 2 * B_pack_size; // TODO: tune
    const bool pack_A = select_pack_A && A_size >= 2 * A_pack_size; // TODO: tune
    sto_t B_pack{pack_B ? static_cast<size_t>(B_pack_size) : 0, batmat::matrix::uninitialized};
    sto_t A_pack{pack_A ? static_cast<size_t>(A_pack_size) : 0, batmat::matrix::uninitialized};
    view<T, Abi, StorageOrder::ColMajor> Bkj_pack;
    view<T, Abi, StorageOrder::RowMajor> Aik_pack;

#if 0
    for (index_t j_cache = 0; j_cache < N; j_cache += N_cache) {
        index_t n_cache = std::min(N_cache, N - j_cache);
        for (index_t p_cache = 0; p_cache < K; p_cache += K_cache) {
            index_t k_cache = std::min(K_cache, K - p_cache);
            auto Bkj        = B.block(p_cache, j_cache, k_cache, n_cache);
            if (pack_B) {
                Bkj_pack.reassign({{
                    .data       = B_pack.data(),
                    .depth      = B.depth(),
                    .rows       = k_cache,
                    .cols       = n_cache,
                    .batch_size = B.batch_size(),
                }});
                copy<T, Abi>(Bkj, Bkj_pack);
            }
            for (index_t i_cache = 0; i_cache < M; i_cache += M_cache) {
                index_t m_cache = std::min(M_cache, M - i_cache);
                auto Cij        = C.block(i_cache, j_cache, m_cache, n_cache);
                auto Aik        = A.block(i_cache, p_cache, m_cache, k_cache);
                if (pack_A) {
                    Aik_pack.reassign({{
                        .data       = A_pack.data(),
                        .depth      = A.depth(),
                        .rows       = m_cache,
                        .cols       = k_cache,
                        .batch_size = A.batch_size(),
                    }});
                    copy<T, Abi>(Aik, Aik_pack);
                }
                if (pack_A && pack_B) {
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(
                        Aik_pack.as_const(), Bkj_pack.as_const(), Cij, init_zero && p_cache == 0);
                } else if (pack_A) {
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik_pack.as_const(), Bkj, Cij,
                                                                     init_zero && p_cache == 0);
                } else if (pack_B) {
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik, Bkj_pack.as_const(), Cij,
                                                                     init_zero && p_cache == 0);
                } else {
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik, Bkj, Cij,
                                                                     init_zero && p_cache == 0);
                }
            }
        }
    }
#elif 0
    if (pack_A && pack_B) {
        foreach_chunked_merged(0, N, N_cache, [&](index_t j_cache, index_t n_cache) {
            foreach_chunked_merged(0, K, K_cache, [&](index_t p_cache, index_t k_cache) {
                auto Bkj = B.block(p_cache, j_cache, k_cache, n_cache);
                Bkj_pack.reassign({{.data = B_pack.data(), .rows = k_cache, .cols = n_cache}});
                copy<T, Abi>(Bkj, Bkj_pack);
                foreach_chunked_merged(0, M, M_cache, [&](index_t i_cache, index_t m_cache) {
                    auto Cij = C.block(i_cache, j_cache, m_cache, n_cache);
                    auto Aik = A.block(i_cache, p_cache, m_cache, k_cache);
                    Aik_pack.reassign({{.data = A_pack.data(), .rows = m_cache, .cols = k_cache}});
                    copy<T, Abi>(Aik, Aik_pack);
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(
                        Aik_pack.as_const(), Bkj_pack.as_const(), Cij, init_zero && p_cache == 0);
                });
            });
        });
    } else if (pack_A) {
        foreach_chunked_merged(0, N, N_cache, [&](index_t j_cache, index_t n_cache) {
            foreach_chunked_merged(0, K, K_cache, [&](index_t p_cache, index_t k_cache) {
                auto Bkj = B.block(p_cache, j_cache, k_cache, n_cache);
                foreach_chunked_merged(0, M, M_cache, [&](index_t i_cache, index_t m_cache) {
                    auto Cij = C.block(i_cache, j_cache, m_cache, n_cache);
                    auto Aik = A.block(i_cache, p_cache, m_cache, k_cache);
                    Aik_pack.reassign({{.data = A_pack.data(), .rows = m_cache, .cols = k_cache}});
                    copy<T, Abi>(Aik, Aik_pack);
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik_pack.as_const(), Bkj, Cij,
                                                                     init_zero && p_cache == 0);
                });
            });
        });
    } else if (pack_B) {
        foreach_chunked_merged(0, N, N_cache, [&](index_t j_cache, index_t n_cache) {
            foreach_chunked_merged(0, K, K_cache, [&](index_t p_cache, index_t k_cache) {
                auto Bkj = B.block(p_cache, j_cache, k_cache, n_cache);
                Bkj_pack.reassign({{.data = B_pack.data(), .rows = k_cache, .cols = n_cache}});
                copy<T, Abi>(Bkj, Bkj_pack);
                foreach_chunked_merged(0, M, M_cache, [&](index_t i_cache, index_t m_cache) {
                    auto Cij = C.block(i_cache, j_cache, m_cache, n_cache);
                    auto Aik = A.block(i_cache, p_cache, m_cache, k_cache);
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik, Bkj_pack.as_const(), Cij,
                                                                     init_zero && p_cache == 0);
                });
            });
        });
    } else {
        foreach_chunked_merged(0, N, N_cache, [&](index_t j_cache, index_t n_cache) {
            foreach_chunked_merged(0, K, K_cache, [&](index_t p_cache, index_t k_cache) {
                auto Bkj = B.block(p_cache, j_cache, k_cache, n_cache);
                foreach_chunked_merged(0, M, M_cache, [&](index_t i_cache, index_t m_cache) {
                    auto Cij = C.block(i_cache, j_cache, m_cache, n_cache);
                    auto Aik = A.block(i_cache, p_cache, m_cache, k_cache);
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik, Bkj, Cij,
                                                                     init_zero && p_cache == 0);
                });
            });
        });
    }
#else
    using micro_kernels::gemm::gemm_register;
    foreach_chunked_merged(0, N, N_cache, [&](index_t j_c, index_t n_c) {
        foreach_chunked_merged(0, K, K_cache, [&](index_t p_c, index_t k_c) {
            auto Bkj = B.block(p_c, j_c, k_c, n_c);
            if (pack_B) {
                Bkj_pack.reassign({{.data = B_pack.data(), .rows = k_c, .cols = n_c}});
                copy<T, Abi>(Bkj, Bkj_pack);
                foreach_chunked_merged(0, M, M_cache, [&](index_t i_c, index_t m_c) {
                    auto Cij = C.block(i_c, j_c, m_c, n_c);
                    auto Aik = A.block(i_c, p_c, m_c, k_c);
                    if (pack_A) {
                        Aik_pack.reassign({{.data = A_pack.data(), .rows = m_c, .cols = k_c}});
                        copy<T, Abi>(Aik, Aik_pack);
                        gemm_register<T, Abi, conf>(Aik_pack.as_const(), Bkj_pack.as_const(), Cij,
                                                    init_zero && p_c == 0);
                    } else {
                        gemm_register<T, Abi, conf>(Aik, Bkj_pack.as_const(), Cij,
                                                    init_zero && p_c == 0);
                    }
                });
            } else {
                foreach_chunked_merged(0, M, M_cache, [&](index_t i_cache, index_t m_cache) {
                    auto Cij = C.block(i_cache, j_c, m_cache, n_c);
                    auto Aik = A.block(i_cache, p_c, m_cache, k_c);
                    if (pack_A) {
                        Aik_pack.reassign({{.data = A_pack.data(), .rows = m_cache, .cols = k_c}});
                        copy<T, Abi>(Aik, Aik_pack);
                        gemm_register<T, Abi, conf>(Aik_pack.as_const(), Bkj, Cij,
                                                    init_zero && p_c == 0);
                    } else {
                        gemm_register<T, Abi, conf>(Aik, Bkj, Cij, init_zero && p_c == 0);
                    }
                });
            }
        });
    });
#endif
}

} // namespace batmat::linalg
