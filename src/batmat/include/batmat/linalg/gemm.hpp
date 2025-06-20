#pragma once

#include <batmat/kib.hpp>
#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/micro-kernels/gemm.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

struct GemmConfig {
    bool negate = false;
    int shift_A = 0;
    int shift_B = 0;
    int shift_C = 0;
    int shift_D = shift_C;
};

template <class T, class Abi, GemmConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC>
void gemm(view<const T, Abi, OA> A, view<const T, Abi, OB> B, view<T, Abi, OC> C, bool init_zero) {
    GUANAQO_TRACE("gemm", 0, A.rows() * A.cols() * B.cols() * A.depth());
    static constexpr micro_kernels::gemm::KernelConfig conf{
        .negate  = Conf.negate,
        .order_A = OA,
        .order_B = OB,
        .order_C = OC,
        .shift_A = Conf.shift_A,
        .shift_B = Conf.shift_B,
        .shift_C = Conf.shift_C,
        .shift_D = Conf.shift_D,
    };
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    const index_t M = C.rows(), N = C.cols(), K = A.cols();
    if (M == 0 || N == 0 || K == 0)
        return;
    // TODO: cache blocking
    static const index_t simd_stride   = simd_view_types<T, Abi>::simd_stride;
    static const index_t simd_align    = simd_view_types<T, Abi>::simd_align;
    static const index_t L1_cache_size = 48_KiB; // TODO: determine dynamically
    static const index_t L2_cache_size = 512_KiB;
    static const index_t L3_cache_size = 16_MiB;
    static const index_t n_cores       = 10; // TODO: OMP
    static const index_t N_reg         = micro_kernels::gemm::ColsReg<T, Abi>;
    static const index_t M_reg         = micro_kernels::gemm::RowsReg<T, Abi>;
    static const index_t K_cache       = L1_cache_size / sizeof(T) / simd_stride / 2 / N_reg;
    static const index_t M_cache =
        L2_cache_size / sizeof(T) / simd_stride / K_cache / 2 / M_reg * M_reg;
    static const index_t N_cache =
        std::max<index_t>(L3_cache_size / sizeof(T) / simd_stride / K_cache / n_cores / M_cache,
                          1) *
        M_cache;

    if (M <= M_cache && N <= N_cache && K <= K_cache) [[likely]]
        return micro_kernels::gemm::gemm_register<T, Abi, conf>(A, B, C, init_zero);

    using sto_t                    = batmat::matrix::aligned_simd_storage<T, simd_align>;
    const index_t B_cache_sto_size = B.ceil_depth() * K_cache * N_cache;
    const index_t A_cache_sto_size = A.ceil_depth() * M_cache * K_cache;
    const index_t B_size           = B.ceil_depth() * K * N;
    const index_t A_size           = A.ceil_depth() * M * K;
    const bool pack_B              = B_size >= 2 * B_cache_sto_size; // TODO: tune
    const bool pack_A              = A_size >= 2 * A_cache_sto_size; // TODO: tune
    sto_t B_cache_sto{pack_B ? static_cast<size_t>(B_cache_sto_size) : 0,
                      batmat::matrix::uninitialized};
    sto_t A_cache_sto{pack_A ? static_cast<size_t>(A_cache_sto_size) : 0,
                      batmat::matrix::uninitialized};
    view<T, Abi, StorageOrder::ColMajor> Bkj_cache;
    view<T, Abi, StorageOrder::RowMajor> Aik_cache;

    for (index_t j_cache = 0; j_cache < N; j_cache += N_cache) {
        index_t n_cache = std::min(N_cache, N - j_cache);
        for (index_t p_cache = 0; p_cache < K; p_cache += K_cache) {
            index_t k_cache = std::min(K_cache, K - p_cache);
            auto Bkj        = B.block(p_cache, j_cache, k_cache, n_cache);
            if (pack_B) {
                Bkj_cache.reassign({{
                    .data       = B_cache_sto.data(),
                    .depth      = B.depth(),
                    .rows       = k_cache,
                    .cols       = n_cache,
                    .batch_size = B.batch_size(),
                }});
                copy<T, Abi>(Bkj, Bkj_cache);
            }
            for (index_t i_cache = 0; i_cache < M; i_cache += M_cache) {
                index_t m_cache = std::min(M_cache, M - i_cache);
                auto Cij        = C.block(i_cache, j_cache, m_cache, n_cache);
                auto Aik        = A.block(i_cache, p_cache, m_cache, k_cache);
                if (pack_A) {
                    Aik_cache.reassign({{
                        .data       = A_cache_sto.data(),
                        .depth      = A.depth(),
                        .rows       = m_cache,
                        .cols       = k_cache,
                        .batch_size = A.batch_size(),
                    }});
                    copy<T, Abi>(Aik, Aik_cache);
                }
                if (pack_A && pack_B) {
                    static constexpr micro_kernels::gemm::KernelConfig conf{
                        .negate  = false,
                        .order_A = StorageOrder::RowMajor,
                        .order_B = StorageOrder::ColMajor,
                        .order_C = OC,
                        .shift_A = Conf.shift_A,
                        .shift_B = Conf.shift_B,
                        .shift_C = Conf.shift_C,
                        .shift_D = Conf.shift_D,
                    };
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik_cache, Bkj_cache, Cij,
                                                                     init_zero && p_cache == 0);
                } else if (pack_A) {
                    static constexpr micro_kernels::gemm::KernelConfig conf{
                        .negate  = false,
                        .order_A = StorageOrder::RowMajor,
                        .order_B = OB,
                        .order_C = OC,
                        .shift_A = Conf.shift_A,
                        .shift_B = Conf.shift_B,
                        .shift_C = Conf.shift_C,
                        .shift_D = Conf.shift_D,
                    };
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik_cache, Bkj, Cij,
                                                                     init_zero && p_cache == 0);
                } else if (pack_B) {
                    static constexpr micro_kernels::gemm::KernelConfig conf{
                        .negate  = false,
                        .order_A = OA,
                        .order_B = StorageOrder::ColMajor,
                        .order_C = OC,
                        .shift_A = Conf.shift_A,
                        .shift_B = Conf.shift_B,
                        .shift_C = Conf.shift_C,
                        .shift_D = Conf.shift_D,
                    };
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik, Bkj_cache, Cij,
                                                                     init_zero && p_cache == 0);
                } else {
                    micro_kernels::gemm::gemm_register<T, Abi, conf>(Aik, Bkj, Cij,
                                                                     init_zero && p_cache == 0);
                }
            }
        }
    }
}

} // namespace batmat::linalg
