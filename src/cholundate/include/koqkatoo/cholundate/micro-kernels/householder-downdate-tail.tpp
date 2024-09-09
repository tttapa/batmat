#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/unroll.h>

#include "householder-downdate.hpp"

#define UNROLL_FOR(...) KOQKATOO_FULLY_UNROLLED_IVDEP_FOR (__VA_ARGS__)
#define UNROLL_FOR_A_COLS(...) for (__VA_ARGS__)

namespace koqkatoo::cholundate::micro_kernels::householder {

template <Config Conf>
[[gnu::hot]] void downdate_tail(mut_matrix_accessor L, matrix_accessor B,
                                mut_matrix_accessor A, index_t colsA,
                                mut_W_accessor<Conf.block_size_r> W) noexcept {
    using simdA                 = tail_simd_A_t<Conf>;
    using simdL                 = tail_simd_L_t<Conf>;
    static constexpr index_t NA = simdA::size(), NL = simdL::size();
    static constexpr index_t S = Conf.block_size_s, R = Conf.block_size_r;
    static_assert(S % NA == 0);
    static_assert(R % NL == 0);
    static_assert(R > 0);
    KOQKATOO_ASSUME(colsA > 0);
    [[maybe_unused]] const auto W_addr = reinterpret_cast<uintptr_t>(W.data);
    KOQKATOO_ASSUME(W_addr % W_align<Conf.block_size_r> == 0);

    // Compute product U = A B
    simdA V[S / NA][R]{};
    UNROLL_FOR_A_COLS (index_t j = 0; j < colsA; ++j)
        UNROLL_FOR (index_t kk = 0; kk < R; kk += NL) {
            auto Akj = B.load<simdL>(kk, j);
            UNROLL_FOR (index_t k = 0; k < NL; ++k)
                UNROLL_FOR (index_t i = 0; i < S; i += NA)
                    V[i / NA][kk + k] += A.load<simdA>(i, j) * Akj[k];
        }
    // Solve system V = (L+U)W⁻¹ (in-place)
    UNROLL_FOR (index_t k = 0; k < R; ++k)
        UNROLL_FOR (index_t i = 0; i < S; i += NA) {
            auto Lik = L.load<simdA>(i, k);
            V[i / NA][k] += Lik;
            UNROLL_FOR (index_t l = 0; l < k; ++l)
                V[i / NA][k] -= V[i / NA][l] * W(l, k);
            V[i / NA][k] *= W(k, k); // diagonal already inverted
            Lik += V[i / NA][k];
            L.store(Lik, i, k);
        }
#if 0
    // Update A -= V Bᵀ
    UNROLL_FOR_A_COLS (index_t j = 0; j < colsA; ++j)
        UNROLL_FOR (index_t i = 0; i < S; i += NA) {
            auto Aij = A.load<simdA>(i, j);
            UNROLL_FOR (index_t kk = 0; kk < R; kk += NL) {
                auto Akj = B.load<simdL>(kk, j);
                UNROLL_FOR (index_t k = 0; k < NL; ++k)
                    Aij -= V[i / NA][kk + k] * Akj[k];
            }
            A.store(Aij, i, j);
        }
#else
    // Update A -= V Bᵀ
    UNROLL_FOR_A_COLS (index_t j = 0; j < colsA; ++j) {
        simdL Akj[R / NL];
        UNROLL_FOR (index_t kk = 0; kk < R; kk += NL)
            Akj[kk / NL] = B.load<simdL>(kk, j);
        UNROLL_FOR (index_t i = 0; i < S; i += NA) {
            auto Aij = A.load<simdA>(i, j);
            UNROLL_FOR (index_t kk = 0; kk < R; kk += NL) {
                UNROLL_FOR (index_t k = 0; k < NL; ++k)
                    Aij -= V[i / NA][kk + k] * Akj[kk / NL][k];
            }
            A.store(Aij, i, j);
        }
    }
#endif
}

} // namespace koqkatoo::cholundate::micro_kernels::householder

#undef UNROLL_FOR
#undef UNROLL_FOR_A_COLS
