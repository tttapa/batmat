#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/unroll.h>

#include "householder-downdate.hpp"

#define UNROLL_FOR(...) KOQKATOO_FULLY_UNROLLED_IVDEP_FOR (__VA_ARGS__)
#define UNROLL_FOR_A_COLS(...) for (__VA_ARGS__)

namespace koqkatoo::cholundate::micro_kernels::householder {

template <index_t R>
[[gnu::hot]] void downdate_full(index_t colsA, real_t *__restrict Ld,
                                index_t ldL, real_t *__restrict Ad,
                                index_t ldA) noexcept {
    using std::sqrt;
    using simd = diag_simd_t<R>;
    const mut_matrix_accessor L{Ld, ldL}, A{Ad, ldA};
    static constexpr index_t N = simd::size();
    static_assert(R % N == 0);
    static_assert(R > 0);
    KOQKATOO_ASSUME(colsA > 0);

    UNROLL_FOR (index_t k = 0; k < R; ++k) {
        // Compute some inner products between A and a
        simd bb[R / N]{};
        UNROLL_FOR_A_COLS (index_t j = 0; j < colsA; ++j) {
            real_t Akj = A(k, j);
            UNROLL_FOR (index_t i = k / N * N; i < R; i += N)
                bb[i / N] += A.load<simd>(i, j) * Akj;
        }
        // Energy condition and Householder coefficients
        const real_t α2 = bb[k / N][k % N], Lkk = L(k, k);
        const real_t L̃kk = sqrt(Lkk * Lkk - α2), β = L̃kk - Lkk;
        const real_t γoβ = 2 * β / (α2 - β * β), γ = β * γoβ, inv_β = 1 / β;
        L(k, k) = L̃kk;
        // Scale Householder vector products
        UNROLL_FOR (index_t i = (k + 1) / N; i < R / N; ++i)
            bb[i] *= γoβ;
        // Compute L̃
        real_t v0[N - 1]{};
        const index_t kp1N = (k + 1 + N - 1) / N;
        UNROLL_FOR (index_t i = k + 1; i < kp1N * N; ++i) // scalar part
            L(i, k) += v0[i - (k + 1)] = bb[i / N][i % N] + γ * L(i, k);
        simd v[R / N];
        UNROLL_FOR (index_t i = kp1N; i < R / N; ++i) { // vectorized
            index_t ii  = i * N;
            auto Lik    = L.load<simd>(ii, k);
            Lik += v[i] = bb[i] + γ * Lik;
            L.store(Lik, ii, k);
        }
        // Update A
        UNROLL_FOR_A_COLS (index_t j = 0; j < colsA; ++j) {
            real_t Akj = A(k, j) *= inv_β; // Scale Householder vector
            UNROLL_FOR (index_t i = k + 1; i < kp1N * N; ++i) // scalar part
                A(i, j) -= v0[i - (k + 1)] * Akj;
            UNROLL_FOR (index_t i = kp1N; i < R / N; ++i) { // vectorized
                index_t ii = i * N;
                auto Aij   = A.load<simd>(ii, j);
                Aij -= v[i] * Akj;
                A.store(Aij, ii, j);
            }
        }
    }
}

} // namespace koqkatoo::cholundate::micro_kernels::householder

#undef UNROLL_FOR
#undef UNROLL_FOR_A_COLS
