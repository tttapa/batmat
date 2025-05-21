#pragma once

#include "xshhud-diag.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/cneg.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::shhud_diag {

template <class Abi, index_t R>
[[gnu::hot]] void
xshhud_diag_diag_microkernel(index_t colsA,
                             triangular_accessor<Abi, real_t, SizeR> W,
                             mut_single_batch_matrix_accessor<Abi> L,
                             mut_single_batch_matrix_accessor<Abi> A,
                             single_batch_vector_accessor<Abi> diag) noexcept {
    using std::copysign;
    using std::sqrt;
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of L
    auto L_cached = with_cached_access<R>(L);
    KOQKATOO_ASSUME(colsA > 0);

    KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
        // Compute all inner products between A and a
        simd bb[R]{};
        for (index_t j = 0; j < colsA; ++j) {
            simd Akj = A.load(k, j) * diag.load(j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < R; ++i)
                bb[i] += A.load(i, j) * Akj;
        }
        // Energy condition and Householder coefficients
        const simd α2 = bb[k], Lkk = L_cached.load(k, k);
        const simd L̃kk = copysign(sqrt(Lkk * Lkk + α2), Lkk), β = Lkk + L̃kk;
        simd γoβ = 2 * β / (β * β + α2), γ = β * γoβ, inv_β = 1 / β;
        L_cached.store(L̃kk, k, k);
        // Compute L̃
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k + 1; i < R; ++i) {
            simd Lik = L_cached.load(i, k);
            bb[i]    = γ * Lik + bb[i] * γoβ;
            L_cached.store(bb[i] - Lik, i, k);
        }
        // Update A
        for (index_t j = 0; j < colsA; ++j) {
            simd Akj = A.load(k, j) * inv_β; // Scale Householder vector
            A.store(Akj, k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k + 1; i < R; ++i) {
                simd Aij = A.load(i, j);
                Aij -= bb[i] * Akj;
                A.store(Aij, i, j);
            }
        }
        // Save block Householder matrix W
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < k + 1; ++i)
            bb[i] *= inv_β;
        bb[k] = γ; // inverse of diagonal
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < k + 1; ++i)
            W.store(bb[i], i, k);
        // TODO: try moving this to before update of A
    }
}

template <class Abi, index_t R>
[[gnu::hot]] void
xshhud_diag_full_microkernel(index_t colsA,
                             mut_single_batch_matrix_accessor<Abi> L,
                             mut_single_batch_matrix_accessor<Abi> A,
                             single_batch_vector_accessor<Abi> diag) noexcept {
    using std::copysign;
    using std::sqrt;
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of L
    auto L_cached = with_cached_access<R>(L);
    KOQKATOO_ASSUME(colsA > 0);

    KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
        // Compute some inner products between A and a
        simd bb[R]{};
        for (index_t j = 0; j < colsA; ++j) {
            simd Akj = A.load(k, j) * diag.load(j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k; i < R; ++i)
                bb[i] += A.load(i, j) * Akj;
        }
        // Energy condition and Householder coefficients
        const simd α2 = bb[k], Lkk = L_cached.load(k, k);
        const simd L̃kk = copysign(sqrt(Lkk * Lkk + α2), Lkk), β = Lkk + L̃kk;
        simd γoβ = 2 * β / (β * β + α2), γ = β * γoβ, inv_β = 1 / β;
        L_cached.store(L̃kk, k, k);
        // Compute L̃
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k + 1; i < R; ++i) {
            simd Lik = L_cached.load(i, k);
            bb[i]    = γ * Lik + bb[i] * γoβ;
            L_cached.store(bb[i] - Lik, i, k);
        }
        // Update A
        for (index_t j = 0; j < colsA; ++j) {
            simd Akj = A.load(k, j) * inv_β; // Scale Householder vector
            A.store(Akj, k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k + 1; i < R; ++i) {
                simd Aij = A.load(i, j);
                Aij -= bb[i] * Akj;
                A.store(Aij, i, j);
            }
        }
    }
}

template <class Abi, index_t R, index_t S>
[[gnu::hot]] void xshhud_diag_tail_microkernel(
    index_t colsA, triangular_accessor<Abi, const real_t, SizeR> W,
    mut_single_batch_matrix_accessor<Abi> L,
    mut_single_batch_matrix_accessor<Abi> A,
    single_batch_matrix_accessor<Abi> B, single_batch_vector_accessor<Abi> diag,
    bool trans_L, index_t L_upper_offset) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of L
    KOQKATOO_ASSUME(colsA > 0);
    KOQKATOO_ASSERT(!trans_L || L_upper_offset >= std::max(R, S));

    // Compute product U = A B
    simd V[S][R]{};
    for (index_t j = 0; j < colsA; ++j) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
            auto Akj = B.load(k, j) * diag.load(j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i)
                V[i][k] += A.load(i, j) * Akj;
        }
    }
    if (trans_L) [[unlikely]] {
        auto L_cached = with_cached_access<S>(L);
        // Solve system V = (L+U)W⁻¹ (in-place)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
            simd Wk[R];
            KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                Wk[l] = W.load(l, k);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                simd Lik = L_cached.load(k, i);
                V[i][k] += Lik;
                KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                    V[i][k] -= V[i][l] * Wk[l];
                V[i][k] *= W.load(k, k); // diagonal already inverted
                Lik = V[i][k] - Lik;
                L_cached.store(Lik, k, i);
            }
        }
    } else {
        auto L_cached = with_cached_access<R>(L);
        // Solve system V = (L+U)W⁻¹ (in-place)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
            simd Wk[R];
            KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                Wk[l] = W.load(l, k);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                using mask       = simd::mask_type;
                constexpr auto N = stdx::simd_size_v<real_t, Abi>;
                using isimd_abi  = stdx::simd_abi::deduce_t<index_t, N>;
                using isimd      = stdx::simd<index_t, isimd_abi>;
                // hack to get better code gen: we want conditional/masked moves
                // instead of branches
                const mask L_nonzero{(isimd{i - k} <= L_upper_offset).__cvt()};
                simd Lik_mem = L_cached.load(i, k);
                simd Lik{0};
                where(L_nonzero, Lik) = Lik_mem;
                V[i][k] += Lik;
                KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                    V[i][k] -= V[i][l] * Wk[l];
                V[i][k] *= W.load(k, k); // diagonal already inverted
                Lik                       = V[i][k] - Lik;
                where(L_nonzero, Lik_mem) = Lik;
                L_cached.store(Lik_mem, i, k);
            }
        }
    }
    // Update A -= V Bᵀ
    for (index_t j = 0; j < colsA; ++j) {
        simd Akj[R];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
            Akj[k] = B.load(k, j);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
            auto Aij = A.load(i, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                Aij -= V[i][k] * Akj[k];
            A.store(Aij, i, j);
        }
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::shhud_diag
