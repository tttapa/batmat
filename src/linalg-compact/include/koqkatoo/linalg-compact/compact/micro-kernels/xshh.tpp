#pragma once

#include "xshh.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::shh {

template <class Abi, index_t R>
[[gnu::hot]] void
xshh_diag_microkernel(index_t colsA, triangular_accessor<Abi, real_t, R> W,
                      mut_single_batch_matrix_accessor<Abi> L,
                      mut_single_batch_matrix_accessor<Abi> A) noexcept {
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
            simd Akj = A.load(k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < R; ++i)
                bb[i] += A.load(i, j) * Akj;
        }
        // Energy condition and Householder coefficients
        const simd α2 = bb[k], Lkk = L_cached.load(k, k);
        const simd L̃kk = copysign(sqrt(Lkk * Lkk - α2), -Lkk), β = L̃kk - Lkk;
        simd γoβ = 2 * β / (α2 - β * β), γ = β * γoβ, inv_β = 1 / β;
        L_cached.store(L̃kk, k, k);
        // Compute L̃
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k + 1; i < R; ++i) {
            simd Lik = L_cached.load(i, k);
            bb[i] *= γoβ;
            bb[i] += γ * Lik;
            L_cached.store(Lik + bb[i], i, k);
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
xshh_full_microkernel(index_t colsA, mut_single_batch_matrix_accessor<Abi> L,
                      mut_single_batch_matrix_accessor<Abi> A) noexcept {
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
            simd Akj = A.load(k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k; i < R; ++i)
                bb[i] += A.load(i, j) * Akj;
        }
        // Energy condition and Householder coefficients
        const simd α2 = bb[k], Lkk = L_cached.load(k, k);
        const simd L̃kk = copysign(sqrt(Lkk * Lkk - α2), -Lkk), β = L̃kk - Lkk;
        simd γoβ = 2 * β / (α2 - β * β), γ = β * γoβ, inv_β = 1 / β;
        L_cached.store(L̃kk, k, k);
        // Compute L̃
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k + 1; i < R; ++i) {
            simd Lik = L_cached.load(i, k);
            bb[i] *= γoβ;
            bb[i] += γ * Lik;
            L_cached.store(Lik + bb[i], i, k);
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
[[gnu::hot]] void
xshh_tail_microkernel(index_t colsA,
                      triangular_accessor<Abi, const real_t, R> W,
                      mut_single_batch_matrix_accessor<Abi> L,
                      mut_single_batch_matrix_accessor<Abi> A,
                      single_batch_matrix_accessor<Abi> B) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of L
    auto L_cached = with_cached_access<R>(L);
    KOQKATOO_ASSUME(colsA > 0);

    // Compute product U = A B
    simd V[S][R]{};
    for (index_t j = 0; j < colsA; ++j) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
            auto Akj = B.load(k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i)
                V[i][k] += A.load(i, j) * Akj;
        }
    }
    // Solve system V = (L+U)W⁻¹ (in-place)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
        simd Wk[R];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
            Wk[l] = W.load(l, k);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
            simd Lik = L.load(i, k);
            V[i][k] += Lik;
            KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                V[i][k] -= V[i][l] * Wk[l];
            V[i][k] *= W.load(k, k); // diagonal already inverted
            Lik += V[i][k];
            L.store(Lik, i, k);
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

} // namespace koqkatoo::linalg::compact::micro_kernels::shh
