#pragma once

#include "rotate.hpp"
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

namespace detail {

template <class T, class Abi, int S>
[[gnu::always_inline]] inline auto rotate(stdx::simd<T, Abi> x,
                                          std::integral_constant<int, S>) {
    return rotr<S>(x);
}

template <class T, class Abi>
[[gnu::always_inline]] inline auto rotate(stdx::simd<T, Abi> x, int s) {
    return rot(x, s);
}

} // namespace detail

template <class Abi, index_t R, index_t S>
[[gnu::hot]] void xshhud_diag_tail_microkernel(
    index_t kA_nonzero_start, index_t kA_nonzero_end, index_t colsA,
    triangular_accessor<Abi, const real_t, SizeR> W,
    mut_single_batch_matrix_accessor<Abi> L,
    single_batch_matrix_accessor<Abi> A_in,
    mut_single_batch_matrix_accessor<Abi> A_out,
    single_batch_matrix_accessor<Abi> B, single_batch_vector_accessor<Abi> diag,
    Structure struc_L, int rotate_A) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    KOQKATOO_ASSUME(colsA > 0);

    // Compute product U = A B
    simd V[S][R]{};
    for (index_t j = kA_nonzero_start; j < kA_nonzero_end; ++j) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
            auto Akj = B.load(k, j) * diag.load(j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i)
                V[i][k] += A_in.load(i, j) * Akj;
        }
    }

    // Solve system V = (L+U)W⁻¹ (in-place)
    auto L_cached = with_cached_access<R>(L);
    switch (struc_L) {
        [[likely]]
        case Structure::General: {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
                simd Wk[R];
                KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                    Wk[l] = W.load(l, k);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                    simd Lik = L_cached.load(i, k);
                    V[i][k] += Lik;
                    KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                        V[i][k] -= V[i][l] * Wk[l];
                    V[i][k] *= W.load(k, k); // diagonal already inverted
                    Lik = V[i][k] - Lik;
                    L_cached.store(Lik, i, k);
                }
            }
        } break;
        case Structure::Zero: {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
                simd Wk[R];
                KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                    Wk[l] = W.load(l, k);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                    KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                        V[i][k] -= V[i][l] * Wk[l];
                    V[i][k] *= W.load(k, k); // diagonal already inverted
                }
            }
        } break;
        case Structure::Upper: {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k) {
                simd Wk[R];
                KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                    Wk[l] = W.load(l, k);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                    simd Lik;
                    if (i <= k) {
                        Lik = L_cached.load(i, k);
                        V[i][k] += Lik;
                    }
                    KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                        V[i][k] -= V[i][l] * Wk[l];
                    V[i][k] *= W.load(k, k); // diagonal already inverted
                    if (i <= k) {
                        Lik = V[i][k] - Lik;
                        L_cached.store(Lik, i, k);
                    }
                }
            }
        } break;
        default: KOQKATOO_ASSUME(false);
    }
    // Update A -= V Bᵀ
    const auto update_A = [&] [[gnu::always_inline]] (auto s) {
        simd Akj[R];
#if 0 // bottom variant generates less code
        for (index_t j = 0; j < kA_nonzero_start; ++j) [[unlikely]] {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                simd Aij{0};
                KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
        for (index_t j = kA_nonzero_start; j < kA_nonzero_end; ++j) [[likely]] {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                auto Aij = A_in.load(i, j);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
        for (index_t j = kA_nonzero_end; j < colsA; ++j) [[unlikely]] {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                simd Aij{0};
                KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
#else
        for (index_t j = kA_nonzero_start; j < kA_nonzero_end; ++j) [[likely]] {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                auto Aij = A_in.load(i, j);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
        for (index_t j = 0; true; ++j) {
            if (j == kA_nonzero_start)
                j = kA_nonzero_end;
            if (j >= colsA)
                break;
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < S; ++i) {
                simd Aij{0};
                KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
#endif
    };
#if defined(__AVX512F__) && 0
    update_A(rotate_A);
#else
    switch (rotate_A) {
        [[likely]] case 0:
            update_A(std::integral_constant<int, 0>{});
            break;
        case -1: update_A(std::integral_constant<int, -1>{}); break;
        case 1: update_A(std::integral_constant<int, 1>{}); break;
        default: KOQKATOO_ASSUME(false);
    }
#endif
}

template <class Abi, index_t R, index_t S>
[[gnu::hot]] void old::xshhud_diag_tail_microkernel(
    index_t colsA, triangular_accessor<Abi, const real_t, SizeR> W,
    mut_single_batch_matrix_accessor<Abi> L,
    mut_single_batch_matrix_accessor<Abi> A,
    single_batch_matrix_accessor<Abi> B, single_batch_vector_accessor<Abi> diag,
    bool trans_L) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of L
    KOQKATOO_ASSUME(colsA > 0);

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
                simd Lik = L_cached.load(i, k);
                V[i][k] += Lik;
                KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < k; ++l)
                    V[i][k] -= V[i][l] * Wk[l];
                V[i][k] *= W.load(k, k); // diagonal already inverted
                Lik = V[i][k] - Lik;
                L_cached.store(Lik, i, k);
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
