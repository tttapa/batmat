#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/hyhound.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/ops/cneg.hpp>
#include <batmat/ops/rotate.hpp>
#include <guanaqo/trace.hpp>
#include <type_traits>
#include <utility>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::hyhound {

template <class T, class Abi, index_t R, StorageOrder OL, StorageOrder OA, KernelConfig Conf>
[[gnu::hot, gnu::flatten]] void
xshhud_diag_diag_microkernel(index_t colsA, triangular_accessor<T, Abi, SizeR<T, Abi>> W,
                             uview<T, Abi, OL> L, uview<T, Abi, OA> A,
                             uview<const T, Abi, StorageOrder::ColMajor> diag) noexcept {
    using batmat::ops::cneg;
    using std::copysign;
    using std::sqrt;
    using simd = datapar::simd<T, Abi>;
    // Pre-compute the offsets of the columns of L
    auto L_cached = with_cached_access<R, R>(L);
    BATMAT_ASSUME(colsA > 0);

    UNROLL_FOR (index_t k = 0; k < R; ++k) {
        // Compute all inner products between A and a
        simd bb[R]{};
        for (index_t j = 0; j < colsA; ++j) {
            simd Akj = Conf.sign_only ? cneg(A.load(k, j), diag.load(j, 0)) //
                                      : A.load(k, j) * diag.load(j, 0);
            UNROLL_FOR (index_t i = 0; i < R; ++i)
                bb[i] += A.load(i, j) * Akj;
        }
        // Energy condition and Householder coefficients
        const simd α2 = bb[k], Lkk = L_cached.load(k, k);
        const simd L̃kk = copysign(sqrt(Lkk * Lkk + α2), Lkk), β = Lkk + L̃kk;
        simd γoβ = simd{2} * β / (β * β + α2), γ = β * γoβ, inv_β = simd{1} / β;
        L_cached.store(L̃kk, k, k);
        // Compute L̃
        UNROLL_FOR (index_t i = k + 1; i < R; ++i) {
            simd Lik = L_cached.load(i, k);
            bb[i]    = γ * Lik + bb[i] * γoβ;
            L_cached.store(bb[i] - Lik, i, k);
        }
        // Update A
        for (index_t j = 0; j < colsA; ++j) {
            simd Akj = A.load(k, j) * inv_β; // Scale Householder vector
            A.store(Akj, k, j);
            UNROLL_FOR (index_t i = k + 1; i < R; ++i) {
                simd Aij = A.load(i, j);
                Aij -= bb[i] * Akj;
                A.store(Aij, i, j);
            }
        }
        // Save block Householder matrix W
        UNROLL_FOR (index_t i = 0; i < k + 1; ++i)
            bb[i] *= inv_β;
        bb[k] = γ; // inverse of diagonal
        UNROLL_FOR (index_t i = 0; i < k + 1; ++i)
            W.store(bb[i], i, k);
        // TODO: try moving this to before update of A
    }
}

template <class T, class Abi, index_t R, StorageOrder OL, StorageOrder OA, KernelConfig Conf>
[[gnu::hot, gnu::flatten]] void
xshhud_diag_full_microkernel(index_t colsA, uview<T, Abi, OL> L, uview<T, Abi, OA> A,
                             uview<const T, Abi, StorageOrder::ColMajor> diag) noexcept {
    using batmat::ops::cneg;
    using std::copysign;
    using std::sqrt;
    using simd = datapar::simd<T, Abi>;
    // Pre-compute the offsets of the columns of L
    auto L_cached = with_cached_access<R, R>(L);
    BATMAT_ASSUME(colsA > 0);

    UNROLL_FOR (index_t k = 0; k < R; ++k) {
        // Compute some inner products between A and a
        simd bb[R]{};
        for (index_t j = 0; j < colsA; ++j) {
            simd Akj = Conf.sign_only ? cneg(A.load(k, j), diag.load(j, 0)) //
                                      : A.load(k, j) * diag.load(j, 0);
            UNROLL_FOR (index_t i = k; i < R; ++i)
                bb[i] += A.load(i, j) * Akj;
        }
        // Energy condition and Householder coefficients
        const simd α2 = bb[k], Lkk = L_cached.load(k, k);
        const simd L̃kk = copysign(sqrt(Lkk * Lkk + α2), Lkk), β = Lkk + L̃kk;
        simd γoβ = simd{2} * β / (β * β + α2), γ = β * γoβ, inv_β = simd{1} / β;
        L_cached.store(L̃kk, k, k);
        // Compute L̃
        UNROLL_FOR (index_t i = k + 1; i < R; ++i) {
            simd Lik = L_cached.load(i, k);
            bb[i]    = γ * Lik + bb[i] * γoβ;
            L_cached.store(bb[i] - Lik, i, k);
        }
        // Update A
        for (index_t j = 0; j < colsA; ++j) {
            simd Akj = A.load(k, j) * inv_β; // Scale Householder vector
            A.store(Akj, k, j);
            UNROLL_FOR (index_t i = k + 1; i < R; ++i) {
                simd Aij = A.load(i, j);
                Aij -= bb[i] * Akj;
                A.store(Aij, i, j);
            }
        }
    }
}

namespace detail {
using ops::rot;
using ops::rotr;

template <class T, class Abi, int S>
[[gnu::always_inline]] inline auto rotate(datapar::simd<T, Abi> x, std::integral_constant<int, S>) {
    return rotr<S>(x);
}

template <class T, class Abi>
[[gnu::always_inline]] inline auto rotate(datapar::simd<T, Abi> x, int s) {
    return rot(x, s);
}

} // namespace detail

template <class T, class Abi, index_t R, index_t S, StorageOrder OL, StorageOrder OA,
          StorageOrder OB, KernelConfig Conf>
[[gnu::hot, gnu::flatten]] void xshhud_diag_tail_microkernel(
    index_t kA_nonzero_start, index_t kA_nonzero_end, index_t colsA,
    triangular_accessor<const T, Abi, SizeR<T, Abi>> W, uview<T, Abi, OL> L,
    uview<const T, Abi, OA> A_in, uview<T, Abi, OA> A_out, uview<const T, Abi, OB> B,
    uview<const T, Abi, StorageOrder::ColMajor> diag, Structure struc_L, int rotate_A) noexcept {
    using batmat::ops::cneg;
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(colsA > 0);

    // Compute product U = A B
    simd V[S][R]{};
    for (index_t j = kA_nonzero_start; j < kA_nonzero_end; ++j) {
        UNROLL_FOR (index_t k = 0; k < R; ++k) {
            auto Akj = Conf.sign_only ? cneg(B.load(k, j), diag.load(j, 0)) //
                                      : B.load(k, j) * diag.load(j, 0);
            UNROLL_FOR (index_t i = 0; i < S; ++i)
                V[i][k] += A_in.load(i, j) * Akj;
        }
    }

    // Solve system V = (L+U)W⁻¹ (in-place)
    auto L_cached = with_cached_access<S, R>(L);
    switch (struc_L) {
        [[likely]]
        case Structure::General: {
            UNROLL_FOR (index_t k = 0; k < R; ++k) {
                simd Wk[R];
                UNROLL_FOR (index_t l = 0; l < k; ++l)
                    Wk[l] = W.load(l, k);
                UNROLL_FOR (index_t i = 0; i < S; ++i) {
                    simd Lik = L_cached.load(i, k);
                    V[i][k] += Lik;
                    UNROLL_FOR (index_t l = 0; l < k; ++l)
                        V[i][k] -= V[i][l] * Wk[l];
                    V[i][k] *= W.load(k, k); // diagonal already inverted
                    Lik = V[i][k] - Lik;
                    L_cached.store(Lik, i, k);
                }
            }
        } break;
        case Structure::Zero: {
            UNROLL_FOR (index_t k = 0; k < R; ++k) {
                simd Wk[R];
                UNROLL_FOR (index_t l = 0; l < k; ++l)
                    Wk[l] = W.load(l, k);
                UNROLL_FOR (index_t i = 0; i < S; ++i) {
                    UNROLL_FOR (index_t l = 0; l < k; ++l)
                        V[i][k] -= V[i][l] * Wk[l];
                    V[i][k] *= W.load(k, k); // diagonal already inverted
                }
            }
        } break;
        case Structure::Upper: {
            UNROLL_FOR (index_t k = 0; k < R; ++k) {
                simd Wk[R];
                UNROLL_FOR (index_t l = 0; l < k; ++l)
                    Wk[l] = W.load(l, k);
                UNROLL_FOR (index_t i = 0; i < S; ++i) {
                    simd Lik;
                    if (i <= k) {
                        Lik = L_cached.load(i, k);
                        V[i][k] += Lik;
                    }
                    UNROLL_FOR (index_t l = 0; l < k; ++l)
                        V[i][k] -= V[i][l] * Wk[l];
                    V[i][k] *= W.load(k, k); // diagonal already inverted
                    if (i <= k) {
                        Lik = V[i][k] - Lik;
                        L_cached.store(Lik, i, k);
                    }
                }
            }
        } break;
        default: BATMAT_ASSUME(false);
    }
    // Update A -= V Bᵀ
    const auto update_A = [&] [[gnu::always_inline]] (auto s) {
        simd Akj[R];
#if 0 // bottom variant generates less code
        for (index_t j = 0; j < kA_nonzero_start; ++j) [[unlikely]] {
            UNROLL_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            UNROLL_FOR (index_t i = 0; i < S; ++i) {
                simd Aij{0};
                UNROLL_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
        for (index_t j = kA_nonzero_start; j < kA_nonzero_end; ++j) [[likely]] {
            UNROLL_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            UNROLL_FOR (index_t i = 0; i < S; ++i) {
                auto Aij = A_in.load(i, j);
                UNROLL_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
        for (index_t j = kA_nonzero_end; j < colsA; ++j) [[unlikely]] {
            UNROLL_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            UNROLL_FOR (index_t i = 0; i < S; ++i) {
                simd Aij{0};rotate
                UNROLL_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
#else
        for (index_t j = kA_nonzero_start; j < kA_nonzero_end; ++j) [[likely]] {
            UNROLL_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            UNROLL_FOR (index_t i = 0; i < S; ++i) {
                auto Aij = A_in.load(i, j);
                UNROLL_FOR (index_t k = 0; k < R; ++k)
                    Aij -= V[i][k] * Akj[k];
                A_out.store(detail::rotate(Aij, s), i, j);
            }
        }
        for (index_t j = 0; true; ++j) {
            if (j == kA_nonzero_start)
                j = kA_nonzero_end;
            if (j >= colsA)
                break;
            UNROLL_FOR (index_t k = 0; k < R; ++k)
                Akj[k] = B.load(k, j);
            UNROLL_FOR (index_t i = 0; i < S; ++i) {
                simd Aij{0};
                UNROLL_FOR (index_t k = 0; k < R; ++k)
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
        // case 1: update_A(std::integral_constant<int, 1>{}); break;
        default: BATMAT_ASSUME(false);
    }
#endif
}

template <class T, class Abi, KernelConfig Conf, StorageOrder OL, StorageOrder OA>
void xshhud_diag_ref(view<T, Abi, OL> L, view<T, Abi, OA> A, view<const T, Abi> D) noexcept {
    static constexpr index_constant<SizeR<T, Abi>> R;
    static constexpr index_constant<SizeS<T, Abi>> S;
    const index_t C = A.cols();
    auto n1 = L.cols(), n2 = L.rows() - n1;
    auto flop_count_diag_11             = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11             = 2 * (C + 1) * n2 * n1 + C * n2;
    [[maybe_unused]] index_t flop_count = flop_count_diag_11 + flop_count_tail_11;
    GUANAQO_TRACE("xshhud_diag", 0, flop_count * L.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    if (L.rows() == L.cols()) {
        foreach_chunked(
            0, L.cols(), R,
            [&](index_t k) {
                // Part of A corresponding to this diagonal block
                // TODO: packing
                auto Ad = A.middle_rows(k, R);
                auto Ld = L.block(k, k, R, R);
                // Process the diagonal block itself
                xshhud_diag_diag_microkernel<T, Abi, R, OL, OA, Conf>(A.cols(), W, Ld, Ad, D);
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    k + R, L.rows(), S,
                    [&](index_t i, auto rem_i) {
                        auto As = A.middle_rows(i, rem_i);
                        auto Ls = L.block(i, k, rem_i, R);
                        microkernel_tail_lut<T, Abi, OL, OA, OA, Conf>[rem_i - 1](
                            0, A.cols(), A.cols(), W, Ls, As, As, Ad, D, Structure::General, 0);
                    },
                    LoopDir::Backward); // TODO: decide on order
            },
            [&](index_t k, index_t rem_k) {
                auto Ad = A.middle_rows(k, rem_k);
                auto Ld = L.block(k, k, rem_k, rem_k);
                microkernel_full_lut<T, Abi, OL, OA, Conf>[rem_k - 1](A.cols(), Ld, Ad, D);
            });
    } else {
        foreach_chunked_merged(0, L.cols(), R, [&](index_t k, auto rem_k) {
            // Part of A corresponding to this diagonal block
            // TODO: packing
            auto Ad = A.middle_rows(k, rem_k);
            auto Ld = L.block(k, k, rem_k, rem_k);
            // Process the diagonal block itself
            microkernel_diag_lut<T, Abi, OL, OA, Conf>[rem_k - 1](A.cols(), W, Ld, Ad, D);
            // Process all rows below the diagonal block (in multiples of S).
            foreach_chunked_merged(
                k + rem_k, L.rows(), S,
                [&](index_t i, auto rem_i) {
                    auto As = A.middle_rows(i, rem_i);
                    auto Ls = L.block(i, k, rem_i, rem_k);
                    microkernel_tail_lut_2<T, Abi, OL, OA, OA, Conf>[rem_k - 1][rem_i - 1](
                        0, A.cols(), A.cols(), W, Ls, As, As, Ad, D, Structure::General, 0);
                },
                LoopDir::Backward); // TODO: decide on order
        });
    }
}

template <class T, class Abi, KernelConfig Conf, StorageOrder OL, StorageOrder OA>
void xshhud_diag_ref(view<T, Abi, OL> L, view<T, Abi, OA> A, view<const T, Abi> D,
                     view<T, Abi> W) noexcept {
    static constexpr index_constant<SizeR<T, Abi>> R;
    const index_t C = A.cols();
    auto n1 = L.cols(), n2 = L.rows() - n1;
    auto flop_count_diag_11             = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11             = 2 * (C + 1) * n2 * n1 + C * n2;
    [[maybe_unused]] index_t flop_count = flop_count_diag_11 + flop_count_tail_11;
    GUANAQO_TRACE("xshhud_diag", 0, flop_count * L.depth());
    if (C == 0)
        return;

    BATMAT_ASSERT(std::make_pair(W.rows(), W.cols()) == (xshhud_W_size<T, Abi>)(L));
    using W_t = triangular_accessor<T, Abi, R>;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L.cols(), R, [&](index_t k, auto nk) {
        static constexpr index_constant<SizeS<T, Abi>> S;
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A.middle_rows(k, nk);
        auto Ld = L.block(k, k, nk, nk);
        auto Wd = W_t{&W(0, 0, k / R)};
        // Process the diagonal block itself
        microkernel_diag_lut<T, Abi, OL, OA, Conf>[nk - 1](A.cols(), Wd, Ld, Ad, D);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            k + nk, L.rows(), S,
            [&](index_t i, auto ni) {
                auto As = A.middle_rows(i, ni);
                auto Ls = L.block(i, k, ni, nk);
                microkernel_tail_lut_2<T, Abi, OL, OA, OA, Conf>[nk - 1][ni - 1](
                    0, A.cols(), A.cols(), Wd, Ls, As, As, Ad, D, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

template <class T, class Abi, KernelConfig Conf, StorageOrder OL, StorageOrder OA>
void xshh_apply_diag_ref(view<T, Abi, OL> L, view<const T, Abi, OA> Ain, view<T, Abi, OA> Aout,
                         view<const T, Abi, OA> B, view<const T, Abi> D, view<const T, Abi> W,
                         index_t kA_nonzero_start, index_t kA_nonzero_end) noexcept {
    static constexpr index_constant<SizeR<T, Abi>> R;
    const index_t C = Ain.cols();
    kA_nonzero_end  = (kA_nonzero_end == -1) ? C : kA_nonzero_end;
    auto n1 = L.cols(), n2 = L.rows();
    [[maybe_unused]] index_t flop_count = 2 * (C + 1) * n2 * n1 + C * n2;
    GUANAQO_TRACE("xshh_apply_diag", 0, flop_count * L.depth());
    if (C == 0)
        return;

    BATMAT_ASSERT(std::make_pair(W.rows(), W.cols()) == (xshhud_W_size<T, Abi>)(L));
    using W_t = triangular_accessor<const T, Abi, R>;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L.cols(), R, [&](index_t k, auto nk) {
        static constexpr index_constant<SizeS<T, Abi>> S;
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = B.middle_rows(k, nk);
        auto Wd = W_t{&W(0, 0, k / R)};
        // Process all rows (in multiples of S).
        foreach_chunked_merged( // TODO: swap loop order?
            0, L.rows(), S,
            [&](index_t i, auto ni) {
                auto Aini  = k == 0 ? Ain.middle_rows(i, ni) : Aout.middle_rows(i, ni);
                auto Aouti = Aout.middle_rows(i, ni);
                auto Ls    = L.block(i, k, ni, nk);
                microkernel_tail_lut_2<T, Abi, OL, OA, OA, Conf>[nk - 1][ni - 1](
                    k == 0 ? kA_nonzero_start : 0, k == 0 ? kA_nonzero_end : C, C, Wd, Ls, Aini,
                    Aouti, Ad, D, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

template <class T, class Abi, StorageOrder OL1, StorageOrder OA1, StorageOrder OL2,
          StorageOrder OA2, KernelConfig Conf>
void xshhud_diag_2_ref(view<T, Abi, OL1> L11, view<T, Abi, OA1> A1, view<T, Abi, OL2> L21,
                       view<T, Abi, OA2> A2, view<const T, Abi> D) noexcept {
    BATMAT_ASSERT(L11.rows() >= L11.cols());
    BATMAT_ASSERT(L11.rows() == A1.rows());
    BATMAT_ASSERT(A1.cols() == D.rows());
    BATMAT_ASSERT(A2.cols() == A1.cols());
    BATMAT_ASSERT(L21.cols() == L11.cols());
    static constexpr index_constant<SizeR<T, Abi>> R;
    const index_t C = A1.cols();
    auto n1 = L11.cols(), n2 = L11.rows() - n1;
    auto flop_count_diag_11 = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11 = 2 * (C + 1) * n2 * n1 + C * n2;
    auto flop_count_tail_21 = 2 * (C + 1) * L21.rows() * n1 + C * L21.rows();
    [[maybe_unused]] index_t flop_count =
        flop_count_diag_11 + flop_count_tail_11 + flop_count_tail_21;
    GUANAQO_TRACE("xshhud_diag_2", 0, flop_count * L11.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t k, auto rem_k) {
        static constexpr index_constant<SizeS<T, Abi>> S;
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1.middle_rows(k, rem_k);
        auto Ld = L11.block(k, k, rem_k, rem_k);
        // Process the diagonal block itself
        microkernel_diag_lut<T, Abi, OL1, OA1, Conf>[rem_k - 1](A1.cols(), W, Ld, Ad, D);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            k + rem_k, L11.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A1.middle_rows(i, rem_i);
                auto Ls = L11.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<T, Abi, OL1, OA1, OA1, Conf>[rem_k - 1][rem_i - 1](
                    0, A1.cols(), A1.cols(), W, Ls, As, As, Ad, D, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L21.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A2.middle_rows(i, rem_i);
                auto Ls = L21.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<T, Abi, OL2, OA2, OA1, Conf>[rem_k - 1][rem_i - 1](
                    0, A1.cols(), A1.cols(), W, Ls, As, As, Ad, D, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

/**
 * Performs a factorization update of the following matrix:
 *
 *     [ A11 A12 | L11 ]     [  0   0  | L̃11 ]
 *     [  0  A22 | L21 ] Q = [ Ã21 Ã22 | L̃21 ]
 *     [ A31  0  | L31 ]     [ Ã31 Ã32 | L̃31 ]
 *           ↑ split_A
 */
template <class T, class Abi, StorageOrder OL, StorageOrder OW, StorageOrder OY, StorageOrder OU,
          KernelConfig Conf>
void xshhud_diag_cyclic(view<T, Abi, OL> L11,      // D
                        view<T, Abi, OW> A1,       // work
                        view<T, Abi, OY> L21,      // Y
                        view<const T, Abi, OW> A2, // work
                        view<T, Abi, OW> A2_out,   // work
                        view<T, Abi, OU> L31,      // U
                        view<const T, Abi, OW> A3, // work
                        view<T, Abi, OW> A3_out,   // work
                        view<const T, Abi> D, index_t split_A) noexcept {
    BATMAT_ASSERT(L11.rows() >= L11.cols());
    BATMAT_ASSERT(L11.rows() == A1.rows());
    BATMAT_ASSERT(L21.rows() == A2.rows());
    BATMAT_ASSERT(L31.rows() == A3.rows());
    BATMAT_ASSERT(A1.cols() == D.rows());
    BATMAT_ASSERT(A2.cols() == A1.cols());
    BATMAT_ASSERT(A3.cols() == A1.cols());
    BATMAT_ASSERT(L21.cols() == L11.cols());
    BATMAT_ASSERT(L31.cols() == L11.cols());
    static constexpr index_constant<SizeR<T, Abi>> R;
    const index_t C = A1.cols();
    auto n1 = L11.cols(), n2 = L11.rows() - n1;
    auto flop_count_diag_11 = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11 = 2 * (C + 1) * n2 * n1 + C * n2;
    auto flop_count_tail_21 = 2 * (C + 1) * L21.rows() * n1 + C * L21.rows();
    auto flop_count_tail_31 = 2 * (C + 1) * L31.rows() * n1 + C * L31.rows();
    // Note: initial zero values of A for simplicity (for large matrices this
    //       does not matter)
    [[maybe_unused]] index_t flop_count =
        flop_count_diag_11 + flop_count_tail_11 + flop_count_tail_21 + flop_count_tail_31;
    GUANAQO_TRACE("xshhud_diag_cyclic", 0, flop_count * L11.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t k, auto rem_k) {
        static constexpr index_constant<SizeS<T, Abi>> S;
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1.middle_rows(k, rem_k);
        auto Ld = L11.block(k, k, rem_k, rem_k);
        // Process the diagonal block itself
        microkernel_diag_lut<T, Abi, OL, OW, Conf>[rem_k - 1](C, W, Ld, Ad, D);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            k + rem_k, L11.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A1.middle_rows(i, rem_i);
                auto Ls = L11.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<T, Abi, OL, OW, OW, Conf>[rem_k - 1][rem_i - 1](
                    0, C, C, W, Ls, As, As, Ad, D, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L21.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As_out = A2_out.middle_rows(i, rem_i);
                auto As     = k == 0 ? A2.middle_rows(i, rem_i) : As_out;
                auto Ls     = L21.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<T, Abi, OY, OW, OW, Conf>[rem_k - 1][rem_i - 1](
                    k == 0 ? split_A : 0, C, C, W, Ls, As, As_out, Ad, D, Structure::General, 0);
                // First half of A2 is implicitly zero in first pass
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L31.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As_out = A3_out.middle_rows(i, rem_i);
                auto As     = k == 0 ? A3.middle_rows(i, rem_i) : As_out;
                auto Ls     = L31.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<T, Abi, OU, OW, OW, Conf>[rem_k - 1][rem_i - 1](
                    0, k == 0 ? split_A : C, C, W, Ls, As, As_out, Ad, D, Structure::General, 0);
                // Second half of A3 is implicitly zero in first pass
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

/**
 * Performs a factorization update of the following matrix:
 *
 *     [ A1 | L11 ]     [  0 | L̃11 ]
 *     [ A2 | L21 ] Q = [ Ã2 | L̃21 ]
 *     [  0 | Lu1 ]     [ Ãu | L̃u1 ]
 * where Lu1 and L̃u1 are upper triangular
 */
template <class T, class Abi, StorageOrder OL, StorageOrder OA, StorageOrder OLu, StorageOrder OAu,
          KernelConfig Conf>
void xshhud_diag_riccati(view<T, Abi, OL> L11, view<T, Abi, OA> A1, view<T, Abi, OL> L21,
                         view<const T, Abi, OA> A2, view<T, Abi, OA> A2_out, view<T, Abi, OLu> Lu1,
                         view<T, Abi, OAu> Au_out, view<const T, Abi> D,
                         bool shift_A_out) noexcept {
    BATMAT_ASSERT(L11.rows() >= L11.cols());
    BATMAT_ASSERT(L11.rows() == A1.rows());
    BATMAT_ASSERT(L21.rows() == A2.rows());
    BATMAT_ASSERT(A2_out.rows() == A2.rows());
    BATMAT_ASSERT(A2_out.cols() == A2.cols());
    BATMAT_ASSERT(Lu1.rows() == Au_out.rows());
    BATMAT_ASSERT(A1.cols() == D.rows());
    BATMAT_ASSERT(A2.cols() == A1.cols());
    BATMAT_ASSERT(L21.cols() == L11.cols());
    BATMAT_ASSERT(Lu1.cols() == L11.cols());
    static constexpr index_constant<SizeR<T, Abi>> R;
    static constexpr index_constant<SizeS<T, Abi>> S;
    static_assert(R == S);
    const index_t C = A1.cols();
    auto n1 = L11.cols(), n2 = L11.rows() - n1;
    auto flop_count_diag_11 = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11 = 2 * (C + 1) * n2 * n1 + C * n2;
    auto flop_count_tail_21 = 2 * (C + 1) * L21.rows() * n1 + C * L21.rows();
    auto flop_count_tail_u1 = 2 * (C + 1) * Lu1.rows() * n1 + C * Lu1.rows();
    // Note: ignoring upper trapezoidal shape and initial zero value of Au for
    //       simplicity (for large matrices this does not matter)
    [[maybe_unused]] index_t flop_count =
        flop_count_diag_11 + flop_count_tail_11 + flop_count_tail_21 + flop_count_tail_u1;
    GUANAQO_TRACE("xshhud_diag_riccati", 0, flop_count * L11.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t k, auto rem_k) {
        const bool do_shift = shift_A_out && k + rem_k == L11.cols();
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1.middle_rows(k, rem_k);
        auto Ld = L11.block(k, k, rem_k, rem_k);
        // Process the diagonal block itself
        microkernel_diag_lut<T, Abi, OL, OA, Conf>[rem_k - 1](C, W, Ld, Ad, D);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            k + rem_k, L11.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A1.middle_rows(i, rem_i);
                auto Ls = L11.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<T, Abi, OL, OA, OA, Conf>[rem_k - 1][rem_i - 1](
                    0, C, C, W, Ls, As, As, Ad, D, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L21.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As_out = A2_out.middle_rows(i, rem_i);
                auto As     = k == 0 ? A2.middle_rows(i, rem_i) : As_out;
                auto Ls     = L21.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<T, Abi, OL, OA, OA, Conf>[rem_k - 1][rem_i - 1](
                    0, C, C, W, Ls, As, As_out, Ad, D, Structure::General, do_shift ? -1 : 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, Lu1.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As_out = Au_out.middle_rows(i, rem_i);
                auto As     = As_out;
                auto Ls     = Lu1.block(i, k, rem_i, rem_k);
                // Au is implicitly zero in first pass
                const auto struc = i == k  ? Structure::Upper
                                   : i < k ? Structure::General
                                           : Structure::Zero;
                microkernel_tail_lut_2<T, Abi, OLu, OAu, OA, Conf>[rem_k - 1][rem_i - 1](
                    0, k == 0 ? 0 : C, C, W, Ls, As, As_out, Ad, D, struc, do_shift ? -1 : 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

} // namespace batmat::linalg::micro_kernels::hyhound
