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

template <class T, class Abi, KernelConfig Conf, index_t R, StorageOrder OL, StorageOrder OA>
[[gnu::hot, gnu::flatten]] void
hyhound_diag_diag_microkernel(index_t kA, triangular_accessor<T, Abi, SizeR<T, Abi>> W,
                              uview<T, Abi, OL> L, uview<T, Abi, OA> A,
                              uview<const T, Abi, StorageOrder::ColMajor> diag) noexcept {
    using batmat::ops::cneg;
    using std::copysign;
    using std::sqrt;
    using simd = datapar::simd<T, Abi>;
    // Pre-compute the offsets of the columns of L
    auto L_cached = with_cached_access<R, R>(L);
    BATMAT_ASSUME(kA > 0);

    UNROLL_FOR (index_t j = 0; j < R; ++j) {
        // Compute all inner products between A and a
        simd bb[R]{};
        for (index_t l = 0; l < kA; ++l) {
            simd Ajl = Conf.sign_only ? cneg(A.load(j, l), diag.load(l, 0)) //
                                      : A.load(j, l) * diag.load(l, 0);
            UNROLL_FOR (index_t i = 0; i < R; ++i)
                bb[i] += A.load(i, l) * Ajl;
        }
        // Energy condition and Householder coefficients
        const simd α2 = bb[j], Ljj = L_cached.load(j, j);
        const simd L̃jj = copysign(sqrt(Ljj * Ljj + α2), Ljj), β = Ljj + L̃jj;
        simd γoβ = simd{2} * β / (β * β + α2), γ = β * γoβ, inv_β = simd{1} / β;
        L_cached.store(L̃jj, j, j);
        // Compute L̃
        UNROLL_FOR (index_t i = j + 1; i < R; ++i) {
            simd Lij = L_cached.load(i, j);
            bb[i]    = γ * Lij + bb[i] * γoβ;
            L_cached.store(bb[i] - Lij, i, j);
        }
        // Update A
        for (index_t l = 0; l < kA; ++l) {
            simd Ajl = A.load(j, l) * inv_β; // Scale Householder vector
            A.store(Ajl, j, l);
            UNROLL_FOR (index_t i = j + 1; i < R; ++i) {
                simd Ail = A.load(i, l);
                Ail -= bb[i] * Ajl;
                A.store(Ail, i, l);
            }
        }
        // Save block Householder matrix W
        UNROLL_FOR (index_t i = 0; i < j + 1; ++i)
            bb[i] *= inv_β;
        bb[j] = γ; // inverse of diagonal
        UNROLL_FOR (index_t i = 0; i < j + 1; ++i)
            W.store(bb[i], i, j);
        // TODO: try moving this to before update of A
    }
}

template <class T, class Abi, KernelConfig Conf, index_t R, StorageOrder OL, StorageOrder OA>
[[gnu::hot, gnu::flatten]] void
hyhound_diag_full_microkernel(index_t kA, uview<T, Abi, OL> L, uview<T, Abi, OA> A,
                              uview<const T, Abi, StorageOrder::ColMajor> diag) noexcept {
    using batmat::ops::cneg;
    using std::copysign;
    using std::sqrt;
    using simd = datapar::simd<T, Abi>;
    // Pre-compute the offsets of the columns of L
    auto L_cached = with_cached_access<R, R>(L);
    BATMAT_ASSUME(kA > 0);

    UNROLL_FOR (index_t j = 0; j < R; ++j) {
        // Compute some inner products between A and a
        simd bb[R]{};
        for (index_t l = 0; l < kA; ++l) {
            simd Ajl = Conf.sign_only ? cneg(A.load(j, l), diag.load(l, 0)) //
                                      : A.load(j, l) * diag.load(l, 0);
            UNROLL_FOR (index_t i = j; i < R; ++i)
                bb[i] += A.load(i, l) * Ajl;
        }
        // Energy condition and Householder coefficients
        const simd α2 = bb[j], Ljj = L_cached.load(j, j);
        const simd L̃jj = copysign(sqrt(Ljj * Ljj + α2), Ljj), β = Ljj + L̃jj;
        simd γoβ = simd{2} * β / (β * β + α2), γ = β * γoβ, inv_β = simd{1} / β;
        L_cached.store(L̃jj, j, j);
        // Compute L̃
        UNROLL_FOR (index_t i = j + 1; i < R; ++i) {
            simd Lij = L_cached.load(i, j);
            bb[i]    = γ * Lij + bb[i] * γoβ;
            L_cached.store(bb[i] - Lij, i, j);
        }
        // Update A
        for (index_t l = 0; l < kA; ++l) {
            simd Ajl = A.load(j, l) * inv_β; // Scale Householder vector
            A.store(Ajl, j, l);
            UNROLL_FOR (index_t i = j + 1; i < R; ++i) {
                simd Ail = A.load(i, l);
                Ail -= bb[i] * Ajl;
                A.store(Ail, i, l);
            }
        }
    }
}

namespace detail {

template <class T, class Abi, int S>
[[gnu::always_inline]] inline auto rotate(datapar::simd<T, Abi> x, std::integral_constant<int, S>) {
    using ops::rotr;
    return rotr<S>(x);
}

template <class T, class Abi>
[[gnu::always_inline]] inline auto rotate(datapar::simd<T, Abi> x, int s) {
    using ops::rot;
    return rot(x, s);
}

} // namespace detail

// A_out and B have the same size. A_in has the same number of rows but may have a different number
// of columns, which means that only a part of A_in is nonzero. The nonzero part is defined by kAin
// and kAin_offset.
template <class T, class Abi, KernelConfig Conf, index_t R, index_t S, StorageOrder OL,
          StorageOrder OA, StorageOrder OB>
[[gnu::hot, gnu::flatten]] void hyhound_diag_tail_microkernel(
    index_t kA_in_offset, index_t kA_in, index_t k,
    triangular_accessor<const T, Abi, SizeR<T, Abi>> W, uview<T, Abi, OL> L,
    uview<const T, Abi, OA> A_in, uview<T, Abi, OA> A_out, uview<const T, Abi, OB> B,
    uview<const T, Abi, StorageOrder::ColMajor> diag, Structure struc_L, int rotate_A) noexcept {
    using batmat::ops::cneg;
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(k > 0);

    // Compute product W = A B
    simd V[S][R]{};
    for (index_t lA = 0; lA < kA_in; ++lA) {
        index_t lB = lA + kA_in_offset;
        UNROLL_FOR (index_t j = 0; j < R; ++j) {
            auto Bjl = Conf.sign_only ? cneg(B.load(j, lB), diag.load(lB, 0)) //
                                      : B.load(j, lB) * diag.load(lB, 0);
            UNROLL_FOR (index_t i = 0; i < S; ++i)
                V[i][j] += A_in.load(i, lA) * Bjl;
        }
    }

    // Solve system V = (L+U)W⁻¹ (in-place)
    auto L_cached = with_cached_access<S, R>(L);
    switch (struc_L) {
        [[likely]]
        case Structure::General: {
            UNROLL_FOR (index_t j = 0; j < R; ++j) {
                simd Wj[R];
                UNROLL_FOR (index_t i = 0; i < j; ++i)
                    Wj[i] = W.load(i, j);
                UNROLL_FOR (index_t i = 0; i < S; ++i) {
                    simd Lij = L_cached.load(i, j);
                    V[i][j] += Lij;
                    UNROLL_FOR (index_t l = 0; l < j; ++l)
                        V[i][j] -= V[i][l] * Wj[l];
                    V[i][j] *= W.load(j, j); // diagonal already inverted
                    Lij = V[i][j] - Lij;
                    L_cached.store(Lij, i, j);
                }
            }
        } break;
        case Structure::Zero: {
            UNROLL_FOR (index_t j = 0; j < R; ++j) {
                simd Wj[R];
                UNROLL_FOR (index_t i = 0; i < j; ++i)
                    Wj[i] = W.load(i, j);
                UNROLL_FOR (index_t i = 0; i < S; ++i) {
                    UNROLL_FOR (index_t l = 0; l < j; ++l)
                        V[i][j] -= V[i][l] * Wj[l];
                    V[i][j] *= W.load(j, j); // diagonal already inverted
                }
            }
        } break;
        case Structure::Upper: {
            UNROLL_FOR (index_t j = 0; j < R; ++j) {
                simd Wj[R];
                UNROLL_FOR (index_t i = 0; i < j; ++i)
                    Wj[i] = W.load(i, j);
                UNROLL_FOR (index_t i = 0; i < S; ++i) {
                    simd Lij;
                    if (i <= j) {
                        Lij = L_cached.load(i, j);
                        V[i][j] += Lij;
                    }
                    UNROLL_FOR (index_t l = 0; l < j; ++l)
                        V[i][j] -= V[i][l] * Wj[l];
                    V[i][j] *= W.load(j, j); // diagonal already inverted
                    if (i <= j) {
                        Lij = V[i][j] - Lij;
                        L_cached.store(Lij, i, j);
                    }
                }
            }
        } break;
        default: BATMAT_ASSUME(false);
    }
    // Update A -= V Bᵀ
    const auto update_A = [&] [[gnu::always_inline]] (auto s) {
        simd Bjl[R];
        for (index_t lB = 0; lB < kA_in_offset; ++lB) [[unlikely]] {
            UNROLL_FOR (index_t j = 0; j < R; ++j)
                Bjl[j] = B.load(j, lB);
            UNROLL_FOR (index_t i = 0; i < S; ++i) {
                simd Ail{0};
                UNROLL_FOR (index_t j = 0; j < R; ++j)
                    Ail -= V[i][j] * Bjl[j];
                A_out.store(detail::rotate(Ail, s), i, lB);
            }
        }
        for (index_t lB = kA_in_offset + kA_in; lB < k; ++lB) [[unlikely]] {
            UNROLL_FOR (index_t j = 0; j < R; ++j)
                Bjl[j] = B.load(j, lB);
            UNROLL_FOR (index_t i = 0; i < S; ++i) {
                simd Ail{0};
                UNROLL_FOR (index_t j = 0; j < R; ++j)
                    Ail -= V[i][j] * Bjl[j];
                A_out.store(detail::rotate(Ail, s), i, lB);
            }
        }
        for (index_t lA = 0; lA < kA_in; ++lA) [[likely]] {
            index_t lB = lA + kA_in_offset;
            UNROLL_FOR (index_t j = 0; j < R; ++j)
                Bjl[j] = B.load(j, lB);
            UNROLL_FOR (index_t i = 0; i < S; ++i) {
                auto Ail = A_in.load(i, lA);
                UNROLL_FOR (index_t j = 0; j < R; ++j)
                    Ail -= V[i][j] * Bjl[j];
                A_out.store(detail::rotate(Ail, s), i, lB);
            }
        }
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

/// Block hyperbolic Householder factorization update using register blocking.
/// This variant does not store the Householder representation W.
template <class T, class Abi, KernelConfig Conf, StorageOrder OL, StorageOrder OA>
void hyhound_diag_register(const view<T, Abi, OL> L, const view<T, Abi, OA> A,
                           const view<const T, Abi> D) noexcept {
    static constexpr index_constant<SizeR<T, Abi>> R;
    static constexpr index_constant<SizeS<T, Abi>> S;
    const index_t k = A.cols();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(L.rows() >= L.cols());
    BATMAT_ASSUME(L.rows() == A.rows());
    BATMAT_ASSUME(A.cols() == D.rows());

    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W[W_t::size()];

    // Sizeless views to partition and pass to the micro-kernels
    const uview<T, Abi, OL> L_                           = L;
    const uview<T, Abi, OA> A_                           = A;
    const uview<const T, Abi, StorageOrder::ColMajor> D_ = D;

    // Process all diagonal blocks (in multiples of R, except the last).
    if (L.rows() == L.cols()) {
        foreach_chunked(
            0, L.cols(), R,
            [&](index_t j) {
                // Part of A corresponding to this diagonal block
                // TODO: packing
                auto Ad = A_.middle_rows(j);
                auto Ld = L_.block(j, j);
                // Process the diagonal block itself
                hyhound_diag_diag_microkernel<T, Abi, Conf, R, OL, OA>(k, W, Ld, Ad, D_);
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    j + R, L.rows(), S,
                    [&](index_t i, auto rem_i) {
                        auto As = A_.middle_rows(i);
                        auto Ls = L_.block(i, j);
                        microkernel_tail_lut<T, Abi, Conf, OL, OA, OA>[rem_i - 1](
                            0, k, k, W, Ls, As, As, Ad, D_, Structure::General, 0);
                    },
                    LoopDir::Backward); // TODO: decide on order
            },
            [&](index_t j, index_t rem_j) {
                auto Ad = A_.middle_rows(j);
                auto Ld = L_.block(j, j);
                microkernel_full_lut<T, Abi, Conf, OL, OA>[rem_j - 1](k, Ld, Ad, D_);
            });
    } else {
        foreach_chunked_merged(0, L.cols(), R, [&](index_t j, auto rem_j) {
            // Part of A corresponding to this diagonal block
            // TODO: packing
            auto Ad = A_.middle_rows(j);
            auto Ld = L_.block(j, j);
            // Process the diagonal block itself
            microkernel_diag_lut<T, Abi, Conf, OL, OA>[rem_j - 1](k, W, Ld, Ad, D_);
            // Process all rows below the diagonal block (in multiples of S).
            foreach_chunked_merged(
                j + rem_j, L.rows(), S,
                [&](index_t i, auto rem_i) {
                    auto As = A_.middle_rows(i);
                    auto Ls = L_.block(i, j);
                    microkernel_tail_lut_2<T, Abi, Conf, OL, OA, OA>[rem_j - 1][rem_i - 1](
                        0, k, k, W, Ls, As, As, Ad, D_, Structure::General, 0);
                },
                LoopDir::Backward); // TODO: decide on order
        });
    }
}

/// Block hyperbolic Householder factorization update using register blocking.
/// This variant stores the Householder representation W.
template <class T, class Abi, KernelConfig Conf, StorageOrder OL, StorageOrder OA>
void hyhound_diag_register(const view<T, Abi, OL> L, const view<T, Abi, OA> A,
                           const view<const T, Abi> D, const view<T, Abi> W) noexcept {
    const index_t k = A.cols();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(L.rows() >= L.cols());
    BATMAT_ASSUME(L.rows() == A.rows());
    BATMAT_ASSUME(D.rows() == k);
    BATMAT_ASSUME(std::make_pair(W.rows(), W.cols()) == (xshhud_W_size<T, Abi>)(L));

    static constexpr index_constant<SizeR<T, Abi>> R;
    using W_t = triangular_accessor<T, Abi, R>;

    // Sizeless views to partition and pass to the micro-kernels
    const uview<T, Abi, OL> L_                           = L;
    const uview<T, Abi, OA> A_                           = A;
    const uview<const T, Abi, StorageOrder::ColMajor> D_ = D;
    const uview<T, Abi, StorageOrder::ColMajor> W_       = W;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L.cols(), R, [&](index_t j, auto nj) {
        static constexpr index_constant<SizeS<T, Abi>> S;
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A_.middle_rows(j);
        auto Ld = L_.block(j, j);
        auto Wd = W_t{W_.middle_cols(j / R).data};
        // Process the diagonal block itself
        microkernel_diag_lut<T, Abi, Conf, OL, OA>[nj - 1](k, Wd, Ld, Ad, D_);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            j + nj, L.rows(), S,
            [&](index_t i, auto ni) {
                auto As = A_.middle_rows(i);
                auto Ls = L_.block(i, j);
                microkernel_tail_lut_2<T, Abi, Conf, OL, OA, OA>[nj - 1][ni - 1](
                    0, k, k, Wd, Ls, As, As, Ad, D_, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

/// Apply a block hyperbolic Householder transformation.
template <class T, class Abi, KernelConfig Conf, StorageOrder OL, StorageOrder OA>
void hyhound_diag_apply_register(const view<T, Abi, OL> L, const view<const T, Abi, OA> Ain,
                                 const view<T, Abi, OA> Aout, const view<const T, Abi, OA> B,
                                 const view<const T, Abi> D, const view<const T, Abi> W,
                                 index_t kA_in_offset) noexcept {
    const index_t k_in = Ain.cols(), k = Aout.cols();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(Aout.rows() == Ain.rows());
    BATMAT_ASSUME(Ain.rows() == L.rows());
    BATMAT_ASSUME(B.rows() == L.cols());
    BATMAT_ASSUME(B.cols() == k);
    BATMAT_ASSUME(D.rows() == k);
    BATMAT_ASSUME(0 <= kA_in_offset);
    BATMAT_ASSUME(kA_in_offset + k_in <= k);

    static constexpr index_constant<SizeR<T, Abi>> R;
    using W_t = triangular_accessor<const T, Abi, R>;
    BATMAT_ASSUME(std::make_pair(W.rows(), W.cols()) == (xshhud_W_size<T, Abi>)(L));

    // Sizeless views to partition and pass to the micro-kernels
    const uview<T, Abi, OL> L_                           = L;
    const uview<const T, Abi, OA> Ain_                   = Ain;
    const uview<T, Abi, OA> Aout_                        = Aout;
    const uview<const T, Abi, OA> B_                     = B;
    const uview<const T, Abi, StorageOrder::ColMajor> D_ = D;
    const uview<const T, Abi, StorageOrder::ColMajor> W_ = W;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L.cols(), R, [&](index_t j, auto nj) {
        static constexpr index_constant<SizeS<T, Abi>> S;
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = B_.middle_rows(j);
        auto Wd = W_t{W_.middle_cols(j / R).data};
        // Process all rows (in multiples of S).
        foreach_chunked_merged( // TODO: swap loop order?
            0, L.rows(), S,
            [&](index_t i, auto ni) {
                auto Aini  = j == 0 ? Ain_.middle_rows(i) : Aout_.middle_rows(i);
                auto Aouti = Aout_.middle_rows(i);
                auto Ls    = L_.block(i, j);
                microkernel_tail_lut_2<T, Abi, Conf, OL, OA, OA>[nj - 1][ni - 1](
                    j == 0 ? kA_in_offset : 0, j == 0 ? k_in : k, k, Wd, Ls, Aini, Aouti, Ad, D_,
                    Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

/// Same as hyhound_diag_register but for two block rows at once.
template <class T, class Abi, StorageOrder OL1, StorageOrder OA1, StorageOrder OL2,
          StorageOrder OA2, KernelConfig Conf>
void hyhound_diag_2_register(const view<T, Abi, OL1> L11, const view<T, Abi, OA1> A1,
                             const view<T, Abi, OL2> L21, const view<T, Abi, OA2> A2,
                             const view<const T, Abi> D) noexcept {
    const index_t k = A1.cols();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(L11.rows() >= L11.cols());
    BATMAT_ASSUME(L11.rows() == A1.rows());
    BATMAT_ASSUME(D.rows() == k);
    BATMAT_ASSUME(A2.cols() == k);
    BATMAT_ASSUME(L21.cols() == L11.cols());

    static constexpr index_constant<SizeR<T, Abi>> R;
    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W[W_t::size()];

    // Sizeless views to partition and pass to the micro-kernels
    const uview<T, Abi, OL1> L11_                        = L11;
    const uview<T, Abi, OA1> A1_                         = A1;
    const uview<T, Abi, OL2> L21_                        = L21;
    const uview<T, Abi, OA2> A2_                         = A2;
    const uview<const T, Abi, StorageOrder::ColMajor> D_ = D;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t j, auto nj) {
        static constexpr index_constant<SizeS<T, Abi>> S;
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1_.middle_rows(j);
        auto Ld = L11_.block(j, j);
        // Process the diagonal block itself
        microkernel_diag_lut<T, Abi, Conf, OL1, OA1>[nj - 1](k, W, Ld, Ad, D_);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            j + nj, L11.rows(), S,
            [&](index_t i, auto ni) {
                auto As = A1_.middle_rows(i);
                auto Ls = L11_.block(i, j);
                microkernel_tail_lut_2<T, Abi, Conf, OL1, OA1, OA1>[nj - 1][ni - 1](
                    0, k, k, W, Ls, As, As, Ad, D_, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L21.rows(), S,
            [&](index_t i, auto ni) {
                auto As = A2_.middle_rows(i);
                auto Ls = L21_.block(i, j);
                microkernel_tail_lut_2<T, Abi, Conf, OL2, OA2, OA1>[nj - 1][ni - 1](
                    0, k, k, W, Ls, As, As, Ad, D_, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

/**
 * Performs a factorization update of the following matrix:
 *
 *     [ A11 A12 | L11 ]     [  0   0  | L̃11 ]
 *     [  0  A22 | L21 ] Q = [ Ã21 Ã22 | L̃21 ]
 *     [ A31  0  | L31 ]     [ Ã31 Ã32 | L̃31 ]
 */
template <class T, class Abi, StorageOrder OL, StorageOrder OW, StorageOrder OY, StorageOrder OU,
          KernelConfig Conf>
void hyhound_diag_cyclic_register(const view<T, Abi, OL> L11,       // D
                                  const view<T, Abi, OW> A1,        // work
                                  const view<T, Abi, OY> L21,       // Y
                                  const view<const T, Abi, OW> A22, // work
                                  const view<T, Abi, OW> A2_out,    // work
                                  const view<T, Abi, OU> L31,       // U
                                  const view<const T, Abi, OW> A31, // work
                                  const view<T, Abi, OW> A3_out,    // work
                                  const view<const T, Abi> D) noexcept {
    const index_t k = A1.cols(), k1 = A31.cols(), k2 = A22.cols();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(L11.rows() >= L11.cols());
    BATMAT_ASSUME(L11.rows() == A1.rows());
    BATMAT_ASSUME(L21.rows() == A22.rows());
    BATMAT_ASSUME(L31.rows() == A31.rows());
    BATMAT_ASSUME(A22.rows() == A2_out.rows());
    BATMAT_ASSUME(A31.rows() == A3_out.rows());
    BATMAT_ASSUME(D.rows() == k);
    BATMAT_ASSUME(L21.cols() == L11.cols());
    BATMAT_ASSUME(L31.cols() == L11.cols());
    BATMAT_ASSUME(k1 + k2 == k);

    static constexpr index_constant<SizeR<T, Abi>> R;
    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W[W_t::size()];

    // Sizeless views to partition and pass to the micro-kernels
    const uview<T, Abi, OL> L11_                         = L11;
    const uview<T, Abi, OW> A1_                          = A1;
    const uview<T, Abi, OY> L21_                         = L21;
    const uview<const T, Abi, OW> A22_                   = A22;
    const uview<T, Abi, OW> A2_out_                      = A2_out;
    const uview<T, Abi, OU> L31_                         = L31;
    const uview<const T, Abi, OW> A31_                   = A31;
    const uview<T, Abi, OW> A3_out_                      = A3_out;
    const uview<const T, Abi, StorageOrder::ColMajor> D_ = D;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t j, auto nj) {
        static constexpr index_constant<SizeS<T, Abi>> S;
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1_.middle_rows(j);
        auto Ld = L11_.block(j, j);
        // Process the diagonal block itself
        microkernel_diag_lut<T, Abi, Conf, OL, OW>[nj - 1](k, W, Ld, Ad, D_);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            j + nj, L11.rows(), S,
            [&](index_t i, auto ni) {
                auto As = A1_.middle_rows(i);
                auto Ls = L11_.block(i, j);
                microkernel_tail_lut_2<T, Abi, Conf, OL, OW, OW>[nj - 1][ni - 1](
                    0, k, k, W, Ls, As, As, Ad, D_, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L21.rows(), S,
            [&](index_t i, auto ni) {
                auto As_out = A2_out_.middle_rows(i);
                auto As     = j == 0 ? A22_.middle_rows(i) : As_out;
                auto Ls     = L21_.block(i, j);
                // First half of A2 is implicitly zero in first pass
                index_t offset_s = j == 0 ? k1 : 0, k_s = j == 0 ? k2 : k;
                microkernel_tail_lut_2<T, Abi, Conf, OY, OW, OW>[nj - 1][ni - 1](
                    offset_s, k_s, k, W, Ls, As, As_out, Ad, D_, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L31.rows(), S,
            [&](index_t i, auto ni) {
                auto As_out = A3_out_.middle_rows(i);
                auto As     = j == 0 ? A31_.middle_rows(i) : As_out;
                auto Ls     = L31_.block(i, j);
                // Second half of A3 is implicitly zero in first pass
                index_t offset_s = 0, k_s = j == 0 ? k1 : k;
                microkernel_tail_lut_2<T, Abi, Conf, OU, OW, OW>[nj - 1][ni - 1](
                    offset_s, k_s, k, W, Ls, As, As_out, Ad, D_, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

/**
 * Performs a factorization update of the following matrix:
 *
 *     [ A1 | L11 ]     [  0 | L̃11 ]
 *     [ A2 | L21 ] Q = [ Ã2 | L̃21 ]
 *     [  0 | Lu1 ]     [ Ãu | L̃u1 ]
 *
 * where Lu1 and L̃u1 are upper triangular
 */
template <class T, class Abi, StorageOrder OL, StorageOrder OA, StorageOrder OLu, StorageOrder OAu,
          KernelConfig Conf>
void hyhound_diag_riccati_register(const view<T, Abi, OL> L11, const view<T, Abi, OA> A1,
                                   const view<T, Abi, OL> L21, const view<const T, Abi, OA> A2,
                                   const view<T, Abi, OA> A2_out, const view<T, Abi, OLu> Lu1,
                                   const view<T, Abi, OAu> Au_out, const view<const T, Abi> D,
                                   bool shift_A_out) noexcept {
    const index_t k = A1.cols();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(L11.rows() >= L11.cols());
    BATMAT_ASSUME(L11.rows() == A1.rows());
    BATMAT_ASSUME(L21.rows() == A2.rows());
    BATMAT_ASSUME(A2_out.rows() == A2.rows());
    BATMAT_ASSUME(A2_out.cols() == A2.cols());
    BATMAT_ASSUME(Lu1.rows() == Au_out.rows());
    BATMAT_ASSUME(A1.cols() == D.rows());
    BATMAT_ASSUME(A2.cols() == A1.cols());
    BATMAT_ASSUME(L21.cols() == L11.cols());
    BATMAT_ASSUME(Lu1.cols() == L11.cols());

    static constexpr index_constant<SizeR<T, Abi>> R;
    static constexpr index_constant<SizeS<T, Abi>> S;
    static_assert(R == S);
    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W[W_t::size()];

    // Sizeless views to partition and pass to the micro-kernels
    const uview<T, Abi, OL> L11_                         = L11;
    const uview<T, Abi, OA> A1_                          = A1;
    const uview<T, Abi, OL> L21_                         = L21;
    const uview<const T, Abi, OA> A2_                    = A2;
    const uview<T, Abi, OA> A2_out_                      = A2_out;
    const uview<T, Abi, OLu> Lu1_                        = Lu1;
    const uview<T, Abi, OAu> Au_out_                     = Au_out;
    const uview<const T, Abi, StorageOrder::ColMajor> D_ = D;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t j, auto nj) {
        const bool do_shift = shift_A_out && j + nj == L11.cols();
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1_.middle_rows(j);
        auto Ld = L11_.block(j, j);
        // Process the diagonal block itself
        microkernel_diag_lut<T, Abi, Conf, OL, OA>[nj - 1](k, W, Ld, Ad, D_);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            j + nj, L11.rows(), S,
            [&](index_t i, auto ni) {
                auto As = A1_.middle_rows(i);
                auto Ls = L11_.block(i, j);
                microkernel_tail_lut_2<T, Abi, Conf, OL, OA, OA>[nj - 1][ni - 1](
                    0, k, k, W, Ls, As, As, Ad, D_, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L21.rows(), S,
            [&](index_t i, auto ni) {
                auto As_out = A2_out_.middle_rows(i);
                auto As     = j == 0 ? A2_.middle_rows(i) : As_out;
                auto Ls     = L21_.block(i, j);
                microkernel_tail_lut_2<T, Abi, Conf, OL, OA, OA>[nj - 1][ni - 1](
                    0, k, k, W, Ls, As, As_out, Ad, D_, Structure::General, do_shift ? -1 : 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, Lu1.rows(), S,
            [&](index_t i, auto ni) {
                auto As_out = Au_out_.middle_rows(i);
                auto As     = As_out;
                auto Ls     = Lu1_.block(i, j);
                // Au is implicitly zero in first pass
                const auto struc = i == j  ? Structure::Upper
                                   : i < j ? Structure::General
                                           : Structure::Zero;
                microkernel_tail_lut_2<T, Abi, Conf, OLu, OAu, OA>[nj - 1][ni - 1](
                    0, j == 0 ? 0 : k, k, W, Ls, As, As_out, Ad, D_, struc, do_shift ? -1 : 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

} // namespace batmat::linalg::micro_kernels::hyhound
