#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/geqrf.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/lut.hpp>
#include <batmat/ops/cneg.hpp>
#include <batmat/ops/rotate.hpp>
#include <guanaqo/trace.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::geqrf {

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD>
inline const constinit auto microkernel_diag_lut =
    make_1d_lut<SizeR<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return geqrf_diag_microkernel<T, Abi, Conf, Row + 1, OA, OD>;
    });

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD>
inline const constinit auto microkernel_full_lut =
    make_1d_lut<SizeR<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return geqrf_full_microkernel<T, Abi, Conf, Row + 1, OA, OD>;
    });

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD, StorageOrder OB>
inline const constinit auto microkernel_tail_lut =
    make_1d_lut<SizeS<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return geqrf_tail_microkernel<T, Abi, Conf, SizeR<T, Abi>, Row + 1, OA, OD, OB>;
    });

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD, StorageOrder OB>
inline const constinit auto microkernel_tail_lut_2 = make_2d_lut<SizeR<T, Abi>, SizeS<T, Abi>>(
    []<index_t NR, index_t NS>(index_constant<NR>, index_constant<NS>) {
        return geqrf_tail_microkernel<T, Abi, Conf, NR + 1, NS + 1, OA, OD, OB>;
    });

template <class T, class Abi, KernelConfig Conf, index_t R, StorageOrder OA, StorageOrder OD>
[[gnu::hot, gnu::flatten]] void
geqrf_diag_microkernel(index_t k, triangular_accessor<T, Abi, SizeR<T, Abi>> W,
                       uview<const T, Abi, OA> A, uview<T, Abi, OD> D) noexcept {
    using std::copysign;
    using std::sqrt;
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(k > 0); // TODO: fast path for k == 1

    UNROLL_FOR (index_t j = 0; j < R; ++j) {
        const bool use_A = j == 0;
        // Compute all inner products between A and a
        simd bb[R]{};
        for (index_t l = j + 1; l < k; ++l) {
            simd Alj = use_A ? A.load(l, j) : D.load(l, j);
            UNROLL_FOR (index_t i = 0; i < R; ++i)
                bb[i] += (use_A ? A.load(l, i) : D.load(l, i)) * Alj;
        }
        simd aa[R];
        UNROLL_FOR (index_t i = 0; i < R; ++i)
            aa[i] = use_A ? A.load(j, i) : D.load(j, i);
        bb[j] += aa[j] * aa[j];
        // Energy condition and Householder coefficients
        const simd ãjj = copysign(sqrt(bb[j]), aa[j]), β = aa[j] + ãjj;
        simd inv_τ = β / ãjj, inv_β = simd{1} / β;
        D.store(ãjj, j, j);
        // Save block Householder matrix W
        UNROLL_FOR (index_t i = 0; i < j; ++i)
            bb[i] = bb[i] * inv_β + aa[i];
        bb[j] = inv_τ; // inverse of diagonal
        UNROLL_FOR (index_t i = 0; i < j + 1; ++i)
            W.store(bb[i], i, j);
        // Replace row j of A by R (and replace bb[j+1:] with w)
        UNROLL_FOR (index_t i = j + 1; i < R; ++i) {
            bb[i] = (aa[i] + bb[i] * inv_β) * inv_τ; // w
            D.store(bb[i] - aa[i], j, i);            // R[j, i]
        }
        // Update trailing part of A
        for (index_t l = j + 1; l < k; ++l) {
            simd Alj = use_A ? A.load(l, j) : D.load(l, j);
            Alj *= inv_β;       // Scale Householder vector
            D.store(Alj, l, j); // V[l, j]
            UNROLL_FOR (index_t i = j + 1; i < R; ++i) {
                simd Ali = use_A ? A.load(l, i) : D.load(l, i);
                Ali -= Alj * bb[i];
                D.store(Ali, l, i);
            }
        }
    }
}

/// A (k×R)
/// D (k×R)
template <class T, class Abi, KernelConfig Conf, index_t R, StorageOrder OA, StorageOrder OD>
[[gnu::hot, gnu::flatten]] void geqrf_full_microkernel(index_t k, uview<const T, Abi, OA> A,
                                                       uview<T, Abi, OD> D) noexcept {
    using std::copysign;
    using std::sqrt;
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(k > 0); // TODO: fast path for k == 1

    UNROLL_FOR (index_t j = 0; j < R; ++j) {
        const bool use_A = j == 0;
        // Compute all inner products between A and a
        simd bb[R];
        UNROLL_FOR (index_t i = j; i < R; ++i)
            bb[i] = simd{0};
        for (index_t l = j + 1; l < k; ++l) {
            simd Alj = use_A ? A.load(l, j) : D.load(l, j);
            UNROLL_FOR (index_t i = j; i < R; ++i)
                bb[i] += (use_A ? A.load(l, i) : D.load(l, i)) * Alj;
        }
        simd aa[R];
        UNROLL_FOR (index_t i = j; i < R; ++i)
            aa[i] = use_A ? A.load(j, i) : D.load(j, i);
        bb[j] += aa[j] * aa[j];
        // Energy condition and Householder coefficients
        const simd ãjj = copysign(sqrt(bb[j]), aa[j]), β = aa[j] + ãjj;
        simd inv_τ = β / ãjj, inv_β = simd{1} / β;
        D.store(ãjj, j, j);
        // Replace row j of A by R (and replace bb[j+1:] with w)
        UNROLL_FOR (index_t i = j + 1; i < R; ++i) {
            bb[i] = (aa[i] + bb[i] * inv_β) * inv_τ; // w
            D.store(bb[i] - aa[i], j, i);            // R[j, i]
        }
        // Update trailing part of A
        for (index_t l = j + 1; l < k; ++l) {
            simd Alj = use_A ? A.load(l, j) : D.load(l, j);
            Alj *= inv_β;       // Scale Householder vector
            D.store(Alj, l, j); // V[l, j]
            UNROLL_FOR (index_t i = j + 1; i < R; ++i) {
                simd Ali = use_A ? A.load(l, i) : D.load(l, i);
                Ali -= Alj * bb[i];
                D.store(Ali, l, i);
            }
        }
    }
}

// Householder vectors are stored in the strict lower triangle of B, with the upper triangle
// implicitly equal to the identity.
// The matrix W completes the block Householder representation Q = BW⁻¹Bᵀ - I. The diagonal of W
// is already inverted to enable efficient application of W⁻¹.
// B = [ 1   0   0   ... ]
//     [ b11 1   0   ... ]
//     [ b21 b22 1   ... ]
//     [ b31 b32 b33 ... ]
// B: k×R (lower trapezoidal, with implicit unit diagonal)
// W: R×R (upper triangular, with inverted diagonal)
// A: k×S
// D: k×S
template <class T, class Abi, KernelConfig Conf, index_t R, index_t S, StorageOrder OA,
          StorageOrder OD, StorageOrder OB>
[[gnu::hot, gnu::flatten]] void geqrf_tail_microkernel(
    index_t k, bool transposed, triangular_accessor<const T, Abi, SizeR<T, Abi>> W,
    uview<const T, Abi, OA> A, uview<T, Abi, OD> D, uview<const T, Abi, OB> B) noexcept {
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(k > 0);

    // Compute product U = BᵀA
    simd V[R][S];
    // Triangular part of B (top R rows)
    UNROLL_FOR (index_t l = 0; l < R; ++l)
        UNROLL_FOR (index_t i = 0; i < S; ++i) {
            V[l][i] = A.load(l, i);                // B[l, l] = 1
            UNROLL_FOR (index_t j = 0; j < l; ++j) // B[l, >l] = 0
                V[j][i] += B.load(l, j) * A.load(l, i);
        }
    // Remaining rectangular part of B
    for (index_t l = R; l < k; ++l)
        UNROLL_FOR (index_t j = 0; j < R; ++j) {
            auto Blj = B.load(l, j);
            UNROLL_FOR (index_t i = 0; i < S; ++i)
                V[j][i] += Blj * A.load(l, i);
        }

    // Solve system V = W⁻¹ U (with W upper triangular, in-place)
    if (!transposed)
        UNROLL_FOR (index_t j = R; j-- > 0;)               // row of W
            UNROLL_FOR (index_t i = 0; i < S; ++i) {       // column of V, U
                UNROLL_FOR (index_t l = j + 1; l < R; ++l) // column of W
                    V[j][i] -= W.load(j, l) * V[l][i];
                V[j][i] *= W.load(j, j); // diagonal already inverted
            }
    // Solve system V = W⁻ᵀ U (with W upper triangular, in-place)
    else
        UNROLL_FOR (index_t j = 0; j < R; ++j)         // row of Wᵀ
            UNROLL_FOR (index_t i = 0; i < S; ++i) {   // column of V, U
                UNROLL_FOR (index_t l = 0; l < j; ++l) // column of Wᵀ
                    V[j][i] -= W.load(l, j) * V[l][i];
                V[j][i] *= W.load(j, j); // diagonal already inverted
            }

    // Update A = B V - A
    simd Bl[R];
    // Top R rows of B
    UNROLL_FOR (index_t l = 0; l < R; ++l) {
        UNROLL_FOR (index_t j = 0; j < l; ++j)
            Bl[j] = B.load(l, j);
        UNROLL_FOR (index_t i = 0; i < S; ++i) {
            simd Dli = V[l][i] - A.load(l, i);
            UNROLL_FOR (index_t j = 0; j < l; ++j)
                Dli += V[j][i] * Bl[j];
            D.store(Dli, l, i);
        }
    }
    // Remaining rectangular part of B
    for (index_t l = R; l < k; ++l) {
        UNROLL_FOR (index_t j = 0; j < R; ++j)
            Bl[j] = B.load(l, j);
        UNROLL_FOR (index_t i = 0; i < S; ++i) {
            simd Dli = -A.load(l, i);
            UNROLL_FOR (index_t j = 0; j < R; ++j)
                Dli += V[j][i] * Bl[j];
            D.store(Dli, l, i);
        }
    }
}

/// Block hyperbolic Householder factorization update using register blocking.
/// This variant does not store the Householder representation W.
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD>
void geqrf_copy_register(const view<const T, Abi, OA> A, const view<T, Abi, OD> D,
                         const view<T, Abi> W) noexcept {
    static constexpr index_constant<SizeR<T, Abi>> R;
    static constexpr index_constant<SizeS<T, Abi>> S;
    const index_t k = A.rows();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(A.rows() >= A.cols());
    BATMAT_ASSUME(A.rows() == D.rows());
    BATMAT_ASSUME(A.cols() == D.cols());
    BATMAT_ASSUME(W.rows() == 0 || (W.cols() == 1 && W.rows() == A.cols()) ||
                  std::make_pair(W.rows(), W.cols()) == (geqrf_W_size<const T, Abi>)(A));

    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W_sto[W_t::size()];

    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_               = A;
    const uview<T, Abi, OD> D_                     = D;
    const uview<T, Abi, StorageOrder::ColMajor> W_ = W;
    const bool store_full_W = std::make_pair(W.rows(), W.cols()) == (geqrf_W_size<const T, Abi>)(A);

    // Process all diagonal blocks (in multiples of R, except the last).
    if (A.rows() == A.cols() && W.rows() == 0) {
        auto Wj = W_t{W_sto};
        foreach_chunked(
            0, A.cols(), R,
            [&](index_t j) {
                auto Djj = D_.block(j, j);
                // Copy result from A to D
                if (j == 0) {
                    // Triangularize block column j (all rows below diagonal)
                    geqrf_diag_microkernel<T, Abi, Conf, R, OA, OD>(k, Wj, A_, Djj);
                    // Update the trailing columns (in multiples of S)
                    foreach_chunked_merged(
                        j + R, A.cols(), S,
                        [&](index_t i, auto rem_i) {
                            auto Dji = D_.block(j, i);
                            microkernel_tail_lut<T, Abi, Conf, OA, OD, OD>[rem_i - 1](
                                k, true, Wj, A_.block(j, i), Dji, Djj);
                        },
                        LoopDir::Backward); // TODO: decide on order
                } else {
                    // Triangularize block column j (all rows below diagonal)
                    geqrf_diag_microkernel<T, Abi, Conf, R, OD, OD>(k - j, Wj, Djj, Djj);
                    // Update the trailing columns (in multiples of S)
                    foreach_chunked_merged(
                        j + R, A.cols(), S,
                        [&](index_t i, auto rem_i) {
                            auto Dji = D_.block(j, i);
                            microkernel_tail_lut<T, Abi, Conf, OD, OD, OD>[rem_i - 1](
                                k - j, true, Wj, Dji, Dji, Djj);
                        },
                        LoopDir::Backward); // TODO: decide on order
                }
            },
            [&](index_t j, index_t rem_j) {
                auto Djj = D_.block(j, j);
                if (j == 0) // copy result from A to D
                    microkernel_full_lut<T, Abi, Conf, OA, OD>[rem_j - 1](k, A_, Djj);
                else
                    microkernel_full_lut<T, Abi, Conf, OD, OD>[rem_j - 1](k - j, Djj, Djj);
            });
    } else {
        foreach_chunked_merged(0, A.cols(), R, [&](index_t j, auto rem_j) {
            auto Wj  = store_full_W ? W_t{W_.middle_cols(j / R).data} : W_t{W_sto};
            auto Djj = D_.block(j, j);
            // Copy result from A to D
            if (j == 0) {
                // Triangularize block column j (all rows below diagonal)
                microkernel_diag_lut<T, Abi, Conf, OA, OD>[rem_j - 1](k, Wj, A_, Djj);
                // Update the trailing columns (in multiples of S)
                foreach_chunked_merged(
                    j + R, A.cols(), S,
                    [&](index_t i, auto rem_i) {
                        auto Dji = D_.block(j, i);
                        microkernel_tail_lut_2<T, Abi, Conf, OA, OD, OD>[rem_j - 1][rem_i - 1](
                            k, true, Wj, A_.block(j, i), Dji, Djj);
                    },
                    LoopDir::Backward); // TODO: decide on order
            } else {
                // Triangularize block column j (all rows below diagonal)
                microkernel_diag_lut<T, Abi, Conf, OD, OD>[rem_j - 1](k - j, Wj, Djj, Djj);
                // Update the trailing columns (in multiples of S)
                foreach_chunked_merged(
                    j + R, A.cols(), S,
                    [&](index_t i, auto rem_i) {
                        auto Dji = D_.block(j, i);
                        microkernel_tail_lut_2<T, Abi, Conf, OD, OD, OD>[rem_j - 1][rem_i - 1](
                            k - j, true, Wj, Dji, Dji, Djj);
                    },
                    LoopDir::Backward); // TODO: decide on order
            }
            if (!store_full_W && W.rows() > 0) [[unlikely]]
                for (index_t l = 0; l < rem_j; ++l)
                    W_.store(Wj.load(l, l), j + l, 0);
        });
    }
}

/// Apply a block Householder transformation.
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD, StorageOrder OB>
void geqrf_apply_register(const view<const T, Abi, OA> A, const view<T, Abi, OD> D,
                          const view<const T, Abi, OB> B, const view<const T, Abi> W,
                          bool transposed) noexcept {
    const index_t k = A.rows();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(A.rows() == D.rows());
    BATMAT_ASSUME(A.cols() == D.cols());
    BATMAT_ASSUME(B.rows() == A.rows());

    static constexpr index_constant<SizeR<T, Abi>> R;
    using W_t = triangular_accessor<const T, Abi, R>;
    BATMAT_ASSUME(std::make_pair(W.rows(), W.cols()) == (geqrf_W_size<const T, Abi>)(B));

    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_                     = A;
    const uview<T, Abi, OD> D_                           = D;
    const uview<const T, Abi, OB> B_                     = B;
    const uview<const T, Abi, StorageOrder::ColMajor> W_ = W;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(
        0, B.cols(), R,
        [&](index_t j, auto nj) {
            const bool first = transposed ? j == 0 : j + nj >= B.cols();
            static constexpr index_constant<SizeS<T, Abi>> S;
            // Part of A corresponding to this diagonal block
            auto Bjj = B_.block(j, j);
            auto Wj  = W_t{W_.middle_cols(j / R).data};
            // Process all rows (in multiples of S).
            foreach_chunked_merged( // TODO: swap loop order?
                0, A.cols(), S,
                [&](index_t i, auto ni) {
                    auto Dji = D_.block(j, i);
                    if (first)
                        microkernel_tail_lut_2<T, Abi, Conf, OA, OD, OB>[nj - 1][ni - 1](
                            k - j, transposed, Wj, A_.block(j, i), Dji, Bjj);
                    else
                        microkernel_tail_lut_2<T, Abi, Conf, OD, OD, OB>[nj - 1][ni - 1](
                            k - j, transposed, Wj, Dji, Dji, Bjj);
                    // TODO: is it better to merge this copy into the next micro-kernel call?
                    if (first && !transposed && D_.data != A_.data)
                        for (index_t l = 0; l < j; ++l)
                            for (index_t ii = i; ii < i + ni; ++ii)
                                D_.store(A_.load(l, ii), l, ii);
                },
                LoopDir::Backward); // TODO: decide on order
        },
        transposed ? LoopDir::Forward : LoopDir::Backward);
}

} // namespace batmat::linalg::micro_kernels::geqrf
