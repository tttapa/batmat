#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/small-potrf.hpp>
#include <batmat/loop.hpp>
#include <batmat/lut.hpp>
#include <batmat/ops/rsqrt.hpp>
#include <batmat/ops/sqrt.hpp>
#include <batmat/simd.hpp>
#include <bit>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::small_potrf {

template <class T, index_t NC> // number of columns to handle at once
[[gnu::flatten, gnu::hot]]
void potrf_trsm_microkernel(index_t k, scalar_view<const T> A, scalar_view<T> L) noexcept {
    using ops::rsqrt;
    using ops::sqrt;
    constexpr index_t NR = 8; // Number of rows in each sub-diagonal block
    T Dr[NC * (NC + 1) / 2];
    static constexpr auto index = [](index_t r, index_t c) { return c * (2 * NC - 1 - c) / 2 + r; };
    /* Load diagonal block into (scalar) registers */
    UNROLL_FOR (index_t j = 0; j < NC; ++j)     // column
        UNROLL_FOR (index_t i = j; i < NC; ++i) // row
            Dr[index(i, j)] = A(i, j);
    /* Cholesky factorization of diagonal block */
    UNROLL_FOR (index_t j = 0; j < NC; ++j) { // column
        const auto pivot     = sqrt(Dr[index(j, j)]);
        const auto inv_pivot = rsqrt(Dr[index(j, j)]);
        Dr[index(j, j)]      = inv_pivot;
        UNROLL_FOR (index_t i = j + 1; i < NC; ++i)
            Dr[index(i, j)] *= inv_pivot;
        UNROLL_FOR (index_t kk = j + 1; kk < NC; ++kk) { // column syrk
            const T fac = Dr[index(kk, j)];
            UNROLL_FOR (index_t i = kk; i < NC; ++i)
                Dr[index(i, kk)] -= Dr[index(i, j)] * fac;
        }
        L(j, j) = pivot;
        UNROLL_FOR (index_t i = j + 1; i < NC; ++i) // row
            L(i, j) = Dr[index(i, j)];
    }
    /* Multiply the sub-diagonal blocks by the inverse of the Cholesky factor */
    auto trsm_tail = [&](auto &trsm_tail, index_t r, auto N) {
        using simdN = datapar::deduced_simd<T, N>;
        for (; r + N <= k; r += N) { // block row
            simdN Xrx[NC];
            UNROLL_FOR (index_t c = 0; c < NC; ++c) // column
                Xrx[c] = datapar::unaligned_load<simdN>(&A(r, c));
            UNROLL_FOR (index_t c = 0; c < NC; ++c) { // column
                simdN &Xij = Xrx[c];
                UNROLL_FOR (index_t kk = 0; kk < c; ++kk) { // column inner
                    const T Aik = Dr[index(c, kk)];
                    Xij -= Aik * Xrx[kk];
                }
                Xij *= Dr[index(c, c)];
                datapar::unaligned_store(Xij, &L(r, c));
            }
        }
        if constexpr (N > 1)
            trsm_tail(trsm_tail, r, std::integral_constant<index_t, N / 2>());
    };
    trsm_tail(trsm_tail, NC, std::integral_constant<index_t, NR>());
}

/// Outer product for updating the bottom right tail during Cholesky factorization.
/// @param A21 rows×ColsReg
/// @param A22 rows×RowsReg
template <class T, index_t RowsReg, index_t ColsReg>
[[gnu::flatten, gnu::hot]]
void potrf_syrk_microkernel(index_t k, scalar_view<const T> L21, scalar_view<const T> A22,
                            scalar_view<T> L22) noexcept {
    constexpr index_t NR = 8; // Number of rows in each sub-diagonal block
    // Pre-compute the offsets of the columns of A21 and A22
    auto L21_cached = with_cached_access<0, ColsReg>(L21);
    auto A22_cached = with_cached_access<0, RowsReg>(A22);
    auto L22_cached = with_cached_access<0, RowsReg>(L22);
    // Load matrix into registers
    T A21_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    UNROLL_FOR (index_t i = 0; i < RowsReg; ++i)
        UNROLL_FOR (index_t j = 0; j < ColsReg; ++j)
            A21_reg[i][j] = L21_cached(i, j);
    // Matrix multiplication of diagonal block
    UNROLL_FOR (index_t i = 0; i < RowsReg; ++i) {
        T A22ix[RowsReg];
        UNROLL_FOR (index_t j = 0; j <= i; ++j)
            A22ix[j] = A22_cached(i, j);
        UNROLL_FOR (index_t j = 0; j <= i; ++j)
            UNROLL_FOR (index_t kk = 0; kk < ColsReg; ++kk)
                A22ix[j] -= A21_reg[i][kk] * A21_reg[j][kk];
        UNROLL_FOR (index_t j = 0; j <= i; ++j)
            L22_cached(i, j) = A22ix[j];
    }
    // Matrix multiplication of sub-diagonal block
    auto gemm_tail = [&](auto &gemm_tail, index_t i, auto N) {
        using simd = datapar::deduced_simd<T, N>;
        for (; i + N <= k; i += N) { // block row
            simd Aix[RowsReg];
            UNROLL_FOR (index_t j = 0; j < RowsReg; ++j)
                Aix[j] = datapar::unaligned_load<simd>(&A22_cached(i, j));
            UNROLL_FOR (index_t j = 0; j < RowsReg; ++j)
                UNROLL_FOR (index_t kk = 0; kk < ColsReg; ++kk) {
                    const simd A21ik = datapar::unaligned_load<simd>(&L21_cached(i, kk));
                    Aix[j] -= A21ik * A21_reg[j][kk];
                }
            UNROLL_FOR (index_t j = 0; j < RowsReg; ++j)
                datapar::unaligned_store(Aix[j], &L22_cached(i, j));
        }
        if constexpr (N > 1)
            gemm_tail(gemm_tail, i, std::integral_constant<index_t, N / 2>());
    };
    gemm_tail(gemm_tail, RowsReg, std::integral_constant<index_t, NR>());
}

template <class T, index_t R>
void small_potrf(view<const T, datapar::scalar_abi<T>> A, view<T, datapar::scalar_abi<T>> L,
                 index_t n) noexcept {
    static const constinit auto microkernel_trsm_lut = make_1d_lut<R>(
        []<index_t Row>(index_constant<Row>) { return potrf_trsm_microkernel<T, Row + 1>; });
    static const constinit auto microkernel_syrk_lut = make_1d_lut<R>(
        []<index_t Row>(index_constant<Row>) { return potrf_syrk_microkernel<T, Row + 1, R>; });
    static const constinit auto microkernel_syrk_lut_2 =
        make_2d_lut<R, R>([]<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
            return potrf_syrk_microkernel<T, Row + 1, Col + 1>;
        });
    (void)microkernel_syrk_lut; // Invalid GCC warning
    (void)microkernel_syrk_lut_2;

    const index_t m = L.rows(), N = L.cols();
    if (n < 0)
        n = N;
    BATMAT_ASSUME(m >= N);
    BATMAT_ASSUME((n == m && m == N) || (n == N && m >= N) || (n < m && m == N));

    scalar_view<const T> A_ = A;
    scalar_view<T> L_       = L;

    // Compute the Cholesky factorization of the very last block (right before
    // the Schur complement block), which has size r×r rather than R×R.
    // If requested, also update the rows below the Cholesky factor, and the
    // Schur complement to the bottom right of the given block.
    // These extra blocks are always sizes (m-n)×r and (m-n)×(m-n) respectively.
    const auto process_bottom_right = [m, N, n](scalar_view<const T> Aii, scalar_view<T> Lii,
                                                index_t r) {
        // Cholesky of last block to be factorized + triangular solve with
        // sub-diagonal block.
        microkernel_trsm_lut[r - 1](r + m - n, Aii, Lii);
        // Update the Schur complement (bottom right) with the outer product
        // of the sub-diagonal block column.
        if (n < N) {
            auto L21 = Lii.middle_rows(r), L22 = Lii.block(r, r);
            auto A22 = Aii.block(r, r);
            foreach_chunked_merged(
                0, m - n, index_constant<R>(),
                [&](index_t j, auto rem) {
                    auto Lj1 = L21.middle_rows(j), Ljj = L22.middle_cols(j);
                    auto Ajj = A22.middle_cols(j);
                    microkernel_syrk_lut_2[rem - 1][r - 1](m - n - j, Lj1, Ajj, Ljj);
                },
                LoopDir::Forward);
        }
    };

    // Base case
    if (n == 0) {
        return;
    } else if (n <= R) {
        process_bottom_right(A_, L_, n);
        return;
    }
    // Loop over columns of H with block size R.
    index_t i;
    for (i = 0; i + R <= n; i += R) {
        auto L11 = L_.block(i, i);
        auto A11 = i == 0 ? A_.block(i, i) : decltype(A_){L11};
        // Factor the diagonal block and update the subdiagonal block
        potrf_trsm_microkernel<T, R>(m - i, A11, L11);
        // Update the Schur complement (bottom right) with the outer product of
        // the subdiagonal block.
        foreach_chunked_merged(
            i + R, N, index_constant<R>(),
            [&](index_t j, auto rem) {
                auto L21 = L_.block(j, i), L22 = L_.block(j, j);
                auto A22 = i == 0 ? A_.block(j, j) : decltype(A_){L22};
                microkernel_syrk_lut[rem - 1](m - j, L21, A22, L22);
            },
            LoopDir::Backward);
        // Loop backwards for cache locality (we'll use the next column in the
        // next interation, so we want the syrk operation to leave it in cache).
        // TODO: verify in benchmark.
    }
    const index_t rem = n - i;
    if (rem > 0) {
        auto Lii = L_.block(i, i);
        auto Aii = i == 0 ? A_.block(i, i) : decltype(A_){Lii};
        process_bottom_right(Aii, Lii, rem);
    }
}

/// Left-looking variant of small_potrf, which updates the current block with the outer product of
/// the previously computed part L21.
/// @param L21 m×k
/// @param A22 m×NC
/// @param L22 m×NC
template <class T, index_t NC, index_t NR>
[[gnu::flatten, gnu::hot]]
void syrk_potrf_trsm_microkernel(index_t m, index_t k, scalar_view<const T> L21,
                                 scalar_view<const T> A22, scalar_view<T> L22) noexcept {
    using ops::sqrt;

    using simd           = datapar::deduced_simd<T, std::bit_ceil(static_cast<unsigned>(NC))>;
    const auto load_mask = datapar::generate_mask_until<simd, NC>();

    /* Load diagonal block into registers */
    simd Dr[NC];
    UNROLL_FOR (index_t j = 0; j < NC; ++j) // column
        Dr[j] = NC == simd::size() ? datapar::unaligned_load<simd>(&A22(0, j))
                                   : datapar::partial_load<simd, NC>(&A22(0, j));
    /* Accumulate previous updates */
    for (index_t l = 0; l < k; ++l) { // syrk update diagonal block
        simd L21l = NC == simd::size() ? datapar::unaligned_load<simd>(&L21(0, l))
                                       : datapar::partial_load<simd, NC>(&L21(0, l));
        UNROLL_FOR (index_t j = 0; j < NC; ++j)
            Dr[j] -= L21l * L21l[j];
    }

    /* Cholesky factorization of diagonal block */
    T inv_pivots[NC];
    auto store_mask = load_mask;
    UNROLL_FOR (index_t j = 0; j < NC; ++j) { // column
        const T Djj = Dr[j][j];
        BATMAT_ASSUME(Djj > T{});
        const T pivot     = sqrt(Djj);
        const T inv_pivot = 1 / pivot;
        inv_pivots[j]     = inv_pivot;
        Dr[j] *= inv_pivot;                         // update current column
        UNROLL_FOR (index_t i = j + 1; i < NC; ++i) // column syrk
            Dr[i] -= Dr[j] * Dr[j][i];
#if BATMAT_WITH_GSI_HPC_SIMD
        const auto mask_j = datapar::generate_mask<simd>(j);
        Dr[j]             = datapar::select(mask_j, simd{pivot}, Dr[j]);
        datapar::masked_unaligned_store(Dr[j], store_mask, &L22(0, j));
        store_mask = store_mask && !mask_j;
#else
        Dr[j][j] = pivot;
        datapar::masked_unaligned_store(Dr[j], store_mask, &L22(0, j));
        store_mask[j] = false;
#endif
    }

    /* Multiply the sub-diagonal blocks by the inverse of the Cholesky factor */
    auto trsm_tail = [&](auto &trsm_tail, index_t r, auto N) {
        using simdN = datapar::deduced_simd<T, N>;
        for (; r + N <= m; r += N) { // block row
            simdN Xrx[NC];
            UNROLL_FOR (index_t c = 0; c < NC; ++c) // column
                Xrx[c] = datapar::unaligned_load<simdN>(&A22(r, c));
            for (index_t l = 0; l < k; ++l) { // syrk update subdiagonal block
                simdN L21rl = datapar::unaligned_load<simdN>(&L21(r, l));
                UNROLL_FOR (index_t j = 0; j < NC; ++j)
                    Xrx[j] -= L21rl * L21(j, l);
            }
            UNROLL_FOR (index_t j = 0; j < NC; ++j) { // column
                simdN &Xij = Xrx[j];
                UNROLL_FOR (index_t i = 0; i < j; ++i) // column inner
                    Xij -= Dr[i][j] * Xrx[i];
                Xij *= inv_pivots[j];
                datapar::unaligned_store(Xij, &L22(r, j));
            }
        }
        if constexpr (N > 1)
            trsm_tail(trsm_tail, r, std::integral_constant<index_t, N / 2>());
    };
    trsm_tail(trsm_tail, NC, std::integral_constant<index_t, NR>());
}

template <class T, index_t R, index_t S>
void small_potrf_left(view<const T, datapar::scalar_abi<T>> A,
                      view<T, datapar::scalar_abi<T>> L) noexcept {
    static const constinit auto microkernel_lut =
        make_1d_lut<R>([]<index_t Row>(index_constant<Row>) {
            return syrk_potrf_trsm_microkernel<T, Row + 1, S>;
        });
    (void)microkernel_lut; // Invalid GCC warning

    const index_t m = L.rows(), N = L.cols();
    BATMAT_ASSUME(m >= N);

    scalar_view<const T> A_ = A;
    scalar_view<T> L_       = L;

    // Loop over columns of H with block size R.
    foreach_chunked(
        0, N, index_constant<R>(),
        [&](index_t i) {
            auto L22 = L_.block(i, i);
            auto A22 = A_.block(i, i);
            auto L21 = L_.block(i, 0);
            syrk_potrf_trsm_microkernel<T, R, S>(m - i, i, L21, A22, L22);
        },
        [&](index_t i, auto rem) {
            auto L22 = L_.block(i, i);
            auto A22 = A_.block(i, i);
            auto L21 = L_.block(i, 0);
            microkernel_lut[rem - 1](m - i, i, L21, A22, L22);
        });
}

} // namespace batmat::linalg::micro_kernels::small_potrf
