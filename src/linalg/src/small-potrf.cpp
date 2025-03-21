#include <koqkatoo/config.hpp>

#include <koqkatoo/linalg/small-potrf.hpp>

#include <koqkatoo/loop.hpp>
#include <koqkatoo/lut.hpp>
#include <guanaqo/trace.hpp>
#include <koqkatoo/unroll.h>

#include <experimental/simd>
#include <cassert>

namespace koqkatoo::linalg {

namespace {

namespace stdx = std::experimental;

#define UNROLL_FOR(...) KOQKATOO_FULLY_UNROLLED_FOR (__VA_ARGS__)

#if 0
template <index_t Size, bool Transpose, class T>
struct cached_mat_access_impl {
    using value_type = T;
    value_type *const data[Size];

    [[gnu::always_inline]] value_type &operator()(index_t r,
                                                  index_t c) const noexcept {
        ptrdiff_t i0 = Transpose ? c : r;
        index_t i1   = Transpose ? r : c;
        assert(i1 < Size);
        return data[i1][i0];
    }

    template <index_t... Is>
    [[gnu::always_inline]] cached_mat_access_impl(
        T *data, index_t outer_stride,
        std::integer_sequence<index_t, Is...>) noexcept
        : data{(data + Is * static_cast<ptrdiff_t>(outer_stride))...} {}
    [[gnu::always_inline]] cached_mat_access_impl(T *data,
                                                  index_t outer_stride) noexcept
        : cached_mat_access_impl{data, outer_stride,
                                 std::make_integer_sequence<index_t, Size>()} {}
};
#else
template <index_t Size, bool Transpose, class T>
struct cached_mat_access_impl {
    using value_type = T;
    value_type *const data;
    index_t outer_stride;

    [[gnu::always_inline]] value_type &operator()(index_t r,
                                                  index_t c) const noexcept {
        ptrdiff_t i0 = Transpose ? c : r;
        index_t i1   = Transpose ? r : c;
        assert(i1 < Size);
        return data[i1 * outer_stride + i0];
    }

    [[gnu::always_inline]] cached_mat_access_impl(T *data,
                                                  index_t outer_stride) noexcept
        : data{data}, outer_stride{outer_stride} {}
};
#endif

template <index_t Size, bool Transpose = false, class T>
[[gnu::always_inline]] inline cached_mat_access_impl<Size, Transpose, T>
with_cached_access(T *data, index_t outer_stride) noexcept {
    return {data, outer_stride};
}

template <index_t NC> // number of columns to handle at once
void potrf_trsm_kernel(real_t *L, index_t ldL, index_t rows) {
    using std::sqrt;
    using stdx::simd_abi::deduce_t;
    constexpr index_t NR  = 8; // Number of rows in each sub-diagonal block
    constexpr auto nalign = stdx::element_aligned;
    real_t Dr[NC * (NC + 1) / 2];
    auto index = [](index_t r, index_t c) {
        return c * (2 * NC - 1 - c) / 2 + r;
    };
    /* Cache the column pointers of L */
    auto L_cached = with_cached_access<NC>(L, ldL);
    /* Load diagonal block into (scalar) registers */
    UNROLL_FOR (index_t j = 0; j < NC; ++j)     // column
        UNROLL_FOR (index_t i = j; i < NC; ++i) // row
            Dr[index(i, j)] = L_cached(i, j);
    /* Cholesky factorization of diagonal block */
    UNROLL_FOR (index_t j = 0; j < NC; ++j) { // column
        const auto pivot     = sqrt(Dr[index(j, j)]);
        const auto inv_pivot = 1 / pivot; // TODO: fast inverse square root?
        Dr[index(j, j)]      = inv_pivot;
        UNROLL_FOR (index_t i = j + 1; i < NC; ++i)
            Dr[index(i, j)] *= inv_pivot;
        UNROLL_FOR (index_t k = j + 1; k < NC; ++k) { // column syrk
            const real_t fac = Dr[index(k, j)];
            UNROLL_FOR (index_t i = k; i < NC; ++i)
                Dr[index(i, k)] -= Dr[index(i, j)] * fac;
        }
        L_cached(j, j) = pivot;
        UNROLL_FOR (index_t i = j + 1; i < NC; ++i) // row
            L_cached(i, j) = Dr[index(i, j)];
    }
    /* Multiply the sub-diagonal blocks by the inverse of the Cholesky factor */
    auto trsm_tail = [&](auto &trsm_tail, index_t r, auto N) {
        using simd = stdx::simd<real_t, deduce_t<real_t, N>>;
        for (; r + N <= rows; r += N) {               // block row
            UNROLL_FOR (index_t c = 0; c < NC; ++c) { // column
                simd Xij{&L_cached(r, c), nalign};
                UNROLL_FOR (index_t k = 0; k < c; ++k) { // column inner
                    const real_t Aik = Dr[index(c, k)];
                    simd Xkj{&L[r + k * ldL], nalign};
                    Xij -= Aik * Xkj;
                }
                Xij *= Dr[index(c, c)];
                Xij.copy_to(&L_cached(r, c), nalign);
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
template <index_t RowsReg, index_t ColsReg>
void potrf_syrk_kernel(const real_t *A21, index_t ldA21, real_t *A22,
                       index_t ldA22, index_t rows) noexcept {
    using stdx::simd_abi::deduce_t;
    constexpr index_t NR  = 8; // Number of rows in each sub-diagonal block
    constexpr auto nalign = stdx::element_aligned;
    // Pre-compute the offsets of the columns of A21 and A22
    auto A21_cached = with_cached_access<ColsReg>(A21, ldA21);
    auto A22_cached = with_cached_access<RowsReg>(A22, ldA22);
    // Load matrix into registers
    real_t A21_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    UNROLL_FOR (index_t i = 0; i < RowsReg; ++i)
        UNROLL_FOR (index_t j = 0; j < ColsReg; ++j)
            A21_reg[i][j] = A21_cached(i, j);
    // Matrix multiplication of diagonal block
    UNROLL_FOR (index_t i = 0; i < RowsReg; ++i) {
        real_t A22i[RowsReg];
        UNROLL_FOR (index_t j = 0; j <= i; ++j)
            A22i[j] = A22_cached(i, j);
        UNROLL_FOR (index_t j = 0; j <= i; ++j)
            UNROLL_FOR (index_t kk = 0; kk < ColsReg; ++kk)
                A22i[j] -= A21_reg[i][kk] * A21_reg[j][kk];
        UNROLL_FOR (index_t j = 0; j <= i; ++j)
            A22_cached(i, j) = A22i[j];
    }
    // Matrix multiplication of sub-diagonal block
    auto gemm_tail = [&](auto &gemm_tail, index_t i, auto N) {
        using simd = stdx::simd<real_t, deduce_t<real_t, N>>;
        for (; i + N <= rows; i += N) { // block row
            simd Aij[RowsReg];
            UNROLL_FOR (index_t j = 0; j < RowsReg; ++j)
                Aij[j].copy_from(&A22_cached(i, j), nalign);
            UNROLL_FOR (index_t j = 0; j < RowsReg; ++j)
                UNROLL_FOR (index_t k = 0; k < ColsReg; ++k) {
                    simd A21ik{&A21_cached(i, k), nalign};
                    Aij[j] -= A21ik * A21_reg[j][k];
                }
            UNROLL_FOR (index_t j = 0; j < RowsReg; ++j)
                Aij[j].copy_to(&A22_cached(i, j), nalign);
        }
        if constexpr (N > 1)
            gemm_tail(gemm_tail, i, std::integral_constant<index_t, N / 2>());
    };
    gemm_tail(gemm_tail, RowsReg, std::integral_constant<index_t, NR>());
}

} // namespace

template <index_t R>
void small_potrf(real_t *L, index_t ldL, index_t m, index_t N, index_t n) {
    static const constinit auto microkernel_trsm_lut =
        make_1d_lut<R>([]<index_t Row>(index_constant<Row>) {
            return potrf_trsm_kernel<Row + 1>;
        });
    static const constinit auto microkernel_syrk_lut =
        make_1d_lut<R>([]<index_t Row>(index_constant<Row>) {
            return potrf_syrk_kernel<Row + 1, R>;
        });
    static const constinit auto microkernel_syrk_lut_2 = make_2d_lut<R, R>(
        []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
            return potrf_syrk_kernel<Row + 1, Col + 1>;
        });

    if (n < 0)
        n = N;
    assert(m >= N);
    assert((n == m && m == N) || (n == N && m >= N) || (n < m && m == N));

    [[maybe_unused]] const auto op_cnt_chol = (n + 1) * n * (n - 1) / 6 +
                                              n * (n - 1) / 2 + 2 * n,
                                op_cnt_trsm = n * (n + 1) * (m - n) / 2,
                                op_cnt_syrk = (N - n) * (N - n + 1) * n / 2;
    GUANAQO_TRACE("xpotrf_small", 0, op_cnt_chol + op_cnt_trsm + op_cnt_syrk);

    // Compute the Cholesky factorization of the very last block (right before
    // the Schur complement block), which has size r×r rather than R×R.
    // If requested, also update the rows below the Cholesky factor, and the
    // Schur complement to the bottom right of the given block.
    // These extra blocks are always sizes (m-n)×r and (m-n)×(m-n) respectively.
    auto process_bottom_right = [ldL, m, N, n](real_t *Lii, index_t r) {
        // Cholesky of last block to be factorized + triangular solve with
        // sub-diagonal block.
        microkernel_trsm_lut[r - 1](Lii, ldL, r + m - n);
        // Update the Schur complement (bottom right) with the outer product
        // of the sub-diagonal block column.
        if (n < N) {
            auto *L21 = Lii + r, *L22 = L21 + r * ldL;
            foreach_chunked_merged(
                0, m - n, index_constant<R>(),
                [&](index_t j, auto rem) {
                    auto *Lj1 = L21 + j, *Ljj = L22 + j * ldL;
                    microkernel_syrk_lut_2[rem - 1][r - 1](Lj1, ldL, Ljj, ldL,
                                                           m - n - j);
                },
                LoopDir::Forward);
        }
    };

    // Base case
    if (n == 0) {
        return;
    } else if (n <= R) {
        process_bottom_right(L, n);
        return;
    }
    // Loop over columns of H with block size R.
    index_t i;
    for (i = 0; i + R <= n; i += R) {
        auto *L11 = L + i + i * ldL;
        // Factor the diagonal block and update the subdiagonal block
        potrf_trsm_kernel<R>(L11, ldL, m - i);
        // Update the Schur complement (bottom right) with the outer product of
        // the subdiagonal block.
        foreach_chunked_merged(
            i + R, N, index_constant<R>(),
            [&](index_t j, auto rem) {
                auto L21 = L + j + i * ldL, L22 = L + j + j * ldL;
                microkernel_syrk_lut[rem - 1](L21, ldL, L22, ldL, m - j);
            },
            LoopDir::Backward);
        // Loop backwards for cache locality (we'll use the next column in the
        // next interation, so we want the syrk operation to leave it in cache).
        // TODO: verify in benchmark.
    }
    const index_t rem = n - i;
    if (rem > 0) {
        auto *Lii = L + i + i * ldL;
        process_bottom_right(Lii, rem);
    }
}

template void small_potrf<4>(real_t *L, index_t ldL, index_t m, index_t N,
                             index_t n);

} // namespace koqkatoo::linalg
