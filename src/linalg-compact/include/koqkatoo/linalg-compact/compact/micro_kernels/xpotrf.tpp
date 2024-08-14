#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact/util.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <experimental/simd>
#include <guanaqo/mat-view.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::potrf {

namespace stdx = std::experimental;

/// H = cholesky(H).
template <class simd, index_t RowsReg>
void xpotrf_RxR_microkernel(auto H) {
    static constexpr auto aligned = stdx::vector_aligned;
    // Load matrix into registers
    simd H_reg[RowsReg * (RowsReg + 1) / 2]; // NOLINT(*-c-arrays)
    auto index = [](index_t r, index_t c) {
        return c * (2 * RowsReg - 1 - c) / 2 + r;
    };
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j)
            H_reg[index(i, j)].copy_from(&H(0, i, j), aligned);
#if 0
    // Actual Cholesky kernel (Cholesky–Banachiewicz)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < j; ++k)
                H_reg[index(i, j)] -= H_reg[index(i, k)] * H_reg[index(j, k)];
            H_reg[index(i, j)] = i == j
                                     ? sqrt(H_reg[index(i, j)])
                                     : H_reg[index(i, j)] / H_reg[index(j, j)];
        }
    }
#else
    // Actual Cholesky kernel (Cholesky–Crout)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < j; ++k)
            H_reg[index(j, j)] -= H_reg[index(j, k)] * H_reg[index(j, k)];
        H_reg[index(j, j)] = sqrt(H_reg[index(j, j)]);
        simd inv_pivot     = simd{1} / H_reg[index(j, j)];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j + 1; i < RowsReg; ++i) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < j; ++k)
                H_reg[index(i, j)] -= H_reg[index(i, k)] * H_reg[index(j, k)];
            H_reg[index(i, j)] = inv_pivot * H_reg[index(i, j)];
        }
    }
#endif
    // Store matrix to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j)
            H_reg[index(i, j)].copy_to(&H(0, i, j), aligned);
}

#ifdef __AVX512F__
// AVX512 has 32 vector registers, we use 28 to cache the lower triangle of
// the matrix:
static constexpr index_t RowsReg = 7;
#else
// AVX2 has 16 vector registers, we use 15 to cache the lower triangle of
// the matrix:
static constexpr index_t RowsReg = 5;
#endif

template <class simd>
void xpotrf_register(auto H) {
    using microkernel_t = void (*)(decltype(H));
    // We need a 1D lookup table to account for all possible remainders when
    // dividing the matrix into tiles of dimensions RowsReg×RowsReg:
    static constinit auto microkernel_lut =
        make_1d_lut<RowsReg>([]<index_t R>(index_constant<R>) -> microkernel_t {
            return xpotrf_RxR_microkernel<simd, R + 1>;
        });
    assert(H.rows() <= RowsReg);
    assert(H.rows() > 0);
    microkernel_lut[H.rows() - 1](H);
}

} // namespace koqkatoo::linalg::compact::micro_kernels::potrf
