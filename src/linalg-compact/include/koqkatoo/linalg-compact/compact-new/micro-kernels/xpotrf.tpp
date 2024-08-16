#pragma once

#include "xpotrf.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::potrf {

/// Cholesky factorization Chol(A).
template <class Abi, index_t RowsReg>
[[gnu::hot]] void
xpotrf_microkernel(const mut_single_batch_matrix_accessor<Abi> A) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A
    auto A_cached = with_cached_access<RowsReg>(A);
    // Load matrix into registers
    simd A_reg[RowsReg * (RowsReg + 1) / 2]; // NOLINT(*-c-arrays)
    auto index = [](index_t r, index_t c) {
        return c * (2 * RowsReg - 1 - c) / 2 + r;
    };
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j)
            A_reg[index(i, j)] = A_cached.load(i, j);
#if 0
    // Actual Cholesky kernel (Cholesky–Banachiewicz)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < j; ++k)
                A_reg[index(i, j)] -= A_reg[index(i, k)] * A_reg[index(j, k)];
            A_reg[index(i, j)] = i == j
                                     ? sqrt(A_reg[index(i, j)])
                                     : A_reg[index(i, j)] / A_reg[index(j, j)];
        }
    }
#else
    // Actual Cholesky kernel (Cholesky–Crout)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < j; ++k)
            A_reg[index(j, j)] -= A_reg[index(j, k)] * A_reg[index(j, k)];
        A_reg[index(j, j)] = sqrt(A_reg[index(j, j)]);
        simd inv_pivot     = simd{1} / A_reg[index(j, j)];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j + 1; i < RowsReg; ++i) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < j; ++k)
                A_reg[index(i, j)] -= A_reg[index(i, k)] * A_reg[index(j, k)];
            A_reg[index(i, j)] = inv_pivot * A_reg[index(i, j)];
        }
    }
#endif
    // Store matrix to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j)
            A_cached.store(A_reg[index(i, j)], i, j);
}

} // namespace koqkatoo::linalg::compact::micro_kernels::potrf
