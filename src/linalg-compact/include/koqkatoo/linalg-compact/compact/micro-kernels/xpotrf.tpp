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

/// Cholesky factorization Chol(A) and triangular solve.
template <class Abi, index_t RowsReg>
[[gnu::hot]] void
xpotrf_xtrsm_microkernel(const mut_single_batch_matrix_accessor<Abi> A11,
                         const mut_single_batch_matrix_accessor<Abi> A21,
                         index_t k) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A
    auto A11_cached = with_cached_access<RowsReg>(A11);
    // Load matrix into registers
    simd A11_reg[RowsReg * (RowsReg + 1) / 2 + RowsReg]; // NOLINT(*-c-arrays)
    auto index = [](index_t r, index_t c) {
        return c * (2 * RowsReg - 1 - c) / 2 + r;
    };
    auto inv_index = [](index_t r) { return RowsReg * (RowsReg + 1) / 2 + r; };
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j)
            A11_reg[index(i, j)] = A11_cached.load(i, j);
    // Actual Cholesky kernel (Cholesky–Crout)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < j; ++l)
            A11_reg[index(j, j)] -= A11_reg[index(j, l)] * A11_reg[index(j, l)];
        auto piv = A11_reg[index(j, j)] = sqrt(A11_reg[index(j, j)]);
        auto inv_piv = A11_reg[inv_index(j)] = simd{1} / piv;
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j + 1; i < RowsReg; ++i) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < j; ++k)
                A11_reg[index(i, j)] -=
                    A11_reg[index(i, k)] * A11_reg[index(j, k)];
            A11_reg[index(i, j)] = inv_piv * A11_reg[index(i, j)];
        }
    }
    // Triangular solve
    auto A21_cached = with_cached_access<RowsReg>(A21);
    for (index_t jj = 0; jj < k; ++jj) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Xij = A21_cached.load(jj, ii);
            simd piv = A11_reg[index(ii, ii)];
            KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ii; ++kk) {
                simd Aik = A11_reg[index(ii, kk)];
                simd Xkj = A21_cached.load(jj, kk);
                Xij -= Aik * Xkj;
            }
            Xij *= A11_reg[inv_index(ii)];
            A21_cached.store(Xij, jj, ii);
        }
    }
    // Store matrix to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j)
            A11_cached.store(A11_reg[index(i, j)], i, j);
}

/// Outer product for updating the bottom right tail during Cholesky factorization.
/// @param A21 num_rows×ColsReg
/// @param A22 num_rows×RowsReg
template <class Abi, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xpotrf_xsyrk_microkernel(const single_batch_matrix_accessor<Abi> A21,
                         const mut_single_batch_matrix_accessor<Abi> A22,
                         index_t num_rows) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A21 and A22
    auto A21_cached = with_cached_access<ColsReg>(A21);
    auto A22_cached = with_cached_access<RowsReg>(A22);
    // Load matrix into registers
    simd A21_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            A21_reg[ii][jj] = A21_cached.load(ii, jj);
    // Matrix multiplication of diagonal block
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
        simd A22ij[RowsReg];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
            A22ij[jj] = A22_cached.load(ii, jj);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
            KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ColsReg; ++kk)
                A22ij[jj] -= A21_cached.load(ii, kk) * A21_reg[jj][kk];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
            A22_cached.store(A22ij[jj], ii, jj);
    }
    // Matrix multiplication of sub-diagonal block
    for (index_t ii = RowsReg; ii < num_rows; ++ii) {
        simd Aij[RowsReg];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < RowsReg; ++jj)
            Aij[jj] = A22_cached.load(ii, jj);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < RowsReg; ++jj)
            KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ColsReg; ++kk)
                Aij[jj] -= A21_cached.load(ii, kk) * A21_reg[jj][kk];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < RowsReg; ++jj)
            A22_cached.store(Aij[jj], ii, jj);
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::potrf
