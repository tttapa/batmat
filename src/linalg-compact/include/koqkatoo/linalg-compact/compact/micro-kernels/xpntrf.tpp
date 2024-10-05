#pragma once

#include "rsqrt.hpp"
#include "xpntrf.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/cneg.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::pntrf {

/// Cholesky factorization Chol(±A).
template <class Abi, index_t RowsReg>
[[gnu::hot]] void
xpntrf_microkernel(const mut_single_batch_matrix_accessor<Abi> A,
                   single_batch_vector_accessor<Abi> signs) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A
    auto A_cached = with_cached_access<RowsReg>(A);
    // Load matrix into registers
    simd Ar[RowsReg * (RowsReg + 1) / 2]; // NOLINT(*-c-arrays)
    auto index = [](index_t r, index_t c) {
        return c * (2 * RowsReg - 1 - c) / 2 + r;
    };
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        auto sj = signs.load(j);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j; i < RowsReg; ++i)
            Ar[index(i, j)] = cneg(A_cached.load(i, j), sj);
    }
    // Actual Cholesky kernel (immediate outer product)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        auto sj         = signs.load(j);
        simd inv_pivot  = rsqrt(Ar[index(j, j)]);
        Ar[index(j, j)] = sqrt(Ar[index(j, j)]);
        // Divide by pivot
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j + 1; i < RowsReg; ++i)
            Ar[index(i, j)] = inv_pivot * Ar[index(i, j)];
        // Schur complement outer product
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = j + 1; k < RowsReg; ++k) {
            auto Akj = cneg(Ar[index(k, j)], cneg(signs.load(k), sj));
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k; i < RowsReg; ++i)
                Ar[index(i, k)] -= Ar[index(i, j)] * Akj;
        }
    }
    // Store matrix to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j; i < RowsReg; ++i)
            A_cached.store(Ar[index(i, j)], i, j);
}

/// Cholesky factorization Chol(±A) and triangular solve.
template <class Abi, index_t RowsReg>
[[gnu::hot]] void
xpntrf_xtrsm_microkernel(const mut_single_batch_matrix_accessor<Abi> A11,
                         const mut_single_batch_matrix_accessor<Abi> A21,
                         index_t k,
                         single_batch_vector_accessor<Abi> signs) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A
    auto A11_cached = with_cached_access<RowsReg>(A11);
    // Load matrix into registers
    simd A11r[RowsReg * (RowsReg + 1) / 2 + RowsReg]; // NOLINT(*-c-arrays)
    auto index = [](index_t r, index_t c) {
        return c * (2 * RowsReg - 1 - c) / 2 + r;
    };
    auto inv_index = [](index_t r) { return RowsReg * (RowsReg + 1) / 2 + r; };
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        auto sj = signs.load(j);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j; i < RowsReg; ++i)
            A11r[index(i, j)] = cneg(A11_cached.load(i, j), sj);
    }
    // Actual Cholesky kernel (immediate outer product)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        auto sj        = signs.load(j);
        simd inv_pivot = A11r[inv_index(j)] = rsqrt(A11r[index(j, j)]);
        A11r[index(j, j)]                   = sqrt(A11r[index(j, j)]);
        // Divide by pivot
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j + 1; i < RowsReg; ++i)
            A11r[index(i, j)] = inv_pivot * A11r[index(i, j)];
        // Schur complement outer product
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = j + 1; k < RowsReg; ++k) {
            auto Akj = cneg(A11r[index(k, j)], cneg(signs.load(k), sj));
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = k; i < RowsReg; ++i)
                A11r[index(i, k)] -= A11r[index(i, j)] * Akj;
        }
    }
    // Negate before triangular solve
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        simd sj = signs.load(j);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j; i < RowsReg; ++i)
            A11r[index(i, j)] = cneg(A11r[index(i, j)], sj);
        A11r[inv_index(j)] = cneg(A11r[inv_index(j)], sj);
    }
    // Triangular solve
    auto A21_cached = with_cached_access<RowsReg>(A21);
    for (index_t jj = 0; jj < k; ++jj) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Xij = A21_cached.load(jj, ii);
            simd piv = A11r[index(ii, ii)];
            KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ii; ++kk) {
                simd Aik = A11r[index(ii, kk)];
                simd Xkj = A21_cached.load(jj, kk);
                Xij -= Aik * Xkj;
            }
            Xij *= A11r[inv_index(ii)];
            A21_cached.store(Xij, jj, ii);
        }
    }
    // Store matrix to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < RowsReg; ++j) {
        simd sj = signs.load(j);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j; i < RowsReg; ++i)
            A11_cached.store(cneg(A11r[index(i, j)], sj), i, j);
    }
}

/// Outer product for updating the bottom right tail during Cholesky factorization.
/// @param A21 num_rows×ColsReg
/// @param A22 num_rows×RowsReg
template <class Abi, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xpntrf_xsyrk_microkernel(const single_batch_matrix_accessor<Abi> A21,
                         const mut_single_batch_matrix_accessor<Abi> A22,
                         index_t num_rows,
                         single_batch_vector_accessor<Abi> signs) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A21 and A22
    auto A21_cached = with_cached_access<ColsReg>(A21);
    // auto A22_cached = with_cached_access<RowsReg>(A22);
    auto A22_cached = A22; // TODO
    // Load matrix into registers
    simd A21r[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            A21r[ii][jj] = A21_cached.load(ii, jj);
    // Matrix multiplication of diagonal block
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
        simd A22ij[RowsReg];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
            A22ij[jj] = A22_cached.load(ii, jj);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ColsReg; ++kk) {
            simd sk  = signs.load(kk);
            simd Aik = cneg(A21r[ii][kk], sk);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
                A22ij[jj] -= Aik * A21r[jj][kk];
        }
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
            A22_cached.store(A22ij[jj], ii, jj);
    }
    // Negate before outer product
    KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ColsReg; ++kk) {
        simd sk = signs.load(kk);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < RowsReg; ++jj)
            A21r[jj][kk] = cneg(A21r[jj][kk], sk);
    }
    // Matrix multiplication of sub-diagonal block
    for (index_t ii = RowsReg; ii < num_rows; ++ii) {
        simd Aij[RowsReg];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < RowsReg; ++jj)
            Aij[jj] = A22_cached.load(ii, jj);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ColsReg; ++kk)
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < RowsReg; ++jj)
                Aij[jj] -= A21_cached.load(ii, kk) * A21r[jj][kk];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < RowsReg; ++jj)
            A22_cached.store(Aij[jj], ii, jj);
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::pntrf
