#pragma once

#include "xtrtri.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::trtri {

/// Triangular inverse, with multiplication of subdiagonal blocks. Replaces A
/// by A⁻¹, and right-multiplies B by -A⁻¹.
template <class Abi, index_t RowsReg>
[[gnu::hot]] void
xtrtri_trmm_microkernel(const mut_single_batch_matrix_accessor<Abi> A,
                        const mut_single_batch_matrix_accessor<Abi> B,
                        const index_t rows_B) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A
    auto A_cached = with_cached_access<RowsReg>(A);
    // Load matrix A into registers
    simd A_reg[RowsReg * (RowsReg + 1) / 2]; // NOLINT(*-c-arrays)
    auto Ar = [&A_reg](index_t r, index_t c) -> simd & {
        return A_reg[c * (2 * RowsReg - 1 - c) / 2 + r];
    };
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j)
            Ar(i, j) = A_cached.load(i, j);

    // Invert A.
    // Recursively apply Fact 2.17.1 from Bernstein 2009 - Matrix mathematics
    // theory, facts, and formulas.
    // [ L₁₁  0   ]⁻¹  =  [  L₁₁⁻¹             0     ]
    // [ L₂₁  L₂₂ ]       [ -L₂₂⁻¹ L₂₁ L₁₁⁻¹   L₂₂⁻¹ ]
    // First apply it to the last column:
    // [ l₁₁                  ]
    // [ l₂₁  l₂₂             ]
    // [ l₃₁  l₃₂  l₃₃        ]
    // [ l₄₁  l₄₂  l₄₃  l₄₄⁻¹ ]
    // Then to the bottom right 2×2 block:
    // [ l₁₁                                ] = [ l₁₁                 ]
    // [ l₂₁  l₂₂                           ]   [ l₂₁  l₂₂            ]
    // [ l₃₁  l₃₂   l₃₃⁻¹                   ]   [ l₃₁  l₃₂  [       ] ]
    // [ l₄₁  l₄₂  -l₄₄⁻¹ l₄₃ l₃₃⁻¹   l₄₄⁻¹ ]   [ l₄₁  l₄₂  [ L₃₃⁻¹ ] ]
    // Then to the bottom right 3×3 block, and so on.
    KOQKATOO_FULLY_UNROLLED_FOR (index_t j = RowsReg - 1; j >= 0; --j) {
        // Invert diagonal element.
        Ar(j, j) = 1 / Ar(j, j);
        // Multiply current diagonal element with column j.
        // -ℓ₂₁ ℓ₁₁⁻¹
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = RowsReg - 1; i > j; --i)
            Ar(i, j) *= -Ar(j, j);
        // Triangular matrix-vector product of bottom right block with column j.
        // -L₂₂⁻¹ ℓ₂₁ ℓ₁₁⁻¹
        KOQKATOO_FULLY_UNROLLED_FOR (index_t c = RowsReg - 1; c > j; --c) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = RowsReg - 1; i > c; --i)
                Ar(i, j) += Ar(i, c) * Ar(c, j);
            Ar(c, j) *= Ar(c, c);
        }
    }

    // Store matrix A to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j)
            A_cached.store(Ar(i, j), i, j);

    // Multiply B by A⁻¹
    auto B_cached = with_cached_access<RowsReg>(B);
    for (index_t k = 0; k < rows_B; ++k) {
        simd Br[RowsReg]; // NOLINT(*-c-arrays)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
            Br[i] = B_cached.load(k, i);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i) {
            Br[i] *= -Ar(i, i);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t j = i + 1; j < RowsReg; ++j)
                Br[i] -= Br[j] * Ar(j, i);
            B_cached.store(Br[i], k, i);
        }
    }
}

/// Multiplication of lower trapezoidal column A and by the top block of B.
template <class Abi, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xtrmm_microkernel(const mut_single_batch_matrix_accessor<Abi> A,
                  const mut_single_batch_matrix_accessor<Abi> B,
                  const index_t rows) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A and B
    auto A_cached = with_cached_access<RowsReg>(A);
    auto B_cached = with_cached_access<ColsReg>(B);
    // Load top block of B into registers
    simd Br[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j)
            Br[i][j] = B_cached.load(i, j);

    // Sub-diagonal (gemm)
    for (index_t i = rows - 1; i >= RowsReg; --i)
        // B(i) += A(i, c) * B(c)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j) {
            auto Bij = B_cached.load(i, j);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 0; k < RowsReg; ++k)
                Bij += A_cached.load(i, k) * Br[k][j];
            B_cached.store(Bij, i, j);
        }
    // Diagonal (trmm)
    // B(c) *= A(c, c)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t c = RowsReg - 1; c >= 0; --c)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = RowsReg - 1; i > c; --i)
                Br[i][j] += A_cached.load(i, c) * Br[c][j];
            Br[c][j] *= A_cached.load(c, c);
        }

    // Store top block of B to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j)
            B_cached.store(Br[i][j], i, j);
}

} // namespace koqkatoo::linalg::compact::micro_kernels::trtri
