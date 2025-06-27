#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/trtri.hpp>
#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::trtri {

/// @param  A k×RowsReg.
/// @param  D k×RowsReg.
/// Invert the top block of A and store it in the top block of D. Then multiply the bottom blocks of
/// D by this block (on the right).
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder OA, StorageOrder OD>
[[gnu::hot, gnu::flatten]] void trtri_copy_microkernel(const uview<const T, Abi, OA> A,
                                                       const uview<T, Abi, OD> D,
                                                       const index_t k) noexcept {
    static_assert(Conf.struc == MatrixStructure::LowerTriangular); // TODO
    static_assert(RowsReg > 0);
    BATMAT_ASSUME(k >= RowsReg);
    using simd = datapar::simd<T, Abi>;
    // Pre-compute the offsets of the columns of A
    const auto A1_cached = with_cached_access<RowsReg, RowsReg>(A);
    const auto A_cached  = with_cached_access<0, RowsReg>(A);
    // Load matrix into registers
    simd A1_reg[RowsReg * (RowsReg + 1) / 2]; // NOLINT(*-c-arrays)
    auto A1r = [&A1_reg](index_t r, index_t c) -> simd & {
        return A1_reg[c * (2 * RowsReg - 1 - c) / 2 + r];
    };
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = 0; jj <= ii; ++jj)
            A1r(ii, jj) = A1_cached.load(ii, jj);

    // Invert A₁.
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
    UNROLL_FOR (index_t jj = RowsReg - 1; jj >= 0; --jj) {
        // Invert diagonal element.
        A1r(jj, jj) = simd{1} / A1r(jj, jj);
        // Multiply current diagonal element with column j.
        // -ℓ₂₁ ℓ₁₁⁻¹
        UNROLL_FOR (index_t ii = RowsReg - 1; ii > jj; --ii)
            A1r(ii, jj) *= -A1r(jj, jj);
        // Triangular matrix-vector product of bottom right block with column j.
        // -L₂₂⁻¹ ℓ₂₁ ℓ₁₁⁻¹
        UNROLL_FOR (index_t ll = RowsReg - 1; ll > jj; --ll) {
            UNROLL_FOR (index_t ii = RowsReg - 1; ii > ll; --ii)
                A1r(ii, jj) += A1r(ii, ll) * A1r(ll, jj);
            A1r(ll, jj) *= A1r(ll, ll);
        }
    }

    // Pre-compute the offsets of the columns of D
    const auto D1_cached = with_cached_access<RowsReg, RowsReg>(D);
    const auto D_cached  = with_cached_access<0, RowsReg>(D);
    // Store matrix A₁⁻¹ to D₁
    UNROLL_FOR (index_t i = 0; i < RowsReg; ++i)
        UNROLL_FOR (index_t j = 0; j <= i; ++j)
            D1_cached.store(A1r(i, j), i, j);

    // Multiply A₂ by -A₁⁻¹ and store in D₂
    for (index_t l = RowsReg; l < k; ++l) {
        simd A2r[RowsReg]; // NOLINT(*-c-arrays)
        UNROLL_FOR (index_t i = 0; i < RowsReg; ++i)
            A2r[i] = A_cached.load(l, i);
        UNROLL_FOR (index_t i = 0; i < RowsReg; ++i) {
            A2r[i] *= -A1r(i, i);
            UNROLL_FOR (index_t j = i + 1; j < RowsReg; ++j)
                A2r[i] -= A2r[j] * A1r(j, i);
            D_cached.store(A2r[i], l, i);
        }
    }
}

/// @param  Dr RowsReg×k lower trapezoidal
/// @param  D  k×ColsReg
/// Compute product Dr D and store the result in the bottom block of D
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OD>
[[gnu::hot, gnu::flatten]]
void trmm_microkernel(const uview<const T, Abi, OD> Dr, const uview<T, Abi, OD> D,
                      const index_t k) noexcept {
    static_assert(Conf.struc == MatrixStructure::LowerTriangular); // TODO
    static_assert(RowsReg > 0 && ColsReg > 0);
    BATMAT_ASSUME(k >= RowsReg);
    using simd = datapar::simd<T, Abi>;
    // Clear accumulator
    simd D_reg[RowsReg][ColsReg]{}; // NOLINT(*-c-arrays)
    // Perform gemm
    const auto A1_cached = with_cached_access<RowsReg, 0>(Dr);
    const auto B1_cached = with_cached_access<0, ColsReg>(D);
    for (index_t l = 0; l < k - RowsReg; ++l)
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A1_cached.load(ii, l);
            UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = D_reg[ii][jj];
                simd Blj  = B1_cached.load(l, jj);
                Cij += Ail * Blj;
            }
        }
    // Perform trmm
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) // TODO: move before gemm
        UNROLL_FOR (index_t ll = 0; ll <= ii; ++ll) {
            simd Ail = A1_cached.load(ii, k - RowsReg + ll);
            UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj)
                D_reg[ii][jj] += Ail * B1_cached.load(k - RowsReg + ll, jj);
        }
    // Store result to memory
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            D.store(D_reg[ii][jj], k - RowsReg + ii, jj);
}

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD>
void trtri_copy_register(const view<const T, Abi, OA> A, const view<T, Abi, OD> D) noexcept {
    using enum MatrixStructure;
    static_assert(Conf.struc == LowerTriangular); // TODO
    constexpr auto Rows = RowsReg<T, Abi>, Cols = ColsReg<T, Abi>;
    // Check dimensions
    const index_t I = A.rows();
    BATMAT_ASSUME(A.rows() == A.cols());
    BATMAT_ASSUME(A.rows() == D.rows());
    BATMAT_ASSUME(A.cols() == D.cols());
    static const auto trtri_microkernel = trtri_copy_lut<T, Abi, Conf, OA, OD>;
    static const auto trmm_microkernel  = trmm_lut<T, Abi, Conf, OD>;
    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_ = A;
    const uview<T, Abi, OD> D_       = D;

    // Optimization for very small matrices
    if (I <= Rows)
        return trtri_microkernel[I - 1](A_, D_, I);

    // Partition:
    //     [ ...        ]        [ ...        ]
    // A = [ ... Ajj    ]    D = [ ... Djj    ]   with the invariant Dp = Ap⁻¹
    //     [ ... Aj  Ap ]        [ ... Dj  Dp ]

    foreach_chunked_merged( // Loop over the diagonal blocks of A, in reverse
        0, I, Cols,
        [&](index_t j, auto nj) {
            const auto jp  = j + nj;
            const auto Ajj = A_.block(j, j);
            const auto Djj = D_.block(j, j);
            const auto Dj  = D_.block(jp, j);
            // Invert Djj = Ajj⁻¹ and multiply Dj = Aj Djj
            trtri_microkernel[nj - 1](Ajj, Djj, I - j);
            // Multiply Dp Dj (with Dp lower triangular)
            foreach_chunked_merged( // Loop over the block rows of Dj, in reverse
                jp, I, Rows,
                [&](index_t i, auto ni) {
                    // Block row of already inverted bottom right corner
                    const auto Dpi = D_.block(i, jp);
                    // Current subdiagonal column to be multiplied by Dp
                    trmm_microkernel[ni - 1][nj - 1](Dpi, Dj, i + ni - jp);
                },
                LoopDir::Backward);
        },
        LoopDir::Backward);
}

} // namespace batmat::linalg::micro_kernels::trtri
