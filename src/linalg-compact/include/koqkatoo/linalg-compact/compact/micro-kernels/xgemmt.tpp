#pragma once

#include "xgemm.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::gemm {

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾ computing only the lower
/// triangle of C. Single register block.
template <class Abi, KernelConfig Conf, index_t RowsReg>
[[gnu::hot]] void
xgemmt_microkernel(const single_batch_matrix_accessor<Abi, Conf.trans_A> A,
                   const single_batch_matrix_accessor<Abi, Conf.trans_B> B,
                   const mut_single_batch_matrix_accessor<Abi> C,
                   const index_t k) noexcept {
    static constexpr index_t ColsReg = RowsReg;
    using simd                       = stdx::simd<real_t, Abi>;
    // The following assumption ensures that there is no unnecessary branch
    // for k == 0 in between the loops. This is crucial for good code
    // generation, otherwise the compiler inserts jumps and labels between
    // the matmul kernel and the loading/storing of C, which will cause it to
    // place C_reg on the stack, resulting in many unnecessary loads and stores.
    KOQKATOO_ASSUME(k > 0);
    // Pre-compute the offsets of the columns of C
    auto C_cached = with_cached_access<ColsReg>(C);
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = jj; ii < RowsReg; ++ii)
            C_reg[ii][jj] = C_cached.load(ii, jj);
    // Actual matrix multiplication kernel
    for (index_t l = 0; l < k; ++l)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A.load(ii, l);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = B.load(l, jj);
                if constexpr (Conf.negate)
                    Cij -= Ail * Blj;
                else
                    Cij += Ail * Blj;
            }
        }
    // Store accumulator to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = jj; ii < RowsReg; ++ii)
            C_cached.store(C_reg[ii][jj], ii, jj);
}

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾ computing only the lower
/// triangle of C. Using register blocking.
template <class Abi, KernelConfig Conf>
void xgemmt_register(single_batch_view<Abi> A, single_batch_view<Abi> B,
                     mut_single_batch_view<Abi> C) noexcept {
    static_assert(RowsReg == ColsReg, "Square blocks required");
    const index_t I = C.rows(), J = C.cols();
    const index_t K = Conf.trans_A ? A.rows() : A.cols();
    KOQKATOO_ASSUME(I == Conf.trans_A ? A.cols() : A.rows());
    KOQKATOO_ASSUME(J == Conf.trans_B ? B.rows() : B.cols());
    KOQKATOO_ASSUME(I > 0);
    KOQKATOO_ASSUME(J > 0);
    KOQKATOO_ASSUME(K > 0);
    static const auto microkernel_t = microkernel_t_lut<Abi, Conf>;
    static const auto microkernel   = microkernel_lut<Abi, Conf>;
    const single_batch_matrix_accessor<Abi, Conf.trans_A> A_ = A;
    const single_batch_matrix_accessor<Abi, Conf.trans_B> B_ = B;
    const mut_single_batch_matrix_accessor<Abi> C_           = C;
    // Optimization for very small matrices
    if (I <= RowsReg && I == J)
        return microkernel_t[I - 1](A_, B_, C_, K);
    // Simply loop over all blocks in the given matrices.
    index_t j;
    for (j = 0; j + ColsReg <= J; j += ColsReg) {
        const auto nj = ColsReg;
        const auto Bj = B_.middle_cols(j);
        for (index_t i = j; i < I; i += RowsReg) {
            const index_t ni = std::min<index_t>(RowsReg, I - i);
            const auto Ai    = A_.middle_rows(i);
            const auto Cij   = C_.block(i, j);
            if (i == j) {
                KOQKATOO_ASSUME(ni == nj);
                microkernel_t[nj - 1](Ai, Bj, Cij, K);
            } else {
                microkernel[ni - 1][nj - 1](Ai, Bj, Cij, K);
            }
        }
    }
    // Final block column is smaller
    const auto nj = J - j;
    if (nj > 0) {
        const auto Bj = B_.middle_cols(j);
        for (index_t i = j; i < I; i += RowsReg) {
            const index_t ni = std::min<index_t>(RowsReg, I - i);
            const auto Ai    = A_.middle_rows(i);
            const auto Cij   = C_.block(i, j);
            if (i == j) {
                KOQKATOO_ASSUME(ni >= nj);
                microkernel_t[nj - 1](Ai, Bj, Cij, K);
                if (nj < ni) [[unlikely]]
                    microkernel[ni - nj - 1][nj - 1](Ai.middle_rows(nj), Bj,
                                                     Cij.middle_rows(nj), K);
            } else {
                microkernel[ni - 1][nj - 1](Ai, Bj, Cij, K);
            }
        }
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::gemm
