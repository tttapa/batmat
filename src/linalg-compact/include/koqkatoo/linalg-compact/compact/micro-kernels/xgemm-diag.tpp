#pragma once

#include "xgemm.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::gemm {

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ diag(d) B⁽ᵀ⁾.
/// Single register block.
template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xgemm_diag_microkernel(const single_batch_matrix_accessor<Abi, Conf.trans_A> A,
                       const single_batch_matrix_accessor<Abi, Conf.trans_B> B,
                       const mut_single_batch_matrix_accessor<Abi> C,
                       const index_t k,
                       const single_batch_vector_accessor<Abi> d) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // The following assumption ensures that there is no unnecessary branch
    // for k == 0 in between the loops. This is crucial for good code
    // generation, otherwise the compiler inserts jumps and labels between
    // the matmul kernel and the loading/storing of C, which will cause it to
    // place C_reg on the stack, resulting in many unnecessary loads and stores.
    KOQKATOO_ASSUME(k > 0);
    static constexpr bool A_smaller = RowsReg <= ColsReg;
    // Pre-compute the offsets of the columns of C
    auto C_cached = with_cached_access<ColsReg>(C);
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            C_reg[ii][jj] = C_cached.load(ii, jj);
    // Actual matrix multiplication kernel
    for (index_t l = 0; l < k; ++l) {
        simd dl = d.load(l);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Ail  = A_smaller ? dl * A.load(ii, l) : A.load(ii, l);
                simd Blj  = A_smaller ? B.load(l, jj) : dl * B.load(l, jj);
                if constexpr (Conf.negate)
                    Cij -= Ail * Blj;
                else
                    Cij += Ail * Blj;
            }
        }
    }
    // Store accumulator to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            C_cached.store(C_reg[ii][jj], ii, jj);
}

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Using register blocking.
template <class Abi, KernelConfig Conf>
void xgemm_diag_register(single_batch_view<Abi> A, single_batch_view<Abi> B,
                         mut_single_batch_view<Abi> C,
                         single_batch_view<Abi> d) noexcept {
    const index_t I = C.rows(), J = C.cols();
    const index_t K = Conf.trans_A ? A.rows() : A.cols();
    KOQKATOO_ASSUME(I > 0);
    KOQKATOO_ASSUME(J > 0);
    KOQKATOO_ASSUME(K > 0);
    static const auto microkernel = microkernel_diag_lut<Abi, Conf>;
    const single_batch_matrix_accessor<Abi, Conf.trans_A> A_ = A;
    const single_batch_matrix_accessor<Abi, Conf.trans_B> B_ = B;
    const mut_single_batch_matrix_accessor<Abi> C_           = C;
    // Optimization for very small matrices
    if (I <= RowsReg && J <= ColsReg)
        return microkernel[I - 1][J - 1](A_, B_, C_, K, d);
    // Simply loop over all blocks in the given matrices.
    for (index_t j = 0; j < J; j += ColsReg) {
        const auto nj = std::min<index_t>(ColsReg, J - j);
        const auto Bj = B_.middle_cols(j);
        for (index_t i = 0; i < I; i += RowsReg) {
            const index_t ni = std::min<index_t>(RowsReg, I - i);
            const auto Ai    = A_.middle_rows(i);
            const auto Cij   = C_.block(i, j);
            microkernel[ni - 1][nj - 1](Ai, Bj, Cij, K, d);
        }
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::gemm
