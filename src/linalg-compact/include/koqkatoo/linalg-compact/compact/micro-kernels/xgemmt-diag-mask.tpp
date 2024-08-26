#pragma once

#include "xgemm.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::gemm {

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ diag(d) B⁽ᵀ⁾ computing only
/// the lower triangle of C. Single register block.
template <class Abi, KernelConfig Conf, index_t RowsReg>
[[gnu::hot]] void xgemmt_diag_mask_microkernel(
    const single_batch_matrix_accessor<Abi, Conf.trans_A> A,
    const single_batch_matrix_accessor<Abi, Conf.trans_B> B,
    const mut_single_batch_matrix_accessor<Abi> C, const index_t k,
    const single_batch_vector_accessor<Abi> d,
    const single_batch_vector_mask_accessor<Abi> m) noexcept {
    static constexpr index_t ColsReg = RowsReg;
    using simd                       = stdx::simd<real_t, Abi>;
    using mask                       = typename simd::mask_type;
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
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = jj; ii < RowsReg; ++ii)
            C_reg[ii][jj] = C_cached.load(ii, jj);
    // Actual matrix multiplication kernel
    for (index_t l = 0; l < k; ++l) {
        simd dl = d.load(l);
        mask ml = m.load(l);
        if (none_of(ml))
            continue;
        where(!ml, dl) = 0;
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj) {
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
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = jj; ii < RowsReg; ++ii)
            C_cached.store(C_reg[ii][jj], ii, jj);
}

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ diag(d) B⁽ᵀ⁾ computing only
/// the lower triangle of C. Using register blocking.
template <class Abi, KernelConfig Conf>
void xgemmt_diag_mask_register(single_batch_view<Abi> A,
                               single_batch_view<Abi> B,
                               mut_single_batch_view<Abi> C,
                               single_batch_view<Abi> d,
                               bool_single_batch_view<Abi> m) noexcept {
    static_assert(RowsReg == ColsReg, "Square blocks required");
    const index_t I = C.rows(), J = C.cols();
    const index_t K = Conf.trans_A ? A.rows() : A.cols();
    KOQKATOO_ASSUME(I > 0);
    KOQKATOO_ASSUME(J > 0);
    KOQKATOO_ASSUME(K > 0);
    KOQKATOO_ASSUME(I == J);
    static const auto microkernel_t = microkernel_t_diag_mask_lut<Abi, Conf>;
    static const auto microkernel   = microkernel_diag_mask_lut<Abi, Conf>;
    const single_batch_matrix_accessor<Abi, Conf.trans_A> A_ = A;
    const single_batch_matrix_accessor<Abi, Conf.trans_B> B_ = B;
    const mut_single_batch_matrix_accessor<Abi> C_           = C;
    // Optimization for very small matrices
    if (I <= RowsReg)
        return microkernel_t[I - 1](A_, B_, C_, K, d, m);
    // Simply loop over all blocks in the given matrices.
    for (index_t j = 0; j < J; j += ColsReg) {
        const auto nj = std::min<index_t>(ColsReg, J - j);
        const auto Bj = B_.middle_cols(j);
        for (index_t i = j; i < I; i += RowsReg) {
            const index_t ni = std::min<index_t>(RowsReg, I - i);
            const auto Ai    = A_.middle_rows(i);
            const auto Cij   = C_.block(i, j);
            if (i == j)
                microkernel_t[ni - 1](Ai, Bj, Cij, K, d, m);
            else
                microkernel[ni - 1][nj - 1](Ai, Bj, Cij, K, d, m);
        }
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::gemm
