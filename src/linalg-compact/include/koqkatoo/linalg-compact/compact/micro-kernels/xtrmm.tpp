#pragma once

#include "xtrmm.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::trmm {

/// Generalized matrix multiplication C = C ± A⁾ B, with B lower triangular.
/// Single register block.
template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xtrmm_rlnn_microkernel(const single_batch_matrix_accessor<Abi, false> A,
                       const single_batch_matrix_accessor<Abi, false> B,
                       const mut_single_batch_matrix_accessor<Abi> C,
                       const index_t k) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    KOQKATOO_ASSUME(k >= ColsReg);
    // Pre-compute the offsets of the columns of C
    auto C_cached = with_cached_access<ColsReg>(C);
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            C_reg[ii][jj] = C_cached.load(ii, jj);
    // Triangular matrix multiplication kernel
    KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < ColsReg; ++l)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A.load(ii, l);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= l; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = B.load(l, jj);
                if constexpr (Conf.negate)
                    Cij -= Ail * Blj;
                else
                    Cij += Ail * Blj;
            }
        }
    // Subdiagonal matrix multiplication kernel
    for (index_t l = ColsReg; l < k; ++l)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A.load(ii, l);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
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
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            C_cached.store(C_reg[ii][jj], ii, jj);
}

} // namespace koqkatoo::linalg::compact::micro_kernels::trmm
