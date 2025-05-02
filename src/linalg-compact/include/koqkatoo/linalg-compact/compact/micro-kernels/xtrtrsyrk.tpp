#pragma once

#include "rotate.hpp"
#include "xtrtrsyrk.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::trtrsyrk {

/// Matrix multiplication C = C ± A Bᵀ, with A upper trapezoidal (wide) and Bᵀ
/// either rectangular or lower trapezoidal (tall).
template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xtrtrsyrk_UL_microkernel(const single_batch_matrix_accessor<Abi, false> A,
                         const single_batch_matrix_accessor<Abi, true> B,
                         const mut_single_batch_matrix_accessor<Abi> C,
                         index_t k, bool trtr, bool init_zero) noexcept {
    static constexpr auto S = Conf.shift;
    using simd              = stdx::simd<real_t, Abi>;
    KOQKATOO_ASSUME(k >= RowsReg);
    if constexpr (RowsReg != ColsReg)
        KOQKATOO_ASSUME(!trtr);
    // Pre-compute the offsets of the columns of C
    auto C_cached = with_cached_access<ColsReg>(C);
    if (trtr) {
        // Load accumulator into registers
        simd C_reg[RowsReg][ColsReg]{}; // NOLINT(*-c-arrays)
        if (!init_zero) [[unlikely]]
            KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j)
                KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j; i < RowsReg; ++i)
                    C_reg[i][j] = rotl<S>(C_cached.load(i, j));
        // Triangular-triangular matrix multiplication kernel
        KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < ColsReg; ++l) {
            KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= l; ++j) {
                simd Blj = B.load(l, j);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j; i < RowsReg; ++i) {
                    simd Ail  = A.load(i, l);
                    simd &Cij = C_reg[i][j];
                    if constexpr (Conf.negate)
                        Cij -= Ail * Blj;
                    else
                        Cij += Ail * Blj;
                }
            }
        }
        // Subdiagonal matrix multiplication kernel
        for (index_t l = RowsReg; l < k; ++l)
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i) {
                simd Ail = A.load(i, l);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j <= i; ++j) {
                    simd &Cij = C_reg[i][j];
                    simd Blj  = B.load(l, j);
                    if constexpr (Conf.negate)
                        Cij -= Ail * Blj;
                    else
                        Cij += Ail * Blj;
                }
            }
        // Store accumulator to memory again
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j)
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = j; i < RowsReg; ++i)
                C_cached.template store<S>(rotr<S>(C_reg[i][j]), i, j);
    } else {
        // Load accumulator into registers
        simd C_reg[RowsReg][ColsReg]{}; // NOLINT(*-c-arrays)
        if (!init_zero) [[unlikely]]
            KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j)
                KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
                    C_reg[i][j] = rotl<S>(C_cached.load(i, j));
        // Triangular-general matrix multiplication kernel
        KOQKATOO_FULLY_UNROLLED_FOR (index_t l = 0; l < RowsReg; ++l)
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i <= l; ++i) {
                simd Ail = A.load(i, l);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j) {
                    simd &Cij = C_reg[i][j];
                    simd Blj  = B.load(l, j);
                    if constexpr (Conf.negate)
                        Cij -= Ail * Blj;
                    else
                        Cij += Ail * Blj;
                }
            }
        // Subdiagonal matrix multiplication kernel
        for (index_t l = RowsReg; l < k; ++l)
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i) {
                simd Ail = A.load(i, l);
                KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j) {
                    simd &Cij = C_reg[i][j];
                    simd Blj  = B.load(l, j);
                    if constexpr (Conf.negate)
                        Cij -= Ail * Blj;
                    else
                        Cij += Ail * Blj;
                }
            }
        // Store accumulator to memory again
        KOQKATOO_FULLY_UNROLLED_FOR (index_t j = 0; j < ColsReg; ++j)
            KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 0; i < RowsReg; ++i)
                C_cached.template store<S>(rotr<S>(C_reg[i][j]), i, j);
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::trtrsyrk
