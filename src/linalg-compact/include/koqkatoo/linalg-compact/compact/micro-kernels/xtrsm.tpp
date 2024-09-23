#pragma once

#include "xtrsm.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact::micro_kernels::trsm {

/// Triangular solve micro-kernel.
/// @param  A Lower-triangular RowsReg×RowsReg.
/// @param  B RowsReg×ColsReg (trans=false) or ColsReg×RowsReg (trans=true).
/// @param  A10 RowsReg×k
/// @param  X01 k×ColsReg (trans=false) or ColsReg×k (trans=true)
template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xtrsm_microkernel(const single_batch_matrix_accessor<Abi> A,
                  const mut_single_batch_matrix_accessor<Abi, Conf.trans> B,
                  const single_batch_matrix_accessor<Abi> A10,
                  const mut_single_batch_matrix_accessor<Abi, Conf.trans> X01,
                  const index_t k) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns/rows of B
    auto B_cached = [&] {
        if constexpr (Conf.trans)
            return with_cached_access<RowsReg>(B);
        else
            return with_cached_access<ColsReg>(B);
    }();
    // Load accumulator into registers
    simd B_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            B_reg[ii][jj] = B_cached.load(ii, jj);
    // Matrix multiplication
    for (index_t l = 0; l < k; ++l)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd Xlj = X01.load(l, jj);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
                simd Ail  = A10.load(ii, l);
                simd &Bij = B_reg[ii][jj];
                Bij -= Ail * Xlj;
            }
        }
    // Triangular solve
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
        simd Aii = 1 / A.load(ii, ii);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd &Xij = B_reg[ii][jj];
            KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ii; ++kk) {
                simd Aik  = A.load(ii, kk);
                simd &Xkj = B_reg[kk][jj];
                Xij -= Aik * Xkj;
            }
            Xij *= Aii; // Diagonal already inverted
        }
    }
    // Store accumulator to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            B_cached.store(B_reg[ii][jj], ii, jj);
}

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Using register blocking.
template <class Abi, KernelConfig Conf>
void xtrsm_register(single_batch_view<Abi> A,
                    mut_single_batch_view<Abi> B) noexcept {
    const index_t I = A.rows();
    const index_t J = Conf.trans ? B.rows() : B.cols();
    KOQKATOO_ASSUME(I > 0);
    KOQKATOO_ASSUME(J > 0);
    static const auto microkernel              = microkernel_lut<Abi, Conf>;
    const single_batch_matrix_accessor<Abi> A_ = A;
    const mut_single_batch_matrix_accessor<Abi, Conf.trans> B_ = B;
    auto do_block = [&](index_t i, index_t j) {
        const index_t nj = std::min<index_t>(ColsReg, J - j);
        const index_t ni = std::min<index_t>(RowsReg, I - i);
        auto Aii         = A_.block(i, i); // diagonal block (lower triangular)
        auto Bij         = B_.block(i, j); // rhs block to solve now
        auto Ais         = A_.middle_rows(i); // subdiagonal block
        auto X0j = B_.middle_cols(j); // first rows of rhs (already solved)
        microkernel[ni - 1][nj - 1](Aii, Bij, Ais, X0j, i);
    };
    if constexpr (Conf.trans)
        // Loop over the diagonal of L.
        for (index_t i = 0; i < I; i += RowsReg)
            // Loop over the columns of B.
            for (index_t j = 0; j < J; j += ColsReg)
                do_block(i, j);
    else
        // Loop over the columns of B.
        for (index_t j = 0; j < J; j += ColsReg)
            // Loop over the diagonal of L.
            for (index_t i = 0; i < I; i += RowsReg)
                do_block(i, j);
}

} // namespace koqkatoo::linalg::compact::micro_kernels::trsm
