#pragma once

#include "xtrsm.hpp"

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/loop.hpp>
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

/// Triangular solve micro-kernel AᵀX = B.
/// Replaces B₁ (the top RowsReg rows of B) by the solution,
/// X₁ = L⁻ᵀ(B₁ - L₂₁ᵀB₂), where B₂ is expected to contain the solution of the
/// bottom equations.
/// @param  A Lower-trapezoidal k×RowsReg.
/// @param  B k×ColsReg.
template <class Abi, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xtrsm_lltn_microkernel(const single_batch_matrix_accessor<Abi> A,
                       const mut_single_batch_matrix_accessor<Abi> B,
                       const index_t k) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A and B
    auto A_cached = with_cached_access<RowsReg>(A);
    auto B_cached = with_cached_access<ColsReg>(B);
    // Load accumulator into registers
    simd B_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            B_reg[ii][jj] = B_cached.load(ii, jj);
    // Matrix multiplication
    for (index_t l = RowsReg; l < k; ++l)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd Xlj = B_cached.load(l, jj);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
                simd Ail  = A_cached.load(l, ii);
                simd &Bij = B_reg[ii][jj];
                Bij -= Ail * Xlj;
            }
        }
    // Triangular solve
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = RowsReg; ii-- > 0;) {
        simd Aii = 1 / A_cached.load(ii, ii);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd &Xij = B_reg[ii][jj];
            KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = ii + 1; kk < RowsReg;
                                         ++kk) {
                simd Aik  = A_cached.load(kk, ii);
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

/// Triangular solve micro-kernel AX = B.
/// Replaces B₁ (the top RowsReg rows of B) by the solution X₁, and updates the
/// bottom rows of the right-hand side: B₂ -= L₂₁ X₁.
/// @param  A Lower-trapezoidal k×RowsReg.
/// @param  B k×ColsReg.
template <class Abi, index_t RowsReg, index_t ColsReg>
[[gnu::hot]] void
xtrsm_llnn_microkernel(const single_batch_matrix_accessor<Abi> A,
                       const mut_single_batch_matrix_accessor<Abi> B,
                       const index_t k) noexcept {
    using simd = stdx::simd<real_t, Abi>;
    // Pre-compute the offsets of the columns of A and B
    auto A_cached = with_cached_access<RowsReg>(A);
    auto B_cached = with_cached_access<ColsReg>(B);
    // Load accumulator into registers
    simd B_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            B_reg[ii][jj] = B_cached.load(ii, jj);
    // Triangular solve
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
        simd Aii = 1 / A_cached.load(ii, ii);
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd &Xij = B_reg[ii][jj];
            Xij *= Aii; // Diagonal already inverted
            KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = ii + 1; kk < RowsReg;
                                         ++kk) {
                simd Aki  = A_cached.load(kk, ii);
                simd &Xkj = B_reg[kk][jj];
                Xkj -= Aki * Xij;
            }
        }
    }
    // Store solution to memory
    KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            B_cached.store(B_reg[ii][jj], ii, jj);
    // Matrix multiplication
    for (index_t l = RowsReg; l < k; ++l)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd Blj = B_cached.load(l, jj);
            KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
                Blj -= A_cached.load(l, ii) * B_reg[ii][jj];
            B_cached.store(Blj, l, jj);
        }
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

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Using register blocking.
template <class Abi>
void xtrsm_llnn_register(single_batch_view<Abi> A,
                         mut_single_batch_view<Abi> B) noexcept {
    const index_t I = A.rows();
    const index_t J = B.cols();
    KOQKATOO_ASSUME(I > 0);
    KOQKATOO_ASSUME(J > 0);
    static const auto microkernel                  = microkernel_llnn_lut<Abi>;
    const single_batch_matrix_accessor<Abi> A_     = A;
    const mut_single_batch_matrix_accessor<Abi> B_ = B;
    auto do_block                                  = [&](index_t i, index_t j) {
        const index_t nj = std::min<index_t>(ColsReg, J - j);
        const index_t ni = std::min<index_t>(RowsReg, I - i);
        auto Aii         = A_.block(i, i); // diagonal block (lower trapezoidal)
        auto Bij         = B_.block(i, j); // rhs block to solve now
        microkernel[ni - 1][nj - 1](Aii, Bij, I - i);
    };
    // Loop over the columns of B.
    for (index_t j = 0; j < J; j += ColsReg)
        // Loop over the diagonal of A.
        for (index_t i = 0; i < I; i += RowsReg)
            do_block(i, j);
}

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Using register blocking.
template <class Abi>
void xtrsm_lltn_register(single_batch_view<Abi> A,
                         mut_single_batch_view<Abi> B) noexcept {
    const index_t I = A.rows();
    const index_t J = B.cols();
    KOQKATOO_ASSUME(I > 0);
    KOQKATOO_ASSUME(J > 0);
    static const auto microkernel                  = microkernel_lltn_lut<Abi>;
    const single_batch_matrix_accessor<Abi> A_     = A;
    const mut_single_batch_matrix_accessor<Abi> B_ = B;
    // Loop over the columns of B.
    foreach_chunked_merged(0, J, ColsReg, [&](index_t j, auto nj) {
        // Loop over the diagonal of A.
        foreach_chunked_merged(
            0, I, RowsReg,
            [&](index_t i, auto ni) {
                microkernel[ni - 1][nj - 1](A_.block(i, i), B_.block(i, j),
                                            I - i);
            },
            LoopDir::Backward);
    });
}

} // namespace koqkatoo::linalg::compact::micro_kernels::trsm
