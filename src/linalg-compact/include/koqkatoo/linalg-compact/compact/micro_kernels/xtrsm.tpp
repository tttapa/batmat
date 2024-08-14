#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact/util.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <experimental/simd>
#include <guanaqo/mat-view.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::trsm {

namespace stdx = std::experimental;

struct KernelConfig {
    bool trans = false;
};

template <class simd, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void // NOLINTNEXTLINE(*-cognitive-complexity)
gemm_trsm_RxC_microkernel(auto A, auto B, auto A10, auto X01) {
    static constexpr auto aligned = stdx::vector_aligned;
    // Load accumulator into registers
    simd B_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            B_reg[ii][jj].copy_from(Conf.trans ? &B(0, jj, ii) : &B(0, ii, jj),
                                    aligned);
    // Matrix multiplication
    for (index_t l = 0; l < A10.cols(); ++l) {
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = {&A10(0, ii, l), aligned};
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Bij = B_reg[ii][jj];
                simd Xlj{Conf.trans ? &X01(0, jj, l) : &X01(0, l, jj), aligned};
                Bij -= Ail * Xlj;
            }
        }
    }
    // Triangular solve
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
        simd Aii = {&A(0, ii, ii), aligned};
        Aii      = simd{1} / Aii;
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd &Xij = B_reg[ii][jj];
            KOQKATOO_FULLY_UNROLLED_FOR (index_t kk = 0; kk < ii; ++kk) {
                simd Aik  = {&A(0, ii, kk), aligned};
                simd &Xkj = B_reg[kk][jj];
                Xij -= Aik * Xkj;
            }
            Xij *= Aii; // Diagonal already inverted
        }
    }
    // Store accumulator to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            B_reg[ii][jj].copy_to(Conf.trans ? &B(0, jj, ii) : &B(0, ii, jj),
                                  aligned);
}

template <class simd, KernelConfig Conf>
void trsm_register(auto L, auto B) {
#ifdef __AVX512F__
    // AVX512 has 32 vector registers, we use 25 registers for a 5×5 accumulator
    // block of matrix C (leaving some registers for loading A and B):
    static constexpr index_t RowsReg = 5, ColsReg = 5;
#else
    // AVX2 has 16 vector registers, we use 9 registers for a 3×3 accumulator
    // block of matrix C (leaving some registers for loading A and B):
    static constexpr index_t RowsReg = 3, ColsReg = 3;
#endif
    using microkernel_t =
        void (*)(decltype(L), decltype(B), decltype(L), decltype(B));
    // We need a 2D lookup table to account for all possible remainders when
    // dividing the matrix into tiles of dimensions RowsReg×ColsReg:
    static constinit auto microkernel_lut = make_2d_lut<RowsReg, ColsReg>(
        []<index_t R, index_t C>(index_constant<R>,
                                 index_constant<C>) -> microkernel_t {
            return gemm_trsm_RxC_microkernel<simd, Conf, R + 1, C + 1>;
        });
    // Loop over the columns of B.
    const index_t J = Conf.trans ? B.rows() : B.cols();
    for (index_t j = 0; j < J; j += ColsReg) {
        const index_t nj = std::min<index_t>(ColsReg, J - j);
        // Loop over the diagonal of L.
        for (index_t i = 0; i < L.rows(); i += RowsReg) {
            const index_t ni = std::min<index_t>(RowsReg, L.rows() - i);
            microkernel_lut[ni - 1][nj - 1](
                L.block(i, i, ni, ni),
                Conf.trans ? B.block(j, i, nj, ni) : B.block(i, j, ni, nj),
                L.block(i, 0, ni, i),
                Conf.trans ? B.block(j, 0, nj, i) : B.block(0, j, i, nj));
        }
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::trsm
