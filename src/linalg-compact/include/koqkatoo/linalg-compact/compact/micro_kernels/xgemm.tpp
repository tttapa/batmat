#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact/util.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <experimental/simd>
#include <guanaqo/mat-view.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::gemm {

namespace stdx = std::experimental;

struct KernelConfig {
    bool negate  = false;
    bool trans_A = false;
    bool trans_B = false;
};

/// C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾.
template <class simd, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xgemm_RxC_microkernel(auto A, auto B, auto C) {
    if constexpr (Conf.trans_A)
        KOQKATOO_ASSUME(A.cols() == RowsReg);
    else
        KOQKATOO_ASSUME(A.rows() == RowsReg);
    if constexpr (Conf.trans_B)
        KOQKATOO_ASSUME(B.rows() == ColsReg);
    else
        KOQKATOO_ASSUME(B.cols() == ColsReg);
    KOQKATOO_ASSUME(C.rows() == RowsReg);
    KOQKATOO_ASSUME(C.rows() == RowsReg);
    static constexpr auto aligned = stdx::vector_aligned;
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            C_reg[ii][jj].copy_from(&C(0, ii, jj), aligned);
    // Actual matrix multiplication kernel
    const index_t L = Conf.trans_A ? A.rows() : A.cols();
    for (index_t l = 0; l < L; ++l)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = {Conf.trans_A ? &A(0, l, ii) : &A(0, ii, l), aligned};
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj{Conf.trans_B ? &B(0, jj, l) : &B(0, l, jj), aligned};
                if constexpr (Conf.negate)
                    Cij -= Ail * Blj;
                else
                    Cij += Ail * Blj;
            }
        }
    // Store accumulator to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            C_reg[ii][jj].copy_to(&C(0, ii, jj), aligned);
}

template <class simd, KernelConfig Conf>
void xgemm_register(auto A, auto B, auto C) {
#ifdef __AVX512F__
    // AVX512 has 32 vector registers, we use 25 registers for a 5×5 accumulator
    // block of matrix C (leaving some registers for loading A and B):
    static constexpr index_t RowsReg = 5, ColsReg = 5;
#else
    // AVX2 has 16 vector registers, we use 9 registers for a 3×3 accumulator
    // block of matrix C (leaving some registers for loading A and B):
    static constexpr index_t RowsReg = 3, ColsReg = 3;
#endif
    using microkernel_t = void (*)(decltype(A), decltype(B), decltype(C));
    // We need a 2D lookup table to account for all possible remainders when
    // dividing the matrix into tiles of dimensions RowsReg×ColsReg:
    static constinit auto microkernel_lut = make_2d_lut<RowsReg + 3, ColsReg + 3>(
        []<index_t R, index_t C>(index_constant<R>,
                                 index_constant<C>) -> microkernel_t {
            return xgemm_RxC_microkernel<simd, Conf, R + 1, C + 1>;
        });
    // Simply loop over all blocks in the given matrices.
    const index_t I = Conf.trans_A ? A.cols() : A.rows();
    for (index_t j = 0; j < C.cols(); j += ColsReg) {
        const auto nj = std::min<index_t>(ColsReg, C.cols() - j);
        for (index_t i = 0; i < I; i += RowsReg) {
            const index_t ni = std::min<index_t>(RowsReg, I - i);
            microkernel_lut[ni - 1][nj - 1](
                Conf.trans_A ? A.middle_cols(i, ni) : A.middle_rows(i, ni),
                Conf.trans_B ? B.middle_rows(j, nj) : B.middle_cols(j, nj),
                C.block(i, j, ni, nj));
        }
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::gemm
