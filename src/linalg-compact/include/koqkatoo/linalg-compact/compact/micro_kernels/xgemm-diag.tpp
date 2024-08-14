#pragma once

#include "xgemm.tpp"

namespace koqkatoo::linalg::compact::micro_kernels::gemm_diag {

using namespace koqkatoo::linalg::compact::micro_kernels::gemm;

/// C = C ± A⁽ᵀ⁾ diag(d) B⁽ᵀ⁾.
template <class simd, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xgemm_diag_RxC_microkernel(auto A, auto d, auto B, auto C) {
    static constexpr auto aligned = stdx::vector_aligned;
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            C_reg[ii][jj].copy_from(&C(0, ii, jj), aligned);
    // Actual matrix multiplication kernel
    const index_t L = Conf.trans_A ? A.rows() : A.cols();
    for (index_t l = 0; l < L; ++l) {
        simd dl = {&d(0, l, 0), aligned};
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = {Conf.trans_A ? &A(0, l, ii) : &A(0, ii, l), aligned};
            Ail *= dl;
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj{Conf.trans_B ? &B(0, jj, l) : &B(0, l, jj), aligned};
                if constexpr (Conf.negate)
                    Cij -= Ail * Blj;
                else
                    Cij += Ail * Blj;
            }
        }
    }
    // Store accumulator to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            C_reg[ii][jj].copy_to(&C(0, ii, jj), aligned);
}

template <class simd, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void // NOLINTNEXTLINE(*-cognitive-complexity)
xgemm_diag_RxC_microkernel_mask(auto A, auto d, auto J, auto B, auto C) {
    using mask                    = typename simd::mask_type;
    static constexpr auto aligned = stdx::vector_aligned;
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            C_reg[ii][jj].copy_from(&C(0, ii, jj), aligned);
    // Actual matrix multiplication kernel
    const index_t L = Conf.trans_A ? A.rows() : A.cols();
    for (index_t l = 0; l < L; ++l) {
        simd dl = {&d(0, l, 0), aligned};
        mask Jl = {&J(0, l, 0), aligned};
        if (none_of(Jl))
            continue;
        where(!Jl, dl) = simd{0};
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = {Conf.trans_A ? &A(0, l, ii) : &A(0, ii, l), aligned};
            Ail *= dl;
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj{Conf.trans_B ? &B(0, jj, l) : &B(0, l, jj), aligned};
                if constexpr (Conf.negate)
                    Cij -= Ail * Blj;
                else
                    Cij += Ail * Blj;
            }
        }
    }
    // Store accumulator to memory again
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            C_reg[ii][jj].copy_to(&C(0, ii, jj), aligned);
}

} // namespace koqkatoo::linalg::compact::micro_kernels::gemm_diag
