#pragma once

#include "xgemm-diag.tpp"

namespace koqkatoo::linalg::compact::micro_kernels::gemmt_diag {

using namespace koqkatoo::linalg::compact::micro_kernels::gemm_diag;

/// C = C ± A⁽ᵀ⁾ diag(d) B⁽ᵀ⁾, but accessing only the lower triangle of C.
template <class simd, KernelConfig Conf, index_t RowsReg>
void xgemmt_diag_RxR_microkernel(auto A, auto d, auto B, auto C) {
    static constexpr auto aligned = stdx::vector_aligned;
    // Load accumulator into registers
    simd C_reg[RowsReg][RowsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
            C_reg[ii][jj].copy_from(&C(0, ii, jj), aligned);
    // Actual matrix multiplication kernel
    const index_t L = Conf.trans_A ? A.rows() : A.cols();
    for (index_t l = 0; l < L; ++l) {
        simd dl = {&d(0, l, 0), aligned};
        KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = {Conf.trans_A ? &A(0, l, ii) : &A(0, ii, l), aligned};
            Ail *= dl;
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj) {
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
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
            C_reg[ii][jj].copy_to(&C(0, ii, jj), aligned);
}

template <class simd, KernelConfig Conf>
void xgemmt_diag_register(auto A, auto d, auto B, auto C) {
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
        void (*)(decltype(A), decltype(d), decltype(B), decltype(C));
    // We need a 2D lookup table to account for all possible remainders when
    // dividing the matrix into tiles of dimensions RowsReg×ColsReg:
    static constinit auto microkernel_lut = make_2d_lut<RowsReg, ColsReg>(
        []<index_t R, index_t C>(index_constant<R>,
                                 index_constant<C>) -> microkernel_t {
            return xgemm_diag_RxC_microkernel<simd, Conf, R + 1, C + 1>;
        });
    static_assert(RowsReg == ColsReg, "Non-square blocks not yet implemented");
    static constinit auto diag_microkernel_lut =
        make_1d_lut<RowsReg>([]<index_t R>(index_constant<R>) -> microkernel_t {
            return xgemmt_diag_RxR_microkernel<simd, Conf, R + 1>;
        });
    // Simply loop over the lower-triangular blocks in C.
    const index_t I = Conf.trans_A ? A.cols() : A.rows();
    for (index_t j = 0; j < C.cols(); j += ColsReg) {
        const auto nj = std::min<index_t>(ColsReg, C.cols() - j);
        for (index_t i = j; i < I; i += RowsReg) {
            const index_t ni = std::min<index_t>(RowsReg, I - i);
            if (i == j) {
                diag_microkernel_lut[ni - 1](
                    Conf.trans_A ? A.middle_cols(i, ni) : A.middle_rows(i, ni),
                    d,
                    Conf.trans_B ? B.middle_rows(j, nj) : B.middle_cols(j, nj),
                    C.block(i, j, ni, nj));
            } else {
                microkernel_lut[ni - 1][nj - 1](
                    Conf.trans_A ? A.middle_cols(i, ni) : A.middle_rows(i, ni),
                    d,
                    Conf.trans_B ? B.middle_rows(j, nj) : B.middle_cols(j, nj),
                    C.block(i, j, ni, nj));
            }
        }
    }
}

/// C = C ± A⁽ᵀ⁾ diag(d) B⁽ᵀ⁾, but accessing only the lower triangle of C.
template <class simd, KernelConfig Conf, index_t RowsReg>
void // NOLINTNEXTLINE(*-cognitive-complexity)
xgemmt_diag_RxR_microkernel_mask(auto A, auto d, auto J, auto B, auto C) {
    using mask                    = typename simd::mask_type;
    static constexpr auto aligned = stdx::vector_aligned;
    // Load accumulator into registers
    simd C_reg[RowsReg][RowsReg]; // NOLINT(*-c-arrays)
    KOQKATOO_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
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
            KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj) {
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
        KOQKATOO_FULLY_UNROLLED_FOR (index_t jj = 0; jj <= ii; ++jj)
            C_reg[ii][jj].copy_to(&C(0, ii, jj), aligned);
}

template <class simd, KernelConfig Conf>
void xgemmt_diag_register_mask(auto A, auto d, auto J, auto B, auto C) {
#ifdef __AVX512F__
    // AVX512 has 32 vector registers, we use 25 registers for a 5×5 accumulator
    // block of matrix C (leaving some registers for loading A and B):
    static constexpr index_t RowsReg = 5, ColsReg = 5;
#else
    // AVX2 has 16 vector registers, we use 9 registers for a 3×3 accumulator
    // block of matrix C (leaving some registers for loading A and B):
    static constexpr index_t RowsReg = 3, ColsReg = 3;
#endif
    using microkernel_t = void (*)(decltype(A), decltype(d), decltype(J),
                                   decltype(B), decltype(C));
    // We need a 2D lookup table to account for all possible remainders when
    // dividing the matrix into tiles of dimensions RowsReg×ColsReg:
    static constinit auto microkernel_lut = make_2d_lut<RowsReg, ColsReg>(
        []<index_t R, index_t C>(index_constant<R>,
                                 index_constant<C>) -> microkernel_t {
            return xgemm_diag_RxC_microkernel_mask<simd, Conf, R + 1, C + 1>;
        });
    static_assert(RowsReg == ColsReg, "Non-square blocks not yet implemented");
    static constinit auto diag_microkernel_lut =
        make_1d_lut<RowsReg>([]<index_t R>(index_constant<R>) -> microkernel_t {
            return xgemmt_diag_RxR_microkernel_mask<simd, Conf, R + 1>;
        });
    // Simply loop over the lower-triangular blocks in C.
    const index_t I = Conf.trans_A ? A.cols() : A.rows();
    for (index_t j = 0; j < C.cols(); j += ColsReg) {
        const auto nj = std::min<index_t>(ColsReg, C.cols() - j);
        for (index_t i = j; i < I; i += RowsReg) {
            const index_t ni = std::min<index_t>(RowsReg, I - i);
            if (i == j) {
                diag_microkernel_lut[ni - 1](
                    Conf.trans_A ? A.middle_cols(i, ni) : A.middle_rows(i, ni),
                    d, J,
                    Conf.trans_B ? B.middle_rows(j, nj) : B.middle_cols(j, nj),
                    C.block(i, j, ni, nj));
            } else {
                microkernel_lut[ni - 1][nj - 1](
                    Conf.trans_A ? A.middle_cols(i, ni) : A.middle_rows(i, ni),
                    d, J,
                    Conf.trans_B ? B.middle_rows(j, nj) : B.middle_cols(j, nj),
                    C.block(i, j, ni, nj));
            }
        }
    }
}

} // namespace koqkatoo::linalg::compact::micro_kernels::gemmt_diag
