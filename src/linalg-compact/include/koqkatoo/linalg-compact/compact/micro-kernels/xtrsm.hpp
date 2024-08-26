#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::trsm {

struct KernelConfig {
    bool trans = false;
};

template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xtrsm_microkernel(single_batch_matrix_accessor<Abi> A,
                       mut_single_batch_matrix_accessor<Abi, Conf.trans> B,
                       single_batch_matrix_accessor<Abi> A10,
                       mut_single_batch_matrix_accessor<Abi, Conf.trans> X01,
                       index_t k) noexcept;

template <class Abi, KernelConfig Conf>
void xtrsm_register(single_batch_view<Abi> A,
                    mut_single_batch_view<Abi> B) noexcept;

#ifdef __AVX512F__
// AVX512 has 32 vector registers, we use 25 registers for a 5×5 accumulator
// block of matrix B (leaving some registers for loading A):
constexpr index_t RowsReg = 5, ColsReg = 5;
#else
// AVX2 has 16 vector registers, we use 9 registers for a 3×3 accumulator
// block of matrix B (leaving some registers for loading A):
constexpr index_t RowsReg = 3, ColsReg = 3;
#endif

// We need a 2D lookup table to account for all possible remainders when
// dividing the matrix into tiles of dimensions RowsReg×ColsReg:
template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_lut = make_2d_lut<RowsReg, ColsReg>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return xtrsm_microkernel<Abi, Conf, Row + 1, Col + 1>;
    });

} // namespace koqkatoo::linalg::compact::micro_kernels::trsm
