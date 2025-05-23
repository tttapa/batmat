#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>

#include "xgemm.hpp"

namespace koqkatoo::linalg::compact::micro_kernels::trmm {

struct KernelConfig {
    bool negate  = false;
    bool trans_A = false;
    bool trans_B = false;
    int shift    = 0;
};

template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xtrmm_rlnn_microkernel(single_batch_matrix_accessor<Abi, Conf.trans_A> A,
                            single_batch_matrix_accessor<Abi, Conf.trans_B> B,
                            mut_single_batch_matrix_accessor<Abi> C, index_t k,
                            bool init_zero) noexcept;

template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xtrmm_lunn_microkernel(single_batch_matrix_accessor<Abi, Conf.trans_A> A,
                            single_batch_matrix_accessor<Abi, Conf.trans_B> B,
                            mut_single_batch_matrix_accessor<Abi> C, index_t k,
                            bool init_zero) noexcept;

using gemm::ColsReg;
using gemm::RowsReg;

// We need a 2D lookup table to account for all possible remainders when
// dividing the matrix into tiles of dimensions RowsReg×ColsReg:
template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_rlnn_lut =
    make_2d_lut<RowsReg, ColsReg>(
        []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
            return xtrmm_rlnn_microkernel<Abi, Conf, Row + 1, Col + 1>;
        });

template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_lunn_lut =
    make_2d_lut<RowsReg, ColsReg>(
        []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
            return xtrmm_lunn_microkernel<Abi, Conf, Row + 1, Col + 1>;
        });

} // namespace koqkatoo::linalg::compact::micro_kernels::trmm
