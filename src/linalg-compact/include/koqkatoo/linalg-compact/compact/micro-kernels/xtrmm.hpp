#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>

#include "xgemm.hpp"

namespace koqkatoo::linalg::compact::micro_kernels::trmm {

struct KernelConfig {
    bool negate = false;
};

template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xtrmm_rlnn_microkernel(single_batch_matrix_accessor<Abi, false> A,
                            single_batch_matrix_accessor<Abi, false> B,
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

} // namespace koqkatoo::linalg::compact::micro_kernels::trmm
