#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>
#include "xpotrf.hpp"

namespace koqkatoo::linalg::compact::micro_kernels::pntrf {

template <class Abi, index_t RowsReg>
void xpntrf_microkernel(mut_single_batch_matrix_accessor<Abi> A,
                        single_batch_vector_accessor<Abi> signs) noexcept;

template <class Abi, index_t RowsReg>
void xpntrf_xtrsm_microkernel(mut_single_batch_matrix_accessor<Abi> A11,
                              mut_single_batch_matrix_accessor<Abi> A21,
                              index_t k,
                              single_batch_vector_accessor<Abi> signs) noexcept;

template <class Abi, index_t RowsReg, index_t ColsReg>
void xpntrf_xsyrk_microkernel(single_batch_matrix_accessor<Abi> A21,
                              mut_single_batch_matrix_accessor<Abi> A22,
                              index_t num_rows,
                              single_batch_vector_accessor<Abi> signs) noexcept;

using potrf::RowsReg;

template <class Abi>
inline const constinit auto microkernel_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xpntrf_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_trsm_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xpntrf_xtrsm_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_syrk_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xpntrf_xsyrk_microkernel<Abi, Row + 1, RowsReg>;
    });

template <class Abi>
inline const constinit auto microkernel_syrk_lut_2 =
    make_2d_lut<RowsReg, RowsReg>(
        []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
            return xpntrf_xsyrk_microkernel<Abi, Row + 1, Col + 1>;
        });

} // namespace koqkatoo::linalg::compact::micro_kernels::pntrf
