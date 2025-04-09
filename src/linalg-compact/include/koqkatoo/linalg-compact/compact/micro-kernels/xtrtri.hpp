#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::trtri {

template <class Abi, index_t RowsReg>
void xtrtri_trmm_microkernel(mut_single_batch_matrix_accessor<Abi> A,
                             index_t rows) noexcept;

template <class Abi, index_t RowsReg, bool TransOut = false>
void xtrtri_trmm_copy_microkernel(
    single_batch_matrix_accessor<Abi> Ain,
    mut_single_batch_matrix_accessor<Abi, TransOut> Aout,
    index_t rows) noexcept;

template <class Abi, index_t RowsReg, index_t ColsReg>
void xtrmm_microkernel(single_batch_matrix_accessor<Abi> A,
                       mut_single_batch_matrix_accessor<Abi> B,
                       index_t rows) noexcept;

template <class Abi, index_t RowsReg>
void xtrtri_trmm_copy_T_microkernel(single_batch_matrix_accessor<Abi> Ain,
                                    mut_single_batch_matrix_accessor<Abi> Aout,
                                    index_t rows) noexcept;

template <class Abi, index_t RowsReg, index_t ColsReg>
void xtrmm_copy_T_microkernel(single_batch_matrix_accessor<Abi> A,
                              single_batch_matrix_accessor<Abi> BT,
                              mut_single_batch_matrix_accessor<Abi> C,
                              index_t rows) noexcept;

#ifdef __AVX512F__
// AVX512 has 32 vector registers, TODO:
static constexpr index_t RowsReg = 5;
#elif defined(__ARM_NEON)
// NEON has 32 vector registers, TODO:
static constexpr index_t RowsReg = 5;
#else
// AVX2 has 16 vector registers, TODO:
static constexpr index_t RowsReg = 4;
#endif

template <class Abi>
inline const constinit auto microkernel_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xtrtri_trmm_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_copy_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xtrtri_trmm_copy_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_trmm_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xtrmm_microkernel<Abi, Row + 1, RowsReg>;
    });

template <class Abi>
inline const constinit auto microkernel_T_copy_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xtrtri_trmm_copy_T_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_trmm_T_copy_lut =
    make_1d_lut<RowsReg>([]<index_t Col>(index_constant<Col>) {
        return xtrmm_copy_T_microkernel<Abi, RowsReg, Col + 1>;
    });

} // namespace koqkatoo::linalg::compact::micro_kernels::trtri
