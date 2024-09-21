#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::trtri {

template <class Abi, index_t RowsReg>
void xtrtri_trmm_microkernel(mut_single_batch_matrix_accessor<Abi> A,
                             mut_single_batch_matrix_accessor<Abi> B,
                             index_t rows_B) noexcept;

template <class Abi, index_t RowsReg, index_t ColsReg>
void xtrmm_microkernel(mut_single_batch_matrix_accessor<Abi> A,
                       mut_single_batch_matrix_accessor<Abi> B,
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
inline const constinit auto microkernel_trmm_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xtrmm_microkernel<Abi, Row + 1, RowsReg>;
    });

} // namespace koqkatoo::linalg::compact::micro_kernels::trtri
