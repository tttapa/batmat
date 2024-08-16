#pragma once

#include <koqkatoo/linalg-compact/compact-new/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::potrf {

template <class Abi, index_t RowsReg>
void xpotrf_microkernel(mut_single_batch_matrix_accessor<Abi> A) noexcept;

#ifdef __AVX512F__
// AVX512 has 32 vector registers, we use 28 to cache the lower triangle of
// the matrix:
static constexpr index_t RowsReg = 7;
#else
// AVX2 has 16 vector registers, we use 15 to cache the lower triangle of
// the matrix:
static constexpr index_t RowsReg = 5;
#endif

template <class Abi>
inline const constinit auto microkernel_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xpotrf_microkernel<Abi, Row + 1>;
    });

} // namespace koqkatoo::linalg::compact::micro_kernels::potrf
