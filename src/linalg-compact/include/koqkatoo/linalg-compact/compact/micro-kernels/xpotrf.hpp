#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::potrf {

template <class Abi, index_t RowsReg>
void xpotrf_microkernel(mut_single_batch_matrix_accessor<Abi> A) noexcept;

template <class Abi, index_t RowsReg>
void xpotrf_xtrsm_microkernel(mut_single_batch_matrix_accessor<Abi> A11,
                              mut_single_batch_matrix_accessor<Abi> A21,
                              index_t k) noexcept;

template <class Abi, index_t RowsReg, index_t ColsReg>
void xpotrf_xsyrk_microkernel(single_batch_matrix_accessor<Abi> A21,
                              mut_single_batch_matrix_accessor<Abi> A22,
                              index_t num_rows) noexcept;

#ifdef __AVX512F__
// AVX512 has 32 vector registers, we use 25 to cache part of the A21 block of
// the matrix:
static constexpr index_t RowsReg = 5;
#elif defined(__ARM_NEON)
// NEON has 32 vector registers, we use 25 to cache part of the A21 block of
// the matrix.
// On the Raspberry Pi 3B+ (Cortex A53) I used for testing, a 4×4 accumulator is
// around 1% slower than a 5×5 accumulator for 20×20 matrices.
static constexpr index_t RowsReg = 5;
#else
// AVX2 has 16 vector registers, we (try to) use 16 to cache part of the A21
// block of the matrix, and spill some elements to load the other data. This
// seems to be ~10% faster than 3×3 on an Intel i7-10750H.
static constexpr index_t RowsReg = 4;
#endif

template <class Abi>
inline const constinit auto microkernel_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xpotrf_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_trsm_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xpotrf_xtrsm_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_syrk_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xpotrf_xsyrk_microkernel<Abi, Row + 1, RowsReg>;
    });

} // namespace koqkatoo::linalg::compact::micro_kernels::potrf
