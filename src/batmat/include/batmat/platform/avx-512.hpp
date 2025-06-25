#pragma once

#include <batmat/config.hpp>
#include <batmat/simd.hpp>

namespace batmat {
namespace linalg::micro_kernels::gemm {

/// Register block size of the matrix-matrix multiplication micro-kernels.
/// AVX-512 has 32 vector registers, we use 25 registers for a 5×5 accumulator
/// block of matrix C (leaving some registers for loading A and B):
template <class T, class Abi>
inline constexpr index_t RowsReg = 5;
// Vectors greater than the physical vector length use more registers, so decrease the block size.
template <class T, class Abi>
    requires(datapar::simd_size<T, Abi>::value * sizeof(T) > 64)
inline constexpr index_t RowsReg<T, Abi> = 3;

} // namespace linalg::micro_kernels::gemm
namespace ops {

template <class T>
inline constexpr index_t RowsRegTranspose = 8;
template <class T>
inline constexpr index_t ColsRegTranspose = 8;

// TODO: we're using the AVX2 implementation for now.
template <>
inline constexpr index_t RowsRegTranspose<double> = 4;
template <>
inline constexpr index_t ColsRegTranspose<double> = 4;

} // namespace ops
} // namespace batmat
