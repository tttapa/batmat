#pragma once

#include <batmat/config.hpp>

#include <experimental/simd>

namespace batmat {
namespace stdx = std::experimental;
namespace linalg::micro_kernels::gemm {

/// Register block size of the matrix-matrix multiplication micro-kernels.
/// AVX2 has 16 vector registers, we use 9 registers for a 3×3 accumulator
/// block of matrix C (leaving some registers for loading A and B):
template <class T, class Abi>
inline constexpr index_t RowsReg = 3;
// Vectors greater than the physical vector length use more registers, so decrease the block size.
template <class T, size_t N>
    requires(N * sizeof(T) > 32)
inline constexpr index_t RowsReg<T, stdx::simd_abi::fixed_size<N>> = 2;

} // namespace linalg::micro_kernels::gemm
namespace ops {

template <class T>
inline constexpr index_t RowsRegTranspose = 4;
template <class T>
inline constexpr index_t ColsRegTranspose = 4;

} // namespace ops
} // namespace batmat
