#pragma once

#include <batmat/config.hpp>

#include <experimental/simd>

namespace batmat {
namespace stdx = std::experimental;
namespace linalg::micro_kernels::gemm {

/// Register block size of the matrix-matrix multiplication micro-kernels.
/// NEON has 32 vector registers, we use 16 registers for a 4×4 accumulator
/// block of matrix C (leaving plenty of registers for loading A and B):
/// @todo   On the Raspberry Pi 3B+ (Cortex A53) I used for testing, a 5×5 accumulator
///         was >6% slower for 15×15 matrix-matrix multiplication, and >5% slower for
///         20×20 matrices.
///         My conjecture is that since pre-loading the elements of A and B requires
///         RowsReg+ColsReg registers, the total number of registers required is then 35
///         for the 5×5 case, and the compiler prevents spilling those three extra
///         registers by interleaving the loads of A and B with FMA instructions, and
///         this is suboptimal because of the higher instruction latencies.
template <class T, class Abi>
inline constexpr index_t RowsReg = 4;
// Vectors greater than the physical vector length use more registers, so decrease the block size.
template <class T, size_t N>
    requires(N * sizeof(T) > 16)
inline constexpr index_t RowsReg<T, stdx::simd_abi::fixed_size<N>> = 3;

} // namespace linalg::micro_kernels::gemm
namespace ops {

template <class T>
inline constexpr index_t RowsRegTranspose = 4;
template <class T>
inline constexpr index_t ColsRegTranspose = 4;

} // namespace ops
} // namespace batmat
