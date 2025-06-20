#pragma once

#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>

namespace batmat::linalg::micro_kernels::gemm {

struct KernelConfig {
    bool negate          = false;
    StorageOrder order_A = StorageOrder::ColMajor;
    StorageOrder order_B = StorageOrder::ColMajor;
    StorageOrder order_C = StorageOrder::ColMajor;
    StorageOrder order_D = order_C;
    int shift_A          = 0;
    int shift_B          = 0;
    int shift_C          = 0;
    int shift_D          = shift_C;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void gemm_microkernel(uview<const T, Abi, Conf.order_A> A, uview<const T, Abi, Conf.order_B> B,
                      uview<T, Abi, Conf.order_C> C, index_t k, bool init_zero) noexcept;

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void gemm_copy_microkernel(uview<const T, Abi, Conf.order_A> A, uview<const T, Abi, Conf.order_B> B,
                           uview<const T, Abi, Conf.order_C> C, uview<T, Abi, Conf.order_D> D,
                           index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf>
void gemm_register(view<const T, Abi, Conf.order_A> A, view<const T, Abi, Conf.order_B> B,
                   view<T, Abi, Conf.order_C> C, bool init_zero) noexcept;

template <class T, class Abi, KernelConfig Conf>
void gemm_copy_register(view<const T, Abi, Conf.order_A> A, view<const T, Abi, Conf.order_B> B,
                        view<const T, Abi, Conf.order_C> C, view<T, Abi, Conf.order_D> D) noexcept;

#ifdef __AVX512F__
// AVX512 has 32 vector registers, we use 25 registers for a 5×5 accumulator
// block of matrix C (leaving some registers for loading A and B):
template <class T, class Abi>
constexpr index_t RowsReg = 5;
// Vectors greater than the physical vector length use more registers, so decrease the block size.
template <class T, size_t N>
    requires(N * sizeof(T) > 64)
constexpr index_t RowsReg<T, stdx::simd_abi::fixed_size<N>> = 3;
#elif defined(__ARM_NEON)
// NEON has 32 vector registers, we use 16 registers for a 4×4 accumulator
// block of matrix C (leaving plenty of registers for loading A and B):
// On the Raspberry Pi 3B+ (Cortex A53) I used for testing, a 5×5 accumulator
// was >6% slower for 15×15 matrix-matrix multiplication, and >5% slower for
// 20×20 matrices.
// My conjecture is that since pre-loading the elements of A and B requires
// RowsReg+ColsReg registers, the total number of registers required is then 35
// for the 5×5 case, and the compiler prevents spilling those three extra
// registers by interleaving the loads of A and B with FMA instructions, and
// this is suboptimal because of the higher instruction latencies.
// TODO: re-evaluate after implementing panel-major storage format.
template <class T, class Abi>
constexpr index_t RowsReg = 4;
// Vectors greater than the physical vector length use more registers, so decrease the block size.
template <class T, size_t N>
    requires(N * sizeof(T) > 16)
constexpr index_t RowsReg<T, stdx::simd_abi::fixed_size<N>> = 3;
#else
// AVX2 has 16 vector registers, we use 9 registers for a 3×3 accumulator
// block of matrix C (leaving some registers for loading A and B):
template <class T, class Abi>
constexpr index_t RowsReg = 3;
#endif

// Square block sizes greatly simplify handling of triangular matrices.
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

template <class T, class Abi, KernelConfig Conf>
inline const constinit auto gemm_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return gemm_microkernel<T, Abi, Conf, Row + 1, Col + 1>;
    });

template <class T, class Abi, KernelConfig Conf>
inline const constinit auto gemm_copy_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return gemm_copy_microkernel<T, Abi, Conf, Row + 1, Col + 1>;
    });

} // namespace batmat::linalg::micro_kernels::gemm
