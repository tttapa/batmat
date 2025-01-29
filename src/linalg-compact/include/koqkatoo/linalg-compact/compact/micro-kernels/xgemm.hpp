#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::linalg::compact::micro_kernels::gemm {

struct KernelConfig {
    bool negate  = false;
    bool trans_A = false;
    bool trans_B = false;
    int shift    = 0;
    int shift_B  = 0;
};

template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xgemm_microkernel(single_batch_matrix_accessor<Abi, Conf.trans_A> A,
                       single_batch_matrix_accessor<Abi, Conf.trans_B> B,
                       mut_single_batch_matrix_accessor<Abi> C,
                       index_t k) noexcept;

template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xgemm_diag_microkernel(single_batch_matrix_accessor<Abi, Conf.trans_A> A,
                            single_batch_matrix_accessor<Abi, Conf.trans_B> B,
                            mut_single_batch_matrix_accessor<Abi> C, index_t k,
                            single_batch_vector_accessor<Abi> d) noexcept;

template <class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
void xgemm_diag_mask_microkernel(
    single_batch_matrix_accessor<Abi, Conf.trans_A> A,
    single_batch_matrix_accessor<Abi, Conf.trans_B> B,
    mut_single_batch_matrix_accessor<Abi> C, index_t k,
    single_batch_vector_accessor<Abi> d,
    single_batch_vector_mask_accessor<Abi> m) noexcept;

template <class Abi, KernelConfig Conf, index_t RowsReg>
void xgemmt_microkernel(single_batch_matrix_accessor<Abi, Conf.trans_A> A,
                        single_batch_matrix_accessor<Abi, Conf.trans_B> B,
                        mut_single_batch_matrix_accessor<Abi> C,
                        index_t k) noexcept;

template <class Abi, KernelConfig Conf, index_t RowsReg>
void xgemmt_diag_microkernel(single_batch_matrix_accessor<Abi, Conf.trans_A> A,
                             single_batch_matrix_accessor<Abi, Conf.trans_B> B,
                             mut_single_batch_matrix_accessor<Abi> C, index_t k,
                             single_batch_vector_accessor<Abi> d) noexcept;

template <class Abi, KernelConfig Conf, index_t RowsReg>
void xgemmt_diag_mask_microkernel(
    single_batch_matrix_accessor<Abi, Conf.trans_A> A,
    single_batch_matrix_accessor<Abi, Conf.trans_B> B,
    mut_single_batch_matrix_accessor<Abi> C, index_t k,
    single_batch_vector_accessor<Abi> d,
    single_batch_vector_mask_accessor<Abi> m) noexcept;

template <class Abi, KernelConfig Conf>
void xgemm_register(single_batch_view<Abi> A, single_batch_view<Abi> B,
                    mut_single_batch_view<Abi> C) noexcept;

template <class Abi, KernelConfig Conf>
void xgemmt_register(single_batch_view<Abi> A, single_batch_view<Abi> B,
                     mut_single_batch_view<Abi> C) noexcept;

template <class Abi, KernelConfig Conf>
void xgemmt_diag_register(single_batch_view<Abi> A, single_batch_view<Abi> B,
                          mut_single_batch_view<Abi> C,
                          single_batch_view<Abi> d) noexcept;

template <class Abi, KernelConfig Conf>
void xgemmt_diag_mask_register(single_batch_view<Abi> A,
                               single_batch_view<Abi> B,
                               mut_single_batch_view<Abi> C,
                               single_batch_view<Abi> d,
                               bool_single_batch_view<Abi> m) noexcept;

#ifdef __AVX512F__
// AVX512 has 32 vector registers, we use 25 registers for a 5×5 accumulator
// block of matrix C (leaving some registers for loading A and B):
constexpr index_t RowsReg = 5, ColsReg = 5;
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
constexpr index_t RowsReg = 5, ColsReg = 5;
#else
// AVX2 has 16 vector registers, we use 9 registers for a 3×3 accumulator
// block of matrix C (leaving some registers for loading A and B):
constexpr index_t RowsReg = 3, ColsReg = 3;
#endif

// We need a 2D lookup table to account for all possible remainders when
// dividing the matrix into tiles of dimensions RowsReg×ColsReg:
template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_lut = make_2d_lut<RowsReg, ColsReg>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return xgemm_microkernel<Abi, Conf, Row + 1, Col + 1>;
    });

template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_diag_lut =
    make_2d_lut<RowsReg, ColsReg>(
        []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
            return xgemm_diag_microkernel<Abi, Conf, Row + 1, Col + 1>;
        });

template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_diag_mask_lut =
    make_2d_lut<RowsReg, ColsReg>(
        []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
            return xgemm_diag_mask_microkernel<Abi, Conf, Row + 1, Col + 1>;
        });

template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_t_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xgemmt_microkernel<Abi, Conf, Row + 1>;
    });

template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_t_diag_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xgemmt_diag_microkernel<Abi, Conf, Row + 1>;
    });

template <class Abi, KernelConfig Conf>
inline const constinit auto microkernel_t_diag_mask_lut =
    make_1d_lut<RowsReg>([]<index_t Row>(index_constant<Row>) {
        return xgemmt_diag_mask_microkernel<Abi, Conf, Row + 1>;
    });

} // namespace koqkatoo::linalg::compact::micro_kernels::gemm
