#pragma once

#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>

namespace batmat::linalg::micro_kernels::gemm {

struct KernelConfig {
    bool negate = false;
    int shift_A = 0;
    int shift_B = 0;
    int shift_C = 0;
    int shift_D = shift_C;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OC>
void gemm_microkernel(uview<const T, Abi, OA> A, uview<const T, Abi, OB> B, uview<T, Abi, OC> C,
                      index_t k, bool init_zero) noexcept;

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OC, StorageOrder OD>
void gemm_copy_microkernel(uview<const T, Abi, OA> A, uview<const T, Abi, OB> B,
                           uview<const T, Abi, OC> C, uview<T, Abi, OD> D, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC>
void gemm_register(view<const T, Abi, OA> A, view<const T, Abi, OB> B, view<T, Abi, OC> C,
                   bool init_zero) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC,
          StorageOrder OD>
void gemm_copy_register(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
                        view<const T, Abi, OC> C, view<T, Abi, OD> D) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC>
inline const constinit auto gemm_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return gemm_microkernel<T, Abi, Conf, Row + 1, Col + 1, OA, OB, OC>;
    });

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC,
          StorageOrder OD>
inline const constinit auto gemm_copy_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return gemm_copy_microkernel<T, Abi, Conf, Row + 1, Col + 1, OA, OB, OC, OD>;
    });

} // namespace batmat::linalg::micro_kernels::gemm
