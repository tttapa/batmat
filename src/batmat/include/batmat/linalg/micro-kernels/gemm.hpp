#pragma once

#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>

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
