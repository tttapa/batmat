#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>

namespace batmat::linalg::micro_kernels::trtri {

struct KernelConfig {
    MatrixStructure struc = MatrixStructure::LowerTriangular;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder OA, StorageOrder OD>
void trtri_copy_microkernel(uview<const T, Abi, OA> A, uview<T, Abi, OD> D, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OD>
void trmm_microkernel(uview<const T, Abi, OD> Dr, uview<T, Abi, OD> D, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD>
void trtri_copy_register(view<const T, Abi, OA> A, view<T, Abi, OD> D) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
using gemm::RowsReg;
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD>
inline const constinit auto trtri_copy_lut =
    make_1d_lut<RowsReg<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return trtri_copy_microkernel<T, Abi, Conf, Row + 1, OA, OD>;
    });

template <class T, class Abi, KernelConfig Conf, StorageOrder OD>
inline const constinit auto trmm_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return trmm_microkernel<T, Abi, Conf, Row + 1, Col + 1, OD>;
    });

} // namespace batmat::linalg::micro_kernels::trtri
