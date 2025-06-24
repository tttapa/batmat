#pragma once

#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>
#include "batmat/linalg/structure.hpp"

namespace batmat::linalg::micro_kernels::trsm {

struct KernelConfig {
    MatrixStructure struc_A = MatrixStructure::LowerTriangular;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OD>
void trsm_copy_microkernel(uview<const T, Abi, OA> A, uview<const T, Abi, OB> B,
                           uview<T, Abi, OD> D, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OD>
void trsm_copy_register(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
                        view<T, Abi, OD> D) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
using gemm::RowsReg;
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OD>
inline const constinit auto trsm_copy_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return trsm_copy_microkernel<T, Abi, Conf, Row + 1, Col + 1, OA, OB, OD>;
    });

} // namespace batmat::linalg::micro_kernels::trsm
