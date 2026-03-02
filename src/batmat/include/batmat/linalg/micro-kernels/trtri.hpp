#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/micro-kernels/trtri/export.h>
#include <batmat/platform/platform.hpp>

namespace batmat::linalg::micro_kernels::trtri {

struct BATMAT_LINALG_TRTRI_EXPORT KernelConfig {
    MatrixStructure struc = MatrixStructure::LowerTriangular;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder OA, StorageOrder OD>
void trtri_copy_microkernel(uview<const T, Abi, OA> A, uview<T, Abi, OD> D, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OD>
void trmm_microkernel(uview<const T, Abi, OD> Dr, uview<T, Abi, OD> D, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD>
BATMAT_LINALG_TRTRI_EXPORT void trtri_copy_register(view<const T, Abi, OA> A,
                                                    view<T, Abi, OD> D) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
using gemm::RowsReg;
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

} // namespace batmat::linalg::micro_kernels::trtri

#include <batmat/linalg/micro-kernels/trtri-decl.hpp>
