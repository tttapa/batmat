#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/micro-kernels/trsm/export.h>
#include <batmat/platform/platform.hpp>

namespace batmat::linalg::micro_kernels::trsm {

struct BATMAT_LINALG_TRSM_EXPORT KernelConfig {
    MatrixStructure struc_A = MatrixStructure::LowerTriangular;
    index_t rotate_B        = 0;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OD>
void trsm_copy_microkernel(uview<const T, Abi, OA> A, uview<const T, Abi, OB> B,
                           uview<T, Abi, OD> D, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OD>
BATMAT_LINALG_TRSM_EXPORT void
trsm_copy_register(view<const T, Abi, OA> A, view<const T, Abi, OB> B, view<T, Abi, OD> D) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
using gemm::RowsReg;
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

} // namespace batmat::linalg::micro_kernels::trsm
