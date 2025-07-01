#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>

namespace batmat::linalg::micro_kernels::syomv {

struct KernelConfig {
    bool negate             = false;
    MatrixStructure struc_A = MatrixStructure::LowerTriangular;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder OA, StorageOrder OB,
          StorageOrder OD>
void syomv_microkernel(uview<const T, Abi, OA> A, uview<const T, Abi, OB> B, uview<T, Abi, OD> D,
                       index_t l0, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OD>
void syomv_register(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
                    view<T, Abi, OD> D) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
using gemm::RowsReg;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OD>
inline const constinit auto syomv_lut =
    make_1d_lut<RowsReg<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return syomv_microkernel<T, Abi, Conf, Row + 1, OA, OB, OD>;
    });

} // namespace batmat::linalg::micro_kernels::syomv
