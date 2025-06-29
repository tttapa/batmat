#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>
#include <optional>

namespace batmat::linalg::micro_kernels::gemm {

struct KernelConfig {
    bool negate             = false;
    int shift_A             = 0;
    int shift_B             = 0;
    int rotate_C            = 0;
    int rotate_D            = rotate_C;
    int mask_D              = rotate_D;
    MatrixStructure struc_A = MatrixStructure::General;
    MatrixStructure struc_B = MatrixStructure::General;
    MatrixStructure struc_C = MatrixStructure::General;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OC, StorageOrder OD>
void gemm_copy_microkernel(uview<const T, Abi, OA> A, uview<const T, Abi, OB> B,
                           std::optional<uview<const T, Abi, OC>> C, uview<T, Abi, OD> D,
                           index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC,
          StorageOrder OD>
void gemm_copy_register(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
                        std::optional<view<const T, Abi, OC>> C, view<T, Abi, OD> D) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC,
          StorageOrder OD>
inline const constinit auto gemm_copy_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return gemm_copy_microkernel<T, Abi, Conf, Row + 1, Col + 1, OA, OB, OC, OD>;
    });

} // namespace batmat::linalg::micro_kernels::gemm
