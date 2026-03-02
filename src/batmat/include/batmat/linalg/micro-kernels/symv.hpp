#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/micro-kernels/symv/export.h>
#include <batmat/platform/platform.hpp>
#include <optional>

namespace batmat::linalg::micro_kernels::symv {

struct BATMAT_LINALG_SYMV_EXPORT KernelConfig {
    bool negate             = false;
    MatrixStructure struc_A = MatrixStructure::LowerTriangular;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder OA>
void symv_copy_microkernel(uview<const T, Abi, OA> A, uview<const T, Abi, StorageOrder::ColMajor> B,
                           std::optional<uview<const T, Abi, StorageOrder::ColMajor>> C,
                           uview<T, Abi, StorageOrder::ColMajor> D, index_t k) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA>
BATMAT_LINALG_SYMV_EXPORT void symv_copy_register(view<const T, Abi, OA> A, view<const T, Abi> B,
                                                  std::optional<view<const T, Abi>> C,
                                                  view<T, Abi> D) noexcept;

template <class T, class Abi>
constexpr index_t RowsReg = 2 * gemm::RowsReg<T, Abi>;

} // namespace batmat::linalg::micro_kernels::symv
