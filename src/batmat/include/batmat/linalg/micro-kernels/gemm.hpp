#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/micro-kernels/gemm/export.h>
#include <batmat/platform/platform.hpp>
#include <optional>

namespace batmat::linalg::micro_kernels::gemm {

struct BATMAT_LINALG_GEMM_EXPORT KernelConfig {
    bool negate             = false;
    int shift_A             = 0;
    int rotate_B            = 0;
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
BATMAT_LINALG_GEMM_EXPORT void
gemm_copy_register(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
                   std::optional<view<const T, Abi, OC>> C, view<T, Abi, OD> D) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

namespace detail {
// Initialization of the LUT. The actual LUT is not defined here, because it needs to be exported
// in the shared library, we don't want the compiler calling the micro-kernels directly, since
// those are not exported. We could move this value to the .tpp file, but then we'd have to spell
// out the type here anyway.
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC,
          StorageOrder OD>
constexpr auto gemm_copy_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return gemm_copy_microkernel<T, Abi, Conf, Row + 1, Col + 1, OA, OB, OC, OD>;
    });
} // namespace detail

#ifndef BATMAT_LINALG_GEMM_NO_DECLARE_LUT
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC,
          StorageOrder OD>
BATMAT_LINALG_GEMM_EXPORT extern const constinit decltype(detail::gemm_copy_lut<T, Abi, Conf, OA,
                                                                                OB, OC, OD>)
    gemm_copy_lut;
#endif

} // namespace batmat::linalg::micro_kernels::gemm
