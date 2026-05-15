#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/micro-kernels/geqrf/export.h>
#include <batmat/platform/platform.hpp>
#include <batmat/simd.hpp>

namespace batmat::linalg::micro_kernels::geqrf {

struct BATMAT_LINALG_GEQRF_EXPORT KernelConfig {};

template <class T, class Abi>
inline constexpr index_t SizeR = gemm::RowsReg<T, Abi>; // TODO
template <class T, class Abi>
inline constexpr index_t SizeS = gemm::RowsReg<T, Abi>; // TODO

template <class T, class Abi, KernelConfig Conf, index_t R, StorageOrder OA, StorageOrder OD>
void geqrf_diag_microkernel(index_t k, triangular_accessor<T, Abi, SizeR<T, Abi>> W,
                            uview<const T, Abi, OA> A, uview<T, Abi, OD> D) noexcept;

template <class T, class Abi, KernelConfig Conf, index_t R, StorageOrder OA, StorageOrder OD>
void geqrf_full_microkernel(index_t k, uview<const T, Abi, OA> A, uview<T, Abi, OD> D) noexcept;

template <class T, class Abi, KernelConfig Conf, index_t R, index_t S, StorageOrder OA,
          StorageOrder OD, StorageOrder OB>
void geqrf_tail_microkernel(index_t k, bool transposed,
                            triangular_accessor<const T, Abi, SizeR<T, Abi>> W,
                            uview<const T, Abi, OA> A, uview<T, Abi, OD> D,
                            uview<const T, Abi, OB> B) noexcept;

// Helper function to compute size of the storage for the matrix W (part of the block
// Householder representation).
template <class T, class Abi, StorageOrder OA>
constexpr std::pair<index_t, index_t> geqrf_W_size(view<T, Abi, OA> A) {
    static constexpr index_constant<SizeR<std::remove_const_t<T>, Abi>> R;
    using W_t = triangular_accessor<std::remove_const_t<T>, Abi, R>;
    return {W_t::num_elem_per_layer(), (A.cols() + R - 1) / R};
}

// Low-level register-blocked routines
template <class T, class Abi, KernelConfig Conf = {}, StorageOrder OA = StorageOrder::ColMajor,
          StorageOrder OD = StorageOrder::ColMajor>
BATMAT_LINALG_GEQRF_EXPORT void geqrf_copy_register(view<const T, Abi, OA> A, view<T, Abi, OD> D,
                                                    view<T, Abi> W) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OD, StorageOrder OB>
void geqrf_apply_register(view<const T, Abi, OA> A, view<T, Abi, OD> D, view<const T, Abi, OB> B,
                          view<const T, Abi> W, bool transposed) noexcept;

} // namespace batmat::linalg::micro_kernels::geqrf
