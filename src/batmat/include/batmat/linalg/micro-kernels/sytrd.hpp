#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/micro-kernels/sytrd/export.h>
#include <batmat/platform/platform.hpp>
#include <batmat/simd.hpp>

namespace batmat::linalg::micro_kernels::sytrd {

struct BATMAT_LINALG_SYTRD_EXPORT KernelConfig {};

template <class T, class Abi>
inline constexpr index_t SizeR = gemm::RowsReg<T, Abi>; // TODO

template <class T, class Abi, KernelConfig Conf, index_t R, StorageOrder OD>
void sytrd_diag_microkernel(index_t k, triangular_accessor<T, Abi, SizeR<T, Abi>> W,
                            uview<T, Abi, OD> D, uview<T, Abi, StorageOrder::ColMajor> Y) noexcept;

// Helper function to compute size of the storage for the matrix W (part of the block
// Householder representation).
template <class T, class Abi, StorageOrder OD>
constexpr std::pair<index_t, index_t> sytrd_W_size(view<T, Abi, OD> D) {
    static constexpr index_constant<SizeR<std::remove_const_t<T>, Abi>> R;
    using W_t = triangular_accessor<std::remove_const_t<T>, Abi, R>;
    return {W_t::num_elem_per_layer(), (std::max<index_t>(D.cols(), 1) - 1 + R - 1) / R};
}

// Helper function to compute size of the storage for the work matrix Y.
template <class T, class Abi, StorageOrder OD>
constexpr std::pair<index_t, index_t> sytrd_Y_size(view<T, Abi, OD> D) {
    static constexpr index_constant<SizeR<std::remove_const_t<T>, Abi>> R;
    return {D.rows(), R};
}

// Low-level register-blocked routines
template <class T, class Abi, KernelConfig Conf = {}, StorageOrder OD = StorageOrder::ColMajor>
BATMAT_LINALG_SYTRD_EXPORT void sytrd_register(view<T, Abi, OD> D, view<T, Abi> W,
                                               view<T, Abi> Y) noexcept;

} // namespace batmat::linalg::micro_kernels::sytrd
