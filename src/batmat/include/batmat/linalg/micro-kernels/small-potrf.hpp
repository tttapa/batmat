#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/micro-kernels/small-potrf/export.h>
#include <batmat/platform/platform.hpp>

namespace batmat::linalg::micro_kernels::small_potrf {

template <class T>
using scalar_view = uview<T, datapar::scalar_abi<std::remove_const_t<T>>, StorageOrder::ColMajor>;

template <class T, index_t NC> // number of columns to handle at once
void potrf_trsm_microkernel(index_t k, scalar_view<const T> A, scalar_view<T> L) noexcept;

template <class T, index_t RowsReg, index_t ColsReg>
void potrf_syrk_microkernel(index_t k, scalar_view<const T> L21, scalar_view<const T> A22,
                            scalar_view<T> L22) noexcept;

template <class T, index_t RowsReg = 4>
BATMAT_LINALG_SMALL_POTRF_EXPORT void small_potrf(view<const T, datapar::scalar_abi<T>> A,
                                                  view<T, datapar::scalar_abi<T>> L,
                                                  index_t n = -1) noexcept;

template <class T, index_t NC, index_t NR>
BATMAT_LINALG_SMALL_POTRF_EXPORT void
syrk_potrf_trsm_microkernel(index_t m, index_t k, scalar_view<const T> L21,
                            scalar_view<const T> A22, scalar_view<T> L22) noexcept;

template <class T, index_t RowsReg = 4, index_t S = 8>
BATMAT_LINALG_SMALL_POTRF_EXPORT void small_potrf_left(view<const T, datapar::scalar_abi<T>> A,
                                                       view<T, datapar::scalar_abi<T>> L) noexcept;

} // namespace batmat::linalg::micro_kernels::small_potrf
