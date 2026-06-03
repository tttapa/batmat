#pragma once

#include <batmat/linalg/uview.hpp>
#include <batmat/micro-kernels/sterf/export.h>

#include <expected>

namespace batmat::linalg::micro_kernels::sterf {

struct SterfOptions {
    /// If <= 0, machine epsilon is used.
    double relative_tolerance             = 0.0;
    index_t max_iterations_per_eigenvalue = 64;
};

template <class T, class Abi>
BATMAT_LINALG_STERF_EXPORT std::expected<index_t, index_t>
sterf(view<T, Abi, StorageOrder::ColMajor> diag, view<T, Abi, StorageOrder::ColMajor> subdiag,
      SterfOptions options) noexcept;

} // namespace batmat::linalg::micro_kernels::sterf
