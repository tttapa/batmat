#pragma once

#include <koqkatoo/config.hpp>
#include <guanaqo/mat-view.hpp>

namespace koqkatoo {

using RealMatrixView        = guanaqo::MatrixView<const real_t, index_t>;
using MutableRealMatrixView = guanaqo::MatrixView<real_t, index_t>;

} // namespace koqkatoo
