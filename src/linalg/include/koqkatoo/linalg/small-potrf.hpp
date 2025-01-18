#pragma once

#include <koqkatoo/config.hpp>

namespace koqkatoo::linalg {

/// Cholesky factorization for small matrices.
/// @param L    Pointer to lower triangular matrix.
/// @param ldL  Leading dimension or outer stride of L.
/// @param m    Number of rows of L.
/// @param N    Number of columns of L.
/// @param n    Number of columns to factorize (to compute Schur complement).
///             Negative means all columns.
template <index_t R = 4>
void small_potrf(real_t *L, index_t ldL, index_t m, index_t N, index_t n = -1);

} // namespace koqkatoo::linalg
