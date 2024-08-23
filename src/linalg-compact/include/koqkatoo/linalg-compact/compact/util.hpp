#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/lut.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/unroll.h>

#include <experimental/simd>

#if KOQKATOO_WITH_OPENMP
#include <omp.h>
#endif

namespace koqkatoo::linalg::compact {

namespace stdx = std::experimental;

using scalar_abi = stdx::simd_abi::scalar;

} // namespace koqkatoo::linalg::compact
