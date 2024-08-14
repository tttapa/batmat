#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/stringify.h>

#if KOQKATOO_WITH_OPENMP
#include <omp.h>
#define KOQKATOO_OMP(X) _Pragma(KOQKATOO_STRINGIFY(omp X))
#define KOQKATOO_OMP_IF_ELSE(X, Y) X
#else
#define KOQKATOO_OMP(X)
#define KOQKATOO_OMP_IF_ELSE(X, Y) Y
#endif
