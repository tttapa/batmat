#pragma once

#include <koqkatoo/config.hpp>
#include <guanaqo/stringify.h>

#if KOQKATOO_WITH_OPENMP
#include <omp.h>
#define KOQKATOO_OMP(X) _Pragma(GUANAQO_STRINGIFY(omp X))
#define KOQKATOO_OMP_IF_ELSE(X, Y) X
#define KOQKATOO_OMP_IF(X) X
#else
#define KOQKATOO_OMP(X)
#define KOQKATOO_OMP_IF_ELSE(X, Y) Y
#define KOQKATOO_OMP_IF(X)                                                     \
    do {                                                                       \
    } while (0)
#endif
