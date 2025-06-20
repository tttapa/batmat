#pragma once

#include <batmat/config.hpp>
#include <guanaqo/stringify.h>

#if BATMAT_WITH_OPENMP
#include <omp.h>
#define BATMAT_OMP(X) _Pragma(GUANAQO_STRINGIFY(omp X))
#define BATMAT_OMP_IF_ELSE(X, Y) X
#define BATMAT_OMP_IF(X) X
#else
#define BATMAT_OMP(X)
#define BATMAT_OMP_IF_ELSE(X, Y) Y
#define BATMAT_OMP_IF(X)                                                                           \
    do {                                                                                           \
    } while (0)
#endif
