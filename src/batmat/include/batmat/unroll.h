#pragma once

#include <guanaqo/stringify.h>

#ifdef NDEBUG

#ifdef __clang__
#define BATMAT_FULLY_UNROLL_LOOP _Pragma("clang loop unroll(full)")
#define BATMAT_FULLY_UNROLLED_FOR(...) BATMAT_FULLY_UNROLL_LOOP for (__VA_ARGS__)
#define BATMAT_FULLY_UNROLLED_IVDEP_FOR(...)                                                       \
    BATMAT_FULLY_UNROLL_LOOP _Pragma("clang loop interleave(enable)") for (__VA_ARGS__)
#define BATMAT_UNROLLED_IVDEP_FOR(N, ...)                                                          \
    _Pragma(GUANAQO_STRINGIFY(clang loop unroll_count(N)))                                         \
        _Pragma("clang loop interleave(enable)") for (__VA_ARGS__)
#else
#define BATMAT_FULLY_UNROLL_LOOP _Pragma("GCC unroll 99")
#define BATMAT_FULLY_UNROLLED_FOR(...) BATMAT_FULLY_UNROLL_LOOP for (__VA_ARGS__)
#define BATMAT_FULLY_UNROLLED_IVDEP_FOR(...)                                                       \
    BATMAT_FULLY_UNROLL_LOOP _Pragma("GCC ivdep") for (__VA_ARGS__)
#define BATMAT_UNROLLED_IVDEP_FOR(N, ...)                                                          \
    _Pragma(GUANAQO_STRINGIFY(GCC unroll N)) _Pragma("GCC ivdep") for (__VA_ARGS__)
#endif

#else

#define BATMAT_FULLY_UNROLL_LOOP
#define BATMAT_FULLY_UNROLLED_FOR(...) for (__VA_ARGS__)
#define BATMAT_FULLY_UNROLLED_IVDEP_FOR(...) for (__VA_ARGS__)
#define BATMAT_UNROLLED_IVDEP_FOR(N, ...) for (__VA_ARGS__)

#endif
