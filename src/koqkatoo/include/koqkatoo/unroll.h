#pragma once

#include <guanaqo/stringify.h>

#ifdef NDEBUG

#ifdef __clang__
#define KOQKATOO_FULLY_UNROLL_LOOP _Pragma("clang loop unroll(full)")
#define KOQKATOO_FULLY_UNROLLED_FOR(...)                                       \
    KOQKATOO_FULLY_UNROLL_LOOP for (__VA_ARGS__)
#define KOQKATOO_FULLY_UNROLLED_IVDEP_FOR(...)                                 \
    KOQKATOO_FULLY_UNROLL_LOOP _Pragma(                                        \
        "clang loop vectorize(enable) interleave(enable)") for (__VA_ARGS__)
#define KOQKATOO_UNROLLED_IVDEP_FOR(N, ...)                                    \
    _Pragma(GUANAQO_STRINGIFY(clang loop unroll_count(N))) _Pragma(            \
        "clang loop vectorize(enable) interleave(enable)") for (__VA_ARGS__)
#else
#define KOQKATOO_FULLY_UNROLL_LOOP _Pragma("GCC unroll 99")
#define KOQKATOO_FULLY_UNROLLED_FOR(...)                                       \
    KOQKATOO_FULLY_UNROLL_LOOP for (__VA_ARGS__)
#define KOQKATOO_FULLY_UNROLLED_IVDEP_FOR(...)                                 \
    KOQKATOO_FULLY_UNROLL_LOOP _Pragma("GCC ivdep") for (__VA_ARGS__)
#define KOQKATOO_UNROLLED_IVDEP_FOR(N, ...)                                    \
    _Pragma(GUANAQO_STRINGIFY(GCC unroll N))                                   \
        _Pragma("GCC ivdep") for (__VA_ARGS__)
#endif

#else

#define KOQKATOO_FULLY_UNROLL_LOOP
#define KOQKATOO_FULLY_UNROLLED_FOR(...) for (__VA_ARGS__)
#define KOQKATOO_FULLY_UNROLLED_IVDEP_FOR(...) for (__VA_ARGS__)
#define KOQKATOO_UNROLLED_IVDEP_FOR(N, ...) for (__VA_ARGS__)

#endif
