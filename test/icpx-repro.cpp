///usr/bin/env icpx -march=x86-64-v3 -O2 -std=c++23 -Wall -Wextra "$0" -o icpx-repro -DBROKEN=1 "$@" && exec ./icpx-repro || exit $?

/*
This is a simple function to set a matrix to zero that results in incorrect codegen on Intel icpx
2025.3 at optimization levels higher than -O1. The specific issue seems to be the stores to all rows
except the last one are optimized away.
Compiling with -O1 works around the issue. Compiling with -fsanitize=address,undefined also works
without any problems. GCC 14 and Clang 21 produce the correct results at all optimization levels.
*/

#ifdef __clang__
#define FULLY_UNROLL_LOOP _Pragma("clang loop unroll(full)")
#else
#define FULLY_UNROLL_LOOP _Pragma("GCC unroll 99")
#endif
#define UNROLL_FOR(...) FULLY_UNROLL_LOOP for (__VA_ARGS__)

enum { M = 3, N = 3 };
[[gnu::noinline]] void matrix_zero(double *A, int ldA) {
#if BROKEN
    double *const A_cached[M] = {A, A + ldA, A + 2 * ldA}; /* cache row pointers */
    UNROLL_FOR (int ii = 0; ii < M; ++ii)
        UNROLL_FOR (int jj = 0; jj < N; ++jj)
            A_cached[ii][jj] = 0;
#else
    UNROLL_FOR (int ii = 0; ii < M; ++ii)
        UNROLL_FOR (int jj = 0; jj < N; ++jj)
            A[ii * ldA + jj] = 0;
#endif
}

#include <algorithm>
#include <print>

int main() {
    double A[M * N];
    std::ranges::fill(A, -1);

    matrix_zero(A, N);

    for (int i = 0; i < M; ++i, std::println())
        for (int j = 0; j < N; ++j)
            std::print("{:7.2f}", A[i * N + j]);
    return std::ranges::all_of(A, [](auto x) { return x == 0; }) ? 0 : 1;
}
