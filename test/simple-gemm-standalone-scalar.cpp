///usr/bin/env icpx -march=x86-64-v3 -fp-model=precise -O2 -DNDEBUG -std=c++23 -g -Wall -Wextra "$0" -o simple-gemm-standalone-scalar -DBROKEN=1 "$@" && exec ./simple-gemm-standalone-scalar || exit $?

/*
This is a simplified GEMM micro-kernel implementation that results in incorrect codegen on Intel
icpx 2025.3 at optimization levels higher than -O1. The specific issue seems to be the step that
stores the accumulator to memory: for some reason, the compiler seems to skip all rows except the
last one.

Example output:

                      +nan                      +nan                      +nan
                      +nan                      +nan                      +nan
  -2.71131653461806647e-03  +5.37292994853599479e-02  -1.58774333480441704e-01

Err: D[0,0] = +nan,     reference Dij = +2.14696802803609677e+00

Expected output:

  +2.14696802803609677e+00  -2.55100910088379207e-01  +1.87795491759574085e+00
  +6.14088366618291831e-02  -8.72430990165916853e-01  +1.21189408820919206e-01
  -2.71131653461806647e-03  +5.37292994853599479e-02  -1.58774333480441704e-01

Compiling with -O1 works around the issue. Compiling with -fsanitize=address,undefined also works
without any problems. GCC and Clang both give the correct result at all optimization levels, only
icpx does not.
*/

#pragma clang fp contract(fast)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <print>
#include <random>

#ifdef __clang__
#define FULLY_UNROLL_LOOP _Pragma("clang loop unroll(full)")
#else
#define FULLY_UNROLL_LOOP _Pragma("GCC unroll 99")
#endif
#define UNROLL_FOR(...) FULLY_UNROLL_LOOP for (__VA_ARGS__)

/** D = C + AB  or  D = AB if C is null. A is M×k, B is k×N, C and D are M×N (row-major). */
[[gnu::flatten, gnu::noinline]] void gemm_microkernel_scalar(const double *A, int ldA,
                                                             const double *B, int ldB,
                                                             const double *C, int ldC, double *D,
                                                             int ldD, int k) {
    enum { M = 3, N = 3 };
    /* Load accumulator into registers */
    double C_acc[M][N] = {};
    if (C)
        UNROLL_FOR (int ii = 0; ii < M; ++ii)
            UNROLL_FOR (int jj = 0; jj < N; ++jj)
                C_acc[ii][jj] = C[ii * ldC + jj];
    /* Rectangular matrix multiplication kernel */
    for (int l = 0; l < k; ++l)
        UNROLL_FOR (int ii = 0; ii < M; ++ii)
            UNROLL_FOR (int jj = 0; jj < N; ++jj)
                C_acc[ii][jj] += A[ii * ldA + l] * B[l * ldB + jj];
    /* Store accumulator to memory again */
#if BROKEN
    double *const D_cached[M] = {D, D + ldD, D + 2 * ldD}; /* cache row pointers */
    UNROLL_FOR (int ii = 0; ii < M; ++ii)
        UNROLL_FOR (int jj = 0; jj < N; ++jj)
            D_cached[ii][jj] = C_acc[ii][jj];
#else
    UNROLL_FOR (int ii = 0; ii < M; ++ii)
        UNROLL_FOR (int jj = 0; jj < N; ++jj)
            D[ii * ldD + jj] = C_acc[ii][jj];
#endif
}

int main() {
    static constexpr int m = 3, n = 3, k = 7;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<double> dist{-1, 1};
    double A[m * k], B[k * n], D[m * n];
    std::ranges::generate(A, [&] { return dist(rng); });
    std::ranges::generate(B, [&] { return dist(rng); });
    std::ranges::fill(D, std::numeric_limits<double>::quiet_NaN());

    gemm_microkernel_scalar(A, k, B, n, nullptr, n, D, n, k);

    const auto idx = [](int i, int j, int stride) { return i * stride + j; };
    for (int i = 0; i < m; ++i, std::println())
        for (int j = 0; j < n; ++j)
            std::print("{:+26.17e}", D[idx(i, j, n)]);
    std::println();

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double Dij = 0;
            for (int p = 0; p < k; ++p)
                Dij += A[idx(i, p, k)] * B[idx(p, j, n)];
            if (!(std::abs(D[idx(i, j, n)] - Dij) < 1e-12))
                return std::println("Err: D[{},{}] = {:+.17e},\treference Dij = {:+.17e}", i, j,
                                    D[idx(i, j, n)], Dij),
                       EXIT_FAILURE;
        }
}
