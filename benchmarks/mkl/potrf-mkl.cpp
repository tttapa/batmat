#include <benchmark/benchmark.h>
#include <mkl.h>

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

namespace flops {
struct FlopCount {
    size_t fma  = 0;
    size_t mul  = 0;
    size_t add  = 0;
    size_t div  = 0;
    size_t sqrt = 0;
};

constexpr size_t total(FlopCount c) { return c.fma + c.mul + c.add + c.div + c.sqrt; }

constexpr FlopCount potrf(size_t m, size_t n) {
    assert(m >= n);
    return {
        .fma = (n + 1) * n * (n - 1) / 6    // Schur complement (square)
               + (m - n) * n * (n - 1) / 2, //                  (bottom)
        .mul = n * (n - 1) / 2              // multiplication by inverse pivot (square)
               + (m - n) * n,               //                                 (bottom)
        .div  = n,                          // inverting pivot
        .sqrt = n,                          // square root pivot
    };
}
} // namespace flops

std::vector<double> init_pos_def_matrix(size_t n) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<double> A(n * n);
    std::generate(A.begin(), A.end(), [&] { return uni(rng); });
    // Make it diagonally dominant (positive definite)
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += 10.0 * static_cast<double>(n);
    return A;
}

static void benchmark_dpotrf(benchmark::State &state) {
    const auto n          = static_cast<size_t>(state.range(0));
    std::vector<double> A = init_pos_def_matrix(n);
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<double> B = A; // Copy to preserve input
        state.ResumeTiming();
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, B.data(), n);
    }
    auto flop_cnt                 = static_cast<double>(total(flops::potrf(n, n)));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {1.0};
}

static void dpotrf_1(benchmark::State &state) {
    mkl_set_num_threads(1);
    benchmark_dpotrf(state);
}

static void dpotrf_8(benchmark::State &state) {
    mkl_set_num_threads(8);
    benchmark_dpotrf(state);
}

#define BM_RANGES()                                                                                \
    DenseRange(1, 127, 1)                                                                          \
        ->DenseRange(128, 255, 16)                                                                 \
        ->DenseRange(256, 511, 32)                                                                 \
        ->DenseRange(512, 1023, 128)                                                               \
        ->DenseRange(1024, 8192, 1024)                                                             \
        ->MeasureProcessCPUTime()                                                                  \
        ->UseRealTime()
BENCHMARK(dpotrf_1)->BM_RANGES();
BENCHMARK(dpotrf_8)->BM_RANGES();

BENCHMARK_MAIN();
