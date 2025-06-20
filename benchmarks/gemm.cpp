#include <batmat/linalg/gemm.hpp>
#include <benchmark/benchmark.h>
#include <random>

#ifndef KQT_BENCHMARK_DEPTH
#define KQT_BENCHMARK_DEPTH 64
#endif

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

template <class Abi, bool TransA, bool TransB>
void dgemm(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    const index_t d = KQT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    using Mat       = matrix<real_t, Abi>;
    Mat A{{
        .depth = d,
        .rows  = n,
        .cols  = n,
    }};
    Mat B{{
        .depth = d,
        .rows  = n,
        .cols  = n,
    }};
    Mat C{{
        .depth = d,
        .rows  = n,
        .cols  = n,
    }};
    std::ranges::generate(A, [&] { return nrml(rng); });
    std::ranges::generate(B, [&] { return nrml(rng); });
    std::ranges::generate(C, [&] { return nrml(rng); });
    for (auto _ : state)
        if constexpr (TransA && !TransB)
            for (index_t l = 0; l < A.num_batches(); ++l)
                gemm<real_t, Abi, {}>(A.batch(l).transposed().as_const(), B.batch(l).as_const(),
                                      C.batch(l), true);
        else if constexpr (!TransA && TransB)
            for (index_t l = 0; l < A.num_batches(); ++l)
                gemm<real_t, Abi, {}>(A.batch(l).as_const(), B.batch(l).transposed().as_const(),
                                      C.batch(l), true);
        else
            invalid_config(Abi{});
    const auto nd = static_cast<double>(n), dd = static_cast<double>(d);
    auto flop_cnt                 = dd * std::pow(nd, 3);
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {dd};
}

using stdx::simd_abi::deduce_t;
using stdx::simd_abi::scalar;

#define BM_RANGES()                                                                                \
    DenseRange(1, 63, 1)                                                                           \
        ->DenseRange(64, 127, 4)                                                                   \
        ->DenseRange(128, 255, 16)                                                                 \
        ->DenseRange(256, 512, 32)                                                                 \
        ->MeasureProcessCPUTime()                                                                  \
        ->UseRealTime()
BENCHMARK(dgemm<deduce_t<real_t, 8>, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 4>, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, false, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 4>, false, true>)->BM_RANGES();
#if 0
BENCHMARK(dgemm<deduce_t<real_t, 2>, true, false>)->BM_RANGES();
BENCHMARK(dgemm<scalar, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 2>, false, true>)->BM_RANGES();
BENCHMARK(dgemm<scalar, false, true>)->BM_RANGES();
#endif
