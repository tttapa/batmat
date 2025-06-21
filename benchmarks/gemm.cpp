#include <batmat/linalg/gemm.hpp>
#include <benchmark/benchmark.h>
#include <random>

#ifndef KQT_BENCHMARK_DEPTH
#define KQT_BENCHMARK_DEPTH 64
#endif

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

template <class Abi, StorageOrder OA, StorageOrder OB, PackingSelector PA, PackingSelector PB,
          bool Tiling>
void dgemm(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    const index_t d = KQT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    matrix<real_t, Abi, OA> A{{
        .depth = d,
        .rows  = n,
        .cols  = n,
    }};
    matrix<real_t, Abi, OB> B{{
        .depth = d,
        .rows  = n,
        .cols  = n,
    }};
    matrix<real_t, Abi> C{{
        .depth = d,
        .rows  = n,
        .cols  = n,
    }};
    std::ranges::generate(A, [&] { return nrml(rng); });
    std::ranges::generate(B, [&] { return nrml(rng); });
    std::ranges::generate(C, [&] { return nrml(rng); });
    for (auto _ : state)
        for (index_t l = 0; l < A.num_batches(); ++l)
            gemm<real_t, Abi, {}>(A.batch(l).as_const(), B.batch(l).as_const(), C.batch(l), true,
                                  {!Tiling, PA, PB});
    const auto nd = static_cast<double>(n), dd = static_cast<double>(d);
    auto flop_cnt                 = dd * std::pow(nd, 3);
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {dd};
}

using stdx::simd_abi::deduce_t;
using stdx::simd_abi::scalar;

using enum StorageOrder;
using enum PackingSelector;
#define BM_RANGES()                                                                                \
    DenseRange(1, 63, 1)                                                                           \
        ->DenseRange(64, 127, 4)                                                                   \
        ->DenseRange(128, 255, 16)                                                                 \
        ->DenseRange(256, 512, 32)                                                                 \
        ->MeasureProcessCPUTime()                                                                  \
        ->UseRealTime()
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, ColMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, RowMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, ColMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, RowMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, ColMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, RowMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, ColMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, RowMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, ColMajor, Never, Always, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, RowMajor, Never, Always, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, ColMajor, Never, Always, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, RowMajor, Never, Always, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, ColMajor, Always, Never, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, RowMajor, Always, Never, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, ColMajor, Always, Never, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, RowMajor, Always, Never, true>)->BM_RANGES();

BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, ColMajor, Never, Never, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, RowMajor, RowMajor, Never, Never, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, ColMajor, Never, Never, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, ColMajor, RowMajor, Never, Never, false>)->BM_RANGES();
