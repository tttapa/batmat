#include <koqkatoo/linalg-compact/compact.hpp>
#include <benchmark/benchmark.h>
#include <random>

#ifndef KQT_BENCHMARK_DEPTH
#define KQT_BENCHMARK_DEPTH 64
#endif

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

template <class Abi, PreferredBackend Backend, bool TransA, bool TransB>
void dgemm(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    const index_t d = KQT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    using Mat = BatchedMatrix<real_t, index_t, stdx::simd_size<real_t, Abi>>;
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
            CompactBLAS<Abi>::xgemm_TN_sub(A, B, C, Backend);
        else if constexpr (!TransA && TransB)
            CompactBLAS<Abi>::xgemm_NT_sub(A, B, C, Backend);
        else
            invalid_config(Abi{});
    const auto nd = static_cast<double>(n), dd = static_cast<double>(d);
    auto flop_cnt                 = dd * std::pow(nd, 3);
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"]      = {1e-9 * flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]       = {dd};
}

using stdx::simd_abi::deduce_t;
using stdx::simd_abi::scalar;
using enum PreferredBackend;

#define BM_RANGES()                                                            \
    DenseRange(1, 63, 1)                                                       \
        ->DenseRange(64, 127, 4)                                               \
        ->DenseRange(128, 255, 16)                                             \
        ->DenseRange(256, 512, 32)                                             \
        ->MeasureProcessCPUTime()                                              \
        ->UseRealTime()
BENCHMARK(dgemm<deduce_t<real_t, 8>, Reference, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 4>, Reference, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 2>, Reference, true, false>)->BM_RANGES();
BENCHMARK(dgemm<scalar, Reference, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, Reference, false, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 4>, Reference, false, true>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 2>, Reference, false, true>)->BM_RANGES();
BENCHMARK(dgemm<scalar, Reference, false, true>)->BM_RANGES();

BENCHMARK(dgemm<deduce_t<real_t, 8>, MKLAll, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, MKLAll, false, true>)->BM_RANGES();
#ifndef __AVX512F__
BENCHMARK(dgemm<deduce_t<real_t, 4>, MKLAll, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 4>, MKLAll, false, true>)->BM_RANGES();
#endif
#ifndef __AVX2__
BENCHMARK(dgemm<deduce_t<real_t, 2>, MKLAll, true, false>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 2>, MKLAll, false, true>)->BM_RANGES();
#endif
BENCHMARK(dgemm<scalar, MKLAll, true, false>)->BM_RANGES();
BENCHMARK(dgemm<scalar, MKLAll, false, true>)->BM_RANGES();
