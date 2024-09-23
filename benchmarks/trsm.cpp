#include <koqkatoo/linalg-compact/compact.hpp>
#include <benchmark/benchmark.h>
#include <random>

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

template <class Abi, PreferredBackend Backend>
void dtrsm(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    auto n    = static_cast<index_t>(state.range(0));
    using Mat = BatchedMatrix<real_t, index_t, stdx::simd_size<real_t, Abi>>;
    Mat L{{
        .depth = 64,
        .rows  = n,
        .cols  = n,
    }};
    Mat C{{
        .depth = 64,
        .rows  = n,
        .cols  = n,
    }};
    std::ranges::generate(L, [&] { return nrml(rng); });
    std::ranges::generate(C, [&] { return nrml(rng); });
    for (auto _ : state)
        CompactBLAS<Abi>::xtrsm_LLNN(L, C, Backend);
    auto flop_cnt = 64e-9 * std::pow(static_cast<double>(n), 3) / 2;
    state.counters["GFLOP count"] = {flop_cnt};
    state.counters["GFLOPS"]      = {flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
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
BENCHMARK(dtrsm<deduce_t<real_t, 8>, Reference>)->BM_RANGES();
BENCHMARK(dtrsm<deduce_t<real_t, 4>, Reference>)->BM_RANGES();
BENCHMARK(dtrsm<deduce_t<real_t, 2>, Reference>)->BM_RANGES();
BENCHMARK(dtrsm<scalar, Reference>)->BM_RANGES();

BENCHMARK(dtrsm<deduce_t<real_t, 8>, MKLAll>)->BM_RANGES();
#ifndef __AVX512F__
BENCHMARK(dtrsm<deduce_t<real_t, 4>, MKLAll>)->BM_RANGES();
#endif
#ifndef __AVX2__
BENCHMARK(dtrsm<deduce_t<real_t, 2>, MKLAll>)->BM_RANGES();
#endif
BENCHMARK(dtrsm<scalar, MKLAll>)->BM_RANGES();
