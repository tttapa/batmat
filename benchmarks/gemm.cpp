#include <koqkatoo/linalg-compact/compact.hpp>
#include <benchmark/benchmark.h>
#include <random>

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

template <class Abi, PreferredBackend Backend>
void dgemm(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    auto n    = static_cast<index_t>(state.range(0));
    using Mat = BatchedMatrix<real_t, index_t, stdx::simd_size<real_t, Abi>>;
    Mat A{{
        .depth = 64,
        .rows  = n,
        .cols  = n,
    }};
    Mat B{{
        .depth = 64,
        .rows  = n,
        .cols  = n,
    }};
    Mat C{{
        .depth = 64,
        .rows  = n,
        .cols  = n,
    }};
    std::ranges::generate(A, [&] { return nrml(rng); });
    std::ranges::generate(B, [&] { return nrml(rng); });
    std::ranges::generate(C, [&] { return nrml(rng); });
    for (auto _ : state)
        CompactBLAS<Abi>::xgemm_add(A, B, C, Backend);
    auto flop_cnt                 = 64e-9 * std::pow(static_cast<double>(n), 3);
    state.counters["GFLOP count"] = {flop_cnt};
    state.counters["GFLOPS"]      = {flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
}

using stdx::simd_abi::deduce_t;
using stdx::simd_abi::scalar;
using enum PreferredBackend;

#define BM_RANGES()                                                            \
    DenseRange(1, 63, 1)                                                       \
        ->DenseRange(64, 255, 4)                                               \
        ->DenseRange(256, 511, 8)                                              \
        ->MeasureProcessCPUTime()
BENCHMARK(dgemm<deduce_t<real_t, 8>, Reference>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 4>, Reference>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 2>, Reference>)->BM_RANGES();
BENCHMARK(dgemm<scalar, Reference>)->BM_RANGES();

BENCHMARK(dgemm<deduce_t<real_t, 8>, MKLAll>)->BM_RANGES();
#ifndef __AVX512F__
BENCHMARK(dgemm<deduce_t<real_t, 4>, MKLAll>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 2>, MKLAll>)->BM_RANGES();
#endif
BENCHMARK(dgemm<scalar, MKLAll>)->BM_RANGES();
