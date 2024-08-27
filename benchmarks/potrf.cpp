#include <koqkatoo/linalg-compact/compact.hpp>
#include <benchmark/benchmark.h>
#include <random>

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

template <class Abi, PreferredBackend Backend>
void dpotrf(benchmark::State &state) {
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
    CompactBLAS<Abi>::xgemm_TN(L, L, C, Backend);
    for (auto _ : state) {
        state.PauseTiming();
        CompactBLAS<Abi>::xcopy(C, L);
        state.ResumeTiming();
        benchmark::DoNotOptimize(L.data());
        CompactBLAS<Abi>::xpotrf(L, Backend);
        benchmark::ClobberMemory();
    }
    auto flop_cnt = 64e-9 * std::pow(static_cast<double>(n), 3) / 6;
    state.counters["GFLOP count"] = {flop_cnt};
    state.counters["GFLOPS"]      = {flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
}

template <class Abi, PreferredBackend Backend>
void dpotrf_recursive(benchmark::State &state) {
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
    CompactBLAS<Abi>::xgemm_TN(L, L, C, Backend);
    for (auto _ : state) {
        state.PauseTiming();
        CompactBLAS<Abi>::xcopy(C, L);
        state.ResumeTiming();
        benchmark::DoNotOptimize(L.data());
        CompactBLAS<Abi>::xpotrf_recursive(L, Backend);
        benchmark::ClobberMemory();
    }
    auto flop_cnt = 64e-9 * std::pow(static_cast<double>(n), 3) / 6;
    state.counters["GFLOP count"] = {flop_cnt};
    state.counters["GFLOPS"]      = {flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
}

template <class Abi, PreferredBackend Backend>
void dpotrf_base(benchmark::State &state) {
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
    CompactBLAS<Abi>::xgemm_TN(L, L, C, Backend);
    for (auto _ : state) {
        state.PauseTiming();
        CompactBLAS<Abi>::xcopy(C, L);
        state.ResumeTiming();
        benchmark::DoNotOptimize(L.data());
        CompactBLAS<Abi>::xpotrf_base(L, Backend);
        benchmark::ClobberMemory();
    }
    auto flop_cnt = 64e-9 * std::pow(static_cast<double>(n), 3) / 6;
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
BENCHMARK(dpotrf<deduce_t<real_t, 8>, Reference>)->BM_RANGES();
BENCHMARK(dpotrf<deduce_t<real_t, 4>, Reference>)->BM_RANGES();
BENCHMARK(dpotrf<deduce_t<real_t, 2>, Reference>)->BM_RANGES();
BENCHMARK(dpotrf<scalar, Reference>)->BM_RANGES();

BENCHMARK(dpotrf<deduce_t<real_t, 8>, MKLAll>)->BM_RANGES();
#ifndef __AVX512F__
BENCHMARK(dpotrf<deduce_t<real_t, 4>, MKLAll>)->BM_RANGES();
BENCHMARK(dpotrf<deduce_t<real_t, 2>, MKLAll>)->BM_RANGES();
#endif
BENCHMARK(dpotrf<scalar, MKLAll>)->BM_RANGES();
