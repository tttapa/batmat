#include <koqkatoo/linalg-compact/compact.hpp>
#include <benchmark/benchmark.h>
#include <random>

#ifndef KQT_BENCHMARK_DEPTH
#define KQT_BENCHMARK_DEPTH 64
#endif

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

template <class Abi, PreferredBackend Backend>
void dtrtri(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    const index_t d = KQT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    using Mat = BatchedMatrix<real_t, index_t, stdx::simd_size<real_t, Abi>>;
    Mat L{{
        .depth = d,
        .rows  = n,
        .cols  = n,
    }};
    std::ranges::generate(L, [&] { return nrml(rng); });
    L.view.add_to_diagonal(5);
    for (auto _ : state)
        CompactBLAS<Abi>::xtrtri(L, Backend);
    const auto nd = static_cast<double>(n), dd = static_cast<double>(d);
    // TODO: double check FLOP count
    auto flop_cnt = dd * (nd + nd * (nd - 1) + nd * (nd - 1) * (nd - 2) / 6);
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"]      = {1e-9 * flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]       = {dd};
}

template <class Abi>
void dtrtri_T(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    const index_t d = KQT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    using Mat = BatchedMatrix<real_t, index_t, stdx::simd_size<real_t, Abi>>;
    Mat L{{
        .depth = d,
        .rows  = n,
        .cols  = n,
    }};
    std::ranges::generate(L, [&] { return nrml(rng); });
    L.view.add_to_diagonal(5);
    for (auto _ : state)
        for (index_t i = 0; i < L.num_batches(); ++i)
            CompactBLAS<Abi>::xtrtri_T_copy_ref(L.batch(i), L.batch(i));
    const auto nd = static_cast<double>(n), dd = static_cast<double>(d);
    // TODO: double check FLOP count
    auto flop_cnt = dd * (nd + nd * (nd - 1) + nd * (nd - 1) * (nd - 2) / 6);
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
BENCHMARK(dtrtri_T<deduce_t<real_t, 8>>)->BM_RANGES();
BENCHMARK(dtrtri<deduce_t<real_t, 8>, Reference>)->BM_RANGES();
BENCHMARK(dtrtri<deduce_t<real_t, 4>, Reference>)->BM_RANGES();
BENCHMARK(dtrtri<deduce_t<real_t, 2>, Reference>)->BM_RANGES();
BENCHMARK(dtrtri<scalar, Reference>)->BM_RANGES();
BENCHMARK(dtrtri<scalar, MKLAll>)->BM_RANGES();
