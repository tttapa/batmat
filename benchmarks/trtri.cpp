#include <koqkatoo/linalg-compact/compact.hpp>
#include <benchmark/benchmark.h>
#include <random>

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

template <class Abi, PreferredBackend Backend>
void dtrtri(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    auto n    = static_cast<index_t>(state.range(0));
    using Mat = BatchedMatrix<real_t, index_t, stdx::simd_size<real_t, Abi>>;
    Mat L{{
        .depth = 64,
        .rows  = n,
        .cols  = n,
    }};
    std::ranges::generate(L, [&] { return nrml(rng); });
    L.view.add_to_diagonal(5);
    for (auto _ : state)
        CompactBLAS<Abi>::xtrtri(L, Backend);
    auto nd = static_cast<double>(n);
    // TODO: double check FLOP count
    auto flop_cnt = 64e-9 * (nd + nd * (nd - 1) + nd * (nd - 1) * (nd - 2) / 6);
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
        ->DenseRange(256, 512, 8)                                              \
        ->MeasureProcessCPUTime()                                              \
        ->UseRealTime()
BENCHMARK(dtrtri<deduce_t<real_t, 8>, Reference>)->BM_RANGES();
BENCHMARK(dtrtri<deduce_t<real_t, 4>, Reference>)->BM_RANGES();
BENCHMARK(dtrtri<deduce_t<real_t, 2>, Reference>)->BM_RANGES();
BENCHMARK(dtrtri<scalar, Reference>)->BM_RANGES();
BENCHMARK(dtrtri<scalar, MKLAll>)->BM_RANGES();
