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
        CompactBLAS<Abi>::xtrsm_RLTN(L, C, Backend);
}

using stdx::simd_abi::deduce_t;
using stdx::simd_abi::scalar;
using enum PreferredBackend;

#define BM_RANGES()                                                            \
    DenseRange(1, 63)                                                          \
        ->RangeMultiplier(2)                                                   \
        ->Range(64, 512)                                                       \
        ->MeasureProcessCPUTime()
BENCHMARK(dtrsm<deduce_t<real_t, 8>, Reference>)->BM_RANGES();
BENCHMARK(dtrsm<deduce_t<real_t, 4>, Reference>)->BM_RANGES();
BENCHMARK(dtrsm<deduce_t<real_t, 2>, Reference>)->BM_RANGES();
BENCHMARK(dtrsm<scalar, Reference>)->BM_RANGES();

BENCHMARK(dtrsm<deduce_t<real_t, 8>, MKLAll>)->BM_RANGES();
BENCHMARK(dtrsm<deduce_t<real_t, 4>, MKLAll>)->BM_RANGES();
BENCHMARK(dtrsm<deduce_t<real_t, 2>, MKLAll>)->BM_RANGES();
BENCHMARK(dtrsm<scalar, MKLAll>)->BM_RANGES();
