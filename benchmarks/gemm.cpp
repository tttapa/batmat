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
}

using stdx::simd_abi::deduce_t;
using stdx::simd_abi::scalar;
using enum PreferredBackend;

#define BM_RANGES()                                                            \
    DenseRange(1, 31)                                                          \
        ->RangeMultiplier(2)                                                   \
        ->Range(32, 1024)                                                      \
        ->MeasureProcessCPUTime()
// BENCHMARK(dgemm<deduce_t<real_t, 16>, Reference>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, Reference>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 4>, Reference>)->BM_RANGES();
BENCHMARK(dgemm<scalar, Reference>)->BM_RANGES();

// BENCHMARK(dgemm<deduce_t<real_t, 16>, MKLAll>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 8>, MKLAll>)->BM_RANGES();
BENCHMARK(dgemm<deduce_t<real_t, 4>, MKLAll>)->BM_RANGES();
BENCHMARK(dgemm<scalar, MKLAll>)->BM_RANGES();

BENCHMARK_MAIN();
