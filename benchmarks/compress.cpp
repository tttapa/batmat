#include <batmat/linalg/compress.hpp>
#include <benchmark/benchmark.h>
#include <random>

using batmat::index_t;
using batmat::real_t;
using batmat::linalg::StorageOrder;

template <class Abi, StorageOrder OA>
constexpr auto compress = [](benchmark::State &state) {
    using namespace batmat::linalg;
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution brnl{0.5};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    matrix<real_t, Abi> Σi{{.depth = d, .rows = n, .cols = 1}};
    matrix<real_t, Abi> Σo{{.depth = d, .rows = n, .cols = 1}};
    matrix<real_t, Abi, OA> Ai{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OA> Ao{{.depth = d, .rows = n, .cols = n}};
    std::ranges::generate(Σi, [&] { return brnl(rng) ? 1.0 : 0.0; });
    std::ranges::generate(Ai, [&] { return nrml(rng); });
    for (auto _ : state)
        for (index_t l = 0; l < Ai.num_batches(); ++l)
            batmat::linalg::compress_masks(Ai.batch(l), Σi.batch(l), Ao.batch(l), Σo.batch(l));
    auto flop_cnt                 = static_cast<double>(d * Ai.rows() * Ai.cols()) * brnl.p();
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};

using enum StorageOrder;
#define BM_RANGES()                                                                                \
    DenseRange(1, 127, 1)                                                                          \
        ->DenseRange(128, 255, 16)                                                                 \
        ->DenseRange(256, 511, 32)                                                                 \
        ->DenseRange(512, 1024, 128)                                                               \
        ->MeasureProcessCPUTime()                                                                  \
        ->UseRealTime()

using scalar = batmat::datapar::scalar_abi<real_t>;
using simd8  = batmat::datapar::deduced_abi<real_t, 8>;
using simd4  = batmat::datapar::deduced_abi<real_t, 4>;

#ifdef __AVX512F__
BENCHMARK(compress<simd8, RowMajor>)->BM_RANGES();
BENCHMARK(compress<simd8, ColMajor>)->BM_RANGES();
#endif
BENCHMARK(compress<simd4, RowMajor>)->BM_RANGES();
BENCHMARK(compress<simd4, ColMajor>)->BM_RANGES();
