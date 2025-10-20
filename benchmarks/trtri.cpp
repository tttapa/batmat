#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/trtri.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <guanaqo/mat-view.hpp>
#include <random>

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

template <class Abi, StorageOrder OA = StorageOrder::ColMajor, StorageOrder OB = OA>
void trtri(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OB> D{{.depth = d, .rows = n, .cols = n}};
    std::ranges::generate(A, [&] { return uni(rng); });
    std::ranges::generate(D, [&] { return uni(rng); });
    A.view().add_to_diagonal(10 * static_cast<real_t>(n));
    for (auto _ : state)
        for (index_t l = 0; l < A.num_batches(); ++l)
            if constexpr (decltype(A)::batch_size_type::value == 1) {
                state.PauseTiming();
                copy(tril(A.batch(l)), tril(D.batch(l)));
                state.ResumeTiming();
                guanaqo::blas::xtrtri_LN(D(l));
            } else {
                trtri(tril(A.batch(l)), tril(D.batch(l)));
            }
    auto flop_cnt                 = static_cast<double>(d * total(flops::trtri(A.rows())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
}

using enum StorageOrder;
#define BM_RANGES()                                                                                \
    DenseRange(1, 63, 1)                                                                           \
        ->DenseRange(64, 127, 4)                                                                   \
        ->DenseRange(128, 255, 16)                                                                 \
        ->DenseRange(256, 511, 32)                                                                 \
        ->DenseRange(512, 1024, 128)                                                               \
        ->MeasureProcessCPUTime()                                                                  \
        ->UseRealTime()

using scalar = batmat::datapar::scalar_abi<real_t>;
using simd8  = batmat::datapar::deduced_abi<real_t, 8>;
using simd4  = batmat::datapar::deduced_abi<real_t, 4>;

#ifdef __AVX512F__
BENCHMARK(trtri<simd8, RowMajor>)->BM_RANGES();
BENCHMARK(trtri<simd8, ColMajor>)->BM_RANGES();
#endif
BENCHMARK(trtri<simd4, RowMajor>)->BM_RANGES();
BENCHMARK(trtri<simd4, ColMajor>)->BM_RANGES();
BENCHMARK(trtri<scalar, ColMajor>)->BM_RANGES();
